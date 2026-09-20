from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

from PIL import Image
from scipy import ndimage


def _find_target_index(targets, target_name):
    target_name = str(target_name).lower()

    for index, name in enumerate(targets):
        if str(name).lower() == target_name:
            return index

    raise ValueError(
        f"Target '{target_name}' was not found.\n"
        f"Available targets:\n{targets}"
    )


def _to_probability(output):
    output = output.float()

    output_min = float(output.min().item())
    output_max = float(output.max().item())

    if output_min < 0.0 or output_max > 1.0:
        output = torch.sigmoid(output)

    return output.clamp(0.0, 1.0)


def _prepare_xrv_input(
    image_path,
    input_size,
    device,
):
    image_path = Path(image_path)

    image = Image.open(image_path).convert("L")

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    original_height, original_width = image_array.shape

    image_array = xrv.utils.normalize(
        image_array,
        maxval=255,
    )

    side = min(
        original_height,
        original_width,
    )

    y0 = (original_height - side) // 2
    x0 = (original_width - side) // 2

    cropped = image_array[
        y0:y0 + side,
        x0:x0 + side,
    ]

    tensor = (
        torch.from_numpy(cropped)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=(int(input_size), int(input_size)),
        mode="bilinear",
        align_corners=False,
    )

    tensor = tensor.to(device)

    metadata = {
        "height": int(original_height),
        "width": int(original_width),
        "side": int(side),
        "x0": int(x0),
        "y0": int(y0),
    }

    return tensor, metadata


def _restore_probability(
    probability,
    metadata,
):
    side = int(metadata["side"])
    original_height = int(metadata["height"])
    original_width = int(metadata["width"])

    x0 = int(metadata["x0"])
    y0 = int(metadata["y0"])

    probability = (
        probability
        .unsqueeze(0)
        .unsqueeze(0)
    )

    probability = F.interpolate(
        probability,
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    )

    probability = (
        probability[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    restored = np.zeros(
        (original_height, original_width),
        dtype=np.float32,
    )

    restored[
        y0:y0 + side,
        x0:x0 + side,
    ] = probability

    return restored


def _postprocess_single_lung(
    mask,
    fill_holes=True,
    min_component_area_ratio=0.002,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if not mask.any():
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    labeled, num_components = ndimage.label(mask)

    if num_components == 0:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    image_area = mask.shape[0] * mask.shape[1]

    minimum_area = (
        image_area
        * float(min_component_area_ratio)
    )

    components = []

    for component_id in range(1, num_components + 1):
        area = int(
            np.sum(labeled == component_id)
        )

        if area >= minimum_area:
            components.append(
                (component_id, area)
            )

    if not components:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    components.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    largest_component_id = components[0][0]

    output = (
        labeled
        == largest_component_id
    )

    if fill_holes:
        output = ndimage.binary_fill_holes(output)

    return output.astype(bool)


def _to_uint8_mask(mask):
    return np.asarray(
        mask,
        dtype=np.uint8,
    ) * 255


class XRVLungSegmenter:
    def __init__(
        self,
        model_name,
        device,
        input_size=512,
        threshold=0.5,
        fill_holes=True,
        min_component_area_ratio=0.002,
    ):
        self.model_name = str(model_name)
        self.device = device
        self.input_size = int(input_size)
        self.threshold = float(threshold)
        self.fill_holes = bool(fill_holes)
        self.min_component_area_ratio = float(
            min_component_area_ratio
        )

        if self.model_name == "xrv_pspnet":
            self.model = (
                xrv.baseline_models
                .chestx_det
                .PSPNet()
            )

            self.left_target = "Left Lung"
            self.right_target = "Right Lung"

        elif self.model_name == "xrv_cxas":
            self.model = (
                xrv.baseline_models
                .chestx_anatomy
                .UNetResNet50()
            )

            self.left_target = "left lung"
            self.right_target = "right lung"

        else:
            raise ValueError(
                f"Unknown segmentation model: "
                f"{self.model_name}"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        if not hasattr(self.model, "targets"):
            raise AttributeError(
                f"{self.model_name} does not provide "
                "'model.targets'."
            )

        self.left_index = _find_target_index(
            targets=self.model.targets,
            target_name=self.left_target,
        )

        self.right_index = _find_target_index(
            targets=self.model.targets,
            target_name=self.right_target,
        )

        print(
            "[INFO] Segmentation model:",
            self.model_name,
        )
        print(
            "[INFO] Left lung channel:",
            self.left_index,
        )
        print(
            "[INFO] Right lung channel:",
            self.right_index,
        )


    @torch.no_grad()
    def _predict_probabilities(
        self,
        image_path,
    ):
        input_tensor, metadata = _prepare_xrv_input(
            image_path=image_path,
            input_size=self.input_size,
            device=self.device,
        )

        output = self.model(input_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim != 4:
            raise RuntimeError(
                f"Unexpected segmentation output shape: "
                f"{tuple(output.shape)}"
            )

        output = _to_probability(output)

        left_probability = output[
            0,
            self.left_index,
        ]

        right_probability = output[
            0,
            self.right_index,
        ]

        left_probability = _restore_probability(
            probability=left_probability,
            metadata=metadata,
        )

        right_probability = _restore_probability(
            probability=right_probability,
            metadata=metadata,
        )

        return (
            left_probability,
            right_probability,
        )


    @torch.no_grad()
    def predict_masks(
        self,
        image_path,
    ):
        left_probability, right_probability = (
            self._predict_probabilities(image_path)
        )

        left_mask = (
            left_probability
            >= self.threshold
        )

        right_mask = (
            right_probability
            >= self.threshold
        )

        left_mask = _postprocess_single_lung(
            mask=left_mask,
            fill_holes=self.fill_holes,
            min_component_area_ratio=(
                self.min_component_area_ratio
            ),
        )

        right_mask = _postprocess_single_lung(
            mask=right_mask,
            fill_holes=self.fill_holes,
            min_component_area_ratio=(
                self.min_component_area_ratio
            ),
        )

        whole_mask = np.logical_or(
            left_mask,
            right_mask,
        )

        left_mask = _to_uint8_mask(left_mask)
        right_mask = _to_uint8_mask(right_mask)
        whole_mask = _to_uint8_mask(whole_mask)

        if self.model_name == "xrv_pspnet":
            return {
                "left_lung": left_mask, 
                "right_lung": right_mask, 
                "whole_lung": whole_mask,
            }

        if self.model_name == "xrv_cxas":
            return {
                "left_lung": left_mask,
                "right_lung": right_mask,
                "whole_lung": whole_mask,
            }

        raise RuntimeError(
            f"Unexpected segmentation model: "
            f"{self.model_name}"
        )


def build_lung_segmenter(
    model_name,
    cfg,
    device,
):
    segmentation_cfg = cfg["segmentation"]
    models_cfg = segmentation_cfg["models"]

    if model_name not in models_cfg:
        raise KeyError(
            f"Model '{model_name}' is not defined in "
            "segmentation.models."
        )

    model_cfg = models_cfg[model_name]

    postprocess_cfg = segmentation_cfg.get(
        "postprocess",
        {},
    )

    return XRVLungSegmenter(
        model_name=model_name,
        device=device,
        input_size=int(
            segmentation_cfg.get("input_size", 512)
        ),
        threshold=float(
            model_cfg.get("threshold", 0.5)
        ),
        fill_holes=bool(
            postprocess_cfg.get("fill_holes", True)
        ),
        min_component_area_ratio=float(
            postprocess_cfg.get(
                "min_component_area_ratio",
                0.002,
            )
        ),
    )