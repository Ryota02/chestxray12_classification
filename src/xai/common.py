from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models
from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]

# ============================================================
# checkpoint読み込み
# 12クラスmodel作成
# Original / Masked画像の読み込み
# ImageNet normalization
# ============================================================


# ============================================================
# Build classifier
# ============================================================

def build_classifier(
    backbone,
    num_classes,
):

    if backbone == "resnet50":

        model = models.resnet50(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model


    if backbone == "densenet201":

        model = models.densenet201(
            weights=None
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

        return model


    if backbone == "vit_b_16":

        model = models.vit_b_16(
            weights=None
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )

        return model


    raise ValueError(
        f"Unknown backbone: {backbone}"
    )


# ============================================================
# Load checkpoint
# ============================================================

def _extract_state_dict(
    checkpoint,
):

    if not isinstance(
        checkpoint,
        dict,
    ):

        return checkpoint

    candidates = [
        "model_state_dict",
        "state_dict",
        "model",
        "weights",
    ]

    for key in candidates:

        if key in checkpoint:

            value = checkpoint[
                key
            ]

            if isinstance(
                value,
                dict,
            ):

                return value

    return checkpoint


def _remove_module_prefix(
    state_dict,
):

    new_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len(
                    "module."
                ):
            ]

        new_state_dict[
            key
        ] = value

    return new_state_dict


def load_classifier(
    backbone,
    num_classes,
    checkpoint_path,
    device,
):

    model = build_classifier(
        backbone=backbone,
        num_classes=num_classes,
    )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    state_dict = (
        _extract_state_dict(
            checkpoint
        )
    )

    state_dict = (
        _remove_module_prefix(
            state_dict
        )
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# Find checkpoint
# ============================================================

def find_checkpoint(
    checkpoint_root,
    backbone,
    experiment_name,
    seed,
    filename=None,
):

    run_dir = (
        Path(
            checkpoint_root
        )
        / backbone
        / experiment_name
        / f"seed{seed}"
        / "checkpoints"
    )

    if not run_dir.exists():

        raise FileNotFoundError(
            f"Run directory not found: "
            f"{run_dir}"
        )

    if filename is not None:

        checkpoint_path = (
            run_dir
            / filename
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        return checkpoint_path


    candidates = [
        "best_model.pth",
        "best_model.pt",
        "best_checkpoint.pth",
        "checkpoint_best.pth",
        "best.pth",
        "best.pt",
    ]

    for candidate in candidates:

        path = (
            run_dir
            / candidate
        )

        if path.exists():

            return path


    # fallback
    paths = sorted(
        list(
            run_dir.glob(
                "*.pth"
            )
        )
        +
        list(
            run_dir.glob(
                "*.pt"
            )
        )
    )

    if len(
        paths
    ) == 1:

        return paths[
            0
        ]

    raise FileNotFoundError(
        "Could not determine checkpoint "
        f"in {run_dir}"
    )


# ============================================================
# Find experiment config
# ============================================================

def find_experiment(
    classification_cfg,
    experiment_name,
):

    experiments = (
        classification_cfg[
            "experiment"
        ][
            "experiments"
        ]
    )

    for experiment in experiments:

        if (
            experiment[
                "name"
            ]
            == experiment_name
        ):

            return experiment

    raise KeyError(
        f"Experiment not found: "
        f"{experiment_name}"
    )


# ============================================================
# Mask utilities
# ============================================================

def get_mask_path(
    relative_path,
    mask_root,
    mask_model,
    mask_region,
):

    return (
        Path(
            mask_root
        )
        / mask_model
        / mask_region
        / Path(
            relative_path
        )
    ).with_suffix(
        ".png"
    )


def apply_lung_mask(
    image,
    mask,
):

    image = image.convert(
        "L"
    )

    mask = mask.convert(
        "L"
    )

    if (
        mask.size
        != image.size
    ):

        mask = mask.resize(
            image.size,
            Image.Resampling.NEAREST,
        )

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask_array = (
        np.asarray(
            mask,
            dtype=np.uint8,
        )
        > 127
    )

    output = np.zeros_like(
        image_array,
        dtype=np.uint8,
    )

    output[
        mask_array
    ] = image_array[
        mask_array
    ]

    return Image.fromarray(
        output
    ).convert(
        "RGB"
    )


def moment_standardize(
    image,
    mask,
    clip_z=3.0,
    eps=1.0e-8,
):

    image = image.convert(
        "L"
    )

    mask = mask.convert(
        "L"
    )

    if (
        mask.size
        != image.size
    ):

        mask = mask.resize(
            image.size,
            Image.Resampling.NEAREST,
        )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    mask_array = (
        np.asarray(
            mask,
            dtype=np.uint8,
        )
        > 127
    )

    pixels = image_array[
        mask_array
    ]

    if pixels.size == 0:

        return Image.new(
            "RGB",
            image.size,
            color=0,
        )

    mean = float(
        pixels.mean()
    )

    std = float(
        pixels.std()
    )

    std = max(
        std,
        eps,
    )

    z = (
        image_array
        - mean
    ) / std

    z = np.clip(
        z,
        -clip_z,
        clip_z,
    )

    normalized = (
        z
        + clip_z
    ) / (
        2.0
        * clip_z
    )

    normalized[
        ~mask_array
    ] = 0.0

    normalized = (
        normalized
        * 255.0
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        normalized
    ).convert(
        "RGB"
    )


# ============================================================
# Load image according to experiment
# ============================================================

def load_experiment_image(
    relative_path,
    classification_cfg,
    experiment_cfg,
):

    dataset_cfg = (
        classification_cfg[
            "dataset"
        ]
    )

    preprocessing_cfg = (
        classification_cfg.get(
            "preprocessing",
            {},
        )
    )

    data_root = Path(
        dataset_cfg[
            "root"
        ]
    )

    image_path = (
        data_root
        / relative_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    mode = experiment_cfg[
        "mode"
    ]

    if mode == "original":

        return image


    mask_model = (
        experiment_cfg[
            "mask_model"
        ]
    )

    mask_region = (
        experiment_cfg.get(
            "mask_region",
            "whole_lung",
        )
    )

    mask_root = Path(
        preprocessing_cfg[
            "mask_root"
        ]
    )

    mask_path = get_mask_path(
        relative_path=relative_path,
        mask_root=mask_root,
        mask_model=mask_model,
        mask_region=mask_region,
    )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask not found: "
            f"{mask_path}"
        )

    mask = Image.open(
        mask_path
    )


    if mode == "lung_only":

        return apply_lung_mask(
            image=image,
            mask=mask,
        )


    if mode == "moment_standardized":

        return moment_standardize(
            image=image,
            mask=mask,
            clip_z=float(
                preprocessing_cfg.get(
                    "clip_z",
                    3.0,
                )
            ),
            eps=float(
                preprocessing_cfg.get(
                    "eps",
                    1.0e-8,
                )
            ),
        )


    raise ValueError(
        f"Unknown preprocessing mode: "
        f"{mode}"
    )


# ============================================================
# Spatial preprocessing
# ============================================================

def prepare_display_image(
    image,
    classification_cfg,
):

    augmentation_cfg = (
        classification_cfg[
            "augmentation"
        ]
    )

    resize_size = int(
        augmentation_cfg.get(
            "resize_size",
            256,
        )
    )

    image_size = int(
        augmentation_cfg.get(
            "image_size",
            224,
        )
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    resize_size,
                    resize_size,
                )
            ),

            transforms.CenterCrop(
                image_size
            ),
        ]
    )

    return transform(
        image
    )


# ============================================================
# PIL -> model tensor
# ============================================================

def image_to_tensor(
    image,
    device,
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        device
    )


# ============================================================
# Batch image conversion for LIME
# ============================================================

def numpy_batch_to_tensor(
    images,
    device,
):

    tensors = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    for image in images:

        image = np.asarray(
            image
        )

        if (
            image.dtype
            != np.uint8
        ):

            if (
                image.max()
                <= 1.0
            ):

                image = (
                    image
                    * 255.0
                )

            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        pil_image = Image.fromarray(
            image
        ).convert(
            "RGB"
        )

        tensors.append(
            transform(
                pil_image
            )
        )

    batch = torch.stack(
        tensors,
        dim=0,
    )

    return batch.to(
        device
    )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    input_tensor,
):

    logits = model(
        input_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    prediction = int(
        probabilities.argmax(
            dim=1
        ).item()
    )

    return (
        prediction,
        probabilities[
            0
        ].detach().cpu().numpy(),
    )