import numpy as np

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)


# ============================================================
# Target layer
# ============================================================

def get_target_layer(
    model,
    backbone,
):

    if backbone == "resnet50":

        return model.layer4[
            -1
        ]


    if backbone == "densenet201":

        return (
            model
            .features
            .denseblock4
        )


    raise ValueError(
        "CAM currently supports "
        "resnet50 and densenet201."
    )


# ============================================================
# CAM
# ============================================================

def explain_cam(
    model,
    input_tensor,
    target_class,
    backbone,
    method="gradcam",
):

    target_layer = (
        get_target_layer(
            model=model,
            backbone=backbone,
        )
    )

    if method == "gradcam":

        cam_class = GradCAM

    elif (
        method
        == "gradcam_plus_plus"
    ):

        cam_class = (
            GradCAMPlusPlus
        )

    else:

        raise ValueError(
            f"Unknown CAM method: "
            f"{method}"
        )


    targets = [
        ClassifierOutputTarget(
            target_class
        )
    ]


    with cam_class(
        model=model,
        target_layers=[
            target_layer
        ],
    ) as cam:

        grayscale_cam = cam(
            input_tensor=(
                input_tensor
            ),
            targets=targets,
        )

    heatmap = (
        grayscale_cam[
            0
        ]
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    return heatmap