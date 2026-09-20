import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,

    densenet201,
    DenseNet201_Weights,

    vit_b_16,
    ViT_B_16_Weights,
)


def build_classifier(
    backbone,
    num_classes,
    pretrained=True,
):
    backbone = backbone.lower()

    # ========================================================
    # ResNet50
    # ========================================================
    if backbone == "resnet50":
        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = resnet50(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    # ========================================================
    # DenseNet201
    # ========================================================

    elif backbone== "densenet201":
        weights = (
            DenseNet201_Weights.DEFAULT
            if pretrained
            else None
        )

        model = densenet201(weights=weights)
        in_features = (
            model
            .classifier
            .in_features
        )

        model.classifier = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

    # ========================================================
    # ViT-B/16
    # ========================================================

    elif backbone in {"vit", "vit_b_16"}:
        weights = (
            ViT_B_16_Weights.DEFAULT
            if pretrained
            else None
        )

        model = vit_b_16(weights=weights)

        in_features = (
            model
            .heads
            .head
            .in_features
        )

        model.heads.head = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

    else:
        raise ValueError(
            f"Unknown backbone: "
            f"{backbone}"
        )

    return model