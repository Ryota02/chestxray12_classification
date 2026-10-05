import numpy as np
import shap
import torch
import torch.nn as nn


class TargetClassModel(
    nn.Module
):

    def __init__(
        self,
        model,
        target_class,
    ):

        super().__init__()

        self.model = model

        self.target_class = int(
            target_class
        )


    def forward(
        self,
        x,
    ):

        logits = self.model(
            x
        )

        return logits[
            :,
            self.target_class:
            self.target_class + 1
        ]


def _convert_shap_to_heatmap(
    shap_values,
):

    if isinstance(
        shap_values,
        list,
    ):

        shap_values = (
            shap_values[
                0
            ]
        )

    values = np.asarray(
        shap_values
    )

    values = np.squeeze(
        values
    )


    # C x H x W
    if (
        values.ndim == 3
        and
        values.shape[
            0
        ] in {
            1,
            3,
        }
    ):

        heatmap = np.mean(
            np.abs(
                values
            ),
            axis=0,
        )


    # H x W x C
    elif (
        values.ndim == 3
        and
        values.shape[
            -1
        ] in {
            1,
            3,
        }
    ):

        heatmap = np.mean(
            np.abs(
                values
            ),
            axis=-1,
        )


    elif values.ndim == 2:

        heatmap = np.abs(
            values
        )


    else:

        raise RuntimeError(
            "Unexpected SHAP shape: "
            f"{values.shape}"
        )


    heatmap = heatmap.astype(
        np.float32
    )

    heatmap -= heatmap.min()

    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )

    return heatmap


def explain_shap(
    model,
    input_tensor,
    target_class,
    nsamples=64,
    background_size=4,
):

    wrapped_model = (
        TargetClassModel(
            model=model,
            target_class=target_class,
        )
    )

    wrapped_model.eval()


    background = torch.zeros(
        (
            int(
                background_size
            ),
            *input_tensor.shape[
                1:
            ],
        ),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


    explainer = (
        shap.GradientExplainer(
            wrapped_model,
            background,
        )
    )


    shap_values = (
        explainer.shap_values(
            input_tensor,
            nsamples=int(
                nsamples
            ),
        )
    )


    heatmap = (
        _convert_shap_to_heatmap(
            shap_values
        )
    )

    return heatmap