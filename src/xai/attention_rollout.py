import math

import numpy as np
import torch
import torch.nn.functional as F


def explain_attention_rollout(
    model,
    input_tensor,
):

    attentions = []

    handles = []


    # ========================================================
    # Hook
    # ========================================================

    def create_hook():

        def hook(
            module,
            inputs,
        ):

            x = inputs[
                0
            ]

            # torchvision ViT applies
            # ln_1 before self-attention
            x_norm = module.ln_1(
                x
            )

            with torch.no_grad():

                _, attention = (
                    module.self_attention(
                        x_norm,
                        x_norm,
                        x_norm,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                )

            attentions.append(
                attention.detach()
            )

        return hook


    # ========================================================
    # Register hooks
    # ========================================================

    for block in (
        model
        .encoder
        .layers
    ):

        handle = (
            block.register_forward_pre_hook(
                create_hook()
            )
        )

        handles.append(
            handle
        )


    # ========================================================
    # Forward
    # ========================================================

    try:

        with torch.no_grad():

            model(
                input_tensor
            )

    finally:

        for handle in handles:

            handle.remove()


    if len(
        attentions
    ) == 0:

        raise RuntimeError(
            "No ViT attention maps "
            "were collected."
        )


    # ========================================================
    # Attention rollout
    # ========================================================

    joint_attention = None


    for attention in attentions:

        # B x Heads x N x N
        attention = attention.mean(
            dim=1
        )

        batch_size = (
            attention.shape[
                0
            ]
        )

        num_tokens = (
            attention.shape[
                -1
            ]
        )

        identity = torch.eye(
            num_tokens,
            device=(
                attention.device
            ),
            dtype=(
                attention.dtype
            ),
        )

        identity = identity.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1,
        )

        attention = (
            attention
            + identity
        )

        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True,
            )
        )


        if joint_attention is None:

            joint_attention = (
                attention
            )

        else:

            joint_attention = torch.bmm(
                attention,
                joint_attention,
            )


    # ========================================================
    # CLS -> image patches
    # ========================================================

    cls_attention = (
        joint_attention[
            0,
            0,
            1:
        ]
    )


    number_of_patches = int(
        cls_attention.numel()
    )

    grid_size = int(
        math.sqrt(
            number_of_patches
        )
    )


    if (
        grid_size
        * grid_size
        != number_of_patches
    ):

        raise RuntimeError(
            "ViT patch count is not "
            f"square: {number_of_patches}"
        )


    heatmap = (
        cls_attention
        .reshape(
            1,
            1,
            grid_size,
            grid_size,
        )
    )


    output_height = int(
        input_tensor.shape[
            -2
        ]
    )

    output_width = int(
        input_tensor.shape[
            -1
        ]
    )


    heatmap = F.interpolate(
        heatmap,
        size=(
            output_height,
            output_width,
        ),
        mode="bilinear",
        align_corners=False,
    )


    heatmap = (
        heatmap[
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
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