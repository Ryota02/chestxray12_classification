import numpy as np
import torch

from lime import lime_image

from src.xai.common import (
    numpy_batch_to_tensor,
)


def explain_lime(
    model,
    image,
    target_class,
    device,
    num_samples=500,
    num_features=10,
):

    explainer = (
        lime_image.LimeImageExplainer()
    )


    # ========================================================
    # Classifier function
    # ========================================================

    def classifier_fn(
        images,
    ):

        batch = numpy_batch_to_tensor(
            images=images,
            device=device,
        )

        with torch.no_grad():

            logits = model(
                batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )


        return (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )


    # ========================================================
    # Image
    # ========================================================

    image_array = np.asarray(
        image.convert(
            "RGB"
        ),
        dtype=np.uint8,
    )


    # ========================================================
    # Explain predicted class
    # ========================================================

    explanation = (
        explainer.explain_instance(
            image_array,
            classifier_fn,
            labels=[
                int(
                    target_class
                )
            ],
            top_labels=None,
            hide_color=0,
            num_samples=int(
                num_samples
            ),
        )
    )


    segments = (
        explanation.segments
    )


    local_exp = (
        explanation
        .local_exp
        .get(
            int(
                target_class
            ),
            [],
        )
    )


    # ========================================================
    # Sort by positive contribution
    #
    # We want regions that SUPPORT the predicted class.
    # ========================================================

    positive_features = [
        (
            segment_id,
            weight,
        )
        for (
            segment_id,
            weight,
        )
        in local_exp
        if weight > 0
    ]


    positive_features.sort(
        key=lambda x: x[1],
        reverse=True,
    )


    positive_features = (
        positive_features[
            :int(
                num_features
            )
        ]
    )


    heatmap = np.zeros(
        segments.shape,
        dtype=np.float32,
    )


    for (
        segment_id,
        weight,
    ) in positive_features:

        heatmap[
            segments
            == segment_id
        ] = float(
            weight
        )


    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )


    return heatmap