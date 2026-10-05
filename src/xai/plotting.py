from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_xai_comparison(
    image,
    heatmaps,
    output_path,
    title=None,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    image_array = np.asarray(
        image.convert(
            "RGB"
        )
    )


    method_names = list(
        heatmaps.keys()
    )


    num_columns = (
        1
        + len(
            method_names
        )
    )


    figure = plt.figure(
        figsize=(
            4 * num_columns,
            4,
        )
    )


    # ========================================================
    # Original / classifier input
    # ========================================================

    axis = figure.add_subplot(
        1,
        num_columns,
        1,
    )

    axis.imshow(
        image_array
    )

    axis.set_title(
        "Input"
    )

    axis.axis(
        "off"
    )


    # ========================================================
    # Explanations
    # ========================================================

    for index, method_name in enumerate(
        method_names,
        start=2,
    ):

        axis = figure.add_subplot(
            1,
            num_columns,
            index,
        )

        axis.imshow(
            image_array
        )

        axis.imshow(
            heatmaps[
                method_name
            ],
            cmap="jet",
            alpha=0.45,
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(
            method_name
        )

        axis.axis(
            "off"
        )


    if title is not None:

        figure.suptitle(
            title
        )


    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )