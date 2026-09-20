from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    output_path,
):
    confusion_matrix = np.asarray(confusion_matrix)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    image = ax.imshow(confusion_matrix)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(
        class_names,
        rotation=90,
    )
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)