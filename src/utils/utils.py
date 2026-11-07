import numpy as np
import torch


def calculate_class_weights(
    train_dataset,
    num_classes,
    device,
):

    labels = (
        train_dataset
        .dataframe["label"]
        .astype(int)
        .to_numpy()
    )

    class_counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    total_samples = int(
        class_counts.sum()
    )

    class_weights = (
        total_samples
        / (num_classes * class_counts)
    )

    print(
        "[INFO] Train class counts:",
        class_counts.tolist(),
    )

    print(
        "[INFO] Class weights:",
        class_weights.tolist(),
    )

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )