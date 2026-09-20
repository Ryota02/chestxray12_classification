import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def compute_multiclass_metrics(
    y_true,
    y_pred,
    y_probability,
    class_names,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probability = np.asarray(y_probability)

    num_classes = len(class_names)
    labels = list(range(num_classes))

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    y_binary = label_binarize(
        y_true,
        classes=labels,
    )

    try:
        roc_auc_macro = float(
            roc_auc_score(
                y_binary,
                y_probability,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        roc_auc_macro = None

    try:
        pr_auc_macro = float(
            average_precision_score(
                y_binary,
                y_probability,
                average="macro",
            )
        )
    except ValueError:
        pr_auc_macro = None

    # Class-wise metrics
    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    class_metrics = {}

    for class_index, class_name in enumerate(class_names):
        binary_true = (y_true == class_index).astype(np.int32)
        try:
            class_auc = float(
                roc_auc_score(
                    binary_true,
                    y_probability[:, class_index],
                )
            )
        except ValueError:
            class_auc = None

        try:
            class_pr_auc = float(
                average_precision_score(
                    binary_true,
                    y_probability[:, class_index],
                )
            )
        except ValueError:
            class_pr_auc = None

        class_metrics[class_name] = {
            "precision": float(precision_per_class[class_index]),
            "recall": float(recall_per_class[class_index]),
            "f1": float(f1_per_class[class_index]),
            "roc_auc": class_auc,
            "pr_auc": class_pr_auc,
        }

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
    }