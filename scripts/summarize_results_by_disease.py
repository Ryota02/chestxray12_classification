from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Settings
# ============================================================

root = Path(
    "outputs/experimentA"
)

output_dir = Path(
    "outputs/experimentA/summary"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


backbones = [
    "resnet50",
    "densenet201",
    "vit_b_16",
]


methods = [
    "original",
    "cxas_lung_only",
    "pspnet_lung_only",
    "niaid_lung_only",
    "cxas_moment",
    "pspnet_moment",
    "niaid_moment",
]


seeds = [
    42,
    43,
    44,
    45,
    46,
]


class_names = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]


metric_names = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


# ============================================================
# Calculate class-wise metrics for one seed
# ============================================================

def calculate_class_metrics(
    dataframe,
    class_names,
):

    y_true = dataframe[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    y_pred = dataframe[
        "pred_label"
    ].to_numpy(
        dtype=int
    )

    rows = []

    for class_index, class_name in enumerate(
        class_names
    ):

        # ----------------------------------------------------
        # One-vs-rest labels
        # ----------------------------------------------------

        y_true_binary = (
            y_true
            == class_index
        ).astype(
            int
        )

        y_pred_binary = (
            y_pred
            == class_index
        ).astype(
            int
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability_column = (
            f"prob_{class_name}"
        )

        if probability_column not in dataframe.columns:

            raise KeyError(
                f"Missing probability column: "
                f"{probability_column}"
            )

        y_probability = dataframe[
            probability_column
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = precision_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        )

        # ----------------------------------------------------
        # Recall
        #
        # This corresponds to class-wise correct rate:
        #
        # TP / (TP + FN)
        # ----------------------------------------------------

        recall = recall_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        )

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1 = f1_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0,
        )

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if len(
            np.unique(
                y_true_binary
            )
        ) == 2:

            roc_auc = roc_auc_score(
                y_true_binary,
                y_probability,
            )

            pr_auc = average_precision_score(
                y_true_binary,
                y_probability,
            )

        else:

            roc_auc = np.nan
            pr_auc = np.nan

        # ----------------------------------------------------
        # Number of samples
        # ----------------------------------------------------

        support = int(
            y_true_binary.sum()
        )

        rows.append(
            {
                "class_name":
                    class_name,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "roc_auc":
                    roc_auc,

                "pr_auc":
                    pr_auc,

                "support":
                    support,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Load all results
# ============================================================

all_rows = []


for backbone in backbones:

    for method in methods:

        for seed in seeds:

            prediction_path = (
                root
                / backbone
                / method
                / f"seed{seed}"
                / "test_predictions.csv"
            )

            if not prediction_path.exists():

                print(
                    "[WARNING] Missing:",
                    prediction_path,
                )

                continue

            dataframe = pd.read_csv(
                prediction_path
            )

            class_metrics = (
                calculate_class_metrics(
                    dataframe=dataframe,
                    class_names=class_names,
                )
            )

            class_metrics.insert(
                0,
                "seed",
                seed,
            )

            class_metrics.insert(
                0,
                "method",
                method,
            )

            class_metrics.insert(
                0,
                "backbone",
                backbone,
            )

            all_rows.append(
                class_metrics
            )


if len(all_rows) == 0:

    raise RuntimeError(
        "No prediction files were found."
    )


# ============================================================
# Raw results
# ============================================================

raw_df = pd.concat(
    all_rows,
    ignore_index=True,
)


raw_output_path = (
    output_dir
    / "disease_results_raw.csv"
)

raw_df.to_csv(
    raw_output_path,
    index=False,
)


# ============================================================
# Mean / SD
# ============================================================

summary_rows = []


group_columns = [
    "backbone",
    "method",
    "class_name",
]


for (
    backbone,
    method,
    class_name,
), group in raw_df.groupby(
    group_columns,
    sort=False,
):

    row = {
        "backbone":
            backbone,

        "method":
            method,

        "class_name":
            class_name,

        "n_seeds":
            group[
                "seed"
            ].nunique(),
    }

    # --------------------------------------------------------
    # Support
    # --------------------------------------------------------

    support_values = group[
        "support"
    ].to_numpy(
        dtype=float
    )

    row[
        "support_mean"
    ] = support_values.mean()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    for metric in metric_names:

        values = (
            group[
                metric
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(values) == 0:

            mean = np.nan
            sd = np.nan

        else:

            mean = values.mean()

            if len(values) > 1:

                sd = values.std(
                    ddof=1
                )

            else:

                sd = np.nan

        row[
            f"{metric}_mean"
        ] = mean

        row[
            f"{metric}_sd"
        ] = sd

        if np.isnan(mean):

            formatted = "NaN"

        elif np.isnan(sd):

            formatted = (
                f"{mean:.4f}"
            )

        else:

            formatted = (
                f"{mean:.4f} "
                f"± "
                f"{sd:.4f}"
            )

        row[
            f"{metric}_mean_sd"
        ] = formatted

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save full summary
# ============================================================

summary_output_path = (
    output_dir
    / "disease_results_summary.csv"
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)


# ============================================================
# Create simple presentation table
# ============================================================

display_columns = [
    "backbone",
    "method",
    "class_name",
    "precision_mean_sd",
    "recall_mean_sd",
    "f1_mean_sd",
    "roc_auc_mean_sd",
    "pr_auc_mean_sd",
]


display_df = summary_df[
    display_columns
].copy()


display_df = display_df.rename(
    columns={
        "class_name":
            "Disease",

        "precision_mean_sd":
            "Precision",

        "recall_mean_sd":
            "Recall",

        "f1_mean_sd":
            "F1",

        "roc_auc_mean_sd":
            "ROC-AUC",

        "pr_auc_mean_sd":
            "PR-AUC",
    }
)


display_output_path = (
    output_dir
    / "disease_results_mean_sd.csv"
)

display_df.to_csv(
    display_output_path,
    index=False,
)


# ============================================================
# Print
# ============================================================

for backbone in backbones:

    print()
    print(
        "=" * 100
    )

    print(
        backbone
    )

    print(
        "=" * 100
    )

    for method in methods:

        subset = display_df[
            (
                display_df[
                    "backbone"
                ]
                == backbone
            )
            &
            (
                display_df[
                    "method"
                ]
                == method
            )
        ]

        if len(subset) == 0:
            continue

        print()
        print(
            f"[{method}]"
        )

        print(
            subset[
                [
                    "Disease",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC-AUC",
                    "PR-AUC",
                ]
            ].to_string(
                index=False
            )
        )


print()
print(
    "========================================"
)

print(
    "Saved:"
)

print(
    raw_output_path
)

print(
    summary_output_path
)

print(
    display_output_path
)