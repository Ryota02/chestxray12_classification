import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Settings
# ============================================================

# output will be root / "summary"
root = Path(
    "outputs/experimentF"
)

backbones = [
    "resnet50",
    # "densenet201",
    # "vit_b_16",
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
    # 43,
    # 44,
    # 45,
    # 46,
]

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro",
    "pr_auc_macro",
]


# ============================================================
# Load results
# ============================================================

rows = []


for backbone in backbones:

    for method in methods:

        for seed in seeds:

            metrics_path = (
                root
                / backbone
                / method
                / f"seed{seed}"
                / "test_metrics.json"
            )

            if not metrics_path.exists():

                print(
                    "[WARNING] Not found:",
                    metrics_path,
                )

                continue

            with metrics_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                metrics = json.load(
                    f
                )

            row = {
                "backbone":
                    backbone,

                "method":
                    method,

                "seed":
                    seed,
            }

            for metric in metric_names:

                row[
                    metric
                ] = metrics.get(
                    metric,
                    np.nan,
                )

            rows.append(
                row
            )


# ============================================================
# Raw dataframe
# ============================================================

raw_df = pd.DataFrame(
    rows
)


if len(raw_df) == 0:

    raise RuntimeError(
        "No result files were found."
    )


print(
    "\n"
    "========================================"
)

print(
    "Raw results"
)

print(
    "========================================"
)

print(
    raw_df.to_string(
        index=False
    )
)


# ============================================================
# Mean / SD
# ============================================================

summary_rows = []


for backbone in backbones:

    for method in methods:

        subset = raw_df[
            (
                raw_df[
                    "backbone"
                ]
                == backbone
            )
            &
            (
                raw_df[
                    "method"
                ]
                == method
            )
        ]

        if len(subset) == 0:
            continue

        summary_row = {
            "backbone":
                backbone,

            "method":
                method,

            "n_seeds":
                len(subset),
        }

        for metric in metric_names:

            values = (
                subset[
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

                mean = (
                    values.mean()
                )

                if len(values) > 1:

                    sd = values.std(
                        ddof=1
                    )

                else:

                    sd = np.nan

            summary_row[
                f"{metric}_mean"
            ] = mean

            summary_row[
                f"{metric}_sd"
            ] = sd

            if len(values) > 1:

                summary_row[
                    f"{metric}_mean_sd"
                ] = (
                    f"{mean:.4f} "
                    f"± "
                    f"{sd:.4f}"
                )

            else:

                summary_row[
                    f"{metric}_mean_sd"
                ] = (
                    f"{mean:.4f}"
                )

        summary_rows.append(
            summary_row
        )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Print summary
# ============================================================

print(
    "\n"
    "========================================"
)

print(
    "Mean ± SD"
)

print(
    "========================================"
)


for backbone in backbones:

    print(
        "\n"
        "########################################"
    )

    print(
        backbone
    )

    print(
        "########################################"
    )

    backbone_df = summary_df[
        summary_df[
            "backbone"
        ]
        == backbone
    ]

    for _, row in backbone_df.iterrows():

        print(
            f"\n[{row['method']}]"
        )

        print(
            f"Seeds: "
            f"{int(row['n_seeds'])}"
        )

        for metric in metric_names:

            print(
                f"{metric:20s}: "
                f"{row[f'{metric}_mean_sd']}"
            )


# ============================================================
# Save
# ============================================================

output_dir = root / "summary"

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

raw_output_path = (
    output_dir
    / "classification_results_raw.csv"
)

summary_output_path = (
    output_dir
    / "classification_results_summary.csv"
)


raw_df.to_csv(
    raw_output_path,
    index=False,
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)


print(
    "\n"
    "========================================"
)

print(
    "Saved"
)

print(
    "========================================"
)

print(
    "Raw results:"
)

print(
    raw_output_path
)

print(
    "\nSummary:"
)

print(summary_output_path)