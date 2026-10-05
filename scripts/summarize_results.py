import json
from pathlib import Path
import numpy as np

# Running this code can show you average of each evaluation metric from seed42 to seed45

root = Path(
    "outputs/classification/resnet50/cxas_moment"
)

seeds = [
    42,
    43,
    44,
    45,
]

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro",
    "pr_auc_macro",
]

results = {
    metric: []
    for metric in metric_names
}

for seed in seeds:

    path = (
        root
        / f"seed{seed}"
        / "test_metrics.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    print(
        f"\nSeed {seed}"
    )

    for metric in metric_names:

        value = metrics[metric]

        results[metric].append(
            value
        )

        print(
            f"{metric:20s}: "
            f"{value:.4f}"
        )


print(
    "\n=============================="
)
print(
    "Mean ± SD"
)
print(
    "=============================="
)

for metric in metric_names:

    values = np.asarray(
        results[metric],
        dtype=float,
    )

    print(
        f"{metric:20s}: "
        f"{values.mean():.4f} "
        f"± "
        f"{values.std(ddof=1):.4f}"
    )