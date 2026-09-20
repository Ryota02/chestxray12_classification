```python
import argparse
import copy
import json
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.classification.dataset import (
    build_classification_datasets,
    build_classification_loaders,
)
from src.classification.model import build_classifier
from src.classification.engine import (
    fit_classifier,
    evaluate_classifier,
)
from src.classification.plots import plot_confusion_matrix
from src.utils.config import load_config
from src.utils.seed import get_device, set_seed


def run_single_experiment(
    base_cfg,
    seed,
    backbone,
    backbone_cfg,
    experiment_cfg,
    device,
):
    cfg = copy.deepcopy(base_cfg)
    set_seed(seed)

    experiment_name = experiment_cfg["name"]
    mode = experiment_cfg["mode"]
    mask_model = experiment_cfg.get("mask_model")
    mask_region = experiment_cfg.get("mask_region")

    cfg["train"]["batch_size"] = int(
        backbone_cfg.get("batch_size", 32)
    )
    cfg["train"]["lr"] = float(
        backbone_cfg.get("lr", 1e-4)
    )

    class_names = list(cfg["dataset"]["classes"])

    manifest_path = (
        Path(cfg["dataset"]["manifest_dir"])
        / f"seed{seed}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    output_dir = (
        Path(cfg["output"]["root"])
        / backbone
        / experiment_name
        / f"seed{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n==========================================")
    print("[RUN]")
    print("Backbone:", backbone)
    print("Experiment:", experiment_name)
    print("Mode:", mode)
    print("Mask model:", mask_model)
    print("Mask region:", mask_region)
    print("Seed:", seed)
    print("==========================================")

    # Dataset
    datasets = build_classification_datasets(
        cfg=cfg,
        manifest=manifest_path,
        experiment_cfg=experiment_cfg,
    )

    loaders = build_classification_loaders(
        datasets=datasets,
        cfg=cfg,
        seed=seed,
    )

    # Model
    model = build_classifier(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=bool(
            backbone_cfg.get("pretrained", True)
        ),
    ).to(device)

    metadata = {
        "seed": seed,
        "backbone": backbone,
        "experiment_name": experiment_name,
        "mode": mode,
        "mask_model": mask_model,
        "mask_region": mask_region,
        "manifest": str(manifest_path),
    }

    # Train
    fit_classifier(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
        metadata=metadata,
    )

    # Best model
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Test
    result = evaluate_classifier(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names=class_names,
    )

    metrics = result["metrics"]

    metrics_path = (
        output_dir
        / "test_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # Predictions
    probabilities = np.asarray(
        result["y_probability"]
    )

    test_dataset = datasets["test"]
    rows = []

    for result_index, dataset_index in enumerate(
        result["sample_indices"]
    ):
        sample = (
            test_dataset
            .dataframe
            .iloc[dataset_index]
        )

        row = {
            "relative_path": sample["relative_path"],
            "true_label": result["y_true"][result_index],
            "pred_label": result["y_pred"][result_index],
        }

        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(
                probabilities[
                    result_index,
                    class_index,
                ]
            )

        rows.append(row)

    predictions_path = (
        output_dir
        / "test_predictions.csv"
    )

    pd.DataFrame(rows).to_csv(
        predictions_path,
        index=False,
    )

    # Confusion matrix
    plot_confusion_matrix(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
    )

    print("[TEST] Accuracy:", metrics["accuracy"])
    print("[TEST] Macro F1:", metrics["f1_macro"])
    print("[TEST] Macro AUC:", metrics["roc_auc_macro"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = get_device(
        cfg.get("device", {}).get("require_cuda", True)
    )

    seeds = [
        int(seed)
        for seed in cfg["experiment"]["seeds"]
    ]

    experiments = [
        experiment
        for experiment in cfg["experiment"]["experiments"]
        if experiment.get("enabled", False)
    ]

    backbones = {
        name: backbone_cfg
        for name, backbone_cfg in cfg["models"].items()
        if backbone_cfg.get("enabled", False)
    }

    print("[INFO] Seeds:", seeds)
    print("[INFO] Backbones:", list(backbones.keys()))
    print(
        "[INFO] Experiments:",
        [experiment["name"] for experiment in experiments],
    )

    for seed in seeds:
        for backbone, backbone_cfg in backbones.items():
            for experiment_cfg in experiments:
                run_single_experiment(
                    base_cfg=cfg,
                    seed=seed,
                    backbone=backbone,
                    backbone_cfg=backbone_cfg,
                    experiment_cfg=experiment_cfg,
                    device=device,
                )


if __name__ == "__main__":
    main()