import argparse
import json
import sys

from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.classification.dataset import (
    build_classification_datasets,
    build_classification_loaders,
)
from src.classification.model import build_classifier
from src.classification.engine import evaluate_classifier
from src.utils.config import load_config
from src.utils.seed import get_device, set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(args.seed)

    device = get_device(
        cfg.get("device", {}).get("require_cuda", True)
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    backbone = checkpoint["backbone"]
    mode = checkpoint["mode"]
    mask_model = checkpoint.get("mask_model")
    class_names = checkpoint["class_names"]
    manifest_path = Path(checkpoint["manifest"])

    datasets = build_classification_datasets(
        cfg=cfg,
        manifest=manifest_path,
        mode=mode,
        mask_model=mask_model,
    )

    loaders = build_classification_loaders(
        datasets=datasets,
        cfg=cfg,
        seed=args.seed,
    )

    model = build_classifier(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    result = evaluate_classifier(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names=class_names,
    )

    print(
        json.dumps(
            result["metrics"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()