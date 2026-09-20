import argparse
import itertools
import sys

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.utils.config import load_config
from src.utils.image import find_images


def dice_score(mask_a, mask_b, eps=1e-8):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    denominator = mask_a.sum() + mask_b.sum()

    return float(
        (2.0 * intersection + eps)
        / (denominator + eps)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_root = Path(cfg["dataset"]["root"])
    mask_root = Path(cfg["output"]["mask_root"])

    model_names = [
        name
        for name, model_cfg in cfg["segmentation"]["models"].items()
        if model_cfg.get("enabled", False)
    ]

    image_paths = []

    for split in cfg["dataset"]["splits"]:
        image_paths.extend(find_images(data_root / split))

    image_paths = sorted(set(image_paths))

    rows = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(data_root)

        for model_a, model_b in itertools.combinations(model_names, 2):
            mask_a_path = (
                mask_root / model_a / relative_path
            ).with_suffix(".png")

            mask_b_path = (
                mask_root / model_b / relative_path
            ).with_suffix(".png")

            if not mask_a_path.exists() or not mask_b_path.exists():
                continue

            mask_a = (
                np.asarray(Image.open(mask_a_path).convert("L"))
                >= 128
            )

            mask_b = (
                np.asarray(Image.open(mask_b_path).convert("L"))
                >= 128
            )

            if mask_a.shape != mask_b.shape:
                raise ValueError(
                    f"Mask shape mismatch: "
                    f"{mask_a_path}, {mask_b_path}"
                )

            dice = dice_score(mask_a, mask_b)

            rows.append(
                {
                    "image": str(relative_path),
                    "model_a": model_a,
                    "model_b": model_b,
                    "dice": dice,
                    "area_ratio_a": float(mask_a.mean()),
                    "area_ratio_b": float(mask_b.mean()),
                }
            )

    dataframe = pd.DataFrame(rows)

    output_dir = Path("outputs/mask_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "pairwise_dice.csv"
    dataframe.to_csv(detail_path, index=False)

    summary = (
        dataframe
        .groupby(["model_a", "model_b"])["dice"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary_path = output_dir / "pairwise_dice_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary)
    print("[SAVE]", detail_path)
    print("[SAVE]", summary_path)


if __name__ == "__main__":
    main()