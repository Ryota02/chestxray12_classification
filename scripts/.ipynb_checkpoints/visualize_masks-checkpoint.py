import argparse
import random
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.utils.config import load_config
from src.utils.image import find_images


def make_overlay(image, mask):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    rgb = np.stack([image] * 3, axis=-1)
    binary = mask > 127

    overlay = rgb.copy()
    overlay[binary, 0] = 1.0
    overlay[binary, 1] *= 0.5
    overlay[binary, 2] *= 0.5

    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    visualization_cfg = cfg.get("visualization", {})
    num_images = int(visualization_cfg.get("num_images", 10))

    data_root = Path(cfg["dataset"]["root"])
    mask_root = Path(cfg["output"]["mask_root"])
    figure_dir = Path(cfg["output"]["figure_dir"])

    model_names = [
        name
        for name, model_cfg in cfg["segmentation"]["models"].items()
        if model_cfg.get("enabled", False)
    ]

    images = []

    for split in cfg["dataset"]["splits"]:
        images.extend(find_images(data_root / split))

    images = sorted(set(images))

    if not images:
        raise RuntimeError("No images found.")

    random.seed(int(cfg.get("seed", 42)))

    selected = random.sample(
        images,
        min(num_images, len(images)),
    )

    rows = len(selected)
    columns = 1 + len(model_names)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
    )

    for row, image_path in enumerate(selected):
        image = np.asarray(
            Image.open(image_path).convert("L")
        )

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(image_path.name)
        axes[row, 0].axis("off")

        relative_path = image_path.relative_to(data_root)

        for column, model_name in enumerate(model_names, start=1):
            mask_path = (
                mask_root / model_name / relative_path
            ).with_suffix(".png")

            if not mask_path.exists():
                axes[row, column].text(
                    0.5,
                    0.5,
                    "Mask missing",
                    ha="center",
                    va="center",
                )
                axes[row, column].axis("off")
                continue

            mask = np.asarray(
                Image.open(mask_path).convert("L")
            )

            if mask.shape != image.shape:
                mask = np.asarray(
                    Image.fromarray(mask).resize(
                        (image.shape[1], image.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                )

            overlay = make_overlay(image, mask)

            axes[row, column].imshow(overlay)
            axes[row, column].set_title(model_name)
            axes[row, column].axis("off")

    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = figure_dir / "mask_comparison.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("[SAVE]", output_path)


if __name__ == "__main__":
    main()