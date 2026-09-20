import argparse
import csv
import sys

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import time
import random

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.segmentation.models import build_lung_segmenter
from src.segmentation.niaid_resunet import NIAIDResUNetSegmenter
from src.utils.config import load_config
from src.utils.seed import get_device, set_seed
from src.utils.image import find_images


def calculate_mask_stats(mask):
    binary = mask > 0

    area_ratio = float(binary.mean())
    _, num_components = ndimage.label(binary)

    if binary.any():
        ys, xs = np.where(binary)

        bbox_width_ratio = float(
            (xs.max() - xs.min() + 1)
            / binary.shape[1]
        )

        bbox_height_ratio = float(
            (ys.max() - ys.min() + 1)
            / binary.shape[0]
        )

    else:
        bbox_width_ratio = 0.0
        bbox_height_ratio = 0.0

    return {
        "area_ratio": area_ratio,
        "num_components": int(num_components),
        "bbox_width_ratio": bbox_width_ratio,
        "bbox_height_ratio": bbox_height_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(
        cfg.get("device", {}).get("require_cuda", True)
    )

    data_root = Path(cfg["dataset"]["root"])
    mask_root = Path(cfg["output"]["mask_root"])

    overwrite = bool(
        cfg.get("runtime", {}).get("overwrite", False)
    )

    save_stats = bool(
        cfg["output"].get("stats", True)
    )

    model_names = [
        model_name
        for model_name, model_cfg
        in cfg["segmentation"]["models"].items()
        if model_cfg.get("enabled", False)
    ]

    if not model_names:
        raise RuntimeError(
            "No segmentation model is enabled in YAML."
        )
    start = time.time()
    
    # ============================================================
    # Collect images
    # ============================================================
    
    image_paths = []
    for split in cfg["dataset"]["splits"]:
    
        split_root = (
            data_root
            / split
        )
    
        image_paths.extend(
            find_images(split_root)
        )
    
    
    image_paths = sorted(
        set(image_paths)
    )
    
    
    print(
        "[INFO] Total images:",
        len(image_paths),
    )
    
    # ============================================================
    # Limit images for testing
    # ============================================================
    
    runtime_cfg = cfg.get("runtime", {})
    
    max_images = runtime_cfg.get("max_images", None,)
    
    sample_mode = str(
        runtime_cfg.get(
            "sample_mode",
            "random",
        )
    ).lower()
    
    
    if max_images is not None:   
        max_images = int(max_images)
    
        if max_images <= 0:
            raise ValueError(
                "runtime.max_images must be "
                "greater than 0 or null."
            )
    
        max_images = min(
            max_images,
            len(image_paths),
        )
    
        # --------------------------------------------------------
        # Random sampling
        # --------------------------------------------------------
    
        if sample_mode == "random":        
            rng = random.Random(seed)
    
            image_paths = rng.sample(
                image_paths,
                max_images,
            )
    
            image_paths = sorted(image_paths)
    
        # --------------------------------------------------------
        # First N images
        # --------------------------------------------------------
    
        elif sample_mode == "first":
            image_paths = image_paths[:max_images]
    
        else:
            raise ValueError(
                "runtime.sample_mode must be "
                "'random' or 'first'."
            )
    
    
    print("[INFO] Device:", device)
    print("[INFO] Num Of Images:", len(image_paths))
    print("[INFO] Models:", model_names)
    print("[INFO] Overwrite:", overwrite)

    for model_name in model_names:
        model_cfg = cfg["segmentation"]["models"][model_name]
        
        save_regions = list(
            model_cfg.get(
                "save_regions",
                ["whole_lung"],
            )
        )
        print("\n====================================")
        print("[INFO] Model:", model_name)
        print("====================================")
        
        # ========================================================
        # NIAID ResUNet
        # ========================================================
    
        if model_name == "niaid_resunet":
            segmenter = (
                NIAIDResUNetSegmenter(
                    cfg=cfg,
                )
            )
    
            segmenter.generate_masks(
                image_paths=image_paths,
                data_root=data_root,
                mask_root=mask_root,
                overwrite=overwrite,
            )
    
            continue

        # ========================================================
        # TorchXRayVision
        #
        # PSPNet / CXAS
        # ========================================================
        segmenter = build_lung_segmenter(
            model_name=model_name,
            cfg=cfg,
            device=device,
        )
        
        stats_rows = []

        for image_path in tqdm(image_paths, desc=model_name):
            relative_path = image_path.relative_to(
                data_root
            )

            expected_paths = {
                region: (
                    mask_root
                    / model_name
                    / region
                    / relative_path
                ).with_suffix(".png")
                for region in save_regions
            }

            all_exist = all(
                path.exists()
                for path in expected_paths.values()
            )

            if all_exist and not overwrite:
                masks = {
                    region: np.asarray(
                        Image.open(mask_path).convert("L"),
                        dtype=np.uint8,
                    )
                    for region, mask_path
                    in expected_paths.items()
                }

            else:
                predicted_masks = segmenter.predict_masks(image_path)
                masks = {}

                for region in save_regions:
                    if region not in predicted_masks:
                        raise RuntimeError(
                            f"{model_name} did not produce "
                            f"region '{region}'."
                        )

                    mask = predicted_masks[region]
                    output_path = expected_paths[region]

                    output_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    Image.fromarray(mask).save(
                        output_path
                    )

                    masks[region] = mask

            if save_stats:
                for region, mask in masks.items():
                    stats = calculate_mask_stats(mask)

                    stats_rows.append(
                        {
                            "image": str(relative_path),
                            "model": model_name,
                            "region": region,
                            "mask": str(
                                expected_paths[region]
                            ),
                            **stats,
                        }
                    )

        if save_stats:
            stats_path = (
                mask_root
                / model_name
                / "mask_stats.csv"
            )

            stats_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fieldnames = [
                "image",
                "model",
                "region",
                "mask",
                "area_ratio",
                "num_components",
                "bbox_width_ratio",
                "bbox_height_ratio",
            ]

            with stats_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(stats_rows)

            print("[SAVE]", stats_path)
            
    end = time.time()
    print(f"TIME: {(end-start)/60}min")


if __name__ == "__main__":
    main()