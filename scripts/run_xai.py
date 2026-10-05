import argparse
import csv
import random
import sys
import numpy as np

from pathlib import Path

import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.utils.config import load_config

from src.xai.common import (
    find_checkpoint,
    find_experiment,
    image_to_tensor,
    load_classifier,
    load_experiment_image,
    predict,
    prepare_display_image,
)
from src.xai.cam import explain_cam
from src.xai.lime import explain_lime
from src.xai.shap import explain_shap
from src.xai.attention_rollout import explain_attention_rollout
from src.xai.plotting import save_xai_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    xai_cfg = load_config(args.config)

    classification_cfg = load_config(
        xai_cfg["classification_config"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)

    seed = int(xai_cfg.get("seed", 42))
    split = str(xai_cfg.get("split", "test"))
    num_images = int(xai_cfg.get("num_images", 5))
    sample_seed = int(
        xai_cfg.get("sample_seed", seed)
    )

    classes = classification_cfg["dataset"]["classes"]
    num_classes = len(classes)

    # Manifest
    manifest_path = (
        Path(
            classification_cfg["dataset"]["manifest_dir"]
        )
        / f"seed{seed}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[
        manifest["split"] == split
    ].copy()

    if len(manifest) == 0:
        raise RuntimeError(
            f"No images found for split={split}"
        )

    # Select the same images for all experiments
    rng = random.Random(sample_seed)

    indices = list(manifest.index)
    rng.shuffle(indices)

    indices = indices[:min(num_images, len(indices))]

    selected = manifest.loc[indices].copy()

    print(
        "[INFO] Number of XAI images:",
        len(selected),
    )

    for _, row in selected.iterrows():
        print(
            "[XAI IMAGE]",
            row["relative_path"],
        )

    methods_cfg = xai_cfg["methods"]
    lime_cfg = xai_cfg.get("lime", {})
    shap_cfg = xai_cfg.get("shap", {})

    output_root = Path(
        xai_cfg["output"]["root"]
    )

    for backbone in xai_cfg["backbones"]:
        print("\n========================================")
        print("[INFO] Backbone:", backbone)
        print("========================================")

        for experiment_name in xai_cfg["experiments"]:
            print("\n----------------------------------------")
            print("[INFO] Experiment:", experiment_name)
            print("----------------------------------------")

            experiment_cfg = find_experiment(
                classification_cfg,
                experiment_name,
            )

            checkpoint_path = find_checkpoint(
                checkpoint_root=xai_cfg["checkpoint"]["root"],
                backbone=backbone,
                experiment_name=experiment_name,
                seed=seed,
                filename=xai_cfg[
                    "checkpoint"
                ].get(
                    "filename",
                    None,
                ),
            )

            print(
                "[INFO] Checkpoint:",
                checkpoint_path,
            )

            model = load_classifier(
                backbone=backbone,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            experiment_output_dir = (
                output_root
                / backbone
                / experiment_name
                / f"seed{seed}"
            )

            experiment_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata_rows = []

            for case_number, (_, row) in enumerate(
                selected.iterrows(),
                start=1,
            ):
                relative_path = str(row["relative_path"])

                ground_truth = int(row["label"]) 
                ground_truth_name = (classes[ground_truth])

                print(
                    f"\n[{case_number}/{len(selected)}] "
                    f"{relative_path}"
                )

                # Load image for this classification condition
                image = load_experiment_image(
                    relative_path=relative_path,
                    classification_cfg=classification_cfg,
                    experiment_cfg=experiment_cfg,
                )

                # Same spatial preprocessing as classifier
                image = prepare_display_image(
                    image=image,
                    classification_cfg=classification_cfg,
                )

                input_tensor = image_to_tensor(
                    image=image,
                    device=device,
                )

                predicted_class, probabilities = predict(
                    model=model,
                    input_tensor=input_tensor,
                )

                predicted_probability = float(
                    probabilities[predicted_class]
                )

                predicted_class_name = classes[
                    predicted_class
                ]

                # Explain predicted class
                target_class = predicted_class

                print(
                    "[PREDICTION]",
                    predicted_class_name,
                    f"{predicted_probability:.4f}",
                )

                print(
                    "[XAI TARGET]",
                    predicted_class_name,
                )

                heatmaps = {}

                # Grad-CAM
                if (
                    methods_cfg["gradcam"]["enabled"]
                    and backbone in {
                        "resnet50",
                        "densenet201",
                    }
                ):
                    heatmaps["Grad-CAM"] = explain_cam(
                        model=model,
                        input_tensor=input_tensor,
                        target_class=target_class,
                        backbone=backbone,
                        method="gradcam",
                    )

                # Grad-CAM++
                if (
                    methods_cfg[
                        "gradcam_plus_plus"
                    ]["enabled"]
                    and backbone in {
                        "resnet50",
                        "densenet201",
                    }
                ):
                    heatmaps["Grad-CAM++"] = explain_cam(
                        model=model,
                        input_tensor=input_tensor,
                        target_class=target_class,
                        backbone=backbone,
                        method="gradcam_plus_plus",
                    )

                # LIME
                if methods_cfg["lime"]["enabled"]:
                    heatmaps["LIME"] = explain_lime(
                        model=model,
                        image=image,
                        target_class=target_class,
                        device=device,
                        num_samples=int(
                            lime_cfg.get(
                                "num_samples",
                                500,
                            )
                        ),
                        num_features=int(
                            lime_cfg.get(
                                "num_features",
                                10,
                            )
                        ),
                    )

                # SHAP
                if methods_cfg["shap"]["enabled"]:
                    heatmaps["SHAP"] = explain_shap(
                        model=model,
                        input_tensor=input_tensor,
                        target_class=target_class,
                        nsamples=int(
                            shap_cfg.get(
                                "nsamples",
                                64,
                            )
                        ),
                        background_size=int(
                            shap_cfg.get(
                                "background_size",
                                4,
                            )
                        ),
                    )

                # Attention Rollout
                if (
                    methods_cfg[
                        "attention_rollout"
                    ]["enabled"]
                    and backbone == "vit_b_16"
                ):
                    heatmaps[
                        "Attention Rollout (class-agnostic)"
                    ] = explain_attention_rollout(
                        model=model,
                        input_tensor=input_tensor,
                    )

                relative_object = Path(relative_path)          
                image_stem = (relative_object.stem)
                
                safe_parent = "__".join(relative_object.parent.parts)
                
                case_id = f"{safe_parent}__{image_stem}"

                output_name = (
                    f"{safe_parent}__"
                    f"{image_stem}_xai.png"
                )

                output_path = (
                    experiment_output_dir
                    / output_name
                )

                title = (
                    f"Ground Truth: "
                    f"{ground_truth_name}"
                    " | "
                    f"Predicted: {predicted_class_name}"
                    " | "
                    f"Probability: "
                    f"{predicted_probability:.4f}"
                )

                # Save raw heatmaps
                
                heatmap_name_map = {
                    "Grad-CAM": "gradcam",
                    "Grad-CAM++": "gradcam_plus_plus",
                    "LIME": "lime",
                    "SHAP": "shap",
                    "Attention Rollout (class-agnostic)": "attention_rollout",
                }
                
                
                for (display_name, heatmap) in heatmaps.items():  
                    if display_name not in heatmap_name_map:
                        continue
                
                    method_key = heatmap_name_map[display_name]
                
                    heatmap_dir = (
                        experiment_output_dir
                        / "heatmaps"
                        / method_key
                    )
                
                    heatmap_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                
                    heatmap_path = (
                        heatmap_dir
                        / f"{case_id}.npy"
                    )
                
                    np.save(
                        heatmap_path,
                        np.asarray(
                            heatmap,
                            dtype=np.float32,
                        ),
                    )

                save_xai_comparison(
                    image=image,
                    heatmaps=heatmaps,
                    output_path=output_path,
                    title=title,
                )

                metadata_rows.append(
                    {
                        "case_id": case_id,
                        "relative_path": relative_path,
                        "backbone": backbone,
                        "experiment": experiment_name,
                        "predicted_class_index": predicted_class,
                        "predicted_class": predicted_class_name,
                        "predicted_probability": predicted_probability,
                        "xai_target_class": predicted_class_name,
                        "xai_target_source": "predicted_class",
                        "output_path": str(output_path),
                    }
                )

                print("[SAVED]", output_path)

            # Save metadata
            metadata_path = (
                experiment_output_dir
                / "xai_metadata.csv"
            )

            if metadata_rows:
                fieldnames = list(
                    metadata_rows[0].keys()
                )

                with metadata_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=fieldnames,
                    )

                    writer.writeheader()
                    writer.writerows(metadata_rows)

            print(
                "[INFO] Metadata:",
                metadata_path,
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()