import argparse
import random
import sys

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from src.utils.config import load_config
from src.utils.image import find_images
from src.utils.seed import set_seed


def collect_by_class(
    split_root,
    classes,
):
    result = {}

    for class_name in classes:
        class_dir = split_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        result[class_name] = find_images(class_dir)

    return result


def sample_reference_count(
    samples_by_class,
    classes,
    reference_class,
    seed,
):
    reference_count = len(samples_by_class[reference_class])

    selected = {}

    for class_index, class_name in enumerate(classes):
        candidates = list(
            samples_by_class[class_name]
        )

        if len(candidates) < reference_count:
            raise RuntimeError(
                f"{class_name}: {len(candidates)} < "
                f"{reference_class}: {reference_count}"
            )

        rng = random.Random(
            seed + class_index * 1009
        )

        selected[class_name] = rng.sample(
            candidates,
            reference_count,
        )

    return selected


def split_train_val(
    selected,
    classes,
    val_ratio,
    seed,
):
    train_rows = []
    val_rows = []

    for label, class_name in enumerate(classes):
        samples = list(selected[class_name])

        rng = random.Random(
            seed + label * 2003
        )
        rng.shuffle(samples)

        n_val = int(
            round(len(samples) * val_ratio)
        )

        if len(samples) > 1:
            n_val = max(1, n_val)
            n_val = min(
                n_val,
                len(samples) - 1,
            )

        for path in samples[:n_val]:
            val_rows.append(
                (path, label, class_name)
            )

        for path in samples[n_val:]:
            train_rows.append(
                (path, label, class_name)
            )

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(val_rows)

    return train_rows, val_rows


def create_manifest(
    cfg,
    seed,
):
    set_seed(seed)

    dataset_cfg = cfg["dataset"]

    data_root = Path(dataset_cfg["root"])
    classes = list(dataset_cfg["classes"])
    reference_class = dataset_cfg["reference_class"]

    train_root = (
        data_root
        / dataset_cfg["train_split"]
    )

    test_root = (
        data_root
        / dataset_cfg["test_split"]
    )

    # Train
    train_by_class = collect_by_class(
        train_root,
        classes,
    )

    selected_train = sample_reference_count(
        samples_by_class=train_by_class,
        classes=classes,
        reference_class=reference_class,
        seed=seed,
    )

    train_rows, val_rows = split_train_val(
        selected=selected_train,
        classes=classes,
        val_ratio=float(
            dataset_cfg.get("val_ratio", 0.2)
        ),
        seed=seed,
    )

    # Test
    test_by_class = collect_by_class(
        test_root,
        classes,
    )

    selected_test = sample_reference_count(
        samples_by_class=test_by_class,
        classes=classes,
        reference_class=reference_class,
        seed=seed + 100000,
    )

    test_rows = []

    for label, class_name in enumerate(classes):
        for path in selected_test[class_name]:
            test_rows.append(
                (path, label, class_name)
            )

    random.Random(seed + 2).shuffle(test_rows)

    rows = []

    for split, split_rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        for path, label, class_name in split_rows:
            rows.append(
                {
                    "split": split,
                    "relative_path": str(
                        path.relative_to(data_root)
                    ),
                    "label": int(label),
                    "class_name": class_name,
                }
            )

    dataframe = pd.DataFrame(rows)

    output_dir = Path(
        dataset_cfg["manifest_dir"]
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"seed{seed}.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("\n================================")
    print(f"Seed {seed}")
    print("================================")

    print(
        dataframe.groupby(
            ["split", "class_name"]
        ).size()
    )

    print("[SAVE]", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg["experiment"]["seeds"]

    for seed in seeds:
        create_manifest(
            cfg=cfg,
            seed=int(seed),
        )


if __name__ == "__main__":
    main()