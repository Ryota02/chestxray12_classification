import argparse
import random
import sys

from pathlib import Path

import pandas as pd


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(ROOT_DIR)
)


from src.utils.config import (
    load_config,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def collect_class_images(
    data_root,
    split,
    class_name,
):

    class_dir = (
        data_root
        / split
        / class_name
    )

    if not class_dir.exists():

        raise FileNotFoundError(
            f"Class directory not found: "
            f"{class_dir}"
        )

    paths = sorted(
        path
        for path in class_dir.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    return paths


def load_excluded_images(
    cfg,
):

    excluded_path = (
        cfg[
            "dataset"
        ].get(
            "excluded_images",
            None,
        )
    )

    if excluded_path is None:
        return set()

    excluded_path = Path(
        excluded_path
    )

    if not excluded_path.exists():

        raise FileNotFoundError(
            f"Excluded image list not found: "
            f"{excluded_path}"
        )

    with excluded_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        excluded = {
            line.strip()
            for line in f
            if line.strip()
        }

    print(
        "[INFO] Excluded images:",
        len(excluded),
    )

    return excluded


def filter_excluded(
    paths,
    data_root,
    excluded,
):

    if not excluded:
        return paths

    filtered = []

    for path in paths:

        relative_path = str(
            path.relative_to(
                data_root
            )
        )

        if relative_path in excluded:
            continue

        filtered.append(
            path
        )

    return filtered


def select_images(
    paths,
    target_count,
    seed,
):

    paths = list(
        paths
    )

    if target_count is None:

        return paths

    if len(paths) < target_count:

        raise RuntimeError(
            "Not enough images for balancing: "
            f"available={len(paths)}, "
            f"required={target_count}"
        )

    rng = random.Random(
        seed
    )

    return rng.sample(
        paths,
        target_count,
    )


def split_train_val(
    paths,
    val_ratio,
    seed,
):

    paths = list(
        paths
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        paths
    )

    if val_ratio <= 0:

        return (
            paths,
            [],
        )

    n_val = int(
        round(
            len(paths)
            * val_ratio
        )
    )

    if (
        len(paths) > 1
        and
        n_val == 0
    ):
        n_val = 1

    if n_val >= len(paths):

        n_val = (
            len(paths)
            - 1
        )

    val_paths = paths[
        :n_val
    ]

    train_paths = paths[
        n_val:
    ]

    return (
        train_paths,
        val_paths,
    )


def make_row(
    path,
    split,
    label,
    class_name,
    data_root,
):

    return {
        "split":
            split,

        "relative_path":
            str(
                path.relative_to(
                    data_root
                )
            ),

        "label":
            label,

        "class_name":
            class_name,
    }


def create_manifest(
    cfg,
    seed,
):

    dataset_cfg = (
        cfg[
            "dataset"
        ]
    )

    data_root = Path(
        dataset_cfg[
            "root"
        ]
    )

    train_split = str(
        dataset_cfg.get(
            "train_split",
            "train",
        )
    )

    test_split = str(
        dataset_cfg.get(
            "test_split",
            "test",
        )
    )

    classes = list(
        dataset_cfg[
            "classes"
        ]
    )

    reference_class = str(
        dataset_cfg.get(
            "reference_class",
            "Pneumonia",
        )
    )

    balance_train = bool(dataset_cfg.get("balance_train", True))
    balance_test = bool(dataset_cfg.get("balance_test", True))

    val_ratio = float(dataset_cfg.get("val_ratio", 0.2))

    excluded = load_excluded_images(cfg)

    # ========================================================
    # Collect all images
    # ========================================================

    train_images = {}
    test_images = {}

    for class_name in classes:

        train_paths = (
            collect_class_images(
                data_root=data_root,
                split=train_split,
                class_name=class_name,
            )
        )

        test_paths = (
            collect_class_images(
                data_root=data_root,
                split=test_split,
                class_name=class_name,
            )
        )

        train_paths = (
            filter_excluded(
                paths=train_paths,
                data_root=data_root,
                excluded=excluded,
            )
        )

        test_paths = (
            filter_excluded(
                paths=test_paths,
                data_root=data_root,
                excluded=excluded,
            )
        )

        train_images[
            class_name
        ] = train_paths

        test_images[
            class_name
        ] = test_paths

    # ========================================================
    # Print original distributions
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        f"[SEED] {seed}"
    )

    print(
        "[ORIGINAL TRAIN]"
    )

    for class_name in classes:

        print(
            f"{class_name:15s}: "
            f"{len(train_images[class_name])}"
        )

    print()
    print(
        "[ORIGINAL TEST]"
    )

    for class_name in classes:

        print(
            f"{class_name:15s}: "
            f"{len(test_images[class_name])}"
        )

    # ========================================================
    # Decide balancing counts
    # ========================================================

    if balance_train:

        train_target_count = len(
            train_images[
                reference_class
            ]
        )

    else:

        train_target_count = None

    if balance_test:

        test_target_count = len(
            test_images[
                reference_class
            ]
        )

    else:

        test_target_count = None

    print()
    print(
        "[CONFIG]"
    )

    print(
        "balance_train:",
        balance_train,
    )

    print(
        "balance_test:",
        balance_test,
    )

    print(
        "train_target_count:",
        train_target_count,
    )

    print(
        "test_target_count:",
        test_target_count,
    )

    # ========================================================
    # Build manifest
    # ========================================================

    rows = []

    for class_index, class_name in enumerate(
        classes
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        selected_train = (
            select_images(
                paths=(
                    train_images[
                        class_name
                    ]
                ),
                target_count=(
                    train_target_count
                ),
                seed=(
                    seed
                    + class_index
                ),
            )
        )

        (
            train_paths,
            val_paths,
        ) = split_train_val(
            paths=selected_train,
            val_ratio=val_ratio,
            seed=(
                seed
                + 1000
                + class_index
            ),
        )

        for path in train_paths:

            rows.append(
                make_row(
                    path=path,
                    split="train",
                    label=class_index,
                    class_name=class_name,
                    data_root=data_root,
                )
            )

        for path in val_paths:

            rows.append(
                make_row(
                    path=path,
                    split="val",
                    label=class_index,
                    class_name=class_name,
                    data_root=data_root,
                )
            )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        selected_test = (
            select_images(
                paths=(
                    test_images[
                        class_name
                    ]
                ),
                target_count=(
                    test_target_count
                ),
                seed=(
                    seed
                    + 2000
                    + class_index
                ),
            )
        )

        for path in selected_test:

            rows.append(
                make_row(
                    path=path,
                    split="test",
                    label=class_index,
                    class_name=class_name,
                    data_root=data_root,
                )
            )

    dataframe = pd.DataFrame(
        rows
    )

    # ========================================================
    # Save
    # ========================================================

    manifest_dir = Path(
        dataset_cfg[
            "manifest_dir"
        ]
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        manifest_dir
        / f"seed{seed}.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # Show resulting distribution
    # ========================================================

    print()
    print(
        "[MANIFEST DISTRIBUTION]"
    )

    summary = (
        dataframe
        .groupby(
            [
                "split",
                "class_name",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        summary
    )

    print()
    print(
        "[SAVED]",
        output_path,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    seeds = list(
        cfg[
            "experiment"
        ][
            "seeds"
        ]
    )

    for seed in seeds:

        create_manifest(
            cfg=cfg,
            seed=int(
                seed
            ),
        )


if __name__ == "__main__":

    main()