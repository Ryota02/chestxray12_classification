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


# ============================================================
# Patient ID
# ============================================================

def extract_patient_id(
    path,
):
    """
    Extract NIH ChestX-ray14 patient ID.

    Example
    -------
    00001234_000.png
        -> 00001234

    00001234_001.png
        -> 00001234
    """

    path = Path(
        path
    )

    stem = path.stem

    if "_" not in stem:
        raise ValueError(
            "Cannot extract patient ID from filename: "
            f"{path.name}"
        )

    patient_id = (
        stem.split(
            "_",
            1,
        )[0]
    )

    return patient_id


# ============================================================
# Collect images
# ============================================================

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


# ============================================================
# Excluded images
# ============================================================

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
            "Excluded image list not found: "
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


# ============================================================
# Patient utilities
# ============================================================

def collect_patient_ids(
    images_by_class,
):

    patient_ids = set()

    for paths in images_by_class.values():

        for path in paths:

            patient_ids.add(
                extract_patient_id(
                    path
                )
            )

    return patient_ids


def count_images_by_patient_set(
    images_by_class,
    patient_ids,
):
    """
    Count images of each class that belong
    to the given patient set.
    """

    counts = {}

    for class_name, paths in images_by_class.items():

        count = sum(
            extract_patient_id(path)
            in patient_ids
            for path in paths
        )

        counts[
            class_name
        ] = count

    return counts


# ============================================================
# Check original train / test patient leakage
# ============================================================

def check_source_train_test_leakage(
    train_images,
    test_images,
):

    train_patient_ids = (
        collect_patient_ids(
            train_images
        )
    )

    test_patient_ids = (
        collect_patient_ids(
            test_images
        )
    )

    overlap = (
        train_patient_ids
        &
        test_patient_ids
    )

    print()
    print(
        "========================================"
    )

    print(
        "[SOURCE PATIENT CHECK]"
    )

    print(
        "Train patients:",
        len(
            train_patient_ids
        ),
    )

    print(
        "Test patients:",
        len(
            test_patient_ids
        ),
    )

    print(
        "Train/Test overlap:",
        len(
            overlap
        ),
    )

    print(
        "========================================"
    )

    if overlap:

        examples = sorted(
            overlap
        )[:20]

        raise RuntimeError(
            "Patient leakage detected between "
            "the original train and test folders.\n"
            f"Number of overlapping patients: "
            f"{len(overlap)}\n"
            f"Examples: {examples}"
        )


# ============================================================
# Patient-level Train / Val split
# ============================================================

def split_train_val_patients(
    images_by_class,
    val_ratio,
    seed,
    max_attempts=1000,
):
    """
    Split patients, not images.

    The function searches multiple random patient splits and
    chooses the split whose per-class validation ratios are
    closest to val_ratio.

    This is useful because one patient may have multiple images
    and may even appear in multiple class folders.
    """

    all_patient_ids = sorted(
        collect_patient_ids(
            images_by_class
        )
    )

    n_patients = len(
        all_patient_ids
    )

    if n_patients == 0:

        raise RuntimeError(
            "No patients found."
        )

    if val_ratio <= 0:

        return (
            set(
                all_patient_ids
            ),
            set(),
        )

    if val_ratio >= 1:

        raise ValueError(
            "val_ratio must be < 1."
        )

    if n_patients < 2:

        raise RuntimeError(
            "At least two patients are required "
            "for train/val splitting."
        )

    # --------------------------------------------------------
    # Number of validation patients
    # --------------------------------------------------------

    n_val_patients = int(
        round(
            n_patients
            * val_ratio
        )
    )

    n_val_patients = max(
        1,
        n_val_patients,
    )

    n_val_patients = min(
        n_patients - 1,
        n_val_patients,
    )

    # --------------------------------------------------------
    # Original image counts
    # --------------------------------------------------------

    total_counts = {
        class_name:
            len(paths)

        for class_name, paths
        in images_by_class.items()
    }

    best_score = float(
        "inf"
    )

    best_train_patients = None
    best_val_patients = None

    # --------------------------------------------------------
    # Search for a reasonably stratified patient split
    # --------------------------------------------------------

    for attempt in range(
        max_attempts
    ):

        patient_ids = (
            all_patient_ids.copy()
        )

        rng = random.Random(
            seed
            + attempt
        )

        rng.shuffle(
            patient_ids
        )

        val_patients = set(
            patient_ids[
                :n_val_patients
            ]
        )

        train_patients = set(
            patient_ids[
                n_val_patients:
            ]
        )

        train_counts = (
            count_images_by_patient_set(
                images_by_class=
                    images_by_class,
                patient_ids=
                    train_patients,
            )
        )

        val_counts = (
            count_images_by_patient_set(
                images_by_class=
                    images_by_class,
                patient_ids=
                    val_patients,
            )
        )

        # ----------------------------------------------------
        # Every class must exist in both train and val
        # ----------------------------------------------------

        valid = True

        for class_name in images_by_class:

            if (
                train_counts[
                    class_name
                ] == 0
                or
                val_counts[
                    class_name
                ] == 0
            ):

                valid = False
                break

        if not valid:

            continue

        # ----------------------------------------------------
        # Score:
        # how close each class is to requested val_ratio
        # ----------------------------------------------------

        score = 0.0

        for class_name in images_by_class:

            total = (
                total_counts[
                    class_name
                ]
            )

            if total == 0:

                continue

            actual_ratio = (
                val_counts[
                    class_name
                ]
                / total
            )

            score += abs(
                actual_ratio
                - val_ratio
            )

        if score < best_score:

            best_score = score

            best_train_patients = (
                train_patients
            )

            best_val_patients = (
                val_patients
            )

    if (
        best_train_patients is None
        or
        best_val_patients is None
    ):

        raise RuntimeError(
            "Could not create a valid "
            "patient-level train/val split.\n"
            "Try increasing max_attempts or "
            "checking the class/patient distribution."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    train_counts = (
        count_images_by_patient_set(
            images_by_class=
                images_by_class,
            patient_ids=
                best_train_patients,
        )
    )

    val_counts = (
        count_images_by_patient_set(
            images_by_class=
                images_by_class,
            patient_ids=
                best_val_patients,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        "[PATIENT-LEVEL TRAIN / VAL SPLIT]"
    )

    print(
        "Total patients:",
        n_patients,
    )

    print(
        "Train patients:",
        len(
            best_train_patients
        ),
    )

    print(
        "Val patients:",
        len(
            best_val_patients
        ),
    )

    print()
    print(
        "[IMAGE COUNTS BEFORE BALANCING]"
    )

    for class_name in images_by_class:

        print(
            f"{class_name:15s} | "
            f"Train: "
            f"{train_counts[class_name]:4d} | "
            f"Val: "
            f"{val_counts[class_name]:4d}"
        )

    print(
        "========================================"
    )

    return (
        best_train_patients,
        best_val_patients,
    )


# ============================================================
# Partition images according to patient split
# ============================================================

def partition_images_by_patient(
    images_by_class,
    train_patient_ids,
    val_patient_ids,
):

    train_images = {}
    val_images = {}

    for class_name, paths in images_by_class.items():

        train_paths = []
        val_paths = []

        for path in paths:

            patient_id = (
                extract_patient_id(
                    path
                )
            )

            if (
                patient_id
                in train_patient_ids
            ):

                train_paths.append(
                    path
                )

            elif (
                patient_id
                in val_patient_ids
            ):

                val_paths.append(
                    path
                )

            else:

                raise RuntimeError(
                    "Patient was not assigned to "
                    "train or val: "
                    f"{patient_id}"
                )

        train_images[
            class_name
        ] = train_paths

        val_images[
            class_name
        ] = val_paths

    return (
        train_images,
        val_images,
    )


# ============================================================
# Image-level sampling AFTER patient split
# ============================================================

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


def get_balance_target(
    images_by_class,
    enabled,
    split_name,
):

    if not enabled:

        return None

    counts = {
        class_name:
            len(paths)

        for class_name, paths
        in images_by_class.items()
    }

    if not counts:

        return None

    minimum_count = min(
        counts.values()
    )

    if minimum_count <= 0:

        raise RuntimeError(
            f"{split_name}: "
            "At least one class has no images "
            "after patient-level splitting."
        )

    minimum_class = min(
        counts,
        key=counts.get,
    )

    print(
        f"[BALANCE] {split_name}: "
        f"target={minimum_count} "
        f"(minimum class="
        f"{minimum_class})"
    )

    return minimum_count


def balance_images_by_class(
    images_by_class,
    classes,
    target_count,
    seed,
):

    selected = {}

    for class_index, class_name in enumerate(
        classes
    ):

        selected[
            class_name
        ] = select_images(
            paths=(
                images_by_class[
                    class_name
                ]
            ),
            target_count=
                target_count,
            seed=(
                seed
                + class_index
            ),
        )

    return selected


# ============================================================
# Manifest row
# ============================================================

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

        "patient_id":
            extract_patient_id(
                path
            ),

        "label":
            label,

        "class_name":
            class_name,
    }


# ============================================================
# Final leakage check
# ============================================================

def check_manifest_patient_leakage(
    dataframe,
):

    split_patient_ids = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_patient_ids[
            split
        ] = set(
            dataframe.loc[
                dataframe[
                    "split"
                ] == split,
                "patient_id",
            ].astype(
                str
            )
        )

    combinations = [
        (
            "train",
            "val",
        ),
        (
            "train",
            "test",
        ),
        (
            "val",
            "test",
        ),
    ]

    print()
    print(
        "========================================"
    )

    print(
        "[FINAL PATIENT LEAKAGE CHECK]"
    )

    for split_a, split_b in combinations:

        overlap = (
            split_patient_ids[
                split_a
            ]
            &
            split_patient_ids[
                split_b
            ]
        )

        print(
            f"{split_a:5s} / "
            f"{split_b:5s}: "
            f"{len(overlap)} "
            "overlapping patients"
        )

        if overlap:

            examples = sorted(
                overlap
            )[:20]

            raise RuntimeError(
                "Patient leakage detected: "
                f"{split_a} vs {split_b}\n"
                f"Examples: {examples}"
            )

    print(
        "[OK] No patient leakage detected."
    )

    print(
        "========================================"
    )


# ============================================================
# Create manifest
# ============================================================

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

    balance_train = bool(
        dataset_cfg.get(
            "balance_train",
            True,
        )
    )

    balance_test = bool(
        dataset_cfg.get(
            "balance_test",
            True,
        )
    )

    val_ratio = float(
        dataset_cfg.get(
            "val_ratio",
            0.2,
        )
    )

    patient_split_attempts = int(
        dataset_cfg.get(
            "patient_split_attempts",
            1000,
        )
    )

    excluded = (
        load_excluded_images(
            cfg
        )
    )

    # ========================================================
    # Collect all images
    # ========================================================

    source_train_images = {}
    source_test_images = {}

    for class_name in classes:

        train_paths = (
            collect_class_images(
                data_root=
                    data_root,
                split=
                    train_split,
                class_name=
                    class_name,
            )
        )

        test_paths = (
            collect_class_images(
                data_root=
                    data_root,
                split=
                    test_split,
                class_name=
                    class_name,
            )
        )

        train_paths = (
            filter_excluded(
                paths=
                    train_paths,
                data_root=
                    data_root,
                excluded=
                    excluded,
            )
        )

        test_paths = (
            filter_excluded(
                paths=
                    test_paths,
                data_root=
                    data_root,
                excluded=
                    excluded,
            )
        )

        source_train_images[
            class_name
        ] = train_paths

        source_test_images[
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

    print()
    print(
        "[ORIGINAL TRAIN]"
    )

    for class_name in classes:

        print(
            f"{class_name:15s}: "
            f"{len(source_train_images[class_name])}"
        )

    print()
    print(
        "[ORIGINAL TEST]"
    )

    for class_name in classes:

        print(
            f"{class_name:15s}: "
            f"{len(source_test_images[class_name])}"
        )

    # ========================================================
    # IMPORTANT:
    # Check existing train/test patient separation
    # ========================================================

    check_source_train_test_leakage(
        train_images=
            source_train_images,
        test_images=
            source_test_images,
    )

    # ========================================================
    # Patient-level train/val split
    # ========================================================

    (
        train_patient_ids,
        val_patient_ids,
    ) = split_train_val_patients(
        images_by_class=
            source_train_images,
        val_ratio=
            val_ratio,
        seed=
            seed,
        max_attempts=
            patient_split_attempts,
    )

    (
        train_images,
        val_images,
    ) = partition_images_by_patient(
        images_by_class=
            source_train_images,
        train_patient_ids=
            train_patient_ids,
        val_patient_ids=
            val_patient_ids,
    )

    # ========================================================
    # Balance AFTER patient-level split
    #
    # Sampling images inside each patient-assigned split does
    # not create patient leakage.
    # ========================================================

    train_target_count = (
        get_balance_target(
            images_by_class=
                train_images,
            enabled=
                balance_train,
            split_name=
                "train",
        )
    )

    val_target_count = (
        get_balance_target(
            images_by_class=
                val_images,
            enabled=
                balance_train,
            split_name=
                "val",
        )
    )

    test_target_count = (
        get_balance_target(
            images_by_class=
                source_test_images,
            enabled=
                balance_test,
            split_name=
                "test",
        )
    )

    selected_train = (
        balance_images_by_class(
            images_by_class=
                train_images,
            classes=
                classes,
            target_count=
                train_target_count,
            seed=
                seed,
        )
    )

    selected_val = (
        balance_images_by_class(
            images_by_class=
                val_images,
            classes=
                classes,
            target_count=
                val_target_count,
            seed=(
                seed
                + 1000
            ),
        )
    )

    selected_test = (
        balance_images_by_class(
            images_by_class=
                source_test_images,
            classes=
                classes,
            target_count=
                test_target_count,
            seed=(
                seed
                + 2000
            ),
        )
    )

    # ========================================================
    # Build manifest
    # ========================================================

    rows = []

    for class_index, class_name in enumerate(
        classes
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        for path in selected_train[
            class_name
        ]:

            rows.append(
                make_row(
                    path=
                        path,
                    split=
                        "train",
                    label=
                        class_index,
                    class_name=
                        class_name,
                    data_root=
                        data_root,
                )
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        for path in selected_val[
            class_name
        ]:

            rows.append(
                make_row(
                    path=
                        path,
                    split=
                        "val",
                    label=
                        class_index,
                    class_name=
                        class_name,
                    data_root=
                        data_root,
                )
            )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        for path in selected_test[
            class_name
        ]:

            rows.append(
                make_row(
                    path=
                        path,
                    split=
                        "test",
                    label=
                        class_index,
                    class_name=
                        class_name,
                    data_root=
                        data_root,
                )
            )

    dataframe = pd.DataFrame(
        rows
    )

    # ========================================================
    # Final safety checks
    # ========================================================

    if dataframe[
        "relative_path"
    ].duplicated().any():

        duplicated = (
            dataframe.loc[
                dataframe[
                    "relative_path"
                ].duplicated(
                    keep=False
                ),
                "relative_path",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate images detected "
            "in manifest.\n"
            f"Examples: "
            f"{duplicated[:20]}"
        )

    check_manifest_patient_leakage(
        dataframe
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
    # Distribution
    # ========================================================

    print()
    print(
        "[MANIFEST IMAGE DISTRIBUTION]"
    )

    image_summary = (
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
        image_summary
    )

    # ========================================================
    # Patient distribution
    # ========================================================

    print()
    print(
        "[MANIFEST PATIENT DISTRIBUTION]"
    )

    patient_summary = (
        dataframe
        .groupby(
            [
                "split",
                "class_name",
            ]
        )[
            "patient_id"
        ]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    print(
        patient_summary
    )

    print()
    print(
        "[UNIQUE PATIENTS PER SPLIT]"
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        n_patients = (
            dataframe.loc[
                dataframe[
                    "split"
                ] == split,
                "patient_id",
            ]
            .nunique()
        )

        print(
            f"{split:5s}: "
            f"{n_patients}"
        )

    print()
    print(
        "[SAVED]",
        output_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    args = (
        parser.parse_args()
    )

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
            cfg=
                cfg,
            seed=
                int(seed),
        )


if __name__ == "__main__":

    main()