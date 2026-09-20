from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.image import (
    get_mask_path,
    apply_lung_mask,
    moment_standardize_image,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayManifestDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        cfg,
        mode,
        mask_model=None,
        mask_region=None,
        training=False,
    ):
        self.cfg = cfg
        self.mode = mode
        self.mask_model = mask_model
        self.mask_region = mask_region

        valid_modes = {
            "original",
            "lung_only",
            "moment_standardized",
        }

        if self.mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {self.mode}"
            )

        if self.mode != "original":
            if self.mask_model is None:
                raise ValueError(
                    f"mask_model is required for mode={self.mode}"
                )

            if self.mask_region is None:
                raise ValueError(
                    f"mask_region is required for mode={self.mode}"
                )

        dataset_cfg = cfg["dataset"]
        preprocessing_cfg = cfg["preprocessing"]

        self.data_root = Path(dataset_cfg["root"])
        self.mask_root = Path(preprocessing_cfg["mask_root"])

        self.mask_threshold = int(
            preprocessing_cfg.get("mask_threshold", 128)
        )
        self.clip_z = float(
            preprocessing_cfg.get("clip_z", 3.0)
        )
        self.eps = float(
            preprocessing_cfg.get("eps", 1e-8)
        )

        dataframe = pd.read_csv(manifest)

        self.dataframe = dataframe[
            dataframe["split"] == split
        ].reset_index(drop=True)

        if len(self.dataframe) == 0:
            raise RuntimeError(
                f"No samples found for split={split}"
            )

        self.transform = self._build_transform(training)


    def _build_transform(self, training):
        aug_cfg = self.cfg["augmentation"]

        resize_size = int(
            aug_cfg.get("resize_size", 256)
        )
        image_size = int(
            aug_cfg.get("image_size", 224)
        )

        transform_list = [
            transforms.Resize(
                (resize_size, resize_size)
            )
        ]

        if training:
            if aug_cfg.get("random_crop", True):
                transform_list.append(
                    transforms.RandomCrop(image_size)
                )
            else:
                transform_list.append(
                    transforms.CenterCrop(image_size)
                )

            if aug_cfg.get("horizontal_flip", True):
                flip_probability = float(
                    aug_cfg.get(
                        "horizontal_flip_probability",
                        0.5,
                    )
                )

                transform_list.append(
                    transforms.RandomHorizontalFlip(
                        p=flip_probability
                    )
                )

        else:
            transform_list.append(
                transforms.CenterCrop(image_size)
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

        return transforms.Compose(transform_list)


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        relative_path = Path(row["relative_path"])
        image_path = self.data_root / relative_path
        label = int(row["label"])

        image = Image.open(image_path)

        if self.mode == "original":
            image = image.convert("RGB")

        else:
            mask_path = get_mask_path(
                image_path=image_path,
                data_root=self.data_root,
                mask_root=self.mask_root,
                mask_model=self.mask_model,
                mask_region=self.mask_region,
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            if self.mode == "lung_only":
                image = apply_lung_mask(
                    image=image,
                    mask_path=mask_path,
                    mask_threshold=self.mask_threshold,
                )

            elif self.mode == "moment_standardized":
                image = moment_standardize_image(
                    image=image,
                    mask_path=mask_path,
                    mask_threshold=self.mask_threshold,
                    clip_z=self.clip_z,
                    eps=self.eps,
                )

        image = self.transform(image)

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            index,
        )


def build_classification_datasets(
    cfg,
    manifest,
    experiment_cfg,
):
    mode = experiment_cfg["mode"]
    mask_model = experiment_cfg.get("mask_model")
    mask_region = experiment_cfg.get("mask_region")

    return {
        split: ChestXrayManifestDataset(
            manifest=manifest,
            split=split,
            cfg=cfg,
            mode=mode,
            mask_model=mask_model,
            mask_region=mask_region,
            training=(split == "train"),
        )
        for split in ["train", "val", "test"]
    }


def build_classification_loaders(
    datasets,
    cfg,
    seed,
):
    train_cfg = cfg["train"]

    batch_size = int(
        train_cfg["batch_size"]
    )
    num_workers = int(
        train_cfg.get("num_workers", 0)
    )
    pin_memory = bool(
        train_cfg.get("pin_memory", True)
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **common_kwargs,
        ),
    }