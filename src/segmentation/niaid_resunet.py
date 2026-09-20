import csv
import os
import shutil
import subprocess
import sys
import tempfile

from collections import defaultdict
from pathlib import Path

import numpy as np

from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

MEDICAL_EXTENSIONS = {
    ".nrrd",
    ".nii",
    ".nii.gz",
}


class NIAIDResUNetSegmenter:

    def __init__(
        self,
        cfg,
    ):

        segmentation_cfg = (
            cfg[
                "segmentation"
            ]
        )

        model_cfg = (
            segmentation_cfg[
                "models"
            ][
                "niaid_resunet"
            ]
        )

        # ====================================================
        # Repository
        # ====================================================

        self.repository_root = Path(
            model_cfg[
                "repository_root"
            ]
        ).expanduser().resolve()

        if not self.repository_root.exists():

            raise FileNotFoundError(
                "NIAID repository not found: "
                f"{self.repository_root}"
            )

        # ====================================================
        # Checkpoint
        # ====================================================

        checkpoint = Path(
            model_cfg[
                "checkpoint"
            ]
        )

        if checkpoint.is_absolute():

            self.checkpoint = checkpoint

        else:

            self.checkpoint = (
                self.repository_root
                / checkpoint
            )

        self.checkpoint = (
            self.checkpoint.resolve()
        )

        if not self.checkpoint.exists():

            raise FileNotFoundError(
                "NIAID checkpoint not found: "
                f"{self.checkpoint}"
            )

        # ====================================================
        # Inference settings
        # ====================================================

        self.img_size = tuple(
            int(x)
            for x
            in model_cfg.get(
                "img_size",
                [
                    224,
                    224,
                ],
            )
        )

        self.patch_size = tuple(
            int(x)
            for x
            in model_cfg.get(
                "patch_size",
                self.img_size,
            )
        )

        if len(
            self.img_size
        ) != 2:

            raise ValueError(
                "img_size must contain "
                "two integers."
            )

        if len(
            self.patch_size
        ) != 2:

            raise ValueError(
                "patch_size must contain "
                "two integers."
            )

        self.post_process = bool(
            model_cfg.get(
                "post_process",
                True,
            )
        )

        self.mask_threshold = float(
            model_cfg.get(
                "mask_threshold",
                0.5,
            )
        )

        self.save_regions = list(
            model_cfg.get(
                "save_regions",
                [
                    "whole_lung",
                ],
            )
        )

        if self.save_regions != [
            "whole_lung"
        ]:

            raise ValueError(
                "niaid_resunet currently "
                "supports only "
                "save_regions: "
                "[whole_lung]"
            )

        self.python_executable = str(
            model_cfg.get(
                "python_executable",
                sys.executable,
            )
        )

        print(
            "[INFO] Segmentation model:",
            "niaid_resunet",
        )

        print(
            "[INFO] Repository:",
            self.repository_root,
        )

        print(
            "[INFO] Checkpoint:",
            self.checkpoint,
        )

        print(
            "[INFO] Image size:",
            self.img_size,
        )

        print(
            "[INFO] Patch size:",
            self.patch_size,
        )

        print(
            "[INFO] Post process:",
            self.post_process,
        )


    # ========================================================
    # Create NIAID input CSV
    # ========================================================

    def _create_input_csv(
        self,
        image_paths,
        csv_path,
    ):

        csv_path = Path(
            csv_path
        )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cxr_file",
                ],
            )

            writer.writeheader()

            for image_path in image_paths:

                writer.writerow(
                    {
                        "cxr_file":
                            str(
                                Path(
                                    image_path
                                ).resolve()
                            )
                    }
                )


    # ========================================================
    # Execute official NIAID inference
    # ========================================================

    def _run_official_inference(
        self,
        image_paths,
        temporary_output_dir,
        temporary_csv,
    ):

        self._create_input_csv(
            image_paths=(
                image_paths
            ),
            csv_path=(
                temporary_csv
            ),
        )

        command = [
            self.python_executable,

            "-m",
            (
                "segment_lung_cxr."
                "inference."
                "inference_lung_segment"
            ),

            str(
                temporary_csv
            ),

            str(
                self.checkpoint
            ),

            "--output_pred_dir",
            str(
                temporary_output_dir
            ),

            "--img_size",
            str(
                self.img_size[
                    0
                ]
            ),
            str(
                self.img_size[
                    1
                ]
            ),

            "--patch_size",
            str(
                self.patch_size[
                    0
                ]
            ),
            str(
                self.patch_size[
                    1
                ]
            ),

            "--post_process",
            (
                "True"
                if self.post_process
                else "False"
            ),
        ]

        env = os.environ.copy()

        current_pythonpath = (
            env.get(
                "PYTHONPATH",
                "",
            )
        )

        if current_pythonpath:

            env[
                "PYTHONPATH"
            ] = (
                str(
                    self.repository_root
                )
                + os.pathsep
                + current_pythonpath
            )

        else:

            env[
                "PYTHONPATH"
            ] = str(
                self.repository_root
            )

        print(
            "[INFO] Running NIAID inference"
        )

        subprocess.run(
            command,
            cwd=str(
                self.repository_root
            ),
            env=env,
            check=True,
        )


    # ========================================================
    # Find predicted mask generated by NIAID
    # ========================================================

    def _find_generated_mask(
        self,
        output_dir,
        image_path,
    ):

        output_dir = Path(
            output_dir
        )

        image_path = Path(
            image_path
        )

        stem = (
            image_path.stem
        )

        # ----------------------------------------------------
        # Common expected names
        # ----------------------------------------------------

        candidate_names = [
            f"{stem}.png",
            f"{stem}_seg.png",
            f"{stem}_mask.png",

            f"{stem}.jpg",
            f"{stem}_seg.jpg",

            f"{stem}.tif",
            f"{stem}_seg.tif",

            f"{stem}.tiff",
            f"{stem}_seg.tiff",

            f"{stem}.nrrd",
            f"{stem}_seg.nrrd",

            f"{stem}.nii",
            f"{stem}_seg.nii",

            f"{stem}.nii.gz",
            f"{stem}_seg.nii.gz",
        ]

        for candidate_name in candidate_names:

            candidate_path = (
                output_dir
                / candidate_name
            )

            if candidate_path.exists():

                return candidate_path

        # ----------------------------------------------------
        # Recursive fallback
        # ----------------------------------------------------

        candidates = []

        for path in (
            output_dir.rglob(
                "*"
            )
        ):

            if not path.is_file():

                continue

            filename = (
                path.name.lower()
            )

            stem_lower = (
                stem.lower()
            )

            if filename.startswith(
                stem_lower
            ):

                candidates.append(
                    path
                )

        if len(
            candidates
        ) == 1:

            return candidates[
                0
            ]

        if len(
            candidates
        ) == 0:

            raise FileNotFoundError(
                "Could not find NIAID "
                "prediction for:\n"
                f"{image_path}\n"
                "Temporary output:\n"
                f"{output_dir}"
            )

        raise RuntimeError(
            "Multiple possible NIAID "
            "prediction files found for "
            f"{image_path.name}:\n"
            + "\n".join(
                str(path)
                for path
                in candidates
            )
        )


    # ========================================================
    # Read predicted mask
    # ========================================================

    def _read_prediction(
        self,
        mask_path,
    ):

        mask_path = Path(
            mask_path
        )

        lower_name = (
            mask_path.name.lower()
        )

        # ----------------------------------------------------
        # Normal image formats
        # ----------------------------------------------------

        if (
            mask_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            mask = Image.open(
                mask_path
            ).convert(
                "L"
            )

            array = np.asarray(
                mask,
                dtype=np.float32,
            )

            return array

        # ----------------------------------------------------
        # NRRD / NIfTI
        # ----------------------------------------------------

        if (
            lower_name.endswith(
                ".nrrd"
            )
            or
            lower_name.endswith(
                ".nii"
            )
            or
            lower_name.endswith(
                ".nii.gz"
            )
        ):

            try:
                import SimpleITK as sitk

            except ImportError as exc:

                raise ImportError(
                    "SimpleITK is required "
                    "to read NIAID output "
                    f"{mask_path}.\n"
                    "Install it with:\n"
                    "pip install SimpleITK"
                ) from exc

            image = sitk.ReadImage(
                str(
                    mask_path
                )
            )

            array = (
                sitk.GetArrayFromImage(
                    image
                )
            )

            array = np.asarray(
                array,
                dtype=np.float32,
            )

            array = np.squeeze(
                array
            )

            if array.ndim != 2:

                raise RuntimeError(
                    "Unexpected NIAID mask "
                    f"shape: {array.shape}"
                )

            return array

        raise ValueError(
            "Unsupported NIAID output "
            f"format: {mask_path}"
        )


    # ========================================================
    # Convert prediction to binary PNG
    # ========================================================

    def _save_binary_png(
        self,
        prediction_path,
        original_image_path,
        output_path,
    ):

        prediction = (
            self._read_prediction(
                prediction_path
            )
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        prediction_min = float(
            np.min(
                prediction
            )
        )

        prediction_max = float(
            np.max(
                prediction
            )
        )

        # Probability map
        if (
            prediction_min >= 0.0
            and
            prediction_max <= 1.0
        ):
            binary = (
                prediction
                >= self.mask_threshold
            )

        # Already a label image
        else:
            binary = (
                prediction
                > 0
            )

        binary = (
            binary.astype(
                np.uint8
            )
            * 255
        )

        # ----------------------------------------------------
        # Restore original image size
        # ----------------------------------------------------

        original_image = Image.open(
            original_image_path
        )

        original_size = (
            original_image.size
        )

        mask_image = Image.fromarray(
            binary
        ).convert(
            "L"
        )

        if mask_image.size != original_size:

            mask_image = (
                mask_image.resize(
                    original_size,
                    Image.Resampling.NEAREST,
                )
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_image.save(output_path)


    # ========================================================
    # Main batch inference
    # ========================================================

    def generate_masks(
        self,
        image_paths,
        data_root,
        mask_root,
        overwrite=False,
    ):

        data_root = Path(data_root).resolve()

        mask_root = Path(mask_root)

        output_root = (
            mask_root
            / "niaid_resunet"
            / "whole_lung"
        )

        # ====================================================
        # Only images that still require inference
        # ====================================================

        pending = []

        for image_path in image_paths:
            image_path = Path(
                image_path
            ).resolve()

            relative_path = (
                image_path.relative_to(data_root)
            )

            output_path = (
                output_root
                / relative_path
            ).with_suffix(
                ".png"
            )

            if (
                output_path.exists()
                and
                not overwrite
            ):

                continue

            pending.append(image_path)

        print(
            "[INFO] NIAID images pending:",
            len(pending),
        )

        if len(pending) == 0:

            return

        # ====================================================
        # Group by original directory
        #
        # Example:
        # source/train/Pneumonia
        #
        # This avoids basename collisions between classes.
        # ====================================================

        grouped = defaultdict(list)

        for image_path in pending:

            relative_path = (
                image_path.relative_to(
                    data_root
                )
            )

            grouped[
                relative_path.parent
            ].append(
                image_path
            )

        # ====================================================
        # NIAID inference
        # ====================================================

        with tempfile.TemporaryDirectory(
            prefix="niaid_resunet_"
        ) as temporary_root:

            temporary_root = Path(temporary_root)
            number_of_groups = len(grouped)

            for group_index, (
                relative_parent,
                group_images,
            ) in enumerate(
                grouped.items(),
                start=1,
            ):

                print(
                    "\n"
                    f"[NIAID] Group "
                    f"{group_index}/"
                    f"{number_of_groups}: "
                    f"{relative_parent}"
                )

                safe_group_name = (
                    str(relative_parent)
                    .replace(
                        "/",
                        "__",
                    )
                    .replace(
                        "\\",
                        "__",
                    )
                )

                group_root = (
                    temporary_root
                    / safe_group_name
                )

                group_output = (
                    group_root
                    / "predictions"
                )

                group_output.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                group_csv = (
                    group_root
                    / "inputs.csv"
                )

                # --------------------------------------------
                # Run official NIAID implementation
                # --------------------------------------------

                self._run_official_inference(
                    image_paths=group_images,
                    temporary_output_dir=group_output,
                    temporary_csv=group_csv,
                )

                # --------------------------------------------
                # Convert outputs to our common PNG structure
                # --------------------------------------------

                for image_path in group_images:

                    relative_path = image_path.relative_to(data_root)

                    final_output_path = (
                        output_root
                        / relative_path
                    ).with_suffix(
                        ".png"
                    )

                    prediction_path = (
                        self._find_generated_mask(
                            output_dir=group_output,
                            image_path=image_path,
                        )
                    )

                    self._save_binary_png(
                        prediction_path=prediction_path,
                        original_image_path=image_path,
                        output_path=final_output_path,
                    )

        print("[INFO] NIAID mask generation completed.")