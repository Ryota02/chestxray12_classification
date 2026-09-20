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

def find_images(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def get_mask_path(
    image_path,
    data_root,
    mask_root,
    mask_model,
    mask_region="whole_lung",
):
    image_path = Path(image_path)
    data_root = Path(data_root)
    mask_root = Path(mask_root)

    relative_path = image_path.relative_to(data_root)

    mask_path = (
        mask_root
        / mask_model
        / mask_region
        / relative_path
    ).with_suffix(".png")

    return mask_path


def load_binary_mask(
    mask_path,
    target_size=None,
    threshold=128,
):
    mask = Image.open(mask_path).convert("L")

    if target_size is not None and mask.size != target_size:
        mask = mask.resize(
            target_size,
            Image.Resampling.NEAREST,
        )

    mask_array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (
        mask_array
        >= threshold
    )


def apply_lung_mask(
    image,
    mask_path,
    mask_threshold=128,
):
    image = image.convert("L")

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask = load_binary_mask(
        mask_path=mask_path,
        target_size=image.size,
        threshold=mask_threshold,
    )

    output = np.zeros_like(
        image_array,
        dtype=np.uint8,
    )

    output[mask] = image_array[mask]

    return Image.fromarray(output).convert("RGB")


def moment_standardize_image(
    image,
    mask_path,
    mask_threshold=128,
    clip_z=3.0,
    eps=1e-8,
):
    image = image.convert("L")

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0

    mask = load_binary_mask(
        mask_path=mask_path,
        target_size=image.size,
        threshold=mask_threshold,
    )

    lung_pixels = image_array[mask]

    if lung_pixels.size == 0:
        raise RuntimeError(
            f"Empty lung mask: "
            f"{mask_path}"
        )

    m1 = float(np.mean(lung_pixels))
    m2 = float(np.mean(lung_pixels ** 2))

    variance = m2 - m1 ** 2
    variance = max(variance, eps)
    
    std = np.sqrt(variance)

    z = (lung_pixels - m1) / std
    z = np.clip(z, -clip_z, clip_z)
    z = (z + clip_z) / (2.0 * clip_z)

    output = np.zeros_like(
        image_array,
        dtype=np.float32,
    )
    output[mask] = z
    output = (output * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(output).convert("RGB")
    