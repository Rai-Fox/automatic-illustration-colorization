from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

LOGGER = logging.getLogger(__name__)


def to_float32_255(image: np.ndarray) -> np.ndarray:
    """Convert a NumPy image to float32 in the [0, 255] range."""
    img = image.astype(np.float32)

    if img.size == 0:
        raise ValueError("Empty image.")

    if img.max() <= 1.0:
        img = img * 255.0

    return np.clip(img, 0.0, 255.0).astype(np.float32)


def rgb_to_luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to luminance in the [0, 255] range."""
    img = to_float32_255(image_rgb)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("rgb_to_luminance expects an RGB image with shape [H, W, 3].")

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return 0.299 * r + 0.587 * g + 0.114 * b


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to grayscale in the [0, 255] range."""
    img = to_float32_255(image)

    if img.ndim == 2:
        return img

    if img.ndim == 3 and img.shape[-1] == 3:
        return rgb_to_luminance(img)

    raise ValueError("Image must have shape [H, W] or [H, W, 3].")


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image to RGB [H, W, 3] in the [0, 255] range."""
    img = to_float32_255(image)

    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[-1] == 3:
        return img

    raise ValueError("Image must have shape [H, W] or [H, W, 3].")


def resize_image(image: np.ndarray, image_size: tuple[int, int] | None) -> np.ndarray:
    """Resize an image if a target size is provided."""
    if image_size is None:
        return image

    import cv2

    height, width = image_size

    return cv2.resize(
        image,
        dsize=(width, height),
        interpolation=cv2.INTER_AREA,
    )


def resolve_torch_device(
    device: str | torch.device,
    *,
    logger: logging.Logger | None = None,
    context: str = "benchmark",
) -> torch.device:
    """
    Resolve the effective torch device, downgrading CUDA requests on CPU-only builds.
    """
    resolved_logger = logger or LOGGER
    if isinstance(device, str) and device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_obj = torch.device(device)

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        resolved_logger.warning(
            "Requested device=%s for %s, but CUDA is unavailable. Falling back to cpu.",
            device_obj,
            context,
        )
        return torch.device("cpu")
    return device_obj


def numpy_images_to_torch_batch(
    images: list[np.ndarray],
    image_size: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert a list of NumPy images to a torch batch [N, 3, H, W] in [0, 1]."""
    effective_device = resolve_torch_device(device)
    processed = []

    for image in images:
        img = ensure_rgb(image)
        img = resize_image(img, image_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)

    batch = np.stack(processed, axis=0)

    return torch.from_numpy(batch).float().to(effective_device)


def torch_batch_to_uint8(batch: torch.Tensor) -> torch.Tensor:
    """Convert a torch image batch to uint8 in the [0, 255] range."""
    x = batch.detach()

    if x.max() <= 1.0:
        x = x * 255.0

    return x.clamp(0, 255).to(torch.uint8)


def rgb_to_lab_image(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to Lab."""
    img = ensure_rgb(image_rgb).astype(np.float32) / 255.0
    return rgb2lab(img)


def median_lab_color(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute the median Lab color inside a mask."""
    lab = rgb_to_lab_image(image_rgb)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("Mask is empty.")

    pixels = lab[mask_bool]

    return np.median(pixels, axis=0)


def delta_e_lab(color_1: np.ndarray, color_2: np.ndarray) -> float:
    """Compute the CIEDE2000 distance between two Lab colors."""
    c1 = np.asarray(color_1).reshape(1, 1, 3)
    c2 = np.asarray(color_2).reshape(1, 1, 3)

    return float(deltaE_ciede2000(c1, c2)[0, 0])


def image_to_pil_grayscale(image: np.ndarray) -> Image.Image:
    """Convert a NumPy image to a grayscale PIL image."""
    gray = to_grayscale(image)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return Image.fromarray(gray)


def get_region_colors_from_images(
    y_images: list[np.ndarray],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    character_id: str,
    region: str,
) -> list[np.ndarray]:
    """Collect Lab colors for a character region across generated images."""
    colors = []

    for image_rgb, masks_i in zip(y_images, image_masks, strict=False):
        if character_id not in masks_i:
            continue

        if region not in masks_i[character_id]:
            continue

        mask_i = masks_i[character_id][region]

        if mask_i.sum() == 0:
            continue

        color_i = median_lab_color(image_rgb, mask_i)
        colors.append(color_i)

    return colors
