from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _to_float32_255(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    if array.size == 0:
        raise ValueError("Empty image.")
    if array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = _to_float32_255(image)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Image must have shape [H, W] or [H, W, 3].")

    return array.astype(np.uint8)


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(to_rgb_uint8(image), cv2.COLOR_RGB2BGR)


def pil_from_numpy(image: np.ndarray) -> Image.Image:
    return Image.fromarray(to_rgb_uint8(image))
