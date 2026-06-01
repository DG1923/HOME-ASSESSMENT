from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array[:, :, :3].copy()


def to_gray(image: Image.Image | np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"))
    if array.ndim == 2:
        gray = array
    else:
        gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
    return gray.astype(np.uint8)


def crop_to_content(gray: np.ndarray, padding: int = 2) -> np.ndarray:
    mask = gray < 245
    if not np.any(mask):
        return gray

    ys, xs = np.where(mask)
    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding + 1, gray.shape[0])
    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, gray.shape[1])
    return gray[y1:y2, x1:x2]


def preprocess_image(
    image: Image.Image | np.ndarray,
    mode: str = "grayscale",
    *,
    crop_content: bool = False,
) -> np.ndarray:
    gray = to_gray(image)
    if crop_content:
        gray = crop_to_content(gray)

    if mode == "grayscale":
        return gray

    if mode == "binary":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    if mode == "edges":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, 50, 150)

    raise ValueError(f"Unsupported preprocessing mode: {mode}")

