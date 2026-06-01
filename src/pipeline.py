from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from src.matcher import (
    DEFAULT_ANGLES,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    build_scales,
    match_template,
)
from src.nms import non_max_suppression
from src.preprocess import preprocess_image, to_rgb_array
from src.visualize import draw_detections


def detect_pattern(
    pattern_image: Image.Image | np.ndarray,
    drawing_image: Image.Image | np.ndarray,
    *,
    threshold: float = 0.72,
    preprocessing: str = "grayscale",
    use_multiscale: bool = True,
    use_rotation: bool = False,
    scale_min: float = DEFAULT_SCALE_MIN,
    scale_max: float = DEFAULT_SCALE_MAX,
    scale_step: float = DEFAULT_SCALE_STEP,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    if pattern_image is None:
        raise ValueError("Pattern image is required")
    if drawing_image is None:
        raise ValueError("Drawing image is required")

    started = time.perf_counter()
    original_drawing = to_rgb_array(drawing_image)
    pattern = preprocess_image(pattern_image, preprocessing, crop_content=True)
    drawing = preprocess_image(drawing_image, preprocessing)

    if pattern.shape[0] > drawing.shape[0] or pattern.shape[1] > drawing.shape[1]:
        raise ValueError("Pattern image must not be larger than drawing image")

    scales = build_scales(scale_min, scale_max, scale_step) if use_multiscale else [1.0]
    angles = DEFAULT_ANGLES if use_rotation else [0]
    candidates, best_match = match_template(
        pattern,
        drawing,
        threshold=threshold,
        scales=scales,
        angles=angles,
    )
    detections = non_max_suppression(candidates, nms_iou_threshold, max_detections)
    visualization = draw_detections(original_drawing, detections)

    result = {
        "detections": [detection.to_dict() for detection in detections],
        "count": len(detections),
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "threshold": threshold,
        "preprocessing": preprocessing,
        "multiscale": use_multiscale,
        "rotation": use_rotation,
        "scales_tested": scales,
        "best_match_before_threshold": best_match,
    }
    return visualization, result
