from __future__ import annotations

import cv2
import numpy as np

from src.detections import Detection


DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 1.5
DEFAULT_SCALE_STEP = 0.05
DEFAULT_ANGLES = [0, 90, 180, 270]


def build_scales(scale_min: float, scale_max: float, scale_step: float) -> list[float]:
    if scale_min <= 0:
        raise ValueError("Scale min must be greater than 0")
    if scale_max < scale_min:
        raise ValueError("Scale max must be greater than or equal to scale min")
    if scale_step <= 0:
        raise ValueError("Scale step must be greater than 0")

    scales: list[float] = []
    current = scale_min
    while current <= scale_max + 1e-9:
        scales.append(round(current, 4))
        current += scale_step
    return scales


def _rotate_right_angle(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return np.rot90(image, 3).copy()
    if normalized == 180:
        return np.rot90(image, 2).copy()
    if normalized == 270:
        return np.rot90(image, 1).copy()
    raise ValueError("Only 0, 90, 180, 270 degree rotation is supported")


def _resize_template(template: np.ndarray, scale: float) -> np.ndarray | None:
    height, width = template.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width < 2 or new_height < 2:
        return None
    return cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _candidate_points(score_map: np.ndarray, threshold: float, max_candidates: int) -> list[tuple[int, int, float]]:
    ys, xs = np.where(score_map >= threshold)
    if len(xs) == 0:
        return []

    scores = score_map[ys, xs]
    order = np.argsort(scores)[::-1][:max_candidates]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


def match_template(
    pattern: np.ndarray,
    drawing: np.ndarray,
    *,
    threshold: float = 0.72,
    scales: list[float] | None = None,
    angles: list[int] | None = None,
    max_candidates_per_variant: int = 200,
) -> tuple[list[Detection], dict]:
    if pattern.ndim != 2 or drawing.ndim != 2:
        raise ValueError("pattern and drawing must be single-channel images")

    active_scales = scales or [1.0]
    active_angles = angles or [0]
    drawing_h, drawing_w = drawing.shape[:2]
    detections: list[Detection] = []
    best_match = {
        "confidence": 0.0,
        "bbox": None,
        "scale": None,
        "angle": None,
    }

    for angle in active_angles:
        rotated = _rotate_right_angle(pattern, angle)
        for scale in active_scales:
            template = _resize_template(rotated, scale)
            if template is None:
                continue

            template_h, template_w = template.shape[:2]
            if template_h > drawing_h or template_w > drawing_w:
                continue
            if float(template.std()) < 1.0:
                continue

            score_map = cv2.matchTemplate(drawing, template, cv2.TM_CCOEFF_NORMED)
            score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
            _, max_score, _, max_location = cv2.minMaxLoc(score_map)
            if max_score > best_match["confidence"]:
                best_match = {
                    "confidence": round(float(max_score), 4),
                    "bbox": [int(max_location[0]), int(max_location[1]), template_w, template_h],
                    "scale": round(float(scale), 4),
                    "angle": angle,
                }

            for x, y, score in _candidate_points(score_map, threshold, max_candidates_per_variant):
                detections.append(
                    Detection(
                        bbox=(x, y, template_w, template_h),
                        confidence=max(0.0, min(1.0, score)),
                        scale=scale,
                        angle=angle,
                    )
                )

    return detections, best_match
