from __future__ import annotations

import cv2
import numpy as np

from src.detections import Detection


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (220, 30, 30), 2)
        label = f"{detection.confidence:.2f}"
        label_y = max(y - 6, 12)
        cv2.putText(
            output,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return output

