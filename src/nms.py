from __future__ import annotations

from src.detections import Detection


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    return inter_area / float(area_a + area_b - inter_area)


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.3,
    max_detections: int = 100,
) -> list[Detection]:
    selected: list[Detection] = []

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_iou(detection.bbox, kept.bbox) <= iou_threshold for kept in selected):
            selected.append(detection)
            if len(selected) >= max_detections:
                break

    return selected

