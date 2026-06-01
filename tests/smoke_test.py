from __future__ import annotations

from PIL import Image, ImageDraw

from src.detections import Detection
from src.nms import non_max_suppression
from src.pipeline import detect_pattern


def _make_pattern() -> Image.Image:
    image = Image.new("RGB", (40, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 32, 32), outline="black", width=3)
    draw.line((8, 20, 32, 20), fill="black", width=3)
    draw.line((20, 8, 20, 32), fill="black", width=3)
    return image


def _make_drawing(pattern: Image.Image) -> Image.Image:
    image = Image.new("RGB", (220, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.line((0, 110, 220, 110), fill="black", width=1)
    image.paste(pattern, (30, 30))
    image.paste(pattern, (130, 70))
    return image


def test_pipeline_finds_synthetic_pattern() -> None:
    pattern = _make_pattern()
    drawing = _make_drawing(pattern)

    _, result = detect_pattern(
        pattern,
        drawing,
        threshold=0.9,
        preprocessing="grayscale",
        use_multiscale=False,
        use_rotation=False,
    )

    assert result["count"] >= 2
    assert all("bbox" in detection for detection in result["detections"])


def test_pipeline_finds_scaled_query_pattern() -> None:
    pattern = _make_pattern()
    drawing = _make_drawing(pattern)
    enlarged_query = pattern.resize((90, 90), Image.Resampling.BICUBIC)

    _, result = detect_pattern(
        enlarged_query,
        drawing,
        threshold=0.7,
        preprocessing="grayscale",
        use_multiscale=True,
        scale_min=0.35,
        scale_max=1.0,
        scale_step=0.05,
        use_rotation=False,
    )

    assert result["count"] >= 2
    assert result["best_match_before_threshold"]["confidence"] >= 0.7


def test_nms_keeps_highest_overlap() -> None:
    detections = [
        Detection((10, 10, 20, 20), 0.8, 1.0, 0),
        Detection((12, 12, 20, 20), 0.9, 1.0, 0),
        Detection((80, 80, 20, 20), 0.7, 1.0, 0),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.3)

    assert len(kept) == 2
    assert kept[0].confidence == 0.9


if __name__ == "__main__":
    test_pipeline_finds_synthetic_pattern()
    test_pipeline_finds_scaled_query_pattern()
    test_nms_keeps_highest_overlap()
    print("smoke tests passed")
