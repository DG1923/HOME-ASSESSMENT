# Zero-Shot Pattern Detection for Technical BOM Drawings

This project detects user-provided visual patterns in black-and-white technical BOM drawings without training, fine-tuning, or hardcoded symbol classes.

Given:

- one pattern/query image
- one technical drawing image

The system returns:

- bounding boxes for matched regions
- confidence score for each detection
- annotated visualization image
- JSON output for downstream inspection

## Method

The detector uses classical computer vision with OpenCV template matching. This is suitable for the assessment because technical drawings are mostly grayscale or binary, line-based, and contain repeated geometric symbols.

Pipeline:

1. Validate pattern and drawing inputs.
2. Convert images to grayscale, binary, or edge representation.
3. Crop blank whitespace around the query pattern.
4. Search the drawing using normalized template matching.
5. Run multi-scale matching from `0.35` to `1.50` by default.
6. Optionally test rotations at `0`, `90`, `180`, and `270` degrees.
7. Collect locations above the confidence threshold.
8. Apply IoU-based non-maximum suppression to remove duplicate boxes.
9. Draw final boxes on the original drawing.
10. Return JSON results with bbox, confidence, scale, and angle.

The uploaded pattern itself defines the search target, so the system remains zero-shot for new symbols.

## Project Structure

```text
src/
  detections.py
  matcher.py
  nms.py
  pipeline.py
  preprocess.py
  visualize.py
data/
  examples/
app.py
docs/system_design_spec.md
requirements.txt
tests/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Gradio Demo

```bash
python app.py
```

Open the local Gradio URL printed in the terminal. The default local URL is usually:

```text
http://127.0.0.1:7860
```

## Demo Usage

1. Upload a cropped pattern image.
2. Upload the target technical drawing.
3. Choose preprocessing mode:
   - `grayscale`: default, usually best first choice.
   - `binary`: useful for clean black-and-white drawings.
   - `edges`: useful when line thickness or screenshot quality differs.
4. Keep multi-scale enabled for patterns that may be larger or smaller than the drawing instance.
5. Enable rotation if the same pattern may appear rotated.
6. Run detection.

Recommended starting settings:

```text
threshold: 0.72
preprocessing: grayscale
multi-scale: enabled
scale min: 0.35
scale max: 1.50
scale step: 0.05
rotation: disabled unless needed
```

If no boxes appear but `best_match_before_threshold.confidence` is close to the threshold, lower the threshold slightly or try `edges` preprocessing.

## Output Format

```json
{
  "detections": [
    {
      "bbox": [120, 240, 32, 45],
      "confidence": 0.92,
      "scale": 1.0,
      "angle": 0
    }
  ],
  "count": 1,
  "runtime_seconds": 0.18,
  "threshold": 0.72,
  "preprocessing": "grayscale",
  "multiscale": true,
  "rotation": false,
  "best_match_before_threshold": {
    "confidence": 0.92,
    "bbox": [120, 240, 32, 45],
    "scale": 0.45,
    "angle": 0
  }
}
```

`best_match_before_threshold` is included to make failed or low-confidence cases easier to analyze.

## Example Data

Example images live in:

```text
data/examples/
```

Expected naming convention:

```text
example1_pattern.png
example1_drawing.png
example2_pattern.png
example2_drawing.png
```

## Verification

Run the smoke test:

```bash
python -m tests.smoke_test
```

The test verifies:

- exact pattern matching
- scaled query matching
- non-maximum suppression behavior

## Design Document

See `design.md` for:

- problem analysis
- architecture
- module descriptions
- requirement mapping
- limitations and future improvements

## Limitations

- Very simple line patterns can match unrelated line segments.
- Crops with extra wires, labels, or surrounding context can reduce the score.
- Strong blur, scan artifacts, or line thickness mismatch can reduce confidence.
- Large drawings with fine scale steps and rotation enabled can increase runtime.
- The current method is image-similarity based; it does not semantically classify gates or symbols.
