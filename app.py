from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.pipeline import detect_pattern
from src.matcher import DEFAULT_SCALE_MAX, DEFAULT_SCALE_MIN, DEFAULT_SCALE_STEP


EXAMPLES_DIR = Path("data/examples")


def _load_examples() -> list[list[str]]:
    examples: list[list[str]] = []
    for pattern_path in sorted(EXAMPLES_DIR.glob("*_pattern.png")):
        drawing_path = pattern_path.with_name(pattern_path.name.replace("_pattern.png", "_drawing.png"))
        if drawing_path.exists():
            examples.append(
                [
                    str(pattern_path),
                    str(drawing_path),
                    0.72,
                    "grayscale",
                    True,
                    DEFAULT_SCALE_MIN,
                    DEFAULT_SCALE_MAX,
                    DEFAULT_SCALE_STEP,
                    False,
                ]
            )
    return examples


def run_detection(
    pattern,
    drawing,
    threshold,
    preprocessing,
    use_multiscale,
    scale_min,
    scale_max,
    scale_step,
    use_rotation,
):
    try:
        image, result = detect_pattern(
            pattern,
            drawing,
            threshold=float(threshold),
            preprocessing=preprocessing,
            use_multiscale=bool(use_multiscale),
            scale_min=float(scale_min),
            scale_max=float(scale_max),
            scale_step=float(scale_step),
            use_rotation=bool(use_rotation),
        )
        return image, json.dumps(result, indent=2)
    except Exception as exc:
        return drawing, json.dumps({"error": str(exc)}, indent=2)


with gr.Blocks(title="Zero-Shot Pattern Detection") as demo:
    gr.Markdown("# Zero-Shot Pattern Detection for Technical Drawings")

    with gr.Row():
        pattern_input = gr.Image(label="Pattern image", type="pil")
        drawing_input = gr.Image(label="Drawing image", type="pil")

    with gr.Row():
        threshold_input = gr.Slider(0.5, 0.99, value=0.72, step=0.01, label="Match threshold")
        preprocessing_input = gr.Dropdown(
            choices=["grayscale", "binary", "edges"],
            value="grayscale",
            label="Preprocessing",
        )
        multiscale_input = gr.Checkbox(value=True, label="Multi-scale")
        rotation_input = gr.Checkbox(value=False, label="Rotation 0/90/180/270")

    with gr.Row():
        scale_min_input = gr.Slider(0.2, 1.0, value=DEFAULT_SCALE_MIN, step=0.05, label="Scale min")
        scale_max_input = gr.Slider(1.0, 2.0, value=DEFAULT_SCALE_MAX, step=0.05, label="Scale max")
        scale_step_input = gr.Slider(0.02, 0.2, value=DEFAULT_SCALE_STEP, step=0.01, label="Scale step")

    run_button = gr.Button("Run detection", variant="primary")

    with gr.Row():
        output_image = gr.Image(label="Detected regions", type="numpy")
        output_json = gr.Code(label="Detections JSON", language="json")

    run_button.click(
        run_detection,
        inputs=[
            pattern_input,
            drawing_input,
            threshold_input,
            preprocessing_input,
            multiscale_input,
            scale_min_input,
            scale_max_input,
            scale_step_input,
            rotation_input,
        ],
        outputs=[output_image, output_json],
    )

    examples = _load_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[
                pattern_input,
                drawing_input,
                threshold_input,
                preprocessing_input,
                multiscale_input,
                scale_min_input,
                scale_max_input,
                scale_step_input,
                rotation_input,
            ],
        )


if __name__ == "__main__":
    demo.launch()
