"""
Gradio GUI for SAM3 segmentation.

Tabs:
  1. Single Image  — upload image, pick classes, see results instantly
  2. Batch Folder  — point to a folder, run over all images, download COCO JSON
  3. Help          — tips and thresholds

Run standalone:
    python gui/app.py
    python gui/app.py --config path/to/config.yaml --port 7860
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from src.utils import load_config, get_image_paths, visualize_results, PALETTE
from src.pipeline import Pipeline
from src.batch_processor import BatchProcessor
from src.coco_writer import COCOWriter

# ── Global state ──────────────────────────────────────────────────────────────
_pipeline: Pipeline | None = None
_batch_proc: BatchProcessor | None = None
_config: dict = {}
_config_path: str = str(ROOT / "config.yaml")


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        raise gr.Error("Models not loaded. Click 'Load Models' first.")
    return _pipeline


def _get_batch_proc() -> BatchProcessor:
    global _batch_proc
    if _batch_proc is None:
        raise gr.Error("Models not loaded. Click 'Load Models' first.")
    return _batch_proc


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(config_path: str) -> str:
    global _pipeline, _batch_proc, _config, _config_path
    try:
        _config_path = config_path
        _config = load_config(config_path)
        status_msg = f"Loading models from config: {config_path}"
        yield gr.update(value=status_msg, visible=True)

        _pipeline = Pipeline(_config)
        yield gr.update(value="Loading SAM3 detector...", visible=True)
        _pipeline.detector.load()

        _batch_proc = BatchProcessor(_pipeline, _config)
        backend = _config.get("models", {}).get("detector", {}).get("backend", "video")
        yield gr.update(value=f"✓ Models loaded successfully! (backend: {backend})", visible=True)
    except Exception as e:
        yield gr.update(value=f"✗ Error loading models: {e}", visible=True)


# ── Single image tab ──────────────────────────────────────────────────────────

def process_single_image(
    image: np.ndarray,
    classes_text: str,
    selected_classes: list[str],
    confidence_threshold: float,
    sam_score_threshold: float,
    alpha: float,
) -> tuple:
    """Run pipeline on one image, return (annotated_image, results_json, coco_json_str)."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Combine text input and checkbox selection
    text_classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    active_classes = list(dict.fromkeys(selected_classes + text_classes))  # dedup, preserve order

    if not active_classes:
        raise gr.Error("Please select or type at least one class.")

    pipeline = _get_pipeline()
    # Override thresholds from sliders
    pipeline.confidence_threshold = confidence_threshold
    pipeline.sam_score_threshold = sam_score_threshold

    pil_image = Image.fromarray(image.astype(np.uint8))
    results = pipeline.process_image(pil_image, active_classes)

    # Annotated visualization
    if results:
        viz = visualize_results(pil_image, results, active_classes, alpha=alpha)
    else:
        viz = pil_image

    # Results JSON for display
    display_results = []
    for r in results:
        display_results.append({
            "class": r["class"],
            "confidence": round(r["confidence"], 3),
            "sam_score": round(r.get("sam_score", 0.0), 3),
            "bbox_normalized": [round(v, 4) for v in r["bbox"]],
            "has_mask": r.get("mask") is not None,
        })

    results_json = json.dumps(display_results, indent=2)

    # Build COCO for download
    writer = COCOWriter(active_classes)
    img_w, img_h = pil_image.size
    writer.add_image_results(
        file_name="uploaded_image.jpg",
        results=results,
        img_w=img_w,
        img_h=img_h,
    )
    coco_dict = writer.to_dict()
    coco_str = json.dumps(coco_dict, indent=2)

    # Save COCO to temp file for download
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="coco_"
    )
    tmp.write(coco_str)
    tmp.close()

    summary = f"Found {len(results)} instance(s): " + ", ".join(
        f"{r['class']} ({r['confidence']:.2f})" for r in display_results
    )

    return np.array(viz), results_json, tmp.name, summary


# ── Batch folder tab ──────────────────────────────────────────────────────────

def run_batch(
    folder_path: str,
    classes_text: str,
    selected_classes: list[str],
    confidence_threshold: float,
    sam_score_threshold: float,
    save_viz: bool,
    output_dir: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple:
    """Run batch processing and return (log_text, coco_json_path)."""
    text_classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    active_classes = list(dict.fromkeys(selected_classes + text_classes))

    if not folder_path:
        raise gr.Error("Please enter a folder path.")
    if not active_classes:
        raise gr.Error("Please select or type at least one class.")
    if not Path(folder_path).exists():
        raise gr.Error(f"Folder not found: {folder_path}")

    batch = _get_batch_proc()
    batch.pipeline.confidence_threshold = confidence_threshold
    batch.pipeline.sam_score_threshold = sam_score_threshold

    image_paths = get_image_paths(folder_path)
    if not image_paths:
        raise gr.Error(f"No images found in {folder_path}")

    log_lines = [f"Starting batch: {len(image_paths)} images", f"Classes: {active_classes}", ""]
    log_text = "\n".join(log_lines)
    out_dir = output_dir.strip() if output_dir.strip() else None

    processed = [0]  # mutable for closure

    def prog_cb(current, total, msg):
        processed[0] = current
        progress(current / total if total > 0 else 0, desc=msg)

    summary = batch.run(
        input_path=folder_path,
        active_classes=active_classes,
        output_dir=out_dir,
        save_viz=save_viz,
        save_coco=True,
        progress_callback=prog_cb,
    )

    lines = [
        f"✓ Batch complete!",
        f"  Processed: {summary['processed']}/{summary['total_images']} images",
        f"  Annotations: {summary['total_annotations']}",
        f"  Output dir: {summary['output_dir']}",
    ]
    if summary.get("coco_summary", {}).get("classes"):
        lines.append("\nClass breakdown:")
        for cls, count in sorted(summary["coco_summary"]["classes"].items()):
            lines.append(f"  {cls}: {count} instances")
    if summary["failed"]:
        lines.append(f"\n⚠ Failures: {summary['failed']}")
        for e in summary["errors"][:5]:
            lines.append(f"  - {e}")

    log_text = "\n".join(lines)
    coco_path = summary.get("coco_path", "")
    return log_text, coco_path if coco_path else None


# ── Build the Gradio UI ───────────────────────────────────────────────────────

def build_app(config_path: str = None) -> gr.Blocks:
    global _config, _config_path
    if config_path:
        _config_path = config_path
    try:
        _config = load_config(_config_path)
    except Exception:
        _config = {}

    default_classes: list[str] = _config.get("classes", [])
    default_output_dir = _config.get("output", {}).get("dir", "outputs")

    with gr.Blocks(
        title="SAM3 Auto-Labeler",
        theme=gr.themes.Soft(),
        css=".result-box {font-family: monospace; font-size: 12px;}",
    ) as demo:
        gr.Markdown(
            """
# SAM3 Auto-Labeler
Detect objects by class name → segment → export COCO JSON
"""
        )

        # ── Top bar: model loading ──
        with gr.Row():
            with gr.Column(scale=4):
                config_input = gr.Textbox(
                    label="Config file path",
                    value=_config_path,
                    placeholder="path/to/config.yaml",
                )
            with gr.Column(scale=2):
                backend_radio = gr.Radio(
                    choices=["image", "video"],
                    value=_config.get("models", {}).get("detector", {}).get("backend", "image"),
                    label="Backend",
                    info="image = faster (no tracker)  |  video = full SAM3 tracker",
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Models", variant="primary", size="lg")
        model_status = gr.Textbox(label="Model status", interactive=False, visible=True)

        def _load_with_backend(config_path: str, backend: str):
            # Patch backend into config before loading
            import yaml
            from pathlib import Path as _Path
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                cfg.setdefault("models", {}).setdefault("detector", {})["backend"] = backend
                # Write back so Pipeline reads the chosen backend
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            except Exception:
                pass
            yield from load_models(config_path)

        load_btn.click(
            fn=_load_with_backend,
            inputs=[config_input, backend_radio],
            outputs=[model_status],
        )

        gr.Markdown("---")

        # ── Class list management (shared across tabs) ──
        with gr.Accordion("Manage Class List", open=False):
            gr.Markdown("Edit the class list below (one class per line), then click **Apply** to update both tabs.")
            with gr.Row():
                class_list_editor = gr.Textbox(
                    label="Class list (one per line)",
                    value="\n".join(default_classes),
                    lines=8,
                    scale=4,
                )
                with gr.Column(scale=1):
                    apply_classes_btn = gr.Button("Apply Class List", variant="primary")
                    reset_classes_btn = gr.Button("Reset to Config Defaults")

        gr.Markdown("---")

        # ── Shared controls helper ──
        def class_controls(prefix: str):
            with gr.Row():
                classes_text = gr.Textbox(
                    label="Additional classes (comma-separated)",
                    placeholder="e.g. helmet, forklift, pallet",
                    scale=3,
                )
            with gr.Row():
                selected_classes = gr.CheckboxGroup(
                    choices=default_classes,
                    value=default_classes,
                    label="Active classes for this run",
                )
            with gr.Row():
                confidence = gr.Slider(
                    0.0, 1.0,
                    value=_config.get("pipeline", {}).get("confidence_threshold", 0.03),
                    step=0.01,
                    label="Min detection confidence  (start: 0.03 · lower → more recall · raise → fewer false positives)",
                )
                sam_score = gr.Slider(
                    0.0, 1.0,
                    value=_config.get("pipeline", {}).get("sam_score_threshold", 0.0),
                    step=0.05,
                    label="Min SAM mask quality score  (0.0 = keep all)",
                )
            return classes_text, selected_classes, confidence, sam_score

        # ────────────────────────────────────────────
        # Tab 1: Single Image
        # ────────────────────────────────────────────
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image", type="numpy")
                    classes_text_1, selected_classes_1, confidence_1, sam_score_1 = class_controls("s1")
                    alpha_1 = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Mask opacity")
                    run_btn = gr.Button("Run Detection + Segmentation", variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Result", type="numpy")
                    summary_text = gr.Textbox(label="Summary", interactive=False)
                    results_json = gr.Code(
                        label="Detections (JSON)",
                        language="json",
                        elem_classes=["result-box"],
                    )
                    coco_download = gr.File(label="Download COCO JSON")

            run_btn.click(
                fn=process_single_image,
                inputs=[
                    input_image,
                    classes_text_1,
                    selected_classes_1,
                    confidence_1,
                    sam_score_1,
                    alpha_1,
                ],
                outputs=[output_image, results_json, coco_download, summary_text],
            )

        # ────────────────────────────────────────────
        # Tab 2: Batch Folder
        # ────────────────────────────────────────────
        with gr.Tab("Batch Folder"):
            with gr.Row():
                with gr.Column(scale=1):
                    folder_input = gr.Textbox(
                        label="Input folder path",
                        placeholder="/path/to/images/",
                    )
                    classes_text_2, selected_classes_2, confidence_2, sam_score_2 = class_controls("s2")
                    with gr.Row():
                        save_viz_check = gr.Checkbox(
                            value=True, label="Save annotated visualizations"
                        )
                        output_dir_input = gr.Textbox(
                            label="Output directory (leave blank for <input_folder>/outputs/)",
                            placeholder="default: <input_folder>/outputs/",
                        )
                    batch_btn = gr.Button(
                        "Start Batch Processing", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    batch_log = gr.Textbox(
                        label="Progress log",
                        lines=20,
                        interactive=False,
                        elem_classes=["result-box"],
                    )
                    coco_batch_download = gr.File(label="Download COCO JSON")

            batch_btn.click(
                fn=run_batch,
                inputs=[
                    folder_input,
                    classes_text_2,
                    selected_classes_2,
                    confidence_2,
                    sam_score_2,
                    save_viz_check,
                    output_dir_input,
                ],
                outputs=[batch_log, coco_batch_download],
            )

        # ── Wire class list management buttons ──
        def _apply_class_list(class_text: str):
            classes = [c.strip() for c in class_text.split("\n") if c.strip()]
            if not classes:
                raise gr.Error("Class list cannot be empty.")
            return (
                gr.update(choices=classes, value=classes),
                gr.update(choices=classes, value=classes),
            )

        def _reset_class_list():
            classes = default_classes
            return (
                "\n".join(classes),
                gr.update(choices=classes, value=classes),
                gr.update(choices=classes, value=classes),
            )

        apply_classes_btn.click(
            fn=_apply_class_list,
            inputs=[class_list_editor],
            outputs=[selected_classes_1, selected_classes_2],
        )
        reset_classes_btn.click(
            fn=_reset_class_list,
            inputs=[],
            outputs=[class_list_editor, selected_classes_1, selected_classes_2],
        )

        # ────────────────────────────────────────────
        # Tab 3: Help / Tips
        # ────────────────────────────────────────────
        with gr.Tab("Help"):
            gr.Markdown(
                """
## Tips for best results

### Class names
- Use short, standard nouns: `person`, `forklift`, `fire extinguisher`
- SAM3 uses CLIP text embeddings — the closer your class name is to ImageNet-style labels, the better

### Confidence threshold
- Start at `0.03`, lower to `0.01` for more recall, raise to `0.08` to cut false positives
- Each image will show different optimal values depending on object density

### SAM mask quality score
- Keep at `0.0` to retain all masks — the confidence threshold is the main filter
- Raise to `0.5` only if you see many poor-quality masks (e.g. thin slivers)

### Visualization
- Each class gets a distinct colour from the palette
- Overlapping masks from different classes both show through

### Batch processing
- Images are processed one at a time (GPU memory constraint)
- Failed images are skipped and logged in `summary.json`

### Output format
- COCO JSON is compatible with **CVAT**, **Label Studio**, **Roboflow**, and standard training pipelines
- Visualizations are saved in `outputs/run_*/visualizations/`

### GPU memory
~6 GB VRAM (SAM3 CLIP+DETR detector + mask decoder)
"""
            )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Gradio GUI")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app(config_path=args.config)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
