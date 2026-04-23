"""
SAM3 Model Comparison GUI.

Side-by-side comparison of two SAM3 checkpoints on the same image.
  - Left panel:  Model A (original / baseline)
  - Right panel: Model B (fine-tuned)
  - Shared:      image, class selection, confidence threshold, mask opacity
  - Per-class breakdown gallery below each combined result

Launch standalone:
    python gui/compare.py
    python gui/compare.py --config path/to/config.yaml --port 7861

Or via run_batch.py:
    python run_batch.py --compare
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from src.models.sam3_image_detector import Sam3ImageDetector
from src.utils import visualize_results, get_class_color, load_config

# ── Global state ──────────────────────────────────────────────────────────────
_det_a: Sam3ImageDetector | None = None
_det_b: Sam3ImageDetector | None = None
_lock = threading.Lock()   # serialize GPU inference between the two models


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_detector(path: str) -> Sam3ImageDetector:
    det = Sam3ImageDetector(
        sam3_local_path=path.strip(),
        device="cuda",
    )
    det.load()
    return det


def load_model_a(path: str):
    global _det_a
    if not path.strip():
        yield gr.update(value="⚠ Enter a model path first.")
        return
    yield gr.update(value="Loading Model A …")
    try:
        with _lock:
            if _det_a is not None:
                _det_a.unload()
                _det_a = None
            _det_a = _build_detector(path)
        yield gr.update(value=f"✓ Model A loaded  ({Path(path.strip()).name})")
    except Exception as e:
        yield gr.update(value=f"✗ {e}")


def load_model_b(path: str):
    global _det_b
    if not path.strip():
        yield gr.update(value="⚠ Enter a model path first.")
        return
    yield gr.update(value="Loading Model B …")
    try:
        with _lock:
            if _det_b is not None:
                _det_b.unload()
                _det_b = None
            _det_b = _build_detector(path)
        yield gr.update(value=f"✓ Model B loaded  ({Path(path.strip()).name})")
    except Exception as e:
        yield gr.update(value=f"✗ {e}")


# ── Inference helpers ─────────────────────────────────────────────────────────

def _run_detector(det: Sam3ImageDetector, pil_img: Image.Image,
                  classes: list[str], conf: float) -> tuple[list[dict], str]:
    """Return (results, error_str). error_str is empty on success."""
    if det is None:
        return [], "Model not loaded."
    try:
        with _lock:
            results = det.detect_and_segment(pil_img, classes,
                                              confidence_threshold=conf)
        return results, ""
    except Exception as e:
        return [], str(e)


def _per_class_gallery(pil_img: Image.Image, results: list[dict],
                       classes: list[str]) -> list[tuple]:
    """
    Build a Gallery-ready list of (np.array, caption) per class.
    Each image shows only that class's detections on the original image.
    Classes with no detections are shown as a dark placeholder.
    """
    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["class"], []).append(r)

    items = []
    for cls in classes:
        dets = by_class.get(cls, [])
        if dets:
            viz = visualize_results(pil_img, dets, classes, alpha=0.55,
                                    draw_bbox=False, draw_labels=False)
            caption = f"{cls}  ·  {len(dets)} det  ·  max {max(d['confidence'] for d in dets):.2f}"
        else:
            # Dark placeholder with class label
            w, h = pil_img.size
            placeholder = Image.new("RGB", (w, h), (30, 30, 30))
            draw = ImageDraw.Draw(placeholder)
            color = get_class_color(cls, classes)
            draw.text((10, 10), f"{cls}\n(not detected)", fill=color)
            viz = placeholder
            caption = f"{cls}  ·  not detected"
        items.append((np.array(viz), caption))
    return items


def _detection_summary(results: list[dict], err: str) -> str:
    if err:
        return f"Error: {err}"
    if not results:
        return "No detections above threshold."
    by_cls: dict[str, list[float]] = {}
    for r in results:
        by_cls.setdefault(r["class"], []).append(r["confidence"])
    lines = [
        f"{cls}: {len(confs)} det  |  max {max(confs):.3f}  avg {sum(confs)/len(confs):.3f}"
        for cls, confs in sorted(by_cls.items())
    ]
    lines.append(f"\nTotal: {len(results)} detections")
    return "\n".join(lines)


# ── Main comparison function ───────────────────────────────────────────────────

def run_comparison(
    image_np,
    classes_text: str,
    selected_classes: list[str],
    confidence: float,
    alpha: float,
):
    if image_np is None:
        raise gr.Error("Upload an image first.")

    text_classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    active_classes = list(dict.fromkeys(selected_classes + text_classes))
    if not active_classes:
        raise gr.Error("Select or type at least one class.")
    if _det_a is None and _det_b is None:
        raise gr.Error("Load at least one model before running.")

    pil_img = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")

    # Run both detectors sequentially (lock inside _run_detector)
    results_a, err_a = _run_detector(_det_a, pil_img, active_classes, confidence)
    results_b, err_b = _run_detector(_det_b, pil_img, active_classes, confidence)

    # Combined visualizations
    viz_a = np.array(
        visualize_results(pil_img, results_a, active_classes, alpha=alpha,
                          draw_bbox=False, draw_labels=False)
        if results_a else pil_img
    )
    viz_b = np.array(
        visualize_results(pil_img, results_b, active_classes, alpha=alpha,
                          draw_bbox=False, draw_labels=False)
        if results_b else pil_img
    )

    # Per-class galleries
    gallery_a = _per_class_gallery(pil_img, results_a, active_classes)
    gallery_b = _per_class_gallery(pil_img, results_b, active_classes)

    summary_a = _detection_summary(results_a, err_a)
    summary_b = _detection_summary(results_b, err_b)

    return viz_a, summary_a, gallery_a, viz_b, summary_b, gallery_b


# ── Class list management ─────────────────────────────────────────────────────

def _apply_class_list(class_text: str):
    classes = [c.strip() for c in class_text.split("\n") if c.strip()]
    if not classes:
        raise gr.Error("Class list cannot be empty.")
    return (
        gr.update(choices=classes, value=classes),
        gr.update(choices=classes, value=classes),
    )


def _reset_class_list(default_classes):
    return (
        "\n".join(default_classes),
        gr.update(choices=default_classes, value=default_classes),
        gr.update(choices=default_classes, value=default_classes),
    )


# ── Build Gradio UI ───────────────────────────────────────────────────────────

_CSS = """
.panel-a { border: 2px solid #4CAF50; border-radius: 8px; padding: 10px; }
.panel-b { border: 2px solid #2196F3; border-radius: 8px; padding: 10px; }
.shared-panel { background: #1e1e1e; border-radius: 8px; padding: 10px; }
.status-box textarea { font-size: 12px !important; }
"""


def build_compare_app(config_path: str | None = None) -> gr.Blocks:
    default_classes: list[str] = []
    default_conf = 0.03

    if config_path and Path(config_path).exists():
        try:
            cfg = load_config(config_path)
            default_classes = cfg.get("classes", [])
            default_conf = cfg.get("pipeline", {}).get("confidence_threshold", 0.03)
        except Exception:
            pass

    with gr.Blocks(
        title="SAM3 Model Comparison",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# SAM3 Model Comparison\n"
            "Load two checkpoints, upload an image, and compare segmentation results side-by-side."
        )

        # ── Model loading ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_classes=["panel-a"]):
                gr.Markdown("### Model A  (Original)")
                path_a = gr.Textbox(
                    label="Checkpoint path",
                    value=str(ROOT / "sam3"),
                    placeholder="/home/.../checkpoints/phase1_h200/best",
                )
                load_btn_a = gr.Button("Load Model A", variant="primary")
                status_a = gr.Textbox(
                    label="Status", interactive=False,
                    elem_classes=["status-box"],
                )

            with gr.Column(elem_classes=["panel-b"]):
                gr.Markdown("### Model B  (Fine-tuned)")
                path_b = gr.Textbox(
                    label="Checkpoint path",
                    placeholder="/home/.../checkpoints/phase2_h200_ft3/best",
                )
                load_btn_b = gr.Button("Load Model B", variant="primary")
                status_b = gr.Textbox(
                    label="Status", interactive=False,
                    elem_classes=["status-box"],
                )

        load_btn_a.click(fn=load_model_a, inputs=[path_a], outputs=[status_a])
        load_btn_b.click(fn=load_model_b, inputs=[path_b], outputs=[status_b])

        gr.Markdown("---")

        # ── Class management ───────────────────────────────────────────────
        with gr.Accordion("Manage Class List", open=False):
            gr.Markdown("One class per line. Click **Apply** to update both panels.")
            with gr.Row():
                class_list_editor = gr.Textbox(
                    label="Class list",
                    value="\n".join(default_classes),
                    lines=6,
                    scale=4,
                )
                with gr.Column(scale=1):
                    apply_btn = gr.Button("Apply", variant="primary")
                    reset_btn = gr.Button("Reset to Config Defaults")

        gr.Markdown("---")

        # ── Shared controls ────────────────────────────────────────────────
        gr.Markdown("### Input & Shared Parameters")
        with gr.Row():
            image_input = gr.Image(
                label="Input Image  (shared)", type="numpy", scale=2,
            )
            with gr.Column(scale=1):
                classes_text = gr.Textbox(
                    label="Extra classes (comma-separated)",
                    placeholder="e.g. helmet, forklift",
                )
                selected_classes = gr.CheckboxGroup(
                    choices=default_classes,
                    value=default_classes,
                    label="Active classes",
                )
                confidence = gr.Slider(
                    0.0, 1.0, value=default_conf, step=0.01,
                    label="Confidence threshold  (shared)",
                    info="Lower → more recall · Higher → fewer false positives",
                )
                alpha = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Mask opacity  (shared)",
                )
                run_btn = gr.Button(
                    "Run Comparison", variant="primary", size="lg",
                )

        gr.Markdown("---")

        # ── Side-by-side results ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_classes=["panel-a"]):
                gr.Markdown("### Model A — Result")
                out_img_a = gr.Image(label="Combined output", type="numpy")
                summary_a = gr.Textbox(
                    label="Detections by class",
                    interactive=False, lines=6,
                    elem_classes=["status-box"],
                )
                gallery_a = gr.Gallery(
                    label="Per-class breakdown",
                    columns=3, height="auto", object_fit="contain",
                )

            with gr.Column(elem_classes=["panel-b"]):
                gr.Markdown("### Model B — Result")
                out_img_b = gr.Image(label="Combined output", type="numpy")
                summary_b = gr.Textbox(
                    label="Detections by class",
                    interactive=False, lines=6,
                    elem_classes=["status-box"],
                )
                gallery_b = gr.Gallery(
                    label="Per-class breakdown",
                    columns=3, height="auto", object_fit="contain",
                )

        run_btn.click(
            fn=run_comparison,
            inputs=[image_input, classes_text, selected_classes, confidence, alpha],
            outputs=[out_img_a, summary_a, gallery_a, out_img_b, summary_b, gallery_b],
        )

        # ── Class list wiring ──────────────────────────────────────────────
        apply_btn.click(
            fn=_apply_class_list,
            inputs=[class_list_editor],
            outputs=[selected_classes, selected_classes],  # both checkbox groups share same component
        )
        reset_btn.click(
            fn=lambda: _reset_class_list(default_classes),
            inputs=[],
            outputs=[class_list_editor, selected_classes, selected_classes],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Model Comparison GUI")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_compare_app(config_path=args.config)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=["/mnt/usbssd", "/tmp"],
    )
