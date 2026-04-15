#!/usr/bin/env python3
"""
train_gui.py — Gradio WebUI for SAM3 fine-tuning.

Wraps train_sam3.py with a visual interface:
  - Data source discovery + selection
  - All training hyperparameters
  - Real-time log streaming
  - Live loss curves (auto-refresh from losses.csv)
  - Checkpoint status

Run:
    conda run -n qwen3vl2sam python3 train_gui.py [--port 7860]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Globals ────────────────────────────────────────────────
_proc: subprocess.Popen | None = None
_current_output_dir: str = "checkpoints/finetune"


# ── Data helpers ───────────────────────────────────────────

def _discover_runs(outputs_dir: str) -> list[tuple[str, str]]:
    """Return list of (label, ann_json_path) for all discoverable run dirs."""
    p = Path(outputs_dir)
    results = []
    if p.exists():
        for ann in sorted(p.rglob("annotations.json")):
            label = str(ann.parent.relative_to(p))
            results.append((label, str(ann)))
    return results


def refresh_sources(outputs_dir: str):
    runs = _discover_runs(outputs_dir)
    choices = [label for label, _ in runs]
    return gr.update(choices=choices, value=choices)   # select all by default


# ── Loss plot ──────────────────────────────────────────────

def _load_losses(output_dir: str):
    csv_path = Path(output_dir) / "losses.csv"
    if not csv_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        train = df[df["phase"] == "train"].copy()
        val   = df[df["phase"] == "val"].copy()
        return train, val
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _best_checkpoint_info(output_dir: str) -> str:
    meta_path = Path(output_dir) / "best" / "training_state.pt"
    if not meta_path.exists():
        return "No checkpoint yet."
    try:
        import torch
        meta = torch.load(meta_path, weights_only=False, map_location="cpu")
        m = meta.get("metrics", {})
        loss_str = f"{m['loss_total']:.4f}" if "loss_total" in m else "—"
        iou_str  = f"{m['mask_iou']:.4f}"   if "mask_iou"   in m else "—"
        return (
            f"**Best checkpoint** — epoch {meta.get('epoch', '?')}\n\n"
            f"- val_loss: `{loss_str}`\n"
            f"- mask_iou: `{iou_str}`\n"
            f"- path: `{Path(output_dir) / 'best'}`"
        )
    except Exception as e:
        return f"Checkpoint found but could not load: {e}"


def _make_loss_figure(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Build a 2-row matplotlib figure: (1) losses, (2) mask IoU."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=100, sharex=False)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#262637")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    ax_loss, ax_iou = axes

    COLORS = {
        "loss_total": "#cba6f7",   # mauve
        "loss_cls":   "#89b4fa",   # blue
        "loss_mask":  "#a6e3a1",   # green
        "loss_dice":  "#fab387",   # peach
        "loss_box":   "#f38ba8",   # red
        "loss_giou":  "#f9e2af",   # yellow
    }

    # --- Top: training losses vs step ---
    if not train_df.empty and "step" in train_df.columns:
        for col, color in COLORS.items():
            if col in train_df.columns:
                ax_loss.plot(
                    train_df["step"], train_df[col],
                    label=f"train {col}", color=color, linewidth=1.2, alpha=0.85,
                )

    # Val total loss as dashed markers (use epoch × steps_per_epoch estimate)
    if not val_df.empty and "loss_total" in val_df.columns and not train_df.empty \
            and "step" in train_df.columns and "epoch" in train_df.columns:
        # Estimate x position: map val epoch to train step
        ep2step = train_df.groupby("epoch")["step"].max().to_dict()
        val_steps = val_df["epoch"].map(ep2step)
        ax_loss.plot(
            val_steps, val_df["loss_total"],
            label="val total", color="#cba6f7",
            linestyle="--", marker="o", markersize=4, linewidth=1.4,
        )

    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.set_title("Loss curves", fontsize=10)
    handles, labels = ax_loss.get_legend_handles_labels()
    if handles:
        ax_loss.legend(handles, labels, loc="upper right", fontsize=7,
                       facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4")
    ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # --- Bottom: val mask IoU vs epoch ---
    if not val_df.empty and "mask_iou" in val_df.columns and "epoch" in val_df.columns:
        ax_iou.plot(
            val_df["epoch"], val_df["mask_iou"],
            color="#89dceb", marker="o", markersize=4, linewidth=1.5, label="val mask IoU",
        )
        ax_iou.fill_between(
            val_df["epoch"], val_df["mask_iou"],
            alpha=0.15, color="#89dceb",
        )
        ax_iou.set_ylim(0, 1)

    ax_iou.set_xlabel("Epoch", fontsize=9)
    ax_iou.set_ylabel("Mask IoU", fontsize=9)
    ax_iou.set_title("Validation mask IoU", fontsize=10)
    handles, labels = ax_iou.get_legend_handles_labels()
    if handles:
        ax_iou.legend(handles, labels, loc="lower right", fontsize=7,
                      facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4")

    fig.tight_layout(pad=1.2)
    return fig


def refresh_monitor(output_dir: str):
    train_df, val_df = _load_losses(output_dir)
    fig = _make_loss_figure(train_df, val_df)
    info = _best_checkpoint_info(output_dir)
    return fig, info


# ── Training process ───────────────────────────────────────

def _build_cmd(
    outputs_dir, selected_sources,
    max_train, max_val, val_split, neg_ratio,
    sam3_path, mask_size,
    finetune_ratio, freeze_vision, freeze_text,
    epochs, batch_size, accum_steps, lr, weight_decay, warmup_steps, max_grad_norm,
    cls_w, box_w, giou_w, mask_w, dice_w, presence_w,
    output_dir, log_interval, save_interval,
) -> list[str]:

    script = str(Path(__file__).parent / "train_sam3.py")
    cmd = [sys.executable, script]

    # Data sources
    runs = _discover_runs(outputs_dir)
    label2ann = {label: ann for label, ann in runs}

    if selected_sources:
        sources_args = []
        for label in selected_sources:
            ann = label2ann.get(label)
            if ann:
                # image search dirs: outputs_dir + run_dir siblings
                ann_path = Path(ann)
                img_dir = str(ann_path.parent.parent)  # e.g. outputs/
                sources_args.append(f"{ann}:{img_dir}")
        if sources_args:
            cmd += ["--sources"] + sources_args
        else:
            cmd += ["--outputs_dir", outputs_dir]
    else:
        cmd += ["--outputs_dir", outputs_dir]

    if max_train and max_train > 0:
        cmd += ["--max_train", str(int(max_train))]
    if max_val and max_val > 0:
        cmd += ["--max_val", str(int(max_val))]

    cmd += [
        "--val_split",    str(val_split),
        "--neg_ratio",    str(neg_ratio),
        "--sam3_path",    sam3_path,
        "--mask_size",    str(int(mask_size)),
        "--finetune_ratio", str(finetune_ratio),
        "--epochs",       str(int(epochs)),
        "--batch_size",   str(int(batch_size)),
        "--accum_steps",  str(int(accum_steps)),
        "--lr",           str(lr),
        "--weight_decay", str(weight_decay),
        "--warmup_steps", str(int(warmup_steps)),
        "--max_grad_norm", str(max_grad_norm),
        "--cls_loss_weight",      str(cls_w),
        "--box_loss_weight",      str(box_w),
        "--giou_loss_weight",     str(giou_w),
        "--mask_loss_weight",     str(mask_w),
        "--dice_loss_weight",     str(dice_w),
        "--presence_loss_weight", str(presence_w),
        "--output_dir",   output_dir,
        "--log_interval", str(int(log_interval)),
        "--save_interval", str(int(save_interval)),
    ]

    if freeze_vision:
        cmd.append("--freeze_vision")
    if freeze_text:
        cmd.append("--freeze_text")

    return cmd


def start_training(
    outputs_dir, selected_sources,
    max_train, max_val, val_split, neg_ratio,
    sam3_path, mask_size,
    finetune_ratio, freeze_vision, freeze_text,
    epochs, batch_size, accum_steps, lr, weight_decay, warmup_steps, max_grad_norm,
    cls_w, box_w, giou_w, mask_w, dice_w, presence_w,
    output_dir, log_interval, save_interval,
):
    global _proc, _current_output_dir
    _current_output_dir = output_dir

    if _proc is not None and _proc.poll() is None:
        yield "⚠️ Training already running. Stop it first.", pd.DataFrame(), "—"
        return

    cmd = _build_cmd(
        outputs_dir, selected_sources,
        max_train, max_val, val_split, neg_ratio,
        sam3_path, mask_size,
        finetune_ratio, freeze_vision, freeze_text,
        epochs, batch_size, accum_steps, lr, weight_decay, warmup_steps, max_grad_norm,
        cls_w, box_w, giou_w, mask_w, dice_w, presence_w,
        output_dir, log_interval, save_interval,
    )

    cmd_str = " ".join(cmd)
    yield f"$ {cmd_str}\n\n", _make_loss_figure(pd.DataFrame(), pd.DataFrame()), "Starting..."

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    log = f"$ {cmd_str}\n\n"
    last_plot_refresh = time.time()
    fig = _make_loss_figure(pd.DataFrame(), pd.DataFrame())

    for line in iter(_proc.stdout.readline, ""):
        log += line

        # Refresh plot every 15 s
        now = time.time()
        if now - last_plot_refresh > 15:
            fig, ckpt_info = refresh_monitor(output_dir)
            last_plot_refresh = now
        else:
            ckpt_info = "—"

        yield log, fig, ckpt_info

    _proc.wait()
    fig, ckpt_info = refresh_monitor(output_dir)
    status = "✓ Complete" if _proc.returncode == 0 else f"✗ Exit code {_proc.returncode}"
    yield log + f"\n\n{status}\n", fig, ckpt_info


def stop_training():
    global _proc
    if _proc is None or _proc.poll() is not None:
        return "No training process running."
    try:
        _proc.send_signal(signal.SIGTERM)
        time.sleep(1)
        if _proc.poll() is None:
            _proc.kill()
        return "⏹ Training stopped."
    except Exception as e:
        return f"Error stopping: {e}"


def poll_monitor(output_dir: str):
    """Called by gr.Timer every N seconds to refresh the plot."""
    fig, ckpt_info = refresh_monitor(output_dir)
    return fig, ckpt_info


# ── Presets ────────────────────────────────────────────────

PRESETS = {
    "phase1": {
        "_label": "Phase 1 — Head Only",
        "_desc": (
            "**Recommended for ~1600 images.**  "
            "Vision + text backbone fully frozen.  "
            "Only trains DETR detector head & mask decoder.  "
            "Low overfitting risk, good starting point."
        ),
        "_output_dir": "checkpoints/phase1",
        "_sam3_path":  "sam3",
        # training
        "epochs":       40,
        "batch_size":   2,
        "accum_steps":  8,
        "lr":           5e-4,
        "weight_decay": 0.05,
        "warmup_steps": 100,
        # data
        "val_split":    0.15,
        "neg_ratio":    0.25,
        # fine-tune control
        "finetune_ratio":  0.0,
        "freeze_vision":   True,
        "freeze_text":     True,
        # loss weights
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        # advanced
        "neg_ratio_adv":   0.25,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   5,
    },
    "phase2": {
        "_label": "Phase 2 — Light Full Fine-tune",
        "_desc": (
            "**Run after Phase 1 converges.**  "
            "Text encoder stays frozen, vision backbone unfreezes at 1% LR.  "
            "Load Phase 1 best checkpoint as starting point."
        ),
        "_output_dir": "checkpoints/phase2",
        "_sam3_path":  "checkpoints/phase1/best",
        # training
        "epochs":       10,
        "batch_size":   2,
        "accum_steps":  8,
        "lr":           1e-4,
        "weight_decay": 0.05,
        "warmup_steps": 50,
        # data
        "val_split":    0.15,
        "neg_ratio":    0.25,
        # fine-tune control
        "finetune_ratio":  0.01,
        "freeze_vision":   False,
        "freeze_text":     True,
        # loss weights
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        # advanced
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   2,
    },
    "large": {
        "_label": "Large Dataset (5000+ images)",
        "_desc": (
            "**For large datasets.**  "
            "Partial backbone unfreeze (5% LR).  "
            "Balanced loss weights, standard regularization."
        ),
        "_output_dir": "checkpoints/large_ft",
        "_sam3_path":  "sam3",
        # training
        "epochs":       30,
        "batch_size":   4,
        "accum_steps":  4,
        "lr":           2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        # data
        "val_split":    0.15,
        "neg_ratio":    0.30,
        # fine-tune control
        "finetune_ratio":  0.05,
        "freeze_vision":   False,
        "freeze_text":     False,
        # loss weights
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     2.0,
        "dice_loss_weight":     2.0,
        "presence_loss_weight": 1.0,
        # advanced
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   5,
    },
}


def _preset_outputs(key: str):
    """Return gr.update() values for all form components from a preset dict."""
    p = PRESETS[key]
    label = f"**Active preset: {p['_label']}**  \n{p['_desc']}"
    return (
        label,                      # preset_info
        p["epochs"],                # epochs_sl
        p["batch_size"],            # batch_sl
        p["accum_steps"],           # accum_sl
        p["lr"],                    # lr_box
        p["weight_decay"],          # wd_box
        p["warmup_steps"],          # warmup_sl
        p["finetune_ratio"],        # ft_ratio_sl
        p["freeze_vision"],         # freeze_vis_cb
        p["freeze_text"],           # freeze_txt_cb
        p["cls_loss_weight"],       # cls_w_sl
        p["box_loss_weight"],       # box_w_sl
        p["giou_loss_weight"],      # giou_w_sl
        p["mask_loss_weight"],      # mask_w_sl
        p["dice_loss_weight"],      # dice_w_sl
        p["presence_loss_weight"],  # pres_w_sl
        p["neg_ratio"],             # neg_ratio_sl
        p["max_grad_norm"],         # max_grad_box
        p["log_interval"],          # log_int_sl
        p["save_interval"],         # save_int_sl
        p["_output_dir"],           # output_dir_box
        p["_sam3_path"],            # sam3_path_box
    )


# ── Gradio app ─────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="SAM3 Fine-tuning") as demo:
        gr.Markdown("# SAM3 Fine-tuning Launcher")

        with gr.Row():
            # ── Left column: config ──────────────────────────
            with gr.Column(scale=4):

                # --- Data sources ---
                gr.Markdown("### Data Sources")
                with gr.Row():
                    outputs_dir_box = gr.Textbox(
                        value="outputs", label="Outputs root dir",
                        info="Directory scanned for run subdirs (each with annotations.json)",
                        scale=3,
                    )
                    scan_btn = gr.Button("Scan", variant="secondary", scale=1)

                source_check = gr.CheckboxGroup(
                    choices=[], value=[],
                    label="Select annotation sources (all selected = use all)",
                    info="Each entry is a run directory found under the outputs root.",
                )

                # --- Presets ---
                gr.Markdown("### Presets")
                with gr.Row():
                    btn_phase1 = gr.Button("Phase 1 — Head Only",         variant="primary",   scale=2)
                    btn_phase2 = gr.Button("Phase 2 — Light Full",         variant="secondary", scale=2)
                    btn_large  = gr.Button("Large Dataset (5k+)",           variant="secondary", scale=2)

                preset_info = gr.Markdown(
                    f"**Active preset: {PRESETS['phase1']['_label']}**  \n"
                    f"{PRESETS['phase1']['_desc']}"
                )

                # --- Training params ---
                gr.Markdown("### Training Parameters")
                with gr.Row():
                    epochs_sl   = gr.Slider(1, 200, value=40,  step=1,  label="Epochs")
                    batch_sl    = gr.Slider(1, 32,  value=2,   step=1,  label="Batch size")
                    accum_sl    = gr.Slider(1, 64,  value=8,   step=1,  label="Accum steps")

                with gr.Row():
                    lr_box      = gr.Number(value=5e-4,  label="Head LR",     precision=6)
                    wd_box      = gr.Number(value=0.05,  label="Weight decay", precision=4)
                    warmup_sl   = gr.Slider(0, 2000, value=100, step=10, label="Warmup steps")

                with gr.Row():
                    max_train_box = gr.Number(value=0, label="Max train samples (0=all)", precision=0)
                    max_val_box   = gr.Number(value=0, label="Max val samples (0=auto)", precision=0)
                    val_split_sl  = gr.Slider(0.05, 0.4, value=0.15, step=0.01, label="Val split")

                # --- Fine-tune ratio ---
                gr.Markdown("### Fine-tune Control")
                with gr.Row():
                    ft_ratio_sl = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.01,
                        label="Fine-tune ratio",
                        info="Backbone LR = head LR × ratio. 0 = freeze backbone.",
                    )
                    freeze_vis_cb  = gr.Checkbox(True,  label="Freeze vision encoder")
                    freeze_txt_cb  = gr.Checkbox(True,  label="Freeze text encoder")

                # --- Loss weights ---
                with gr.Accordion("Loss weights", open=False):
                    with gr.Row():
                        cls_w_sl  = gr.Slider(0, 10, value=1.0, step=0.1, label="Classification")
                        box_w_sl  = gr.Slider(0, 20, value=5.0, step=0.5, label="Box L1")
                        giou_w_sl = gr.Slider(0, 10, value=2.0, step=0.5, label="GIoU")
                    with gr.Row():
                        mask_w_sl = gr.Slider(0, 10, value=3.0, step=0.5, label="Mask BCE")
                        dice_w_sl = gr.Slider(0, 10, value=3.0, step=0.5, label="Dice")
                        pres_w_sl = gr.Slider(0, 10, value=1.0, step=0.5, label="Presence")

                # --- Advanced ---
                with gr.Accordion("Advanced", open=False):
                    sam3_path_box = gr.Textbox(value="sam3",   label="SAM3 model path",
                                               info="For Phase 2 set to checkpoints/phase1/best")
                    mask_size_sl  = gr.Slider(64, 512, value=288, step=32, label="Mask size")
                    neg_ratio_sl  = gr.Slider(0.0, 0.8, value=0.25, step=0.05, label="Negative sample ratio")
                    max_grad_box  = gr.Number(value=1.0, label="Max grad norm", precision=2)
                    log_int_sl    = gr.Slider(5, 200, value=10,  step=5, label="Log interval (steps)")
                    save_int_sl   = gr.Slider(1, 50,  value=5,   step=1, label="Save interval (epochs)")

                gr.Markdown("### Output")
                output_dir_box = gr.Textbox(
                    value="checkpoints/phase1",
                    label="Checkpoint output dir",
                )

                with gr.Row():
                    start_btn = gr.Button("▶  Start Training", variant="primary", scale=3)
                    stop_btn  = gr.Button("⏹  Stop",           variant="stop",    scale=1)

            # ── Right column: monitor ────────────────────────
            with gr.Column(scale=6):

                gr.Markdown("### Live Monitor")

                with gr.Row():
                    refresh_btn  = gr.Button("⟳ Refresh plot", variant="secondary")
                    ckpt_md      = gr.Markdown("No checkpoint yet.")

                loss_plot = gr.Plot(
                    value=_make_loss_figure(pd.DataFrame(), pd.DataFrame()),
                    label="Loss curves",
                )

                log_box = gr.Textbox(
                    label="Training Log",
                    lines=28,
                    max_lines=500,
                    autoscroll=True,
                    interactive=False,
                )

                stop_status = gr.Textbox(label="", lines=1, interactive=False, visible=False)

        # ── Auto-refresh timer ─────────────────────────────────
        timer = gr.Timer(value=20, active=False)

        # ── Event wiring ───────────────────────────────────────

        # Scan button
        scan_btn.click(
            fn=refresh_sources,
            inputs=[outputs_dir_box],
            outputs=[source_check],
        )

        # Auto-scan on load
        demo.load(
            fn=refresh_sources,
            inputs=[outputs_dir_box],
            outputs=[source_check],
        )

        # Start training (streaming generator)
        train_inputs = [
            outputs_dir_box, source_check,
            max_train_box, max_val_box, val_split_sl, neg_ratio_sl,
            sam3_path_box, mask_size_sl,
            ft_ratio_sl, freeze_vis_cb, freeze_txt_cb,
            epochs_sl, batch_sl, accum_sl,
            lr_box, wd_box, warmup_sl, max_grad_box,
            cls_w_sl, box_w_sl, giou_w_sl, mask_w_sl, dice_w_sl, pres_w_sl,
            output_dir_box, log_int_sl, save_int_sl,
        ]
        start_btn.click(
            fn=start_training,
            inputs=train_inputs,
            outputs=[log_box, loss_plot, ckpt_md],
            show_progress=False,
        ).then(
            fn=lambda od: refresh_monitor(od)[1],
            inputs=[output_dir_box],
            outputs=[ckpt_md],
        )

        # Stop button
        stop_btn.click(fn=stop_training, outputs=[log_box])

        # Manual refresh
        refresh_btn.click(
            fn=poll_monitor,
            inputs=[output_dir_box],
            outputs=[loss_plot, ckpt_md],
        )

        # Auto-refresh timer (activates once training starts)
        start_btn.click(
            fn=lambda: gr.update(active=True),
            outputs=[timer],
        )
        timer.tick(
            fn=poll_monitor,
            inputs=[output_dir_box],
            outputs=[loss_plot, ckpt_md],
        )

        # ── Preset buttons ─────────────────────────────────────
        # All three share the same output list; only the key differs.
        _preset_out_components = [
            preset_info,
            epochs_sl, batch_sl, accum_sl,
            lr_box, wd_box, warmup_sl,
            ft_ratio_sl, freeze_vis_cb, freeze_txt_cb,
            cls_w_sl, box_w_sl, giou_w_sl, mask_w_sl, dice_w_sl, pres_w_sl,
            neg_ratio_sl, max_grad_box, log_int_sl, save_int_sl,
            output_dir_box, sam3_path_box,
        ]

        btn_phase1.click(
            fn=lambda: _preset_outputs("phase1"),
            outputs=_preset_out_components,
        )
        btn_phase2.click(
            fn=lambda: _preset_outputs("phase2"),
            outputs=_preset_out_components,
        )
        btn_large.click(
            fn=lambda: _preset_outputs("large"),
            outputs=_preset_out_components,
        )

    return demo


# ── Entry point ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SAM3 Training WebUI")
    ap.add_argument("--port", type=int, default=7861,
                    help="Port to serve the WebUI on (default: 7861)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Host to bind to (default: 0.0.0.0)")
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link")
    args = ap.parse_args()

    demo = build_app()

    print(f"\n{'='*50}")
    print(f"  SAM3 Training GUI  →  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(Path.cwd())],
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
