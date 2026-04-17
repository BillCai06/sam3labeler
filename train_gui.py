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
import csv as _csv
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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
        rows = []
        with open(csv_path, newline="") as _f:
            _reader = _csv.reader(_f)
            try:
                _header = next(_reader)
            except StopIteration:
                return pd.DataFrame(), pd.DataFrame()
            _n = len(_header)
            for _row in _reader:
                if len(_row) == _n:
                    rows.append(dict(zip(_header, _row)))
                elif len(_row) == _n + 1 and "lr" in _header:
                    # Val rows written with an extra mask_iou field before lr
                    _lr_idx = _header.index("lr")
                    _ext = _header[:_lr_idx] + ["mask_iou"] + _header[_lr_idx:]
                    rows.append(dict(zip(_ext, _row)))
        if not rows:
            return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(rows)
        _NON_NUMERIC = {"phase", "timestamp"}
        for _c in df.columns:
            if _c not in _NON_NUMERIC:
                df[_c] = pd.to_numeric(df[_c], errors="coerce")

        # Detect run boundaries: step resets to a lower value → new run.
        # Compute a cumulative offset so the plot always moves right.
        if "step" in df.columns:
            steps = df["step"].fillna(0).reset_index(drop=True)
            step_offset = pd.Series(0.0, index=steps.index)
            cumulative = 0.0
            run_id = pd.Series(0, index=steps.index)
            current_run = 0
            for i in range(1, len(steps)):
                if steps[i] < steps[i - 1]:
                    cumulative = step_offset[i - 1] + steps[i - 1] + 1
                    current_run += 1
                step_offset[i] = cumulative
                run_id[i] = current_run
            df = df.reset_index(drop=True)
            df["run"] = run_id
            df["step_plot"] = steps + step_offset
        else:
            df["run"] = 0
            df["step_plot"] = pd.Series(range(len(df)), index=df.index)

        if "epoch" in df.columns:
            epochs = df["epoch"].fillna(0).reset_index(drop=True)
            ep_offset = pd.Series(0.0, index=epochs.index)
            cumulative_ep = 0.0
            for i in range(1, len(epochs)):
                if epochs[i] < epochs[i - 1]:
                    cumulative_ep = ep_offset[i - 1] + epochs[i - 1] + 1
                ep_offset[i] = cumulative_ep
            df["epoch_plot"] = epochs + ep_offset
        else:
            df["epoch_plot"] = pd.Series(range(len(df)), index=df.index)

        train = df[df["phase"] == "train"].copy()
        val   = df[df["phase"] == "val"].copy()

        # Run boundary x-positions for vertical markers
        boundaries = df[df["run"] > 0].groupby("run")["step_plot"].first().tolist()
        train.attrs["run_boundaries"] = boundaries
        val.attrs["run_boundaries"]   = boundaries

        return train, val
    except Exception as e:
        import traceback; traceback.print_exc()
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


def _ema(values, alpha: float = 0.05):
    """Exponential moving average. Lower alpha = smoother."""
    out, s = [], None
    for v in values:
        s = float(v) if s is None else alpha * float(v) + (1.0 - alpha) * s
        out.append(s)
    return out


def _make_loss_figure(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Build a 2-row matplotlib figure: (1) losses, (2) mask IoU."""
    fig = Figure(figsize=(9, 6), dpi=100)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=False)
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

    run_boundaries = train_df.attrs.get("run_boundaries", []) if not train_df.empty else []

    def _draw_run_boundaries(ax, boundaries, ymin, ymax):
        for i, x in enumerate(boundaries):
            ax.axvline(x, color="#6c7086", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(x + (ymax - ymin) * 0.002, ymax * 0.97,
                    f"run {i + 2}", color="#6c7086", fontsize=6, va="top")

    # --- Top: training losses vs step ---
    step_col = "step_plot" if "step_plot" in train_df.columns else "step"
    if not train_df.empty and step_col in train_df.columns:
        steps = train_df[step_col].to_numpy()
        for col, color in COLORS.items():
            if col not in train_df.columns:
                continue
            raw = train_df[col].to_numpy()
            ax_loss.plot(steps, raw, color=color, linewidth=0.6, alpha=0.2)
            smoothed = _ema(raw, alpha=0.05)
            ax_loss.plot(steps, smoothed,
                         label=f"train {col}", color=color, linewidth=1.6, alpha=0.9)

    # Val total loss — map via step_plot offsets
    ep_col = "epoch_plot" if "epoch_plot" in val_df.columns else "epoch"
    if not val_df.empty and "loss_total" in val_df.columns and not train_df.empty \
            and step_col in train_df.columns and ep_col in train_df.columns:
        ep2step = train_df.groupby(ep_col)[step_col].max().to_dict()
        val_steps = val_df[ep_col].map(ep2step)
        ax_loss.plot(
            val_steps, val_df["loss_total"],
            label="val total", color="#cba6f7",
            linestyle="--", marker="o", markersize=4, linewidth=1.4,
        )

    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.set_title("Loss curves", fontsize=10)
    if run_boundaries:
        ymin, ymax = ax_loss.get_ylim()
        _draw_run_boundaries(ax_loss, run_boundaries, ymin, ymax)
    handles, labels = ax_loss.get_legend_handles_labels()
    if handles:
        ax_loss.legend(handles, labels, loc="upper right", fontsize=7,
                       facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4")
    ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # --- Bottom: val mask IoU vs epoch ---
    has_iou = (not val_df.empty
               and "mask_iou" in val_df.columns
               and ep_col in val_df.columns
               and val_df["mask_iou"].notna().any())

    ax_iou.set_xlabel("Epoch (cumulative)", fontsize=9)
    ax_iou.set_ylabel("Mask IoU", fontsize=9)
    ax_iou.set_title("Validation mask IoU", fontsize=10)

    if has_iou:
        ax_iou.plot(
            val_df[ep_col], val_df["mask_iou"],
            color="#89dceb", marker="o", markersize=4, linewidth=1.5, label="val mask IoU",
        )
        ax_iou.fill_between(
            val_df[ep_col], val_df["mask_iou"],
            alpha=0.15, color="#89dceb",
        )
        ax_iou.set_ylim(0, 1)
        if run_boundaries:
            ep_boundaries = val_df.groupby(
                val_df["run"] if "run" in val_df.columns else pd.Series(0, index=val_df.index)
            )[ep_col].first().iloc[1:].tolist() if "run" in val_df.columns else []
            _draw_run_boundaries(ax_iou, ep_boundaries, 0, 1)
        handles, labels = ax_iou.get_legend_handles_labels()
        ax_iou.legend(handles, labels, loc="lower right", fontsize=7,
                      facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4")
    else:
        ax_iou.text(0.5, 0.5, "Waiting for first validation epoch…",
                    ha="center", va="center", transform=ax_iou.transAxes,
                    color="#6c7086", fontsize=10, style="italic")
        ax_iou.set_xlim(0, 1)
        ax_iou.set_ylim(0, 1)
        ax_iou.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

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
    output_dir, log_interval, save_interval, num_workers, num_gpus,
) -> list[str]:

    script = str(Path(__file__).parent / "train_sam3.py")
    if int(num_gpus) > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(int(num_gpus)), script]
    else:
        cmd = [sys.executable, script]

    # Data sources
    runs = _discover_runs(outputs_dir)
    label2ann = {label: ann for label, ann in runs}

    if selected_sources:
        sources_args = []
        for label in selected_sources:
            ann = label2ann.get(label)
            if ann:
                # primary image search dir is the dataset folder itself
                ann_path = Path(ann)
                img_dir = str(ann_path.parent)
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
        "--log_interval",  str(int(log_interval)),
        "--save_interval", str(int(save_interval)),
        "--num_workers",   str(int(num_workers)),
    ]

    if freeze_vision:
        cmd.append("--freeze_vision")
    if freeze_text:
        cmd.append("--freeze_text")

    return cmd


_LOG_MAX_LINES = 500   # keep only last N lines in the UI textbox
_LOG_YIELD_LINES = 30  # batch: yield after this many new lines …
_LOG_YIELD_SECS = 2.0  # … or after this many seconds, whichever comes first


def _trim_log(log: str) -> str:
    lines = log.splitlines(keepends=True)
    if len(lines) > _LOG_MAX_LINES:
        return "".join(lines[-_LOG_MAX_LINES:])
    return log


def start_training(
    outputs_dir, selected_sources,
    max_train, max_val, val_split, neg_ratio,
    sam3_path, mask_size,
    finetune_ratio, freeze_vision, freeze_text,
    epochs, batch_size, accum_steps, lr, weight_decay, warmup_steps, max_grad_norm,
    cls_w, box_w, giou_w, mask_w, dice_w, presence_w,
    output_dir, log_interval, save_interval, num_workers, num_gpus,
):
    global _proc, _current_output_dir
    _current_output_dir = output_dir

    if _proc is not None and _proc.poll() is None:
        yield "⚠️ Training already running. Stop it first.", _make_loss_figure(pd.DataFrame(), pd.DataFrame()), "—"
        return

    cmd = _build_cmd(
        outputs_dir, selected_sources,
        max_train, max_val, val_split, neg_ratio,
        sam3_path, mask_size,
        finetune_ratio, freeze_vision, freeze_text,
        epochs, batch_size, accum_steps, lr, weight_decay, warmup_steps, max_grad_norm,
        cls_w, box_w, giou_w, mask_w, dice_w, presence_w,
        output_dir, log_interval, save_interval, num_workers, num_gpus,
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
    last_yield_time = time.time()
    pending = 0
    fig = _make_loss_figure(pd.DataFrame(), pd.DataFrame())
    ckpt_info = "—"

    for line in iter(_proc.stdout.readline, ""):
        log += line
        pending += 1

        now = time.time()
        if pending < _LOG_YIELD_LINES and (now - last_yield_time) < _LOG_YIELD_SECS:
            continue  # batch: don't yield yet

        # Trim log and refresh plot on a slower cadence
        log = _trim_log(log)
        if now - last_plot_refresh > 15:
            fig, ckpt_info = refresh_monitor(output_dir)
            last_plot_refresh = now

        yield log, fig, ckpt_info
        last_yield_time = now
        pending = 0

    _proc.wait()
    log = _trim_log(log)
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
    # ── Single-GPU presets (RTX PRO 6000, 96 GB) ──────────
    "test_8gpu": {
        "_label": "Multi-GPU Smoke Test  [8× H200]",
        "_desc": (
            "**100 images · 2 epochs · 8 GPU** — verifies DDP, dataloader, "
            "and full forward/backward pass without committing to a long run.  "
            "Backbone frozen, tiny batch, no checkpoint saved after epoch 1."
        ),
        "_output_dir": "checkpoints/test_8gpu",
        "_sam3_path":  "sam3",
        "max_train":   85,
        "max_val":     15,
        "epochs":       2,
        "batch_size":   8,
        "accum_steps":  1,
        "lr":           1.2e-3,
        "weight_decay": 0.05,
        "warmup_steps": 10,
        "val_split":    0.15,
        "neg_ratio":    0.25,
        "finetune_ratio":  0.0,
        "freeze_vision":   True,
        "freeze_text":     True,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        "neg_ratio_adv":   0.25,
        "max_grad_norm":   1.0,
        "log_interval":    1,
        "save_interval":   999,
        "num_workers":     8,
        "num_gpus":        8,
    },
    "phase1": {
        "_label": "Phase 1 — Head Only  [1× GPU, 96 GB]",
        "_desc": (
            "**Backbone frozen — ~30 GB VRAM @ batch 16, 1 GPU.**  "
            "Vision + text backbone fully frozen.  "
            "Only trains DETR detector head & mask decoder.  "
            "Low overfitting risk, good starting point."
        ),
        "_output_dir": "checkpoints/phase1",
        "_sam3_path":  "sam3",
        "max_train":   0,
        "max_val":     0,
        # batch=16 / accum=2 → effective 32
        "epochs":       40,
        "batch_size":   16,
        "accum_steps":  2,
        "lr":           5e-4,
        "weight_decay": 0.05,
        "warmup_steps": 200,
        "val_split":    0.15,
        "neg_ratio":    0.25,
        "finetune_ratio":  0.0,
        "freeze_vision":   True,
        "freeze_text":     True,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        "neg_ratio_adv":   0.25,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   5,
        "num_workers":     4,
        "num_gpus":        1,
    },
    "phase2": {
        "_label": "Phase 2 — Light Fine-tune  [1× GPU, 96 GB]",
        "_desc": (
            "**Vision unfrozen — ~45 GB VRAM @ batch 8, 1 GPU.**  "
            "Text encoder stays frozen, vision backbone unfreezes at 1% LR.  "
            "Load Phase 1 best checkpoint as starting point."
        ),
        "_output_dir": "checkpoints/phase2",
        "_sam3_path":  "checkpoints/phase1/best",
        "max_train":   0,
        "max_val":     0,
        # batch=8 / accum=4 → effective 32; vision grads need more VRAM
        "epochs":       10,
        "batch_size":   8,
        "accum_steps":  4,
        "lr":           1e-4,
        "weight_decay": 0.05,
        "warmup_steps": 100,
        "val_split":    0.15,
        "neg_ratio":    0.25,
        "finetune_ratio":  0.01,
        "freeze_vision":   False,
        "freeze_text":     True,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   2,
        "num_workers":     4,
        "num_gpus":        1,
    },
    "large": {
        "_label": "Large Dataset  [1× GPU, 96 GB]",
        "_desc": (
            "**Full fine-tune — ~50 GB VRAM @ batch 8, 1 GPU.**  "
            "Partial backbone unfreeze (5% LR).  "
            "Balanced loss weights, standard regularization."
        ),
        "_output_dir": "checkpoints/large_ft",
        "_sam3_path":  "sam3",
        "max_train":   0,
        "max_val":     0,
        # batch=8 / accum=4 → effective 32; full backbone grads
        "epochs":       30,
        "batch_size":   8,
        "accum_steps":  4,
        "lr":           2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 300,
        "val_split":    0.15,
        "neg_ratio":    0.30,
        "finetune_ratio":  0.05,
        "freeze_vision":   False,
        "freeze_text":     False,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     2.0,
        "dice_loss_weight":     2.0,
        "presence_loss_weight": 1.0,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   5,
        "num_workers":     4,
        "num_gpus":        1,
    },
    # ── Multi-GPU presets (8× H200, 144 GB each, 1.5 TB RAM) ─
    "h200_phase1": {
        "_label": "Phase 1 — Head Only  [8× H200, 144 GB]",
        "_desc": (
            "**8× H200 (144 GB each, 1.5 TB RAM) — backbone frozen.**  "
            "batch=64/GPU, accum=1 → effective batch = 512.  "
            "LR scaled √16× from single-GPU baseline.  "
            "~1-2 h for 40 epochs total."
        ),
        "_output_dir": "checkpoints/phase1_h200",
        "_sam3_path":  "sam3",
        "max_train":   0,
        "max_val":     0,
        # batch=64/GPU, accum=1 → effective 64×8 = 512
        # LR: 5e-4 × sqrt(512/32) = 5e-4 × 4 = 2e-3
        "epochs":       40,
        "batch_size":   64,
        "accum_steps":  1,
        "lr":           2e-3,
        "weight_decay": 0.05,
        "warmup_steps": 1000,
        "val_split":    0.15,
        "neg_ratio":    0.25,
        "finetune_ratio":  0.0,
        "freeze_vision":   True,
        "freeze_text":     True,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        "neg_ratio_adv":   0.25,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   5,
        "num_workers":     8,
        "num_gpus":        8,
    },
    "h200_phase2": {
        "_label": "Phase 2 — Light Fine-tune  [8× H200, 144 GB]",
        "_desc": (
            "**8× H200 — vision backbone unfrozen, gradient checkpointing on.**  "
            "batch=8/GPU × accum=4 → effective batch = 256.  "
            "Load Phase 1 H200 checkpoint as starting point."
        ),
        "_output_dir": "checkpoints/phase2_h200",
        "_sam3_path":  "checkpoints/phase1_h200/best",
        "max_train":   0,
        "max_val":     0,
        # batch=8/GPU (gradient checkpointing saves ~3x VRAM vs batch=32)
        # accum=4 → effective 8×8×4 = 256
        # LR: 1e-4 × sqrt(256/32) = ~2.8e-4
        "epochs":       15,
        "batch_size":   8,
        "accum_steps":  4,
        "lr":           2.8e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "val_split":    0.15,
        "neg_ratio":    0.25,
        "finetune_ratio":  0.01,
        "freeze_vision":   False,
        "freeze_text":     True,
        "cls_loss_weight":      1.0,
        "box_loss_weight":      5.0,
        "giou_loss_weight":     2.0,
        "mask_loss_weight":     3.0,
        "dice_loss_weight":     3.0,
        "presence_loss_weight": 1.0,
        "max_grad_norm":   1.0,
        "log_interval":    10,
        "save_interval":   2,
        "num_workers":     8,
        "num_gpus":        8,
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
        p["num_workers"],           # num_workers_sl
        p["num_gpus"],              # num_gpus_sl
        p.get("max_train", 0),      # max_train_box
        p.get("max_val",   0),      # max_val_box
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
                        value="/home/sc3568/sam3labeler/outputs", label="Outputs root dir",
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
                gr.Markdown("**Single GPU (96 GB)**")
                with gr.Row():
                    btn_phase1 = gr.Button("Phase 1 — Head Only",    variant="primary",   scale=2)
                    btn_phase2 = gr.Button("Phase 2 — Light Fine-tune", variant="secondary", scale=2)
                    btn_large  = gr.Button("Large Dataset (5k+)",    variant="secondary", scale=2)
                gr.Markdown("**8× H200 (144 GB · 1.5 TB RAM)**")
                with gr.Row():
                    btn_h200_p1 = gr.Button("H200 Phase 1",  variant="primary",   scale=2)
                    btn_h200_p2 = gr.Button("H200 Phase 2",  variant="secondary", scale=2)
                    gr.Button("", variant="secondary", scale=2, interactive=False, visible=False)
                gr.Markdown("**8× H200 — Multi-GPU**")
                with gr.Row():
                    btn_test_8gpu = gr.Button("🧪 Smoke Test (100 imgs · 8 GPU)", variant="primary", scale=2)
                    gr.Button("", variant="secondary", scale=2, interactive=False, visible=False)
                    gr.Button("", variant="secondary", scale=2, interactive=False, visible=False)

                preset_info = gr.Markdown(
                    f"**Active preset: {PRESETS['phase1']['_label']}**  \n"
                    f"{PRESETS['phase1']['_desc']}"
                )

                # --- Training params ---
                gr.Markdown("### Training Parameters")
                with gr.Row():
                    epochs_sl   = gr.Slider(1, 200, value=40,  step=1,  label="Epochs")
                    batch_sl    = gr.Slider(1, 64,  value=2,   step=1,  label="Batch size")
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
                    log_int_sl    = gr.Slider(1, 200, value=10,  step=1, label="Log interval (steps)")
                    save_int_sl   = gr.Slider(1, 999, value=5,   step=1, label="Save interval (epochs)")
                    num_workers_sl = gr.Slider(
                        0, 8, value=4, step=1,
                        label="DataLoader workers",
                        info="Each worker pre-fetches batches into RAM. "
                             "Keep ≤4 with 32 GB system RAM to avoid OOM.",
                    )
                    num_gpus_sl = gr.Slider(
                        1, 8, value=1, step=1,
                        label="GPUs (multi-GPU via torchrun)",
                        info="1 = single GPU. >1 uses torchrun DDP automatically. "
                             "Batch size is per-GPU; effective total = batch × accum × GPUs.",
                    )

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
            output_dir_box, log_int_sl, save_int_sl, num_workers_sl, num_gpus_sl,
        ]
        start_btn.click(
            fn=lambda: gr.update(active=True),
            outputs=[timer],
        )
        start_btn.click(
            fn=start_training,
            inputs=train_inputs,
            outputs=[log_box, loss_plot, ckpt_md],
            show_progress=False,
        ).then(
            fn=lambda od: (refresh_monitor(od)[1], gr.update(active=False)),
            inputs=[output_dir_box],
            outputs=[ckpt_md, timer],
        )

        # Stop button — deactivate timer and show status in stop_status row
        stop_btn.click(
            fn=stop_training,
            outputs=[stop_status],
        ).then(
            fn=lambda: gr.update(active=False),
            outputs=[timer],
        )

        # Make stop_status visible when there's a message
        stop_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[stop_status],
        )

        # Manual refresh
        refresh_btn.click(
            fn=poll_monitor,
            inputs=[output_dir_box],
            outputs=[loss_plot, ckpt_md],
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
            output_dir_box, sam3_path_box, num_workers_sl, num_gpus_sl,
            max_train_box, max_val_box,
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
        btn_h200_p1.click(
            fn=lambda: _preset_outputs("h200_phase1"),
            outputs=_preset_out_components,
        )
        btn_h200_p2.click(
            fn=lambda: _preset_outputs("h200_phase2"),
            outputs=_preset_out_components,
        )
        btn_test_8gpu.click(
            fn=lambda: _preset_outputs("test_8gpu"),
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
