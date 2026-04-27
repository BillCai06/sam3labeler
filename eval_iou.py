#!/usr/bin/env python3
"""
eval_iou.py — Evaluate SAM3 mask IoU across checkpoints, split by data source.

Outputs:
  - Per-checkpoint IoU: Drone (UAV) / Husky / Overall
  - Printed as a markdown table

Usage:
  python eval_iou.py
  python eval_iou.py --max_samples 500   # fast smoke-test
  python eval_iou.py --batch_size 8
"""

import argparse
import gc
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse data pipeline from train_sam3.py
sys.path.insert(0, str(Path(__file__).parent))
from train_sam3 import (
    Sam3Dataset,
    collate_fn,
    build_samples,
    load_model,
)

BASE_DIR   = Path(__file__).parent
OUTPUTS    = BASE_DIR / "outputs" / "20260320_125939_dataset"
DRONE_ANN  = OUTPUTS / "drone_frames" / "annotations"
HUSKY_ANN  = OUTPUTS / "husky_frames" / "annotations"
DRONE_IMGS = OUTPUTS / "drone_frames"
HUSKY_IMGS = OUTPUTS / "husky_frames"

CHECKPOINTS = [
    ("Base SAM3",       str(BASE_DIR / "sam3")),
    ("Phase1 h200",     str(BASE_DIR / "checkpoints/phase1_prompt_mask_v3/best")),
    ("Phase2 h200",      str(BASE_DIR / "checkpoints/phase2_prompt_mask_v3c_continue/best")),
    # ("Phase2 ft5",      str(BASE_DIR / "checkpoints/phase2_h200_ft5/best")),
]


def build_source_samples(ann_dir: Path, img_dir: Path, max_samples=None, seed=42):
    """Build samples from a single per-image annotation directory."""
    source = (ann_dir, [img_dir, ann_dir.parent])
    return build_samples(
        [source],
        max_samples=max_samples,
        neg_ratio=0.0,      # positives only for eval
        seed=seed,
    )


@torch.no_grad()
def evaluate(model, processor, samples, batch_size, mask_size, device, desc="eval",
             score_threshold=0.3):
    """
    Compute mIoU using the standard confusion-matrix approach:
      1. Accumulate per-class TP (intersection) and TP+FP+FN (union) over all images.
      2. Per-class IoU = sum(TP_i) / sum(union_i)   ← not averaged per sample
      3. mIoU = mean over classes that appear in the data.

    This avoids the bias of per-sample averaging where classes with many
    samples (e.g. grass: 40 instances/image) dominate the mean.
    """
    ds = Sam3Dataset(samples, processor, mask_size=mask_size, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    raw_model = model.module if hasattr(model, "module") else model

    # Per-class accumulators: class_name → [total_intersection, total_union]
    class_inter = defaultdict(float)
    class_union = defaultdict(float)

    for batch in tqdm(loader, desc=desc, leave=False):
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        class_names  = batch["class_names"]

        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                            dtype=dtype, enabled="cuda" in str(device)):
            vision_embeds = raw_model.get_vision_features(pixel_values=pixel_values)

        unique_cls = list(dict.fromkeys(class_names))
        text_cache = {}
        for cls in unique_cls:
            tokens = processor.tokenizer(
                cls, return_tensors="pt",
                padding="max_length", max_length=32,
            ).to(device)
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                dtype=dtype, enabled="cuda" in str(device)):
                emb = raw_model.get_text_features(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    return_dict=True,
                ).pooler_output
            text_cache[cls] = emb

        text_embeds = torch.cat([text_cache[cn] for cn in class_names], dim=0)

        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                            dtype=dtype, enabled="cuda" in str(device)):
            out = raw_model(vision_embeds=vision_embeds, text_embeds=text_embeds)

        logits = out.pred_logits
        if logits.dim() == 3:
            logits = logits[..., 0]
        scores = logits.sigmoid() * out.presence_logits.sigmoid()

        mask_h, mask_w = out.pred_masks.shape[-2], out.pred_masks.shape[-1]

        for i in range(len(class_names)):
            gt_m = batch["gt_masks"][i].to(device)
            if gt_m.shape[0] == 0:
                continue

            # Merge all queries above threshold (handles multi-instance classes).
            # Falls back to top-1 if no query exceeds the threshold.
            confident = (scores[i] > score_threshold).nonzero(as_tuple=True)[0]
            if confident.numel() == 0:
                confident = scores[i].argmax().unsqueeze(0)
            pred_soft = out.pred_masks[i][confident].sigmoid()        # (K, H, W)
            pred_mask = (pred_soft.max(dim=0).values > 0.5).float()   # union of instances

            gt_r = F.interpolate(
                gt_m.unsqueeze(1).float(), size=(mask_h, mask_w), mode="nearest"
            ).squeeze(1)
            gt_any = (gt_r.sum(0) > 0).float()   # union of all GT instances

            # Accumulate TP and TP+FP+FN per class (confusion-matrix style)
            inter = float((pred_mask * gt_any).sum())
            union = float((pred_mask + gt_any).clamp(max=1).sum())
            cls   = class_names[i]
            class_inter[cls] += inter
            class_union[cls] += union

    # Per-class IoU = accumulated_inter / accumulated_union (confusion-matrix style)
    per_class_iou = {
        cls: class_inter[cls] / (class_union[cls] + 1e-6)
        for cls in class_inter
    }
    miou = sum(per_class_iou.values()) / len(per_class_iou) if per_class_iou else 0.0
    return miou, per_class_iou, dict(class_inter), dict(class_union)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per source for quick testing")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--mask_size", type=int, default=288)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score_threshold", type=float, default=0.3,
                   help="Queries above this score are merged into the predicted mask.")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading samples ...")
    drone_samples = build_source_samples(DRONE_ANN, DRONE_IMGS, cfg.max_samples, cfg.seed)
    husky_samples = build_source_samples(HUSKY_ANN, HUSKY_IMGS, cfg.max_samples, cfg.seed)
    all_samples   = drone_samples + husky_samples
    print(f"  Drone: {len(drone_samples)} samples")
    print(f"  Husky: {len(husky_samples)} samples")
    print(f"  Total: {len(all_samples)} samples\n")

    results = []   # [(name, drone_iou, husky_iou, overall_iou)]

    for name, ckpt_path in CHECKPOINTS:
        print(f"{'='*50}")
        print(f"Evaluating: {name}")
        print(f"  Path: {ckpt_path}")

        model, processor = load_model(ckpt_path, str(device))
        model.eval()

        drone_miou, drone_per_cls, drone_inter, drone_union = evaluate(
            model, processor, drone_samples,
            cfg.batch_size, cfg.mask_size, device,
            desc="  Drone", score_threshold=cfg.score_threshold)
        husky_miou, husky_per_cls, husky_inter, husky_union = evaluate(
            model, processor, husky_samples,
            cfg.batch_size, cfg.mask_size, device,
            desc="  Husky", score_threshold=cfg.score_threshold)

        # Overall mIoU: merge raw inter/union from both sources first,
        # then compute per-class IoU, then class-mean.
        # This is correct when sources have different class pixel counts.
        all_cls = set(drone_inter) | set(husky_inter)
        overall_per_cls = {
            cls: (drone_inter.get(cls, 0.0) + husky_inter.get(cls, 0.0))
                 / (drone_union.get(cls, 0.0) + husky_union.get(cls, 0.0) + 1e-6)
            for cls in all_cls
        }
        overall_miou = sum(overall_per_cls.values()) / len(overall_per_cls) \
                       if overall_per_cls else 0.0

        results.append((name, drone_miou, husky_miou, overall_miou,
                        drone_per_cls, husky_per_cls, overall_per_cls))
        print(f"  Drone  mIoU: {drone_miou:.4f}")
        print(f"  Husky  mIoU: {husky_miou:.4f}")
        print(f"  Overall mIoU:{overall_miou:.4f}")
        print(f"  {'Class':<16} {'Drone':>8} {'Husky':>8} {'Overall':>8}")
        for cls in sorted(all_cls):
            d = f"{drone_per_cls[cls]:.4f}" if cls in drone_per_cls else "   —  "
            h = f"{husky_per_cls[cls]:.4f}" if cls in husky_per_cls else "   —  "
            o = f"{overall_per_cls[cls]:.4f}"
            print(f"    {cls:<14} {d:>8} {h:>8} {o:>8}")
        print()

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── Results table ────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS  (standard mIoU: confusion-matrix per class, then class-mean)")
    print("="*60)
    header = f"{'Model':<20} {'Drone (UAV)':>12} {'Husky':>12} {'Overall':>12}"
    print(header)
    print("-" * len(header))
    for name, d, h, o, *_ in results:
        print(f"{name:<20} {d:>12.4f} {h:>12.4f} {o:>12.4f}")

    if len(results) > 1:
        base_d, base_h, base_o = results[0][1], results[0][2], results[0][3]
        print()
        print(f"{'Delta vs Base':<20} {'Drone (UAV)':>12} {'Husky':>12} {'Overall':>12}")
        print("-" * len(header))
        for name, d, h, o, *_ in results[1:]:
            print(f"{name:<20} {d-base_d:>+12.4f} {h-base_h:>+12.4f} {o-base_o:>+12.4f}")

    print()
    print("NOTE: This metric is class-conditioned binary mask IoU")
    print("(class-prompted segmentation), not full semantic segmentation mIoU.")
    print(f"Confident queries merged at score_threshold={cfg.score_threshold};"
          " fallback to top-1 if none exceed threshold.")


if __name__ == "__main__":
    main()
