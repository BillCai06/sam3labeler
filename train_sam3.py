#!/usr/bin/env python3
"""
train_sam3.py — SAM3 fine-tuning script for the qwen3vl2sam pipeline.

Supports:
  - Multiple data sources (auto-discover run output dirs or specify manually)
  - Per-source and global sampling limits
  - Flexible fine-tune ratio (backbone vs head learning rate)
  - Mixed precision (bfloat16), gradient accumulation
  - Cosine LR schedule with linear warmup
  - TensorBoard + CSV loss tracking
  - Periodic checkpoint saves + best-model tracking
  - Resume from checkpoint

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CHECKPOINT OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <output_dir>/
  ├── best/                   ← best val_loss  ★ USE THIS
  │   ├── model.safetensors
  │   ├── config.json
  │   ├── processor_config.json
  │   ├── tokenizer*.json / vocab.json / merges.txt
  │   └── training_state.pt   (epoch, metrics, optimizer/scheduler states)
  ├── epoch_0005/             ← periodic saves (every --save_interval epochs)
  ├── epoch_0010/
  ├── final/                  ← last epoch regardless of val_loss
  ├── losses.csv              ← all loss values (train + val per step/epoch)
  ├── tb/                     ← TensorBoard logs  (tensorboard --logdir tb/)
  └── train_config.json       ← exact args used for this run

 LOAD THE TRAINED MODEL:
  from transformers import Sam3Model, Sam3Processor
  model     = Sam3Model.from_pretrained("checkpoints/ft_v1/best",
                                         torch_dtype=torch.bfloat16)
  processor = Sam3Processor.from_pretrained("checkpoints/ft_v1/best")

 USE IN PIPELINE (sam3_image_detector.py):
  detector = Sam3ImageDetector(sam3_local_path="checkpoints/ft_v1/best")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RECOMMENDED HYPERPARAMETERS  (~1000–2000 images)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase 1 — HEAD ONLY (safe starting point, low overfit risk):
   --freeze_vision --freeze_text   ← freeze entire backbone
   --epochs 40  --lr 5e-4  --weight_decay 0.05
   --batch_size 2  --accum_steps 8          (effective batch = 16)
   --warmup_steps 100
   --mask_loss_weight 3.0  --dice_loss_weight 3.0
   --neg_ratio 0.25
   --output_dir checkpoints/phase1

 Phase 2 — LIGHT FULL FINE-TUNE (optional, after phase 1 converges):
   --finetune_ratio 0.01   ← backbone LR = 5e-6 (very conservative)
   --freeze_text           ← text encoder stays frozen
   --epochs 10  --lr 1e-4  --weight_decay 0.05
   --resume checkpoints/phase1/best
   --output_dir checkpoints/phase2

 Large dataset (5000+ images):
   --finetune_ratio 0.05   ← backbone LR = 10e-6
   --epochs 30  --lr 2e-4  --weight_decay 0.01

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATA SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Auto-discover all annotations.json under --outputs_dir (default).
  Or specify manually:
    --sources path/to/annotations.json:path/to/images_dir

 FINE-TUNE RATIO
  --finetune_ratio 0.0  → backbone fully frozen, only head trains
  --finetune_ratio 0.01 → backbone LR = 1% of head LR  (small dataset)
  --finetune_ratio 0.1  → backbone LR = 10% of head LR (large dataset)
  --finetune_ratio 1.0  → backbone and head share same LR (never for small data)

 RESUME
  --resume checkpoints/phase1/best   ← continue from any saved checkpoint
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sam3")


# ─────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for binary classification (DETR-style)."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.mean() if reduction == "mean" else loss.sum()


def box_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 loss on boxes (N, 4)."""
    return F.l1_loss(pred, gt, reduction="mean")


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    GIoU between two sets of boxes in [cx, cy, w, h] normalized format.
    Returns (N,) GIoU scores.
    """
    # Convert to [x1, y1, x2, y2]
    def to_xyxy(b):
        return torch.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], dim=-1)

    b1 = to_xyxy(boxes1)
    b2 = to_xyxy(boxes2)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter

    iou = inter / (union + 1e-6)

    # Enclosing box
    enc_x1 = torch.min(b1[..., 0], b2[..., 0])
    enc_y1 = torch.min(b1[..., 1], b2[..., 1])
    enc_x2 = torch.max(b1[..., 2], b2[..., 2])
    enc_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Dice loss for mask predictions.
    pred_logits: (N, H, W) raw logits
    targets:     (N, H, W) float in {0, 1}
    """
    pred = pred_logits.sigmoid().flatten(1)  # (N, H*W)
    tgt = targets.flatten(1)
    numerator = 2 * (pred * tgt).sum(1)
    denominator = pred.sum(1) + tgt.sum(1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def mask_bce_loss(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE loss for mask predictions."""
    return F.binary_cross_entropy_with_logits(pred_logits, targets)


# ─────────────────────────────────────────────────────────
# Hungarian matcher (DETR-style)
# ─────────────────────────────────────────────────────────

class HungarianMatcher:
    """
    Matches predicted queries to ground-truth boxes using the Hungarian algorithm.

    Cost = λ_cls * focal_cost + λ_box * L1_cost + λ_giou * giou_cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,   # (num_queries,) — already extracted for one image
        pred_boxes: torch.Tensor,    # (num_queries, 4) cxcywh normalized
        gt_boxes: torch.Tensor,      # (K, 4) cxcywh normalized
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pred_indices, gt_indices) for matched pairs.
        Both are 1-D LongTensors of the same length.
        """
        if gt_boxes.shape[0] == 0:
            device = pred_logits.device
            return torch.zeros(0, dtype=torch.long, device=device), \
                   torch.zeros(0, dtype=torch.long, device=device)

        Q = pred_logits.shape[0]
        K = gt_boxes.shape[0]

        # Classification cost (focal cost, no α weighting)
        prob = pred_logits.sigmoid()  # (Q,)
        # For each query × GT pair: cost is focal-style
        prob_ex = prob.unsqueeze(1).expand(Q, K)          # (Q, K)
        ones_ex = torch.ones(Q, K, device=pred_logits.device)
        cost_cls = (
            -ones_ex * ((1 - prob_ex + 1e-8).log())
            + -prob_ex * ((1 - prob_ex + 1e-8).log() * 0)
        )
        # Simpler: negative log prob (higher prob = lower cost)
        cost_cls = -prob_ex  # (Q, K)

        # Box L1 cost
        pred_boxes_ex = pred_boxes.unsqueeze(1).expand(Q, K, 4)   # (Q, K, 4)
        gt_boxes_ex = gt_boxes.unsqueeze(0).expand(Q, K, 4)       # (Q, K, 4)
        cost_box = torch.cdist(pred_boxes, gt_boxes, p=1)          # (Q, K)

        # GIoU cost
        pred_flat = pred_boxes_ex.reshape(-1, 4)   # (Q*K, 4)
        gt_flat = gt_boxes_ex.reshape(-1, 4)       # (Q*K, 4)
        giou = generalized_box_iou(pred_flat, gt_flat).reshape(Q, K)
        cost_giou = -giou  # (Q, K)

        # Total cost
        C = (
            self.cost_class * cost_cls
            + self.cost_bbox * cost_box
            + self.cost_giou * cost_giou
        )
        C_np = C.cpu().float().numpy()
        row_idx, col_idx = linear_sum_assignment(C_np)

        return (
            torch.as_tensor(row_idx, dtype=torch.long, device=pred_logits.device),
            torch.as_tensor(col_idx, dtype=torch.long, device=pred_logits.device),
        )


# ─────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────

def coco_bbox_to_cxcywh_norm(bbox: list, img_w: int, img_h: int) -> list:
    """COCO [x, y, w, h] pixel → [cx, cy, w, h] normalized [0, 1]."""
    x, y, w, h = bbox
    # Clamp to image bounds
    x = max(0.0, x)
    y = max(0.0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return [cx, cy, nw, nh]


def polygon_to_bbox(polygons: list) -> Optional[list]:
    """
    Derive a tight bounding box [x, y, w, h] (pixel) from polygon coordinates.
    Used when the annotation has no 'bbox' field (labeler per-image JSON format).
    """
    xs, ys = [], []
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
        xs.extend(coords[:, 0].tolist())
        ys.extend(coords[:, 1].tolist())
    if not xs:
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_to_mask(polygons: list, img_h: int, img_w: int) -> np.ndarray:
    """Convert COCO polygon format to binary mask (H, W) uint8."""
    try:
        from pycocotools import mask as coco_mask
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        rle = coco_mask.merge(rles)
        return coco_mask.decode(rle).astype(np.uint8)
    except Exception:
        # Fallback: manual rasterization using OpenCV
        import cv2
        m = np.zeros((img_h, img_w), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
        return m


def find_image_file(filename: str, search_dirs: list[Path]) -> Optional[Path]:
    """Search for an image file across multiple directories."""
    for d in search_dirs:
        if not d.exists():
            continue
        p = d / filename
        if p.exists():
            return p
        # One level deep
        for sub in d.iterdir():
            if sub.is_dir():
                p2 = sub / filename
                if p2.exists():
                    return p2
    return None


def _ann_search_dirs(ann_path: Path, outputs_dir: Path) -> list[Path]:
    """Build image search dirs for a given annotation file/directory."""
    run_dir = ann_path if ann_path.is_dir() else ann_path.parent
    search = [run_dir, run_dir.parent, outputs_dir]
    # Add siblings of run_dir (e.g. drone_frames/ next to the dataset folder)
    if run_dir.parent.exists():
        for sibling in run_dir.parent.iterdir():
            if sibling.is_dir() and sibling != run_dir:
                search.append(sibling)
    return search


def discover_sources(outputs_dir: Path) -> list[tuple[Path, list[Path]]]:
    """
    Auto-discover annotation sources from an outputs directory.

    Supports two formats:
      A) Global   — a single annotations.json covering all images
      B) Per-image — an 'annotations/' directory with one .json per image
                     (output of the labeler WebUI)

    Returns list of (source_path, image_search_dirs).
    For format A: source_path is the annotations.json file.
    For format B: source_path is the annotations/ directory.
    """
    seen: set[Path] = set()
    sources = []

    # Format A: global annotations.json
    for ann_path in sorted(outputs_dir.rglob("annotations.json")):
        key = ann_path.resolve()
        if key not in seen:
            seen.add(key)
            sources.append((ann_path, _ann_search_dirs(ann_path, outputs_dir)))

    # Format B: annotations/ directories containing per-image .json files
    for ann_dir in sorted(outputs_dir.rglob("annotations")):
        if not ann_dir.is_dir():
            continue
        jsons = list(ann_dir.glob("*.json"))
        if not jsons:
            continue
        key = ann_dir.resolve()
        if key not in seen:
            seen.add(key)
            sources.append((ann_dir, _ann_search_dirs(ann_dir, outputs_dir)))

    return sources


def _load_coco_json(ann_path: Path) -> dict:
    """Load and return a COCO JSON file."""
    with open(ann_path) as f:
        return json.load(f)


def _coco_to_samples(
    coco: dict,
    search_dirs: list[Path],
    rng: random.Random,
    neg_ratio: float,
) -> tuple[list[dict], int]:
    """
    Convert a loaded COCO dict to a list of training sample dicts.
    Returns (samples, n_missing_images).

    Handles both:
      - Annotations WITH 'bbox' field (run_batch.py output)
      - Annotations WITHOUT 'bbox' field (labeler output — bbox derived from polygon)
    """
    id2img = {img["id"]: img for img in coco["images"]}
    id2cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    all_class_names = [cat["name"] for cat in coco["categories"]]

    img2class2anns: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        cls = id2cat.get(ann["category_id"])
        if cls:
            img2class2anns[ann["image_id"]][cls].append(ann)

    samples = []
    missing = 0

    for img_id, img_info in id2img.items():
        filename = img_info["file_name"]
        img_path = find_image_file(filename, search_dirs)
        if img_path is None:
            missing += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        classes_present = img2class2anns.get(img_id, {})

        # Positive samples
        for cls_name, anns in classes_present.items():
            boxes, segs = [], []
            for ann in anns:
                seg = ann.get("segmentation") or []
                if not seg:
                    continue
                # Derive bbox from polygon if not present
                bbox = ann.get("bbox") or polygon_to_bbox(seg)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                boxes.append(coco_bbox_to_cxcywh_norm(bbox, img_w, img_h))
                segs.append(seg)
            if boxes:
                samples.append({
                    "image_path": str(img_path),
                    "class_name": cls_name,
                    "gt_boxes": boxes,
                    "gt_segs": segs,
                    "img_w": img_w,
                    "img_h": img_h,
                    "is_negative": False,
                })

        # Negative samples
        absent = [c for c in all_class_names if c not in classes_present]
        n_neg = max(1, int(len(classes_present) * neg_ratio / max(1 - neg_ratio, 1e-6)))
        for cls_name in rng.sample(absent, min(n_neg, len(absent))):
            samples.append({
                "image_path": str(img_path),
                "class_name": cls_name,
                "gt_boxes": [],
                "gt_segs": [],
                "img_w": img_w,
                "img_h": img_h,
                "is_negative": True,
            })

    return samples, missing


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class Sam3Dataset(Dataset):
    """
    COCO-format dataset for SAM3 fine-tuning.

    Each sample is a (image, class_name) pair. The class may have zero GT
    instances in the image (negative sample), which teaches the model to
    output low confidence when a class is absent.

    Returns:
        image_path: str
        class_name: str
        gt_boxes:   (K, 4) float32 cxcywh normalized  — empty tensor if K=0
        gt_segs:    list of polygon lists               — empty list if K=0
        img_w, img_h: image dimensions from annotation
    """

    def __init__(
        self,
        samples: list[dict],
        processor,
        mask_size: int = 288,
        augment: bool = False,
    ):
        """
        samples: list of dicts with keys:
            image_path, class_name, gt_boxes (list of [cx,cy,w,h] normalized),
            gt_segs (list of polygon lists), img_w, img_h
        """
        self.samples = samples
        self.processor = processor
        self.mask_size = mask_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img_path = s["image_path"]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Processor input (normalises + resizes to model input size)
        proc = self.processor(images=img, return_tensors="pt")
        pixel_values = proc["pixel_values"][0]  # (C, H, W)

        # Ground-truth boxes
        gt_boxes = torch.tensor(s["gt_boxes"], dtype=torch.float32) if s["gt_boxes"] \
                   else torch.zeros((0, 4), dtype=torch.float32)

        # Ground-truth masks at low res (mask_size × mask_size)
        gt_masks_list = []
        for seg in s["gt_segs"]:
            m = polygon_to_mask(seg, img_h, img_w)  # (H, W) uint8
            m_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            m_lr = F.interpolate(
                m_t, size=(self.mask_size, self.mask_size),
                mode="bilinear", align_corners=False,
            )[0, 0]  # (mask_size, mask_size)
            gt_masks_list.append((m_lr > 0.5).float())

        gt_masks = torch.stack(gt_masks_list, 0) if gt_masks_list \
                   else torch.zeros((0, self.mask_size, self.mask_size), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "class_name": s["class_name"],
            "gt_boxes": gt_boxes,       # (K, 4)
            "gt_masks": gt_masks,       # (K, mask_size, mask_size)
            "img_w": img_w,
            "img_h": img_h,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack pixel_values; keep boxes/masks as lists (variable K)."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "class_names": [b["class_name"] for b in batch],
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_masks": [b["gt_masks"] for b in batch],
    }


# ─────────────────────────────────────────────────────────
# Sample builder
# ─────────────────────────────────────────────────────────

def build_samples(
    sources: list[tuple[Path, list[Path]]],
    max_samples: Optional[int] = None,
    neg_ratio: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """
    Build flat list of training samples from COCO annotation sources.

    For each annotated image:
      - One positive sample per class present in the image
      - Randomly sampled negative samples for classes absent from the image
        (up to neg_ratio × number of positive samples)
    """
    rng = random.Random(seed)
    all_samples = []

    # When max_samples is set, cap how many JSON files we read.
    # Each image yields at least 1 sample, so reading max_samples files
    # is a safe upper bound. We shuffle first so we get a random subset.
    max_images = max_samples  # None = no cap

    for source_path, search_dirs in sources:
        source_samples: list[dict] = []
        missing_images = 0

        if source_path.is_dir():
            # ── Format B: per-image JSON directory ──────────────
            json_files = sorted(source_path.glob("*.json"))
            total_files = len(json_files)
            if max_images is not None and total_files > max_images:
                rng_files = random.Random(seed)
                json_files = rng_files.sample(json_files, max_images)
                logger.info(
                    f"Loading per-image JSONs from {source_path}  "
                    f"({max_images}/{total_files} files, capped by --max_train/--max_val)"
                )
            else:
                logger.info(f"Loading per-image JSONs from {source_path}  ({total_files} files)")
            for jf in json_files:
                try:
                    coco = _load_coco_json(jf)
                except Exception as e:
                    logger.warning(f"  Skip {jf.name}: {e}")
                    continue
                s, m = _coco_to_samples(coco, search_dirs, rng, neg_ratio)
                source_samples.extend(s)
                missing_images += m
        else:
            # ── Format A: global annotations.json ───────────────
            logger.info(f"Loading global annotations from {source_path}")
            try:
                coco = _load_coco_json(source_path)
            except Exception as e:
                logger.error(f"  Cannot load {source_path}: {e}")
                continue
            source_samples, missing_images = _coco_to_samples(coco, search_dirs, rng, neg_ratio)

        if missing_images:
            logger.warning(f"  {missing_images} images not found in search dirs")
        n_pos = sum(1 for s in source_samples if not s["is_negative"])
        n_neg = sum(1 for s in source_samples if s["is_negative"])
        logger.info(f"  → {len(source_samples)} samples  ({n_pos} pos, {n_neg} neg)")
        all_samples.extend(source_samples)

    rng.shuffle(all_samples)

    if max_samples and len(all_samples) > max_samples:
        # Preserve positive/negative balance when trimming
        pos = [s for s in all_samples if not s["is_negative"]]
        neg = [s for s in all_samples if s["is_negative"]]
        n_neg_keep = int(max_samples * neg_ratio)
        n_pos_keep = max_samples - n_neg_keep
        all_samples = pos[:n_pos_keep] + neg[:n_neg_keep]
        rng.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)} "
                f"({sum(1 for s in all_samples if not s['is_negative'])} pos, "
                f"{sum(1 for s in all_samples if s['is_negative'])} neg)")
    return all_samples


# ─────────────────────────────────────────────────────────
# Loss computation
# ─────────────────────────────────────────────────────────

def compute_loss(
    out,
    gt_boxes_list: list[torch.Tensor],
    gt_masks_list: list[torch.Tensor],
    matcher: HungarianMatcher,
    cfg: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute total training loss for a batch of images.

    out.pred_logits:   (N, Q) or (N, Q, 1)
    out.pred_boxes:    (N, Q, 4) cxcywh normalized
    out.pred_masks:    (N, Q, Hm, Wm)
    out.presence_logits: (N, 1)
    """
    N = len(gt_boxes_list)

    # Normalise shapes
    logits = out.pred_logits  # (N, Q) or (N, Q, 1)
    if logits.dim() == 3:
        logits = logits[..., 0]  # (N, Q)

    boxes = out.pred_boxes       # (N, Q, 4)
    masks = out.pred_masks       # (N, Q, Hm, Wm)
    presence = out.presence_logits  # (N, 1)

    mask_h, mask_w = masks.shape[-2], masks.shape[-1]

    losses = {
        "loss_cls": torch.tensor(0.0, device=device),
        "loss_box": torch.tensor(0.0, device=device),
        "loss_giou": torch.tensor(0.0, device=device),
        "loss_mask": torch.tensor(0.0, device=device),
        "loss_dice": torch.tensor(0.0, device=device),
        "loss_presence": torch.tensor(0.0, device=device),
    }
    num_matched = 0

    for i in range(N):
        gt_b = gt_boxes_list[i].to(device)   # (K, 4)
        gt_m = gt_masks_list[i].to(device)   # (K, Hm0, Wm0) — may need resize
        K = gt_b.shape[0]

        # Resize GT masks to match pred_masks resolution
        if K > 0 and (gt_m.shape[-2] != mask_h or gt_m.shape[-1] != mask_w):
            gt_m = F.interpolate(
                gt_m.unsqueeze(1).float(), size=(mask_h, mask_w),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        # Presence loss
        has_gt = float(K > 0)
        losses["loss_presence"] = losses["loss_presence"] + F.binary_cross_entropy_with_logits(
            presence[i], torch.tensor([has_gt], device=device)
        )

        if K == 0:
            # All-negative: push all query logits toward 0
            neg_target = torch.zeros_like(logits[i])
            losses["loss_cls"] = losses["loss_cls"] + sigmoid_focal_loss(
                logits[i], neg_target, reduction="mean"
            )
            continue

        # Hungarian matching
        pred_idx, gt_idx = matcher.match(logits[i], boxes[i], gt_b)
        num_matched += len(pred_idx)

        # --- Classification loss ---
        # Positive (matched) queries → target 1
        # All others → target 0
        cls_target = torch.zeros_like(logits[i])
        if len(pred_idx) > 0:
            cls_target[pred_idx] = 1.0
        losses["loss_cls"] = losses["loss_cls"] + sigmoid_focal_loss(
            logits[i], cls_target, reduction="mean"
        )

        if len(pred_idx) == 0:
            continue

        # --- Box losses ---
        pred_b_match = boxes[i][pred_idx]    # (M, 4)
        gt_b_match = gt_b[gt_idx]            # (M, 4)

        losses["loss_box"] = losses["loss_box"] + F.l1_loss(pred_b_match, gt_b_match)

        giou = generalized_box_iou(pred_b_match, gt_b_match)
        losses["loss_giou"] = losses["loss_giou"] + (1 - giou).mean()

        # --- Mask losses ---
        pred_m_match = masks[i][pred_idx]    # (M, Hm, Wm)
        gt_m_match = gt_m[gt_idx]            # (M, Hm, Wm)

        losses["loss_mask"] = losses["loss_mask"] + mask_bce_loss(pred_m_match, gt_m_match)
        losses["loss_dice"] = losses["loss_dice"] + dice_loss(pred_m_match, gt_m_match)

    # Normalise by batch size
    for k in losses:
        losses[k] = losses[k] / N

    # Weighted total loss
    total = (
        cfg.cls_loss_weight    * losses["loss_cls"]
        + cfg.box_loss_weight  * losses["loss_box"]
        + cfg.giou_loss_weight * losses["loss_giou"]
        + cfg.mask_loss_weight * losses["loss_mask"]
        + cfg.dice_loss_weight * losses["loss_dice"]
        + cfg.presence_loss_weight * losses["loss_presence"]
    )
    losses["loss_total"] = total
    return losses


# ─────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────

def load_model(sam3_path: str, device: str):
    from transformers import Sam3Model, Sam3Processor
    path_obj = Path(sam3_path).expanduser().resolve()
    logger.info(f"Loading Sam3Model from: {path_obj}")

    # Always treat local filesystem paths as local checkpoints.
    # This avoids accidental Hugging Face Hub repo-id validation errors.
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {path_obj}. "
            f"Use a valid local checkpoint directory (for example: checkpoints/phase1_h200/best) "
            f"or the base model directory (sam3)."
        )

    required = ["config.json", "model.safetensors", "processor_config.json"]
    missing = [name for name in required if not (path_obj / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory is missing required files: {missing}. "
            f"Directory checked: {path_obj}"
        )

    load_kwargs = {}
    if "cuda" in device:
        load_kwargs["torch_dtype"] = torch.bfloat16
    load_kwargs["local_files_only"] = True
    model = Sam3Model.from_pretrained(str(path_obj), **load_kwargs)
    model.to(device)
    model.train()
    processor = Sam3Processor.from_pretrained(str(path_obj), local_files_only=True)
    return model, processor


def _get_param_groups(model, cfg) -> list[dict]:
    """
    Split parameters into backbone (vision + text) and head groups.
    Head LR = cfg.lr, backbone LR = cfg.lr × cfg.finetune_ratio.
    Returns param_groups for AdamW.
    """
    backbone_names = {"vision_encoder", "text_encoder", "backbone", "vision_model",
                      "text_model", "vit", "clip"}
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Heuristic: if any backbone keyword in the module name → backbone group
        parts = name.lower().split(".")
        if any(kw in name.lower() for kw in backbone_names):
            backbone_params.append(p)
        else:
            head_params.append(p)

    backbone_lr = cfg.lr * cfg.finetune_ratio

    logger.info(
        f"Param groups: backbone={len(backbone_params)} params (lr={backbone_lr:.2e}), "
        f"head={len(head_params)} params (lr={cfg.lr:.2e})"
    )

    groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_lr > 0 and backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    elif backbone_params:
        # Zero LR: effectively frozen but still in param groups so grad is computed
        for p in backbone_params:
            p.requires_grad_(False)
        logger.info("Backbone parameters frozen (finetune_ratio=0)")

    return groups


def freeze_module(model, keyword: str):
    """Freeze all parameters whose name contains keyword."""
    n = 0
    for name, p in model.named_parameters():
        if keyword.lower() in name.lower():
            p.requires_grad_(False)
            n += 1
    logger.info(f"Froze {n} parameters containing '{keyword}'")


# ─────────────────────────────────────────────────────────
# Loss tracker + CSV logger
# ─────────────────────────────────────────────────────────

class LossTracker:
    """Running average of named loss values."""

    def __init__(self):
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, losses: dict[str, float]):
        for k, v in losses.items():
            self._sums[k] += v
            self._counts[k] += 1

    def means(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class CSVLogger:
    """Append loss entries to a CSV file."""

    def __init__(self, path: Path):
        self.path = path
        self._fieldnames: list | None = None
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.reader(f)
                try:
                    self._fieldnames = next(reader)
                except StopIteration:
                    pass

    def log(self, row: dict):
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            write_header = True
        else:
            write_header = False
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames,
                                    extrasaction="ignore", restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ─────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────

def _reduce_dict(d: dict[str, float], device: torch.device) -> dict[str, float]:
    """All-reduce a dict of scalar floats across DDP ranks (in-place average)."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return d
    world = dist.get_world_size()
    keys = sorted(d.keys())
    t = torch.tensor([d[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world
    return {k: float(t[i]) for i, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    processor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    matcher: HungarianMatcher,
    cfg: argparse.Namespace,
    epoch: int,
    device: torch.device,
    writer,
    csv_logger: CSVLogger,
    global_step: list[int],
    is_main: bool = True,
) -> dict[str, float]:

    model.train()
    tracker = LossTracker()
    dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    is_ddp = isinstance(model, DDP)
    raw_model = model.module if is_ddp else model

    # Set epoch for DistributedSampler so shuffling differs each epoch
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch",
                dynamic_ncols=True, disable=not is_main)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        class_names = batch["class_names"]

        # Get unique classes in this batch and encode text once per class
        unique_classes = list(dict.fromkeys(class_names))  # preserves order
        text_embed_cache: dict[str, torch.Tensor] = {}
        for cls in unique_classes:
            tokens = processor.tokenizer(
                cls, return_tensors="pt",
                padding="max_length", max_length=32,
            ).to(device)
            with torch.no_grad():
                embed = raw_model.get_text_features(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    return_dict=True,
                ).pooler_output  # (1, seq, dim)
            text_embed_cache[cls] = embed

        # Build per-image text embeds matching class_names order
        text_embeds = torch.cat(
            [text_embed_cache[cn] for cn in class_names], dim=0
        )  # (N, seq, dim)

        # Forward pass: pass pixel_values through DDP so the full vision encoder
        # computation graph is tracked inside DDP's forward. Pre-computing
        # vision_embeds on raw_model outside DDP's forward caused DDP's backward
        # hooks to do in-place ops on tensors still needed by autograd (version
        # mismatch error on the [H*W, C] spatial feature tensor).
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                            dtype=dtype, enabled="cuda" in str(device)):
            out = model(
                pixel_values=pixel_values,
                text_embeds=text_embeds,
            )

            losses = compute_loss(
                out,
                batch["gt_boxes"],
                batch["gt_masks"],
                matcher,
                cfg,
                device,
            )

        # Gradient accumulation
        # Use no_sync on non-sync steps to skip expensive all_reduce mid-accumulation
        loss = losses["loss_total"] / cfg.accum_steps
        sync_step = (step + 1) % cfg.accum_steps == 0
        ctx = (model.no_sync() if is_ddp and not sync_step
               else torch.no_grad.__new__(torch.no_grad))  # null context
        # Simple approach: just call backward; no_sync avoids redundant comms
        if is_ddp and not sync_step:
            with model.no_sync():
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
        else:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        loss_vals = {k: float(v.detach()) for k, v in losses.items()}
        tracker.update(loss_vals)

        if (step + 1) % cfg.accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step[0] += 1

        # Logging
        if global_step[0] % cfg.log_interval == 0 and global_step[0] > 0:
            means = _reduce_dict(tracker.means(), device)  # avg across ranks
            if is_main:
                pbar.set_postfix({
                    "loss": f"{means.get('loss_total', 0):.4f}",
                    "cls": f"{means.get('loss_cls', 0):.4f}",
                    "mask": f"{means.get('loss_mask', 0):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                if writer is not None:
                    for k, v in means.items():
                        writer.add_scalar(f"train/{k}", v, global_step[0])
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step[0])

            if is_main:
                csv_logger.log({
                "epoch": epoch,
                "step": global_step[0],
                "phase": "train",
                    **means,
                    "mask_iou": "",
                    "lr": scheduler.get_last_lr()[0],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            tracker.reset()

    return _reduce_dict(tracker.means(), device)


@torch.no_grad()
def validate(
    model,
    processor,
    loader: DataLoader,
    matcher: HungarianMatcher,
    cfg: argparse.Namespace,
    epoch: int,
    device: torch.device,
    writer,
    csv_logger: CSVLogger,
    is_main: bool = True,
) -> dict[str, float]:

    model.eval()
    tracker = LossTracker()
    iou_scores = []
    dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    raw_model = model.module if isinstance(model, DDP) else model

    for batch in tqdm(loader, desc=f"Val   {epoch}", unit="batch",
                      dynamic_ncols=True, disable=not is_main):
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        class_names = batch["class_names"]

        autocast_ctx = torch.autocast(
            device_type="cuda" if "cuda" in str(device) else "cpu",
            dtype=dtype, enabled="cuda" in str(device),
        )

        with autocast_ctx:
            vision_embeds = raw_model.get_vision_features(pixel_values=pixel_values)

        unique_classes = list(dict.fromkeys(class_names))
        text_embed_cache: dict[str, torch.Tensor] = {}
        for cls in unique_classes:
            tokens = processor.tokenizer(
                cls, return_tensors="pt",
                padding="max_length", max_length=32,
            ).to(device)
            with autocast_ctx:
                embed = raw_model.get_text_features(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    return_dict=True,
                ).pooler_output
            text_embed_cache[cls] = embed

        text_embeds = torch.cat([text_embed_cache[cn] for cn in class_names], dim=0)

        with autocast_ctx:
            out = model(vision_embeds=vision_embeds, text_embeds=text_embeds)
        losses = compute_loss(
            out, batch["gt_boxes"], batch["gt_masks"], matcher, cfg, device
        )

        # Mask IoU metric on positive samples
        logits = out.pred_logits
        if logits.dim() == 3:
            logits = logits[..., 0]
        scores = logits.sigmoid() * out.presence_logits.sigmoid()

        for i in range(len(class_names)):
            gt_m = batch["gt_masks"][i].to(device)
            if gt_m.shape[0] == 0:
                continue
            # Best matching query by score
            top_q = scores[i].argmax().item()
            pred_mask = (out.pred_masks[i, top_q].sigmoid() > 0.5).float()

            mask_h, mask_w = pred_mask.shape
            gt_r = F.interpolate(
                gt_m.unsqueeze(1).float(), size=(mask_h, mask_w),
                mode="nearest",
            ).squeeze(1)
            gt_any = (gt_r.sum(0) > 0).float()

            inter = (pred_mask * gt_any).sum()
            union = (pred_mask + gt_any).clamp(max=1).sum()
            iou_scores.append(float(inter / (union + 1e-6)))

        tracker.update({k: float(v) for k, v in losses.items()})

    # Aggregate metrics across all ranks
    means = _reduce_dict(tracker.means(), device)
    local_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    iou_t = torch.tensor(local_iou, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(iou_t, op=dist.ReduceOp.SUM)
        iou_t /= dist.get_world_size()
    means["mask_iou"] = float(iou_t)

    if is_main:
        if writer is not None:
            for k, v in means.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        csv_logger.log({
            "epoch": epoch,
            "step": -1,
            "phase": "val",
            **means,
            "lr": "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(
            f"[Val epoch {epoch}] loss={means.get('loss_total', 0):.4f}  "
            f"mask_iou={means.get('mask_iou', 0):.4f}"
        )
    return means


# ─────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────

def save_checkpoint(model, processor, optimizer, scheduler, epoch, metrics, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.save_pretrained(str(path))
    processor.save_pretrained(str(path))
    meta = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(meta, path / "training_state.pt")
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, processor, optimizer, scheduler, path: Path) -> int:
    """Load checkpoint and return next epoch number."""
    from transformers import Sam3Model, Sam3Processor
    meta = torch.load(path / "training_state.pt", weights_only=False)
    optimizer.load_state_dict(meta["optimizer_state"])
    scheduler.load_state_dict(meta["scheduler_state"])
    epoch = meta["epoch"]
    logger.info(f"Resumed from {path} (epoch {epoch})")
    return epoch + 1


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM3 fine-tuning script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    grp = p.add_argument_group("Data")
    grp.add_argument(
        "--outputs_dir", default="outputs",
        help="Root directory to auto-discover run output subdirs (each must contain "
             "annotations.json). Scanned recursively.",
    )
    grp.add_argument(
        "--sources", nargs="+", default=None,
        metavar="ANN_JSON:IMAGES_DIR",
        help="Override auto-discovery with explicit sources. Format: "
             "path/to/annotations.json:path/to/images_dir. "
             "Multiple sources separated by spaces.",
    )
    grp.add_argument(
        "--max_train", type=int, default=None,
        help="Max number of training samples (balanced pos/neg). None = all.",
    )
    grp.add_argument(
        "--max_val", type=int, default=None,
        help="Max number of validation samples. None = 15%% of all data.",
    )
    grp.add_argument(
        "--val_split", type=float, default=0.15,
        help="Fraction of data held out for validation.",
    )
    grp.add_argument(
        "--neg_ratio", type=float, default=0.3,
        help="Fraction of training samples that are negatives (absent-class).",
    )

    # --- Model ---
    grp = p.add_argument_group("Model")
    grp.add_argument("--sam3_path", default="sam3",
                     help="Path to SAM3 model directory (local).")
    grp.add_argument("--mask_size", type=int, default=288,
                     help="Low-res mask size used for GT mask resizing.")

    # --- Fine-tune control ---
    grp = p.add_argument_group("Fine-tune control")
    grp.add_argument(
        "--finetune_ratio", type=float, default=0.1,
        help="Backbone LR = lr × finetune_ratio. "
             "0 = backbone frozen. 1 = all params same LR.",
    )
    grp.add_argument(
        "--freeze_vision", action="store_true",
        help="Hard-freeze the vision encoder (overrides --finetune_ratio for vision).",
    )
    grp.add_argument(
        "--freeze_text", action="store_true",
        help="Hard-freeze the text encoder (overrides --finetune_ratio for text).",
    )

    # --- Training hyper-parameters ---
    grp = p.add_argument_group("Training")
    grp.add_argument("--epochs", type=int, default=20)
    grp.add_argument("--batch_size", type=int, default=2,
                     help="Images per GPU step.")
    grp.add_argument("--accum_steps", type=int, default=8,
                     help="Gradient accumulation steps. Effective batch = batch_size × accum_steps.")
    grp.add_argument("--lr", type=float, default=2e-4, help="Head learning rate.")
    grp.add_argument("--weight_decay", type=float, default=0.01)
    grp.add_argument("--warmup_steps", type=int, default=200,
                     help="Linear LR warmup steps.")
    grp.add_argument("--max_grad_norm", type=float, default=1.0)
    grp.add_argument("--num_workers", type=int, default=4,
                     help="DataLoader worker processes.")
    grp.add_argument("--seed", type=int, default=42)

    # --- Loss weights ---
    grp = p.add_argument_group("Loss weights")
    grp.add_argument("--cls_loss_weight",      type=float, default=1.0)
    grp.add_argument("--box_loss_weight",      type=float, default=5.0)
    grp.add_argument("--giou_loss_weight",     type=float, default=2.0)
    grp.add_argument("--mask_loss_weight",     type=float, default=2.0)
    grp.add_argument("--dice_loss_weight",     type=float, default=2.0)
    grp.add_argument("--presence_loss_weight", type=float, default=1.0)

    # --- Logging & checkpoints ---
    grp = p.add_argument_group("Output")
    grp.add_argument("--output_dir", default="checkpoints/finetune",
                     help="Root directory for checkpoints and logs.")
    grp.add_argument("--log_interval", type=int, default=10,
                     help="Global steps between log writes.")
    grp.add_argument("--save_interval", type=int, default=5,
                     help="Epochs between periodic checkpoint saves.")
    grp.add_argument("--resume", default=None,
                     help="Path to a checkpoint directory to resume from.")
    grp.add_argument("--no_tensorboard", action="store_true",
                     help="Disable TensorBoard logging.")

    return p.parse_args()


def main():
    cfg = parse_args()

    # ── Distributed setup ─────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_dist = local_rank >= 0
    if is_dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
    is_main = (rank == 0)

    # Reproducibility
    random.seed(cfg.seed + rank)   # different seed per rank for data augmentation
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)

    if is_main:
        logger.info(f"Device: {device}  |  world_size: {world_size}")

    out_dir = Path(cfg.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()  # wait for rank 0 to create dir

    # ── Data sources ──────────────────────────────────────
    if cfg.sources:
        sources = []
        for s in cfg.sources:
            parts = s.split(":", 1)
            ann_path = Path(parts[0])
            img_dirs = [Path(parts[1])] if len(parts) > 1 else [ann_path.parent]
            sources.append((ann_path, img_dirs))
    else:
        sources = discover_sources(Path(cfg.outputs_dir))

    if not sources:
        logger.error("No data sources found. Use --outputs_dir or --sources.")
        sys.exit(1)

    # ── Build sample list ─────────────────────────────────
    # If both max_train and max_val are set, cap JSON loading upfront
    # so we don't read thousands of files just to discard them.
    _load_cap = None
    if cfg.max_train and cfg.max_val:
        _load_cap = cfg.max_train + cfg.max_val
    elif cfg.max_train:
        _load_cap = int(cfg.max_train / max(1 - cfg.val_split, 0.01))
    all_samples = build_samples(sources, max_samples=_load_cap, neg_ratio=cfg.neg_ratio, seed=cfg.seed)

    # Train/val split
    n_val = cfg.max_val or max(1, int(len(all_samples) * cfg.val_split))
    n_train = cfg.max_train or (len(all_samples) - n_val)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:n_val + n_train]

    logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # ── Model ─────────────────────────────────────────────
    sam3_path = cfg.sam3_path
    if not Path(sam3_path).is_absolute():
        sam3_path = str(Path(__file__).parent / sam3_path)

    model, processor = load_model(sam3_path, str(device))

    if cfg.freeze_vision:
        freeze_module(model, "vision")
    if cfg.freeze_text:
        freeze_module(model, "text")

    # Try to enable gradient checkpointing when backbone is unfrozen.
    # Not all model classes support this; skip silently if unsupported.
    if not cfg.freeze_vision or not cfg.freeze_text:
        for mod in [model] + list(model.children()):
            if hasattr(mod, "gradient_checkpointing_enable"):
                try:
                    mod.gradient_checkpointing_enable()
                    logger.info(f"Gradient checkpointing enabled on {mod.__class__.__name__}")
                    break
                except (ValueError, AttributeError):
                    pass

    # ── Wrap with DDP ─────────────────────────────────────
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ── Datasets & loaders ────────────────────────────────
    train_ds = Sam3Dataset(train_samples, processor, mask_size=cfg.mask_size, augment=True)
    val_ds   = Sam3Dataset(val_samples,   processor, mask_size=cfg.mask_size, augment=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_dist else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Optimizer ─────────────────────────────────────────
    param_groups = _get_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    total_steps = (len(train_loader) // cfg.accum_steps) * cfg.epochs

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed-precision scaler (disabled for bfloat16 — it doesn't need it)
    scaler = None  # bfloat16 is stable without GradScaler

    matcher = HungarianMatcher(
        cost_class=cfg.cls_loss_weight,
        cost_bbox=cfg.box_loss_weight,
        cost_giou=cfg.giou_loss_weight,
    )

    # ── Logging (rank 0 only) ─────────────────────────────
    tb_writer = None
    if is_main and not cfg.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = out_dir / "tb"
            tb_dir.mkdir(exist_ok=True)
            tb_writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard logs → {tb_dir}")
            logger.info(f"  Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available (pip install tensorboard)")

    csv_logger = CSVLogger(out_dir / "losses.csv")

    # Save config for reproducibility (rank 0 only)
    if is_main:
        with open(out_dir / "train_config.json", "w") as f:
            json.dump(vars(cfg), f, indent=2)

    # ── Resume ────────────────────────────────────────────
    start_epoch = 1
    if cfg.resume:
        start_epoch = load_checkpoint(model, processor, optimizer, scheduler, Path(cfg.resume))

    # ── Training ──────────────────────────────────────────
    global_step = [0]
    best_val_loss = float("inf")
    best_val_iou = 0.0

    if is_main:
        logger.info("=" * 60)
        logger.info(f"SAM3 Fine-tuning  |  {cfg.epochs} epochs  |  "
                    f"world_size={world_size}  |  "
                    f"effective batch = {cfg.batch_size * cfg.accum_steps * world_size}")
        logger.info(f"Output → {out_dir}")
        logger.info("=" * 60)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, processor, train_loader,
            optimizer, scheduler, scaler,
            matcher, cfg, epoch, device,
            tb_writer, csv_logger, global_step,
            is_main=is_main,
        )

        val_metrics = validate(
            model, processor, val_loader,
            matcher, cfg, epoch, device,
            tb_writer, csv_logger,
            is_main=is_main,
        )

        val_loss = val_metrics.get("loss_total", float("inf"))
        val_iou  = val_metrics.get("mask_iou", 0.0)

        # Best-model checkpoint (rank 0 only)
        if is_main and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, processor, optimizer, scheduler, epoch,
                            val_metrics, out_dir / "best")
            logger.info(f"  ★ New best model (val_loss={best_val_loss:.4f}, "
                        f"mask_iou={val_iou:.4f})")

        # Periodic checkpoint
        if is_main and epoch % cfg.save_interval == 0:
            save_checkpoint(model, processor, optimizer, scheduler, epoch,
                            val_metrics, out_dir / f"epoch_{epoch:04d}")

    # Final save (rank 0 only)
    if is_main:
        save_checkpoint(model, processor, optimizer, scheduler, cfg.epochs,
                        {}, out_dir / "final")

        if tb_writer:
            tb_writer.close()

        logger.info("=" * 60)
        logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
        logger.info(f"Best model → {out_dir / 'best'}")
        logger.info(f"Loss log   → {out_dir / 'losses.csv'}")
        logger.info("=" * 60)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
