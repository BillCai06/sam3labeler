"""
Visualization, I/O helpers, and color palette.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Fixed per-class colors (RGB) — stable regardless of class_list order or length.
# Unknown classes fall back to PALETTE via hash.
CLASS_COLORS: dict[str, tuple] = {
    "trail":      (194, 154, 108),  # tan
    "grass":      ( 52, 199,  89),  # green
    "tree":       ( 34, 139,  34),  # dark green
    "underbrush": (107, 142,  35),  # olive
    "mulch":      (101,  67,  33),  # dark brown
    "log":        (139,  90,  43),  # medium brown
    "rock":       (128, 128, 128),  # gray
    "gravel":     (180, 180, 165),  # light gray
    "mud":        (116,  92,  72),  # muddy brown
    "water":      (  0, 122, 255),  # blue
    "snow":       (200, 225, 255),  # icy white-blue
    "sky":        (135, 206, 235),  # sky blue
    "fence":      (205, 170, 100),  # tan-yellow
    "bush":       ( 85, 160,  60),  # medium green
    "pole":       ( 90,  90,  90),  # dark gray
    "sign":       (255, 204,   0),  # yellow
    "concrete":   (176, 196, 222),  # steel blue-gray
    "asphalt":    ( 60,  60,  60),  # near-black gray
    "building":   (210, 140,  80),  # orange-brown
    "robot":      (  0, 210, 210),  # cyan
    "car":        (255,  59,  48),  # red
    "person":     (255,  45,  85),  # pink
    "animal":     (255, 149,   0),  # orange
    "other":      (175,  82, 222),  # purple
}

# Fallback palette for classes not listed above
PALETTE = [
    (255, 59,  48),
    (52,  199, 89),
    (0,   122, 255),
    (255, 149, 0),
    (175, 82,  222),
    (255, 45,  85),
    (90,  200, 250),
    (255, 204, 0),
    (100, 210, 255),
    (48,  209, 88),
    (255, 55,  95),
    (0,   199, 190),
    (255, 179, 64),
    (191, 90,  242),
    (100, 100, 255),
    (255, 69,  58),
    (50,  173, 230),
    (255, 159, 10),
    (172, 148, 248),
    (88,  86,  214),
]


def get_class_color(class_name: str, class_list: list[str] = None) -> tuple:
    """Get a fixed color for a class name. class_list is kept for API compatibility."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    # Unknown class: deterministic hash so it's still stable across runs
    return PALETTE[hash(class_name) % len(PALETTE)]


def visualize_results(
    image: Image.Image,
    results: list[dict],
    class_list: list[str],
    alpha: float = 0.5,
    draw_bbox: bool = True,
    draw_labels: bool = True,
) -> Image.Image:
    """
    Draw segmentation masks and bounding boxes on an image.

    Args:
        image: original PIL image (RGB)
        results: list of detection dicts from the pipeline
        class_list: ordered list of all possible classes (for color assignment)
        alpha: mask transparency (0=invisible, 1=opaque)
        draw_bbox: draw bounding box rectangles
        draw_labels: draw class name + confidence labels

    Returns:
        annotated PIL image
    """
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    img_h, img_w = img_np.shape[:2]

    # Blend each mask independently so overlapping masks from different classes
    # both show through (each at `alpha` strength on whatever is beneath it).
    for det in results:
        mask = det.get("mask")
        if mask is None:
            continue
        color = np.array(get_class_color(det["class"], class_list), dtype=np.float32)
        mask3 = mask[:, :, None]  # (H, W, 1) for broadcasting
        img_np = np.where(mask3, img_np * (1 - alpha) + color * alpha, img_np)

    img_np = img_np.clip(0, 255).astype(np.uint8)

    # Draw bboxes and labels
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    font = _load_font(14)

    for det in results:
        color = get_class_color(det["class"], class_list)
        bbox_norm = det.get("bbox", [])
        if len(bbox_norm) != 4:
            continue

        x1 = int(bbox_norm[0] * img_w)
        y1 = int(bbox_norm[1] * img_h)
        x2 = int(bbox_norm[2] * img_w)
        y2 = int(bbox_norm[3] * img_h)

        if draw_bbox:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if draw_labels:
            conf = det.get("confidence", 0.0)
            sam_score = det.get("sam_score", None)
            label = f"{det['class']} {conf:.2f}"
            if sam_score is not None:
                label += f" [{sam_score:.2f}]"

            # Background for label
            text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font)

    return pil_img


_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _load_font(size: int = 14):
    """Load a font at the given size, cached after the first call."""
    if size not in _font_cache:
        for path in _FONT_PATHS:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """
    Convert binary mask to COCO polygon format.

    Args:
        mask: (H, W) bool or uint8 array

    Returns:
        list of polygons, each polygon is a flat list [x1, y1, x2, y2, ...]
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def mask_area(mask: np.ndarray) -> int:
    """Count non-zero pixels in a mask."""
    return int(np.count_nonzero(mask))


def bbox_norm_to_coco(bbox_norm: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert normalized [x1,y1,x2,y2] to COCO [x,y,w,h] in pixels."""
    x1 = bbox_norm[0] * img_w
    y1 = bbox_norm[1] * img_h
    x2 = bbox_norm[2] * img_w
    y2 = bbox_norm[3] * img_h
    return [x1, y1, x2 - x1, y2 - y1]


def get_image_paths(folder: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> list[Path]:
    """Recursively collect all image files from a folder."""
    folder = Path(folder)
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f"**/*{ext}"))
        paths.extend(folder.glob(f"**/*{ext.upper()}"))
    return sorted(set(paths))


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    Non-maximum suppression on detections from the same image.
    Removes duplicate detections by class and IoU overlap.
    """
    if len(detections) <= 1:
        return detections

    # Group by class
    by_class: dict[str, list] = {}
    for i, det in enumerate(detections):
        cls = det["class"]
        by_class.setdefault(cls, []).append((i, det))

    kept_indices = []
    for cls, items in by_class.items():
        # Sort by confidence descending
        items.sort(key=lambda x: x[1]["confidence"], reverse=True)
        suppressed = set()
        for i, (idx_i, det_i) in enumerate(items):
            if i in suppressed:
                continue
            kept_indices.append(idx_i)
            for j, (idx_j, det_j) in enumerate(items[i + 1:], start=i + 1):
                if j not in suppressed and _iou(det_i["bbox"], det_j["bbox"]) > iou_threshold:
                    suppressed.add(j)

    return [detections[i] for i in sorted(kept_indices)]


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two normalized bboxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
