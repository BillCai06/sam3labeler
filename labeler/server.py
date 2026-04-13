"""
Labeler FastAPI backend.

Serves the WebUI and provides REST endpoints for reading/writing
per-image COCO annotations and running SAM on demand.

Config path is resolved from the LABELER_CONFIG env var (set by run_batch.py
before launching uvicorn so it survives the module reload).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Set by run_batch.py via env var before uvicorn import
_CONFIG_PATH = os.environ.get("LABELER_CONFIG", "config.yaml")

# Generic probe classes used when no text class is specified (class-free mode)
_GENERIC_PROBES = [
    "object", "thing", "item", "structure",
    "vehicle", "machine", "equipment", "device", "robot",
    "animal", "person", "plant", "debris",
]

# Lazy-loaded SAM detector (first /api/sam call loads it)
_sam_detector = None

STATIC_DIR = Path(__file__).parent / "static"

# ──────────────────────────────────────────────
# App & static files
# ──────────────────────────────────────────────

app = FastAPI(title="Labeler", docs_url=None, redoc_url=None)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ──────────────────────────────────────────────
# /api/config  — class list + colors
# ──────────────────────────────────────────────

@app.get("/api/config")
async def api_config():
    _ensure_src_path()
    from src.utils import load_config, get_class_color, CLASS_COLORS

    try:
        config = load_config(_CONFIG_PATH)
    except Exception:
        config = {}

    classes = config.get("classes", list(CLASS_COLORS.keys()))
    result = []
    for cls in classes:
        r, g, b = get_class_color(cls)
        result.append({
            "name": cls,
            "rgb": f"{r},{g},{b}",
            "hex": f"#{r:02x}{g:02x}{b:02x}",
        })
    return {"classes": result}


# ──────────────────────────────────────────────
# /api/datasets  — discover output dirs
# ──────────────────────────────────────────────

@app.get("/api/datasets")
async def api_datasets(base: str = "outputs"):
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    results = []
    if base_path.exists():
        for d in sorted(base_path.iterdir()):
            if d.is_dir() and (d / "annotations").is_dir():
                results.append({"path": str(d.resolve()), "name": d.name})
    return {"datasets": results}


# ──────────────────────────────────────────────
# /api/images  — list annotated images in dataset
# ──────────────────────────────────────────────

@app.get("/api/images")
async def api_images(dataset: str, image_dir: str = ""):
    ann_dir = Path(dataset) / "annotations"
    if not ann_dir.exists():
        raise HTTPException(404, detail="No annotations/ directory found in dataset")

    search_dirs: list[Path] = []
    if image_dir:
        search_dirs.append(Path(image_dir))
    search_dirs.append(Path(dataset).parent)   # co-located layout
    search_dirs.append(Path(dataset))          # in-place layout

    items = []
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            img_info = (data.get("images") or [{}])[0]
            file_name = img_info.get("file_name", "")
            img_path = _resolve_image(file_name, search_dirs)
            items.append({
                "filename": file_name,
                "jsonPath": str(jf.resolve()),
                "imagePath": img_path,
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
                "annCount": len(data.get("annotations", [])),
            })
        except Exception:
            continue

    return {"images": items, "count": len(items)}


def _resolve_image(file_name: str, search_dirs: list[Path]) -> Optional[str]:
    name = Path(file_name).name
    for d in search_dirs:
        for candidate in [d / file_name, d / name]:
            try:
                if candidate.exists():
                    return str(candidate.resolve())
            except Exception:
                continue
    return None


# ──────────────────────────────────────────────
# /api/image  — serve raw image bytes
# ──────────────────────────────────────────────

@app.get("/api/image")
async def api_image(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, detail=f"Image not found: {path}")
    ext = p.suffix.lower()
    media = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    return FileResponse(
        str(p), media_type=media,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ──────────────────────────────────────────────
# /api/annotations  GET + PUT
# ──────────────────────────────────────────────

@app.get("/api/annotations")
async def api_get_annotations(json_path: str):
    p = Path(json_path)
    if not p.exists():
        return JSONResponse({"images": [], "annotations": [], "categories": []})
    return JSONResponse(json.loads(p.read_text()))


class SaveBody(BaseModel):
    json_path: str
    data: dict


@app.put("/api/annotations")
async def api_save_annotations(body: SaveBody):
    p = Path(body.json_path)
    if not p.parent.exists():
        raise HTTPException(400, detail="Parent directory does not exist")
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(body.data, indent=2))
        tmp.replace(p)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, detail=str(e))
    return {"ok": True}


# ──────────────────────────────────────────────
# /api/sam  — run SAM3 on a bbox region
# ──────────────────────────────────────────────

class SamBody(BaseModel):
    image_path: str
    class_name: str
    bbox: list[float]          # [x1n, y1n, x2n, y2n] normalized 0-1
    sam_threshold: float = 0.40


@app.post("/api/sam")
async def api_sam(body: SamBody):
    det = await _ensure_sam()

    try:
        from PIL import Image as PILImage

        img_orig = PILImage.open(body.image_path).convert("RGB")
        W, H = img_orig.size

        # Crop to user's drawn bbox (+ 20% padding), then cap resolution
        crop, ox, oy = _crop_region(img_orig, body.bbox, padding=0.20)
        crop_sm, sx, sy = _downsample_for_sam(crop)
        logger.debug("SAM bbox: orig=%dx%d  crop=%dx%d  sam_in=%dx%d",
                     W, H, crop.width, crop.height, crop_sm.width, crop_sm.height)

        # Try specific class first; fall back to all classes + generic probes
        dets = det.detect_and_segment(
            image=crop_sm, classes=[body.class_name],
            confidence_threshold=body.sam_threshold,
        )
        if not dets:
            all_cls = _all_classes()
            for g in _GENERIC_PROBES:
                if g not in all_cls:
                    all_cls.append(g)
            if all_cls:
                dets = det.detect_and_segment(
                    image=crop_sm, classes=all_cls,
                    confidence_threshold=body.sam_threshold,
                )
        if not dets:
            return {"found": False, "message": "No detections found (tried all classes)"}

        # Highest confidence (crop is already the region of interest — no IoU needed)
        best = max(dets, key=lambda d: d["confidence"])
        polys, area = _extract_polys(best, crop_sm, ox, oy, sx, sy)

        return {
            "found": True,
            "segmentation": polys,
            "area": area,
            "confidence": float(best["confidence"]),
            "sam_score": float(best.get("sam_score", 0.0)),
            "detected_class": best["class"],
        }
    except Exception as e:
        logger.exception("SAM bbox inference failed")
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────
# /api/sam_point  — click-center pure segmentation
# ──────────────────────────────────────────────

class SamPointBody(BaseModel):
    image_path: str
    x_norm: float              # normalized click x (0-1)
    y_norm: float              # normalized click y (0-1)
    radius_norm: float = 0.12  # half-size of synthetic bbox (fraction of min dim)
    class_name: str = ""       # empty = try all classes
    sam_threshold: float = 0.40


@app.post("/api/sam_point")
async def api_sam_point(body: SamPointBody):
    det = await _ensure_sam()

    try:
        from PIL import Image as PILImage

        img_orig = PILImage.open(body.image_path).convert("RGB")
        W, H = img_orig.size

        # Crop to the radius circle's bounding box, then cap resolution
        r = body.radius_norm
        bbox_n = [
            max(0.0, body.x_norm - r), max(0.0, body.y_norm - r),
            min(1.0, body.x_norm + r), min(1.0, body.y_norm + r),
        ]
        crop, ox, oy = _crop_region(img_orig, bbox_n, padding=0.10)
        crop_sm, sx, sy = _downsample_for_sam(crop)
        logger.debug("SAM point: orig=%dx%d  crop=%dx%d  sam_in=%dx%d",
                     W, H, crop.width, crop.height, crop_sm.width, crop_sm.height)

        # Which classes to query
        if body.class_name:
            classes = [body.class_name]
        else:
            classes = _all_classes()
            for g in _GENERIC_PROBES:
                if g not in classes:
                    classes.append(g)

        dets = det.detect_and_segment(
            image=crop_sm, classes=classes,
            confidence_threshold=body.sam_threshold,
        )
        if not dets:
            return {"found": False, "message": "No detections found"}

        # Click point in crop_sm space: subtract crop offset, divide by scale
        xp = max(0, min(crop_sm.width  - 1, int((body.x_norm * W - ox) / sx)))
        yp = max(0, min(crop_sm.height - 1, int((body.y_norm * H - oy) / sy)))

        mask_hits = [d for d in dets if d.get("mask") is not None and d["mask"][yp, xp]]
        if mask_hits:
            # Most specific mask covering the click point
            best = min(mask_hits, key=lambda d: int(np.count_nonzero(d["mask"])))
        else:
            # Fallback: highest confidence in crop
            best = max(dets, key=lambda d: d["confidence"])

        polys, area = _extract_polys(best, crop_sm, ox, oy, sx, sy)

        return {
            "found": True,
            "segmentation": polys,
            "area": area,
            "confidence": float(best["confidence"]),
            "sam_score": float(best.get("sam_score", 0.0)),
            "detected_class": best["class"],
        }
    except Exception as e:
        logger.exception("SAM point inference failed")
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────
# /api/propagate  — re-detect prev-frame annotations in next frame
# ──────────────────────────────────────────────

class PropagateItem(BaseModel):
    class_name: str
    bbox: list[float]   # [x1n, y1n, x2n, y2n] normalized, from prev frame


class PropagateBody(BaseModel):
    image_path: str
    items: list[PropagateItem]
    sam_threshold: float = 0.20


@app.post("/api/propagate")
async def api_propagate(body: PropagateBody):
    det = await _ensure_sam()

    try:
        from PIL import Image as PILImage

        img = PILImage.open(body.image_path).convert("RGB")
        results = []

        for item in body.items:
            crop, ox, oy = _crop_region(img, item.bbox, padding=0.30)
            crop_sm, sx, sy = _downsample_for_sam(crop)

            dets = det.detect_and_segment(
                image=crop_sm,
                classes=[item.class_name],
                confidence_threshold=body.sam_threshold,
            )
            if not dets:
                all_cls = _all_classes()
                for g in _GENERIC_PROBES:
                    if g not in all_cls:
                        all_cls.append(g)
                dets = det.detect_and_segment(
                    image=crop_sm, classes=all_cls,
                    confidence_threshold=body.sam_threshold,
                )
            if not dets:
                continue

            best = max(dets, key=lambda d: d["confidence"])
            polys, area = _extract_polys(best, crop_sm, ox, oy, sx, sy)
            results.append({
                "class": item.class_name,
                "confidence": float(best["confidence"]),
                "segmentation": polys,
                "area": area,
            })

        return {"found": len(results) > 0, "annotations": results}
    except Exception as e:
        logger.exception("Propagate inference failed")
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _ensure_sam():
    """Lazy-load SAM3 detector; raise HTTP 500 on failure."""
    global _sam_detector
    if _sam_detector is None:
        _ensure_src_path()
        try:
            from src.utils import load_config
            from src.models.sam3_image_detector import Sam3ImageDetector
            config = load_config(_CONFIG_PATH)
            mc = config.get("models", {}).get("detector", {})
            _sam_detector = Sam3ImageDetector(
                sam3_local_path=mc.get("sam3_local_path", "sam3"),
                device=mc.get("device", "cuda"),
                score_threshold=mc.get("score_threshold", 0.05),
            )
            _sam_detector.load()
            logger.info("SAM3 loaded for labeler")
        except Exception as e:
            raise HTTPException(500, detail=f"SAM3 load failed: {e}")
    return _sam_detector


def _all_classes() -> list[str]:
    """Return full class list from config."""
    _ensure_src_path()
    try:
        from src.utils import load_config
        return load_config(_CONFIG_PATH).get("classes", [])
    except Exception:
        return []


# ── Crop-based SAM helpers ─────────────────────────────────────────────────
#
# Strategy: pass ONLY the user-selected region to SAM.
#
#   orig image (e.g. 3840×2160)
#     └─ crop to bbox + padding  →  small region (e.g. 400×300)
#          └─ downsample if still > _SAM_MAX_DIM  →  SAM input
#               └─ mask/polygon in SAM space
#                    └─ scale back  →  crop space
#                         └─ shift by crop offset  →  original image space
#
# This keeps VRAM proportional to the selected bbox, not the whole image.

_SAM_MAX_DIM = 1024   # cap on longest side fed to SAM vision encoder


def _crop_region(img, bbox_n: list, padding: float = 0.20):
    """
    Crop image to a normalized bbox with proportional padding.
    Returns (crop_img, ox, oy) — ox/oy are the pixel offset of the crop
    top-left corner in the original image.
    """
    from PIL import Image as PILImage
    W, H = img.size
    x1n, y1n, x2n, y2n = bbox_n
    pw = (x2n - x1n) * padding
    ph = (y2n - y1n) * padding
    x1 = max(0, int((x1n - pw) * W))
    y1 = max(0, int((y1n - ph) * H))
    x2 = min(W, int((x2n + pw) * W))
    y2 = min(H, int((y2n + ph) * H))
    return img.crop((x1, y1, x2, y2)), x1, y1


def _downsample_for_sam(img):
    """
    Resize so longest side ≤ _SAM_MAX_DIM.
    Returns (resized_img, sx, sy) where sx/sy convert small→crop pixel coords.
    """
    from PIL import Image as PILImage
    W, H = img.size
    scale = min(_SAM_MAX_DIM / W, _SAM_MAX_DIM / H, 1.0)
    if scale >= 1.0:
        return img, 1.0, 1.0
    nw, nh = int(W * scale), int(H * scale)
    return img.resize((nw, nh), PILImage.LANCZOS), W / nw, H / nh


def _extract_polys(det: dict, crop_sm, ox: int, oy: int, sx: float, sy: float):
    """
    Convert a detection's mask (in crop_sm space) to polygons in original image space.
    Falls back to crop_sm bbox rectangle if no mask.
    Returns (polys, area).
    """
    from src.utils import mask_to_polygon
    mask = det.get("mask")
    if mask is not None:
        polys = mask_to_polygon(mask)           # crop_sm pixel coords
        area  = int(np.count_nonzero(mask))
    else:
        # bbox is normalized inside crop_sm; convert to crop_sm pixels
        b = det["bbox"]
        cw, ch = crop_sm.size
        x1c, y1c = b[0]*cw, b[1]*ch
        x2c, y2c = b[2]*cw, b[3]*ch
        polys = [[x1c, y1c, x2c, y1c, x2c, y2c, x1c, y2c]]
        area  = int((x2c - x1c) * (y2c - y1c))

    # 1. Scale from crop_sm space → crop space
    if sx != 1.0 or sy != 1.0:
        polys = [
            [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(seg)]
            for seg in polys
        ]
        area = int(area * sx * sy)

    # 2. Shift from crop space → original image space
    polys = [
        [v + (ox if i % 2 == 0 else oy) for i, v in enumerate(seg)]
        for seg in polys
    ]
    return polys, area


def _ensure_src_path():
    """Add project root to sys.path so src.* imports work."""
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
