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

    try:
        from PIL import Image as PILImage
        from src.utils import mask_to_polygon

        img = PILImage.open(body.image_path).convert("RGB")
        W, H = img.size

        dets = _sam_detector.detect_and_segment(
            image=img,
            classes=[body.class_name],
            confidence_threshold=body.sam_threshold,
        )
        if not dets:
            return {"found": False, "message": "No detections above threshold"}

        # Pick detection with highest IoU vs user-drawn bbox
        best = max(dets, key=lambda d: _iou(d["bbox"], body.bbox))
        mask = best.get("mask")

        if mask is not None:
            polys = mask_to_polygon(mask)
            area = int(np.count_nonzero(mask))
        else:
            x1, y1 = body.bbox[0] * W, body.bbox[1] * H
            x2, y2 = body.bbox[2] * W, body.bbox[3] * H
            polys = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            area = int((x2 - x1) * (y2 - y1))

        return {
            "found": True,
            "segmentation": polys,
            "area": area,
            "confidence": float(best["confidence"]),
            "sam_score": float(best.get("sam_score", 0.0)),
        }
    except Exception as e:
        logger.exception("SAM inference failed")
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _ensure_src_path():
    """Add project root to sys.path so src.* imports work."""
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
