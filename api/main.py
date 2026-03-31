"""
FastAPI backend for the SAM3 segmentation pipeline.

Endpoints:
  POST /process       — single image (multipart upload + class list)
  POST /batch         — batch folder (JSON body)
  GET  /jobs/{job_id} — batch job status
  GET  /health        — health check
"""

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from src.pipeline import Pipeline
from src.batch_processor import BatchProcessor
from src.utils import visualize_results, load_config
from api.schemas import BatchRequest, BatchResponse, DetectionResult, JobStatusResponse, ProcessResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory job store (replace with Redis/DB for production)
_jobs: dict[str, dict] = {}
_pipeline: Optional[Pipeline] = None
_batch_processor: Optional[BatchProcessor] = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, unload on shutdown."""
    global _pipeline, _batch_processor, _config
    logger.info("Loading SAM3 models...")
    _config = load_config("config.yaml")
    _pipeline = Pipeline(_config)
    _pipeline.load_models()
    _batch_processor = BatchProcessor(_pipeline, _config)
    logger.info("SAM3 ready.")
    yield
    logger.info("Shutting down, unloading models...")
    if _pipeline:
        _pipeline.unload_models()


app = FastAPI(
    title="SAM3 Segmentation API",
    description="Detect and segment objects with SAM3 in a single pass",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _pipeline is not None}


@app.post("/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(..., description="Image file"),
    classes: str = Form(..., description="Comma-separated class names"),
    return_viz: bool = Form(False, description="Return visualization as response header path"),
):
    """
    Detect and segment objects in a single uploaded image.
    """
    if _pipeline is None:
        raise HTTPException(503, "Models not loaded")

    active_classes = [c.strip() for c in classes.split(",") if c.strip()]
    if not active_classes:
        raise HTTPException(400, "No valid classes provided")

    # Read image
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Run pipeline
    results = _pipeline.process_image(image, active_classes)

    detections = [
        DetectionResult(
            **{
                "class": r["class"],
                "bbox": r["bbox"],
                "confidence": r["confidence"],
                "sam_score": r.get("sam_score", 0.0),
                "has_mask": r.get("mask") is not None,
            }
        )
        for r in results
    ]

    return ProcessResponse(
        image_width=image.width,
        image_height=image.height,
        detections=detections,
        num_detections=len(detections),
    )


@app.post("/batch", response_model=BatchResponse)
async def start_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Start a batch processing job on a folder of images.
    Returns a job_id to poll for status.
    """
    if _batch_processor is None:
        raise HTTPException(503, "Models not loaded")

    folder = Path(request.folder_path)
    if not folder.exists():
        raise HTTPException(404, f"Folder not found: {folder}")

    from src.utils import get_image_paths
    image_paths = get_image_paths(str(folder))
    if not image_paths:
        raise HTTPException(400, f"No images found in {folder}")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "processed": 0,
        "total": len(image_paths),
        "summary": None,
        "error": None,
    }

    def run_job():
        _jobs[job_id]["status"] = "running"
        try:
            def progress_cb(current, total, msg):
                _jobs[job_id]["processed"] = current
                _jobs[job_id]["total"] = total
                _jobs[job_id]["progress"] = current / total if total > 0 else 0.0

            summary = _batch_processor.run(
                input_path=request.folder_path,
                active_classes=request.classes,
                output_dir=request.output_dir,
                save_viz=request.save_viz,
                save_coco=request.save_coco,
                progress_callback=progress_cb,
            )
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["summary"] = summary
            _jobs[job_id]["progress"] = 1.0
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            logger.error(f"Batch job {job_id} failed: {e}")

    background_tasks.add_task(run_job)

    return BatchResponse(
        job_id=job_id,
        status="pending",
        total_images=len(image_paths),
        message=f"Job {job_id} queued for {len(image_paths)} images",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    """Poll batch job status."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = _jobs[job_id]
    return JobStatusResponse(job_id=job_id, **job)


@app.get("/jobs/{job_id}/download")
def download_coco(job_id: str):
    """Download the COCO JSON for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job {job_id} not done (status: {job['status']})")
    coco_path = job.get("summary", {}).get("coco_path")
    if not coco_path or not Path(coco_path).exists():
        raise HTTPException(404, "COCO file not found")
    return FileResponse(coco_path, filename=f"annotations_{job_id}.json")


if __name__ == "__main__":
    import uvicorn
    cfg = load_config("config.yaml")
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,
    )
