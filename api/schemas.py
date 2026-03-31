"""Pydantic schemas for the FastAPI backend."""

from typing import Optional
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    class_name: str = Field(..., alias="class")
    bbox: list[float] = Field(..., description="Normalized [x1,y1,x2,y2] in [0,1]")
    confidence: float
    sam_score: float = 0.0
    has_mask: bool = False

    class Config:
        populate_by_name = True


class ProcessResponse(BaseModel):
    image_width: int
    image_height: int
    detections: list[DetectionResult]
    num_detections: int


class BatchRequest(BaseModel):
    folder_path: str
    classes: list[str] = Field(..., min_length=1)
    save_viz: bool = True
    save_coco: bool = True
    output_dir: Optional[str] = None


class BatchResponse(BaseModel):
    job_id: str
    status: str
    total_images: int
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "done", "failed"
    progress: float = 0.0  # 0.0-1.0
    processed: int = 0
    total: int = 0
    summary: Optional[dict] = None
    error: Optional[str] = None
