"""
SAM3 detection + segmentation pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models.sam3_image_detector import Sam3ImageDetector
from .models.sam3_video_detector import Sam3VideoDetector
from .utils import apply_nms, merge_masks_by_class, load_config

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end pipeline: image + active classes → segmented instances.

    Uses Sam3VideoDetector which detects and segments in a single forward pass
    (CLIP+DETR detection + SAM3 mask prediction).

    Result format (list of dicts per instance):
    {
        "class":       str,
        "bbox":        [x1, y1, x2, y2],  # normalised [0, 1]
        "confidence":  float,
        "mask":        np.ndarray | None,  # (H, W) bool
        "sam_score":   float,
    }
    """

    def __init__(self, config: dict):
        self.config = config
        det_cfg = config["models"]["detector"]
        pipe_cfg = config.get("pipeline", {})

        self.confidence_threshold = pipe_cfg.get("confidence_threshold", 0.03)
        self.nms_iou_threshold = pipe_cfg.get("nms_iou_threshold", 0.5)
        self.sam_score_threshold = pipe_cfg.get("sam_score_threshold", 0.0)
        self.merge_same_class = pipe_cfg.get("merge_same_class", False)

        backend = det_cfg.get("backend", "video")
        if backend == "image":
            self.detector = Sam3ImageDetector(
                sam3_local_path=det_cfg.get("sam3_local_path", "sam3"),
                device=det_cfg.get("device", "cuda"),
                score_threshold=det_cfg.get("score_threshold", 0.05),
            )
        else:
            self.detector = Sam3VideoDetector(
                sam3_local_path=det_cfg.get("sam3_local_path", "sam3"),
                device=det_cfg.get("device", "cuda"),
                score_threshold=det_cfg.get("score_threshold", 0.05),
                new_det_thresh=det_cfg.get("new_det_thresh", 0.05),
            )
        self.backend = backend

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> "Pipeline":
        config = load_config(config_path)
        return cls(config)

    def load_models(self):
        """Explicitly load models (otherwise lazy-loaded on first call)."""
        self.detector.load()

    def process_image(
        self,
        image: Image.Image,
        active_classes: list[str],
    ) -> list[dict]:
        """
        Run the full pipeline on one image.

        Args:
            image: PIL RGB image
            active_classes: class names to detect

        Returns:
            list of instance dicts (see class docstring)
        """
        img_np = np.array(image.convert("RGB"))
        img_h, img_w = img_np.shape[:2]
        logger.info(f"Processing {img_w}x{img_h} image, classes={active_classes}")

        raw = self.detector.detect_and_segment(
            image, active_classes, confidence_threshold=self.confidence_threshold
        )

        results = apply_nms(raw, self.nms_iou_threshold)
        logger.info(f"After NMS: {len(results)} instances")

        if self.merge_same_class:
            before = len(results)
            results = merge_masks_by_class(results)
            logger.info(f"After class merge: {len(results)}/{before} instances")

        if self.sam_score_threshold > 0:
            before = len(results)
            results = [r for r in results if r["sam_score"] >= self.sam_score_threshold]
            logger.info(f"After score filter: {len(results)}/{before} instances retained")

        return results

    def process_image_path(
        self,
        image_path: str | Path,
        active_classes: list[str],
    ) -> tuple[Image.Image, list[dict]]:
        """Convenience wrapper: load image from path, run pipeline, return (image, results)."""
        image = Image.open(image_path).convert("RGB")
        results = self.process_image(image, active_classes)
        return image, results

    def unload_models(self):
        """Free GPU memory."""
        self.detector.unload()
