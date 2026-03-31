"""
Main detection + segmentation pipeline.
Orchestrates: Qwen VL detection → NMS → SAM segmentation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models.qwen_detector import QwenDetector
from .models.sam3_video_detector import Sam3VideoDetector
from .models.sam_segmentor import SAMSegmentor
from .utils import apply_nms, load_config

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end pipeline: image + active classes → segmented instances.

    Two detector backends are supported, selected via config:

      detector.backend: "qwen"       (default) Qwen VL → bbox → SAMSegmentor
      detector.backend: "sam3_video" SAM3VideoModel → bbox + mask in one pass

    Result format (list of dicts per instance):
    {
        "class":       str,
        "bbox":        [x1, y1, x2, y2],  # normalized [0,1]
        "confidence":  float,
        "mask":        np.ndarray | None,  # (H, W) bool
        "sam_score":   float,
    }
    """

    def __init__(self, config: dict):
        self.config = config
        det_cfg = config["models"]["detector"]
        seg_cfg = config["models"]["segmentor"]
        pipe_cfg = config.get("pipeline", {})

        self.confidence_threshold = pipe_cfg.get("confidence_threshold", 0.25)
        self.nms_iou_threshold = pipe_cfg.get("nms_iou_threshold", 0.5)
        self.use_box_prompt = pipe_cfg.get("use_box_prompt", True)
        self.sam_score_threshold = pipe_cfg.get("sam_score_threshold", 0.5)

        det_backend = det_cfg.get("backend", "qwen")
        self._det_backend = det_backend

        if det_backend == "sam3_video":
            self.detector = Sam3VideoDetector(
                sam3_local_path=det_cfg.get("sam3_local_path", "sam3"),
                device=det_cfg.get("device", "cuda"),
                score_threshold=det_cfg.get("score_threshold", 0.05),
                new_det_thresh=det_cfg.get("new_det_thresh", 0.05),
            )
            self.segmentor = None  # masks come from the detector
        else:
            self.detector = QwenDetector(
                model_id=det_cfg["model_id"],
                device=det_cfg.get("device", "cuda"),
                dtype=det_cfg.get("dtype", "bfloat16"),
                max_new_tokens=det_cfg.get("max_new_tokens", 2048),
                quantization=det_cfg.get("quantization"),
            )
            self.segmentor = SAMSegmentor(
                backend=seg_cfg.get("backend", "sam3"),
                sam3_local_path=seg_cfg.get("sam3_local_path", "sam3"),
                sam2_model_id=seg_cfg.get("sam2_model_id", "facebook/sam2-hiera-large"),
                sam1_checkpoint=seg_cfg.get("sam1_checkpoint", "checkpoints/sam_vit_h.pth"),
                sam1_model_type=seg_cfg.get("sam1_model_type", "vit_h"),
                device=seg_cfg.get("device", "cuda"),
                multimask_output=seg_cfg.get("multimask_output", False),
            )

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> "Pipeline":
        config = load_config(config_path)
        return cls(config)

    def load_models(self):
        """Explicitly load all models (otherwise lazy-loaded on first call)."""
        self.detector.load()
        if self.segmentor is not None:
            self.segmentor.load()

    def process_image(
        self,
        image: Image.Image,
        active_classes: list[str],
    ) -> list[dict]:
        """
        Run the full pipeline on one image.

        Args:
            image: PIL RGB image
            active_classes: subset of class names to detect for this image

        Returns:
            list of instance dicts (see class docstring)
        """
        img_np = np.array(image.convert("RGB"))
        img_h, img_w = img_np.shape[:2]
        logger.info(f"Processing {img_w}x{img_h} image, classes={active_classes}")

        if self._det_backend == "sam3_video":
            return self._process_sam3_video(image, active_classes)
        else:
            return self._process_qwen_sam(img_np, image, active_classes, img_w, img_h)

    def _process_sam3_video(
        self,
        image: Image.Image,
        active_classes: list[str],
    ) -> list[dict]:
        """Detect + segment in one pass with Sam3VideoDetector."""
        raw = self.detector.detect_and_segment(
            image, active_classes, confidence_threshold=self.confidence_threshold
        )

        # NMS on the boxes that came back
        results = apply_nms(raw, self.nms_iou_threshold)
        logger.info(f"After NMS: {len(results)} instances")

        # SAM-score filter (reuse confidence as proxy since it's the same model)
        if self.sam_score_threshold > 0:
            before = len(results)
            results = [r for r in results if r["sam_score"] >= self.sam_score_threshold]
            logger.info(f"After score filter: {len(results)}/{before} instances retained")

        return results

    def _process_qwen_sam(
        self,
        img_np: np.ndarray,
        image: Image.Image,
        active_classes: list[str],
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        """Original Qwen VL → NMS → SAM segmentation path."""
        # === Step 1: Detect with Qwen VL ===
        raw_detections = self.detector.detect(image, active_classes)

        # === Step 2: Filter by confidence ===
        detections = [
            d for d in raw_detections
            if d["confidence"] >= self.confidence_threshold
        ]
        logger.info(f"After confidence filter: {len(detections)}/{len(raw_detections)} detections")

        # === Step 3: NMS per class ===
        detections = apply_nms(detections, self.nms_iou_threshold)
        logger.info(f"After NMS: {len(detections)} detections")

        if not detections:
            return []

        # === Step 4: SAM segmentation ===
        self.segmentor.set_image(img_np)

        results = []
        for det in detections:
            mask, sam_score = self.segmentor.predict(
                bbox_px=det["bbox"],
                img_w=img_w,
                img_h=img_h,
            )

            if mask is None:
                logger.warning(f"SAM failed for {det['class']}, using bbox-only result")

            results.append({
                "class": det["class"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "mask": mask,
                "sam_score": sam_score,
            })

        self.segmentor.clear_image()

        if self.sam_score_threshold > 0:
            before = len(results)
            results = [
                r for r in results
                if r["mask"] is None or r["sam_score"] >= self.sam_score_threshold
            ]
            logger.info(f"After SAM score filter: {len(results)}/{before} instances retained")

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
        """Free GPU memory for all models."""
        self.detector.unload()
        if self.segmentor is not None:
            self.segmentor.unload()
