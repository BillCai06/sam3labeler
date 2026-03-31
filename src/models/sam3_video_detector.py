"""
SAM3 combined detect+segment backend.

Uses Sam3VideoModel to detect objects by text class name AND produce
segmentation masks in a single forward pass — no Qwen VL or separate
SAMSegmentor needed.

Detection is powered by SAM3's built-in CLIP text encoder + DETR detector.
Segmentation is the SAM3 tracker head.  Masks are returned at original
image resolution, ready to drop into the standard pipeline result format.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Sam3VideoDetector:
    """
    Detect and segment objects in a single image using SAM3VideoModel.

    Interface mirrors the combined detect+segment result format:
        [
          {
            "class":      str,
            "bbox":       [x1, y1, x2, y2],  # normalized [0, 1]
            "confidence": float,
            "mask":       np.ndarray | None,  # (H, W) bool, original resolution
            "sam_score":  float,
          },
          ...
        ]
    """

    def __init__(
        self,
        sam3_local_path: str = "sam3",
        device: str = "cuda",
        score_threshold: float = 0.05,
        new_det_thresh: float = 0.05,
    ):
        self.sam3_local_path = sam3_local_path
        self.device = device
        # Keep detection thresholds low; the pipeline's confidence_threshold
        # does the final filtering so we don't discard real detections early.
        self.score_threshold = score_threshold
        self.new_det_thresh = new_det_thresh

        self.model = None
        self.processor = None
        self._tracker_processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self):
        """Load Sam3VideoModel.  Called lazily on first use."""
        import torch
        from transformers import Sam3VideoModel, Sam3Processor, Sam3TrackerProcessor

        path = self.sam3_local_path
        if not Path(path).is_absolute():
            path = str(Path(__file__).parent.parent.parent / path)

        logger.info(f"Loading Sam3VideoModel from: {path}")
        load_kwargs = {}
        if "cuda" in self.device:
            load_kwargs["torch_dtype"] = torch.bfloat16
        self.model = Sam3VideoModel.from_pretrained(path, **load_kwargs)
        # Patch detection thresholds before inference
        self.model.score_threshold_detection = self.score_threshold
        self.model.new_det_thresh = self.new_det_thresh

        self.model.to(self.device)
        self.model.eval()

        self.processor = Sam3Processor.from_pretrained(path)
        self._tracker_processor = Sam3TrackerProcessor.from_pretrained(path)
        logger.info("Sam3VideoDetector loaded successfully")

    def unload(self):
        """Release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Sam3VideoDetector unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        image: Image.Image,
        classes: list[str],
        confidence_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Detect and segment all instances of *classes* in *image*.

        Args:
            image: PIL RGB image.
            classes: List of text class names to look for.
            confidence_threshold: Keep detections at or above this score.

        Returns:
            List of result dicts (class, bbox, confidence, mask, sam_score).
        """
        import torch
        from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession

        if self.model is None:
            self.load()

        W, H = image.size
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = img_inputs["pixel_values"]

        dtype = next(self.model.parameters()).dtype

        session = Sam3VideoInferenceSession(
            video=pixel_values.to(dtype),
            video_height=H,
            video_width=W,
            dtype=dtype,
            inference_device=self.device,
            inference_state_device=self.device,
        )

        # Tokenize and register each class as a text prompt
        for cls in classes:
            prompt_id = session.add_prompt(cls)
            tokens = self.processor.tokenizer(
                cls, return_tensors="pt", padding="max_length", max_length=32
            )
            session.prompt_input_ids[prompt_id] = tokens.input_ids.to(self.device)
            session.prompt_attention_masks[prompt_id] = tokens.attention_mask.to(self.device)

        with torch.inference_mode():
            out = self.model(inference_session=session, frame_idx=0)

        results = []
        for obj_id in out.object_ids:
            score = float(out.obj_id_to_score.get(obj_id, 0.0))
            if score < confidence_threshold:
                continue

            prompt_id = session.obj_id_to_prompt_id.get(obj_id)
            class_name = session.prompts.get(prompt_id, "unknown")
            raw_mask = out.obj_id_to_mask.get(obj_id)  # [1, 288, 288] logits

            mask_np, bbox = self._process_mask(raw_mask, H, W)

            results.append({
                "class": class_name,
                "bbox": bbox,
                "confidence": score,
                "mask": mask_np,
                "sam_score": score,
            })

        logger.info(f"Sam3VideoDetector: {len(results)} detections above threshold")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_mask(
        self,
        raw_mask,  # torch.Tensor [1, 288, 288] logits, or None
        img_h: int,
        img_w: int,
    ):
        """Upscale mask logits to original resolution and derive tight bbox."""
        if raw_mask is None:
            return None, [0.0, 0.0, 1.0, 1.0]

        # post_process_masks expects [batch, num_objects, num_masks, H, W]
        masks_up = self._tracker_processor.post_process_masks(
            raw_mask.unsqueeze(0).unsqueeze(0).cpu(),  # [1,1,1,288,288]
            original_sizes=[[img_h, img_w]],
            binarize=True,
        )
        mask_np = masks_up[0][0, 0].numpy().astype(bool)  # (H, W)

        # Derive tight bounding box from mask pixels
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if rows.any() and cols.any():
            y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
            bbox = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
        else:
            bbox = [0.0, 0.0, 1.0, 1.0]

        return mask_np, bbox
