"""
SAM3 segmentor wrapper.
Uses centre point + bounding box as dual prompt for best mask accuracy.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SAMSegmentor:
    """
    Wraps Sam3TrackerModel with a unified predict interface.

    Usage pattern (important for performance):
        segmentor.set_image(image)               # cache image for this batch
        masks = segmentor.predict_batch(bboxes)  # one forward pass for ALL instances
        segmentor.clear_image()                  # release cached image
    """

    def __init__(
        self,
        sam3_local_path: str = "sam3",
        device: str = "cuda",
        multimask_output: bool = False,
    ):
        self.sam3_local_path = sam3_local_path
        self.device = device
        self.multimask_output = multimask_output
        self.predictor = None
        self._processor = None
        self._image: Optional[Image.Image] = None
        self._image_set = False

    def load(self):
        """Load the SAM3 model. Called lazily on first use."""
        try:
            import torch
            from transformers import Sam3VideoModel, Sam3TrackerModel, Sam3TrackerProcessor
            from pathlib import Path

            path = self.sam3_local_path
            if not Path(path).is_absolute():
                path = str(Path(__file__).parent.parent.parent / path)
            logger.info(f"Loading SAM3 from: {path}")

            # The on-disk checkpoint is Sam3VideoModel. Load to CPU to extract weights,
            # then remap into Sam3TrackerModel — avoids a ~15 GB VRAM spike from having
            # both models on-device simultaneously.
            logger.info("Extracting weights from Sam3VideoModel checkpoint (CPU)...")
            video_model = Sam3VideoModel.from_pretrained(path, device_map="cpu")
            full_sd = video_model.state_dict()
            remapped = {}
            for k, v in full_sd.items():
                if k.startswith("tracker_model."):
                    remapped[k[len("tracker_model."):]] = v
                elif k.startswith("detector_model.vision_encoder."):
                    remapped[k[len("detector_model."):]] = v
            del video_model, full_sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            target_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
            logger.info(f"Loading remapped weights into Sam3TrackerModel ({target_dtype})...")
            self.predictor = Sam3TrackerModel.from_pretrained(
                path, ignore_mismatched_sizes=True, torch_dtype=target_dtype
            )
            self.predictor.load_state_dict(remapped, strict=False)
            del remapped
            self.predictor.to(self.device)
            self.predictor.eval()
            self._processor = Sam3TrackerProcessor.from_pretrained(path)
            logger.info("SAM3 loaded successfully")
        except ImportError:
            raise ImportError(
                "SAM3 requires transformers>=5. Run:\n"
                "  pip install --upgrade transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 from {self.sam3_local_path}: {e}") from e

    def set_image(self, image: np.ndarray):
        """
        Cache the image for this prediction batch.
        Call ONCE per image, then predict_batch() for all instances.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8
        """
        if self.predictor is None:
            self.load()
        self._image = Image.fromarray(image)
        self._image_set = True

    def predict_batch(
        self, boxes_px: list[list[float]], img_w: int, img_h: int
    ) -> list[Tuple[Optional[np.ndarray], float]]:
        """
        Predict masks for all instances in a SINGLE forward pass.

        Args:
            boxes_px: list of [x1, y1, x2, y2] bboxes (normalised [0,1] or pixel coords)
            img_w: image width (used to denormalise if coords are in [0,1])
            img_h: image height

        Returns:
            list of (mask, score) tuples — binary mask (H, W) bool and SAM quality score
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before predict_batch()")
        if not boxes_px:
            return []

        import torch

        # Denormalise and clamp all boxes
        pixel_boxes = []
        for bbox in boxes_px:
            x1, y1, x2, y2 = bbox
            if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                x1, x2 = x1 * img_w, x2 * img_w
                y1, y2 = y1 * img_h, y2 * img_h
            x1 = max(0.0, min(img_w - 1, x1))
            x2 = max(0.0, min(img_w - 1, x2))
            y1 = max(0.0, min(img_h - 1, y1))
            y2 = max(0.0, min(img_h - 1, y2))
            pixel_boxes.append([x1, y1, x2, y2])

        # SAM3 processor: input_boxes = [batch=1, num_objects=N, 4]
        # Single processor call + single forward pass for all N instances.
        input_boxes = [[b for b in pixel_boxes]]
        inputs = self._processor(
            images=self._image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.predictor(**inputs, multimask_output=self.multimask_output)

        # pred_masks: [batch, num_objects, num_masks, H, W] (logits)
        masks_tensor = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # [num_objects, num_masks, H, W]

        results = []
        for i in range(len(pixel_boxes)):
            mask_np = masks_tensor[i, 0].numpy().astype(bool)
            score = float(outputs.iou_scores[0, i, 0])
            results.append((mask_np, score))
        return results

    def predict(
        self, bbox_px: list[float], img_w: int, img_h: int
    ) -> Tuple[Optional[np.ndarray], float]:
        """Single-instance predict. Prefer predict_batch() when processing multiple instances."""
        results = self.predict_batch([bbox_px], img_w, img_h)
        if not results:
            return None, 0.0
        return results[0]

    def clear_image(self):
        """Release cached image."""
        self._image = None
        self._image_set = False

    def unload(self):
        """Free all GPU memory."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        self._image = None
        self._image_set = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("SAM segmentor unloaded")
