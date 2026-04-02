"""
SAM3 image-mode detector backend.

Uses Sam3Model directly (no video InferenceSession / tracker overhead).
Vision embeddings are computed once per image and reused across all classes.
Text embeddings are cached across images.

Same result format as Sam3VideoDetector:
    [{"class", "bbox", "confidence", "mask", "sam_score"}, ...]
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class Sam3ImageDetector:
    """
    Detect and segment objects using Sam3Model (image backend).

    Faster than the video backend because it skips the SAM3 tracker
    (propagation, planning, execution) that is only useful for video.
    """

    def __init__(
        self,
        sam3_local_path: str = "sam3",
        device: str = "cuda",
        score_threshold: float = 0.05,
    ):
        self.sam3_local_path = sam3_local_path
        self.device = device
        self.score_threshold = score_threshold

        self.model = None
        self.processor = None
        self._text_embed_cache: dict[str, torch.Tensor] = {}  # class → pooled text embed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self):
        from transformers import Sam3Model, Sam3Processor

        path = self.sam3_local_path
        if not Path(path).is_absolute():
            path = str(Path(__file__).parent.parent.parent / path)

        logger.info(f"Loading Sam3Model (image backend) from: {path}")
        load_kwargs = {}
        if "cuda" in self.device:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Sam3Model.from_pretrained(path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.processor = Sam3Processor.from_pretrained(path)
        logger.info("Sam3ImageDetector loaded successfully")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._text_embed_cache.clear()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Sam3ImageDetector unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        image: Image.Image,
        classes: list[str],
        confidence_threshold: float = 0.25,
    ) -> list[dict]:
        if self.model is None:
            self.load()

        W, H = image.size
        dtype = next(self.model.parameters()).dtype

        # Vision encoding — once per image
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs["pixel_values"].to(dtype)
            )

        results = []

        for cls in classes:
            # Text embedding — cached across images
            if cls not in self._text_embed_cache:
                tokens = self.processor.tokenizer(
                    cls, return_tensors="pt", padding="max_length", max_length=32
                ).to(self.device)
                with torch.inference_mode():
                    text_embeds = self.model.get_text_features(
                        input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        return_dict=True,
                    ).pooler_output  # (1, hidden_dim)
                self._text_embed_cache[cls] = text_embeds
            text_embeds = self._text_embed_cache[cls]

            with torch.inference_mode():
                out = self.model(
                    vision_embeds=vision_embeds,
                    text_embeds=text_embeds,
                )

            # pred_logits: (batch, num_queries) or (batch, num_queries, 1)
            # presence_logits: (batch, 1)
            logits = out.pred_logits.sigmoid()
            if logits.dim() == 3:
                logits = logits[..., 0]  # (batch, num_queries)
            scores = (logits * out.presence_logits.sigmoid())[0]  # (num_queries,)

            above = (scores >= confidence_threshold).nonzero(as_tuple=False).squeeze(-1)
            if above.numel() == 0:
                continue

            for qi in above.tolist():
                score = float(scores[qi])
                raw_mask = out.pred_masks[0, qi]   # (H_low, W_low)
                bbox_xyxy = out.pred_boxes[0, qi].tolist()  # [x1,y1,x2,y2] normalised

                mask_np = self._upscale_mask(raw_mask, H, W)

                results.append({
                    "class": cls,
                    "bbox": bbox_xyxy,
                    "confidence": score,
                    "mask": mask_np,
                    "sam_score": score,
                })

        logger.info(f"Sam3ImageDetector: {len(results)} detections above threshold")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _upscale_mask(self, raw_mask: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
        """Bilinear upscale low-res mask logits to original resolution on GPU."""
        mask_up = F.interpolate(
            raw_mask.float().unsqueeze(0).unsqueeze(0),  # (1,1,H_low,W_low)
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return (mask_up > 0).cpu().numpy().astype(bool)
