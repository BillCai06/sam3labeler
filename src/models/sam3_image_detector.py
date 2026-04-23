"""
SAM3 image-mode detector backend.

Uses Sam3Model directly (no video InferenceSession / tracker overhead).

Key optimisations:
  - Vision encoder runs once per image (or once per batch of images)
  - Text embeddings are cached across images
  - detect_and_segment_batch() processes N images in a single GPU forward pass
    per class, exploiting spare VRAM (e.g. 93 GB free on RTX PRO 6000 Black)

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

    Faster than the video backend because it skips the SAM3 tracker.
    Supports batched inference across multiple images simultaneously.
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
        self._text_embed_cache: dict[str, torch.Tensor] = {}  # class → (1, hidden_dim)

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
    # Single-image inference (used by GUI single-image tab)
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        image: Image.Image,
        classes: list[str],
        confidence_threshold: float = 0.25,
    ) -> list[dict]:
        return self.detect_and_segment_batch([image], classes, confidence_threshold)[0]

    # ------------------------------------------------------------------
    # Batched inference (N images, all classes, one forward pass per class)
    # ------------------------------------------------------------------

    def detect_and_segment_batch(
        self,
        images: list[Image.Image],
        classes: list[str],
        confidence_threshold: float = 0.25,
        return_raw_scores: bool = False,
        scores_only: bool = False,
    ) -> "list[list[dict]] | tuple[list[list[dict]], list[dict[str, float]]]":
        """
        Process N images simultaneously.

        Args:
            images: list of PIL RGB images (can be different sizes).
            classes: class names to detect.
            confidence_threshold: keep detections at or above this score.
            return_raw_scores: also return per-class max scores (pre-threshold).
            scores_only: skip mask/bbox extraction entirely — only compute raw
                         per-class max scores. Implies return_raw_scores=True.
                         Use this for confidence-score backfilling (rescore tool).

        Returns:
            list of length N; each element is the result list for that image.
            If return_raw_scores or scores_only: returns (results, raw_scores).
        """
        if scores_only:
            return_raw_scores = True
        if self.model is None:
            self.load()

        N = len(images)
        sizes = [(img.size[1], img.size[0]) for img in images]  # (H, W) per image
        dtype = next(self.model.parameters()).dtype

        # Batch vision encoding — one forward pass for all N images
        img_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs["pixel_values"].to(dtype)
            )

        # Ensure text cache is warm for all classes before the main loop
        self._cache_text_embeds(classes)

        batch_results: list[list[dict]] = [[] for _ in range(N)]
        # raw_scores[i][cls] = max query score for that class in image i (pre-threshold)
        raw_scores: list[dict[str, float]] = [{} for _ in range(N)]

        for cls in classes:
            text_embeds = self._text_embed_cache[cls].expand(N, -1, -1)

            with torch.inference_mode():
                out = self.model(
                    vision_embeds=vision_embeds,
                    text_embeds=text_embeds,
                )

            # pred_logits: (N, num_queries) or (N, num_queries, 1)
            # presence_logits: (N, 1)
            logits = out.pred_logits.sigmoid()
            if logits.dim() == 3:
                logits = logits[..., 0]                      # (N, num_queries)
            scores = logits * out.presence_logits.sigmoid()  # (N, num_queries)

            for i in range(N):
                H, W = sizes[i]
                img_scores = scores[i]                       # (num_queries,)
                raw_scores[i][cls] = float(img_scores.max())

                if scores_only:
                    continue  # skip mask/bbox — caller only needs raw scores

                above = (img_scores >= confidence_threshold).nonzero(as_tuple=False).view(-1)
                for qi in above.tolist():
                    score = float(img_scores[qi])
                    mask_np = self._upscale_mask(out.pred_masks[i, qi], H, W)
                    bbox = out.pred_boxes[i, qi].tolist()
                    batch_results[i].append({
                        "class": cls,
                        "bbox": bbox,
                        "confidence": score,
                        "mask": mask_np,
                        "sam_score": score,
                    })

        if not scores_only:
            total = sum(len(r) for r in batch_results)
            logger.info(f"Sam3ImageDetector batch={N}: {total} detections above threshold")
        if return_raw_scores:
            return batch_results, raw_scores
        return batch_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_text_embeds(self, classes: list[str]):
        """Tokenize and encode any classes not yet in the cache."""
        missing = [cls for cls in classes if cls not in self._text_embed_cache]
        if not missing:
            return
        for cls in missing:
            tokens = self.processor.tokenizer(
                cls, return_tensors="pt", padding="max_length", max_length=32
            ).to(self.device)
            with torch.inference_mode():
                embed = self.model.get_text_features(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    return_dict=True,
                ).pooler_output  # (1, seq_len, 256) — projected, matches k_proj input dim
            self._text_embed_cache[cls] = embed

    def _upscale_mask(self, raw_mask: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
        """Bilinear upscale low-res mask logits to original resolution on GPU."""
        mask_up = F.interpolate(
            raw_mask.float().unsqueeze(0).unsqueeze(0),  # (1, 1, H_low, W_low)
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return (mask_up > 0).cpu().numpy().astype(bool)
