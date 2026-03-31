"""
SAM segmentor wrapper — supports SAM2 and SAM1.
Uses center point + bounding box as dual prompt for best mask accuracy.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SAMSegmentor:
    """
    Wraps SAM2 (preferred) or SAM1 with a unified interface.

    Usage pattern (important for performance):
        segmentor.set_image(image)          # compute embeddings ONCE per image
        for bbox in detections:
            mask, score = segmentor.predict(bbox)   # cheap per-instance
        segmentor.clear_image()             # release embedding memory
    """

    def __init__(
        self,
        backend: str = "sam3",
        sam2_model_id: str = "facebook/sam2-hiera-large",
        sam1_checkpoint: str = "checkpoints/sam_vit_h.pth",
        sam1_model_type: str = "vit_h",
        sam3_local_path: str = "sam3",
        device: str = "cuda",
        multimask_output: bool = False,
    ):
        self.backend = backend
        self.sam2_model_id = sam2_model_id
        self.sam1_checkpoint = sam1_checkpoint
        self.sam1_model_type = sam1_model_type
        self.sam3_local_path = sam3_local_path
        self.device = device
        self.multimask_output = multimask_output
        self.predictor = None
        self._image_set = False
        self._sam3_processor = None
        self._sam3_image = None  # PIL Image stored for SAM3 per-call inference

    def load(self):
        """Load the SAM model. Called lazily on first use."""
        if self.backend == "sam3":
            self._load_sam3()
        elif self.backend == "sam2":
            self._load_sam2()
        else:
            self._load_sam1()

    def _load_sam3(self):
        try:
            import torch
            from transformers import Sam3VideoModel, Sam3TrackerModel, Sam3TrackerProcessor
            from pathlib import Path
            # Resolve relative path against project root
            path = self.sam3_local_path
            if not Path(path).is_absolute():
                path = str(Path(__file__).parent.parent.parent / path)
            logger.info(f"Loading SAM3 from: {path}")

            # The on-disk checkpoint is Sam3VideoModel (model_type=sam3_video), so its
            # state dict has keys prefixed with "tracker_model." and
            # "detector_model.vision_encoder.".  We load it first to get all
            # weights, then remap them into Sam3TrackerModel which exposes the
            # simpler per-image forward API we need.
            #
            # Load to CPU only — avoids a ~15 GB VRAM spike from having both the
            # full video model and the tracker model on-device simultaneously.
            logger.info("Extracting weights from Sam3VideoModel checkpoint (CPU)...")
            video_model = Sam3VideoModel.from_pretrained(path, device_map="cpu")
            full_sd = video_model.state_dict()
            remapped = {}
            for k, v in full_sd.items():
                if k.startswith("tracker_model."):
                    remapped[k[len("tracker_model."):]] = v
                elif k.startswith("detector_model.vision_encoder."):
                    remapped[k[len("detector_model."):]] = v
            del video_model
            del full_sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load directly in bfloat16 to halve peak VRAM during init
            target_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
            logger.info(f"Loading remapped weights into Sam3TrackerModel ({target_dtype})...")
            self.predictor = Sam3TrackerModel.from_pretrained(
                path, ignore_mismatched_sizes=True, torch_dtype=target_dtype
            )
            self.predictor.load_state_dict(remapped, strict=False)
            del remapped
            self.predictor.to(self.device)
            self.predictor.eval()
            self._sam3_processor = Sam3TrackerProcessor.from_pretrained(path)
            logger.info("SAM3 loaded successfully")
        except ImportError:
            raise ImportError(
                "SAM3 requires transformers>=5. Run:\n"
                "  pip install --upgrade transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 from {self.sam3_local_path}: {e}") from e

    def _load_sam2(self):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            logger.info(f"Loading SAM2: {self.sam2_model_id}")
            self.predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_id)
            logger.info("SAM2 loaded successfully")
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 {self.sam2_model_id}: {e}") from e

    def _load_sam1(self):
        try:
            from segment_anything import SamPredictor, sam_model_registry
            logger.info(f"Loading SAM1: {self.sam1_model_type} @ {self.sam1_checkpoint}")
            sam = sam_model_registry[self.sam1_model_type](checkpoint=self.sam1_checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            logger.info("SAM1 loaded successfully")
        except ImportError:
            raise ImportError(
                "segment-anything not installed. Run:\n"
                "  pip install segment-anything"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SAM1 checkpoint not found: {self.sam1_checkpoint}\n"
                "Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )

    def set_image(self, image: np.ndarray):
        """
        Compute image embeddings. Call ONCE per image, then predict() for each instance.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8
        """
        if self.predictor is None:
            self.load()

        if self.backend == "sam3":
            # SAM3 has no separate embedding step; cache PIL image for per-call inference
            self._sam3_image = Image.fromarray(image)
        elif self.backend == "sam2":
            import torch
            with torch.inference_mode():
                self.predictor.set_image(image)
        else:
            self.predictor.set_image(image)

        self._image_set = True

    def predict(
        self, bbox_px: list[float], img_w: int, img_h: int
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Predict segmentation mask for one instance using dual prompt: center point + bbox.

        Args:
            bbox_px: [x1, y1, x2, y2] in pixel coordinates (or normalized [0,1] if img_w given)
            img_w: image width (used to denormalize if bbox_px is normalized)
            img_h: image height

        Returns:
            (mask, score): binary mask (H, W) bool and SAM quality score [0,1]
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before predict()")

        x1, y1, x2, y2 = bbox_px

        # Denormalize if coordinates are in [0,1] range
        if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h

        # Clamp to image bounds
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))

        # Center point of the bounding box (foreground prompt)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        try:
            if self.backend == "sam3":
                masks, scores, _ = self._predict_sam3(box)
            elif self.backend == "sam2":
                masks, scores, _ = self._predict_sam2(point_coords, point_labels, box)
            else:
                masks, scores, _ = self._predict_sam1(point_coords, point_labels, box)

            best_mask = masks[0]
            best_score = float(scores[0])
            return best_mask, best_score

        except Exception as e:
            logger.warning(f"SAM prediction failed for bbox {bbox_px}: {e}")
            return None, 0.0

    def _predict_sam3(self, box):
        import torch
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        input_boxes = [[[x1, y1, x2, y2]]]  # [batch=1, num_objects=1, 4]
        inputs = self._sam3_processor(
            images=self._sam3_image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.predictor(**inputs, multimask_output=False)
        # pred_masks: [batch, num_objects, num_masks, H, W] (logits)
        # post_process_masks resizes to original image size and binarizes
        masks_tensor = self._sam3_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # [num_objects, num_masks, H, W]
        mask_np = masks_tensor[0, 0].numpy().astype(bool)  # (H, W)
        score = float(outputs.iou_scores[0, 0, 0])
        return np.array([mask_np]), np.array([score]), None

    def _predict_sam2(self, point_coords, point_labels, box):
        import torch
        with torch.inference_mode():
            # SAM2 expects batched box: shape (1, 4)
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box[None, :],  # add batch dim
                multimask_output=self.multimask_output,
            )
        return masks, scores, logits

    def _predict_sam1(self, point_coords, point_labels, box):
        # SAM1 expects box shape (4,) — no batch dim
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=self.multimask_output,
        )
        return masks, scores, logits

    def clear_image(self):
        """Release image embedding from GPU memory."""
        if self._image_set and self.predictor is not None:
            if self.backend == "sam3":
                self._sam3_image = None
            elif self.backend == "sam2":
                self.predictor.reset_predictor()
            # SAM1 doesn't have an explicit clear — just flag as unset
            self._image_set = False

    def unload(self):
        """Free all GPU memory."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        self._image_set = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("SAM segmentor unloaded")
