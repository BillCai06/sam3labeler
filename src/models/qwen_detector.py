"""
Qwen VL wrapper for bounding box detection.
Supports Qwen2.5-VL and Qwen3-VL via transformers.
"""

import json
import logging
import re
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class QwenDetector:
    """
    Detects objects from a provided class list using Qwen VL.
    Returns normalized bounding boxes [x1, y1, x2, y2] in [0, 1] range.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        quantization: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor. Called lazily on first detect() call."""
        from transformers import AutoProcessor

        logger.info(f"Loading detector: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        model_kwargs = dict(
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        # Try Qwen2.5-VL specific class first (works reliably), then generic
        self.model = self._load_model(model_kwargs)
        self.model.eval()
        logger.info("Detector loaded successfully")

    def _load_model(self, kwargs: dict):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            logger.info("Using Qwen2_5_VLForConditionalGeneration")
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        except (ImportError, Exception) as e:
            logger.warning(f"Qwen2.5-VL class failed ({e}), trying AutoModelForCausalLM")

        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Could not load model {self.model_id}: {e}") from e

    def detect(self, image: Image.Image, classes: list[str]) -> list[dict]:
        """
        Detect objects from the given class list in the image.

        Args:
            image: PIL RGB image
            classes: list of class names to detect (subset of your full class list)

        Returns:
            list of dicts: [{"class": str, "bbox": [x1,y1,x2,y2], "confidence": float}]
            bbox coordinates are normalized [0, 1]
        """
        if self.model is None:
            self.load()

        prompt = self._build_prompt(classes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._prepare_inputs(messages)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy decoding for reliable JSON
            )

        # Decode only newly generated tokens
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        logger.debug(f"Qwen raw response: {response[:500]}")
        detections = self._parse_response(response, classes)
        logger.info(f"Detected {len(detections)} objects from classes {classes}")
        return detections

    def _prepare_inputs(self, messages: list) -> dict:
        """Prepare model inputs, with qwen_vl_utils if available."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except ImportError:
            # Fall back to basic processor (works for most use cases)
            image = messages[0]["content"][0]["image"]
            return self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)

    def _build_prompt(self, classes: list[str]) -> str:
        class_list = ", ".join(f'"{c}"' for c in classes)
        return f"""Detect all visible instances of these object classes in the image: {class_list}.

Respond with ONLY a valid JSON array. No explanation, no markdown, no code fences.
Each detected object:
{{"class": "<class_name>", "bbox": [x1, y1, x2, y2], "confidence": <0.0-1.0>}}

Rules:
- bbox coordinates are normalized floats [0.0, 1.0] relative to image size
- x1,y1 = top-left corner; x2,y2 = bottom-right corner
- Only use class names from the provided list
- Include every visible instance, even partially occluded ones
- If nothing from the list is visible, respond exactly: []

Response:"""

    def _parse_response(self, text: str, valid_classes: list[str]) -> list[dict]:
        raw = self._extract_json(text)
        detections = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cls = item.get("class", "").strip()
            # Fuzzy class matching (handles capitalization differences)
            matched_cls = next(
                (c for c in valid_classes if c.lower() == cls.lower()), None
            )
            if matched_cls is None:
                logger.debug(f"Skipping unknown class: {cls}")
                continue

            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                continue

            bbox = [float(v) for v in bbox]
            # Handle Qwen native grounding format (0-1000 range)
            if any(v > 1.5 for v in bbox):
                bbox = [v / 1000.0 for v in bbox]
            # Clamp and validate
            bbox = [max(0.0, min(1.0, v)) for v in bbox]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                logger.debug(f"Skipping invalid bbox: {bbox}")
                continue

            confidence = float(item.get("confidence", 0.5))
            detections.append(
                {"class": matched_cls, "bbox": bbox, "confidence": confidence}
            )
        return detections

    def _extract_json(self, text: str) -> list:
        """Extract JSON array from LLM response, handling common formats."""
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

        # Try direct JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("detections", "objects", "results", "annotations"):
                    if isinstance(result.get(key), list):
                        return result[key]
        except json.JSONDecodeError:
            pass

        # Find the first JSON array in the text
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to find the largest JSON array
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not extract JSON from response: {text[:300]}")
        return []

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Detector unloaded")
