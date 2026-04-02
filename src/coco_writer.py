"""
Assembles COCO-format JSON for instance segmentation output.
Compatible with CVAT, Label Studio, Roboflow, and standard training pipelines.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .utils import mask_to_polygon, mask_area, bbox_norm_to_coco

logger = logging.getLogger(__name__)


class COCOWriter:
    """
    Builds a COCO instance segmentation JSON incrementally.

    Usage:
        writer = COCOWriter(class_list)
        img_id = writer.add_image("frame001.jpg", 1920, 1080)
        writer.add_annotation(img_id, "car", bbox_norm, mask)
        writer.save("output/annotations.json")
    """

    def __init__(self, class_list: list[str]):
        self.class_list = class_list
        self._category_map = {
            name: i + 1 for i, name in enumerate(class_list)
        }
        self._images = []
        self._annotations = []
        self._ann_id = 1

    def add_image(self, file_name: str, width: int, height: int) -> int:
        """Register an image and return its COCO image ID."""
        image_id = len(self._images) + 1
        self._images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })
        return image_id

    def add_annotation(
        self,
        image_id: int,
        class_name: str,
        bbox_norm: list[float],
        mask: np.ndarray,
        confidence: float = 1.0,
        sam_score: float = 1.0,
    ) -> int:
        """
        Add one instance annotation.

        Args:
            image_id: from add_image()
            class_name: must be in class_list
            bbox_norm: [x1, y1, x2, y2] normalized [0,1]
            mask: (H, W) bool array
            confidence: Qwen detection confidence
            sam_score: SAM mask quality score

        Returns:
            annotation id
        """
        cat_id = self._category_map.get(class_name)
        if cat_id is None:
            logger.warning(f"Unknown class '{class_name}', skipping annotation")
            return -1

        # Image dimensions for conversion
        img = next((i for i in self._images if i["id"] == image_id), None)
        if img is None:
            logger.warning(f"Unknown image_id {image_id}")
            return -1

        img_w, img_h = img["width"], img["height"]

        # COCO bbox: [x, y, w, h] in pixels
        coco_bbox = bbox_norm_to_coco(bbox_norm, img_w, img_h)

        # Segmentation as polygons
        if mask is not None:
            segmentation = mask_to_polygon(mask)
            area = mask_area(mask)
        else:
            # Fall back to bbox polygon if no mask
            x, y, w, h = coco_bbox
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            area = w * h

        if not segmentation:
            logger.debug(f"Empty polygon for {class_name}, using bbox fallback")
            x, y, w, h = coco_bbox
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

        ann_id = self._ann_id
        self._annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "segmentation": segmentation,
            "bbox": [round(v, 2) for v in coco_bbox],
            "area": float(area),
            "iscrowd": 0,
            # Extra metadata (non-standard but useful)
            "attributes": {
                "detection_confidence": round(confidence, 4),
                "sam_score": round(sam_score, 4),
            },
        })
        self._ann_id += 1
        return ann_id

    def add_image_results(self, file_name: str, results: list[dict], img_w: int, img_h: int) -> int:
        """
        Convenience method: add an image and all its detection results at once.

        Args:
            file_name: image file name (relative path)
            results: list of detection dicts from Pipeline.process_image()
            img_w, img_h: image dimensions

        Returns:
            image_id
        """
        image_id = self.add_image(file_name, img_w, img_h)
        for det in results:
            if det.get("mask") is not None:
                self.add_annotation(
                    image_id=image_id,
                    class_name=det["class"],
                    bbox_norm=det["bbox"],
                    mask=det["mask"],
                    confidence=det.get("confidence", 1.0),
                    sam_score=det.get("sam_score", 1.0),
                )
        return image_id

    def load(self, path: str | Path):
        """Load an existing COCO JSON to resume accumulation."""
        with open(path) as f:
            data = json.load(f)
        self._images = data.get("images", [])
        self._annotations = data.get("annotations", [])
        self._ann_id = max((a["id"] for a in self._annotations), default=0) + 1

    def to_dict(self) -> dict:
        """Return the complete COCO JSON as a dict."""
        categories = [
            {
                "id": i + 1,
                "name": name,
                "supercategory": "object",
            }
            for i, name in enumerate(self.class_list)
        ]
        return {
            "info": {
                "description": "Generated by qwen3vl2sam",
                "date_created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "licenses": [],
            "categories": categories,
            "images": self._images,
            "annotations": self._annotations,
        }

    def save(self, path: str | Path):
        """Write COCO JSON to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            f"Saved COCO JSON: {path} "
            f"({len(self._images)} images, {len(self._annotations)} annotations)"
        )
        return path

    def summary(self) -> dict:
        """Return a summary of what's been collected."""
        class_counts: dict[str, int] = {}
        for ann in self._annotations:
            cat_id = ann["category_id"]
            name = self.class_list[cat_id - 1]
            class_counts[name] = class_counts.get(name, 0) + 1
        return {
            "images": len(self._images),
            "annotations": len(self._annotations),
            "classes": class_counts,
        }
