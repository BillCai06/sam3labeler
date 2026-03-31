"""
Batch processing: run the pipeline over a folder of images, save results.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .coco_writer import COCOWriter
from .pipeline import Pipeline
from .utils import get_image_paths, visualize_results, load_config

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes a folder of images through the SAM3 pipeline.

    Outputs (in output_dir/run_TIMESTAMP/):
        annotations.json     — COCO format with all instances
        visualizations/      — annotated images with mask overlays
        summary.json         — run stats (images, annotations, errors)
    """

    def __init__(self, pipeline: Pipeline, config: dict):
        self.pipeline = pipeline
        self.config = config
        self.out_cfg = config.get("output", {})

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> "BatchProcessor":
        config = load_config(config_path)
        pipeline = Pipeline(config)
        return cls(pipeline, config)

    def run(
        self,
        input_path: str | Path,
        active_classes: list[str],
        output_dir: Optional[str | Path] = None,
        save_viz: bool = True,
        save_coco: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        Process all images in input_path (file or folder).

        Args:
            input_path: path to a single image or folder of images
            active_classes: class names to detect
            output_dir: where to save results (default: from config)
            save_viz: save annotated visualization images
            save_coco: save COCO JSON annotations
            progress_callback: optional callable(current, total, status_str) for GUI

        Returns:
            summary dict with stats and output paths
        """
        input_path = Path(input_path)

        # Collect image paths
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = get_image_paths(str(input_path))
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

        logger.info(f"Processing {len(image_paths)} images, classes: {active_classes}")

        # Set up output directory
        if output_dir is None:
            output_dir = Path(self.out_cfg.get("dir", "outputs"))
        output_dir = Path(output_dir)
        run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        viz_dir = run_dir / "visualizations"

        if save_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        # COCO writer
        coco_writer = COCOWriter(active_classes)

        # Stats tracking
        summary = {
            "total_images": len(image_paths),
            "processed": 0,
            "failed": 0,
            "total_annotations": 0,
            "errors": [],
            "output_dir": str(run_dir),
            "coco_path": None,
        }

        viz_alpha = self.out_cfg.get("viz_alpha", 0.5)

        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for i, img_path in enumerate(image_paths):
                status = f"Processing {img_path.name}"
                pbar.set_description(status)

                if progress_callback:
                    progress_callback(i, len(image_paths), status)

                try:
                    image, results = self.pipeline.process_image_path(
                        img_path, active_classes
                    )
                    img_w, img_h = image.size

                    # Add to COCO
                    if save_coco:
                        rel_name = img_path.name
                        coco_writer.add_image_results(rel_name, results, img_w, img_h)

                    # Save visualization
                    if save_viz and results:
                        viz_img = visualize_results(
                            image, results, active_classes, alpha=viz_alpha
                        )
                        viz_path = viz_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                        viz_img.save(viz_path)

                    summary["processed"] += 1
                    summary["total_annotations"] += len(results)

                except Exception as e:
                    error_msg = f"{img_path.name}: {e}"
                    logger.error(f"Failed to process {img_path}: {traceback.format_exc()}")
                    summary["failed"] += 1
                    summary["errors"].append(error_msg)

                pbar.update(1)

        # Save COCO JSON
        if save_coco:
            coco_path = run_dir / "annotations.json"
            coco_writer.save(coco_path)
            summary["coco_path"] = str(coco_path)
            summary["coco_summary"] = coco_writer.summary()

        # Save run summary
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if progress_callback:
            progress_callback(len(image_paths), len(image_paths), "Done")

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict):
        print(f"\n{'='*50}")
        print(f"Batch complete: {summary['processed']}/{summary['total_images']} images")
        print(f"Total annotations: {summary['total_annotations']}")
        if summary.get("coco_summary"):
            cs = summary["coco_summary"]
            print(f"Class breakdown:")
            for cls, count in sorted(cs.get("classes", {}).items()):
                print(f"  {cls}: {count}")
        if summary["failed"]:
            print(f"Failures: {summary['failed']}")
            for e in summary["errors"][:5]:
                print(f"  - {e}")
        print(f"Output: {summary['output_dir']}")
        print(f"{'='*50}\n")
