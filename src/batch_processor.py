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
from .models.sam3_image_detector import Sam3ImageDetector
from .pipeline import Pipeline
from .utils import get_image_paths, visualize_results, load_config

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes a folder of images through the SAM3 pipeline.

    When the image backend is active, images are processed in GPU batches
    (batch_size from config, default 8) for maximum throughput.

    Outputs (in output_dir/run_TIMESTAMP/):
        annotations.json     — COCO format with all instances
        visualizations/      — annotated images with mask overlays
        summary.json         — run stats (images, annotations, errors)
    """

    def __init__(self, pipeline: Pipeline, config: dict):
        self.pipeline = pipeline
        self.config = config
        self.out_cfg = config.get("output", {})
        self.inference_batch_size = config.get("pipeline", {}).get("inference_batch_size", 8)

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
        input_path = Path(input_path)

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
            base = input_path if input_path.is_dir() else input_path.parent
            candidate = base / self.out_cfg.get("dir", "outputs")
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                _test = candidate / ".write_test"
                _test.touch()
                _test.unlink()
                output_dir = candidate
            except OSError:
                fallback = Path(__file__).parent.parent / self.out_cfg.get("dir", "outputs")
                logger.warning(
                    f"No write permission for {candidate}, falling back to {fallback.resolve()}"
                )
                output_dir = fallback
        output_dir = Path(output_dir)
        run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        viz_dir = run_dir / "visualizations"

        if save_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_images": len(image_paths),
            "processed": 0,
            "failed": 0,
            "total_annotations": 0,
            "errors": [],
            "output_dir": str(run_dir),
            "coco_path": str(run_dir / "annotations") if save_coco else None,
        }

        viz_alpha = self.out_cfg.get("viz_alpha", 0.5)
        combined_writer = COCOWriter(active_classes) if save_coco else None

        use_batched = isinstance(self.pipeline.detector, Sam3ImageDetector)
        batch_size = self.inference_batch_size if use_batched else 1
        logger.info(
            f"Inference mode: {'batched (image backend), batch_size=' + str(batch_size) if use_batched else 'sequential (video backend)'}"
        )

        total = len(image_paths)
        processed_count = 0

        with tqdm(total=total, desc="Processing images") as pbar:
            for batch_start in range(0, total, batch_size):
                batch_paths = image_paths[batch_start: batch_start + batch_size]

                if use_batched:
                    self._run_image_batch(
                        batch_paths, active_classes,
                        run_dir, viz_dir, viz_alpha,
                        save_coco, save_viz,
                        combined_writer, summary,
                        progress_callback, processed_count, total,
                    )
                else:
                    self._run_single(
                        batch_paths[0], active_classes,
                        run_dir, viz_dir, viz_alpha,
                        save_coco, save_viz,
                        combined_writer, summary,
                        progress_callback, processed_count, total,
                    )

                processed_count += len(batch_paths)
                pbar.update(len(batch_paths))

        if save_coco:
            combined_path = run_dir / "annotations.json"
            combined_writer.save(combined_path)
            summary["coco_path"] = str(combined_path)
            summary["coco_summary"] = combined_writer.summary()

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if progress_callback:
            progress_callback(total, total, "Done")

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_image_batch(
        self, paths, active_classes,
        run_dir, viz_dir, viz_alpha,
        save_coco, save_viz,
        combined_writer, summary,
        progress_callback, processed_so_far, total,
    ):
        """Load and process a batch of images using Sam3ImageDetector.detect_and_segment_batch."""
        images, loaded_paths = [], []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                loaded_paths.append(p)
            except Exception as e:
                summary["failed"] += 1
                summary["errors"].append(f"{p.name}: {e}")
                logger.error(f"Failed to load {p}: {e}")

        if not images:
            return

        if progress_callback:
            progress_callback(
                processed_so_far, total,
                f"Processing {loaded_paths[0].name}" + (f" (+{len(loaded_paths)-1} more)" if len(loaded_paths) > 1 else ""),
            )

        try:
            batch_results = self.pipeline.detector.detect_and_segment_batch(
                images, active_classes,
                confidence_threshold=self.pipeline.confidence_threshold,
            )
        except Exception as e:
            logger.error(f"Batch inference failed: {traceback.format_exc()}")
            for p in loaded_paths:
                summary["failed"] += 1
                summary["errors"].append(f"{p.name}: batch error: {e}")
            return

        from .utils import apply_nms
        json_dir = run_dir / "annotations"

        for img_path, image, raw_results in zip(loaded_paths, images, batch_results):
            try:
                results = apply_nms(raw_results, self.pipeline.nms_iou_threshold)
                if self.pipeline.sam_score_threshold > 0:
                    results = [r for r in results if r["sam_score"] >= self.pipeline.sam_score_threshold]

                img_w, img_h = image.size

                if save_coco:
                    per_writer = COCOWriter(active_classes)
                    per_writer.add_image_results(img_path.name, results, img_w, img_h)
                    json_dir.mkdir(parents=True, exist_ok=True)
                    per_writer.save(json_dir / f"{img_path.stem}.json")
                    combined_writer.add_image_results(img_path.name, results, img_w, img_h)

                if save_viz and results:
                    viz_img = visualize_results(image, results, active_classes, alpha=viz_alpha)
                    viz_img.save(viz_dir / f"{img_path.stem}_annotated{img_path.suffix}")

                summary["processed"] += 1
                summary["total_annotations"] += len(results)

            except Exception as e:
                logger.error(f"Failed to save results for {img_path}: {traceback.format_exc()}")
                summary["failed"] += 1
                summary["errors"].append(f"{img_path.name}: {e}")

    def _run_single(
        self, img_path, active_classes,
        run_dir, viz_dir, viz_alpha,
        save_coco, save_viz,
        combined_writer, summary,
        progress_callback, processed_so_far, total,
    ):
        """Process one image (video backend fallback)."""
        if progress_callback:
            progress_callback(processed_so_far, total, f"Processing {img_path.name}")
        try:
            image, results = self.pipeline.process_image_path(img_path, active_classes)
            img_w, img_h = image.size

            json_dir = run_dir / "annotations"
            if save_coco:
                per_writer = COCOWriter(active_classes)
                per_writer.add_image_results(img_path.name, results, img_w, img_h)
                json_dir.mkdir(parents=True, exist_ok=True)
                per_writer.save(json_dir / f"{img_path.stem}.json")
                combined_writer.add_image_results(img_path.name, results, img_w, img_h)

            if save_viz and results:
                viz_img = visualize_results(image, results, active_classes, alpha=viz_alpha)
                viz_img.save(viz_dir / f"{img_path.stem}_annotated{img_path.suffix}")

            summary["processed"] += 1
            summary["total_annotations"] += len(results)

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {traceback.format_exc()}")
            summary["failed"] += 1
            summary["errors"].append(f"{img_path.name}: {e}")

    def _print_summary(self, summary: dict):
        print(f"\n{'='*50}")
        print(f"Batch complete: {summary['processed']}/{summary['total_images']} images")
        print(f"Total annotations: {summary['total_annotations']}")
        if summary.get("coco_summary"):
            for cls, count in sorted(summary["coco_summary"].get("classes", {}).items()):
                print(f"  {cls}: {count}")
        if summary["failed"]:
            print(f"Failures: {summary['failed']}")
            for e in summary["errors"][:5]:
                print(f"  - {e}")
        print(f"Output: {summary['output_dir']}")
        print(f"{'='*50}\n")
