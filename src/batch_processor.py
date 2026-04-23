"""
Batch processing: run the pipeline over a folder of images, save results.

Single-folder mode:
    Pass a folder of images — output is written alongside the input folder.

Multi-folder mode (auto):
    Pass a parent directory that contains sub-folders of images.
    Each sub-folder is processed independently; output goes into each sub-folder.
    Use run_auto() to get this behaviour automatically.

Output directory naming:
    <input_dir>/<parent_name>_<input_name>/
    e.g. drone_frames/ → drone_frames/sensor_record_20260318_140904_dataset_drone_frames/

Resume: if checkpoint.json exists in the output dir, already-processed images are skipped.
Delete checkpoint.json (or the whole output dir) to start fresh.
"""

import gc
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from .coco_writer import COCOWriter
from .models.sam3_image_detector import Sam3ImageDetector
from .pipeline import Pipeline
from .utils import get_image_paths, visualize_results, load_config

logger = logging.getLogger(__name__)


class BatchProcessor:

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
        input_dir = input_path if input_path.is_dir() else input_path.parent

        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = get_image_paths(str(input_path))
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

        # ── Output directory ──────────────────────────────────────────
        run_dir, actual_location = self._resolve_output_dir(input_dir, output_dir)
        viz_dir = run_dir / "visualizations"
        json_dir = run_dir / "annotations"

        if save_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
        if save_coco:
            json_dir.mkdir(parents=True, exist_ok=True)
        if not save_viz and not save_coco:
            run_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume: load checkpoint ───────────────────────────────────
        checkpoint_path = run_dir / "checkpoint.json"
        done_set: set[str] = set()
        if checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text())
                done_set = set(cp.get("done", []))
                logger.info(f"Resuming: {len(done_set)}/{len(image_paths)} images already done")
            except Exception:
                logger.warning("Could not read checkpoint.json, starting fresh")

        remaining = [p for p in image_paths if str(p.relative_to(input_dir)) not in done_set]
        skipped = len(image_paths) - len(remaining)

        logger.info(
            f"Processing {len(remaining)} images "
            f"({skipped} skipped / already done), classes: {active_classes}"
        )

        summary = {
            "total_images": len(image_paths),
            "skipped": skipped,
            "processed": 0,
            "failed": 0,
            "total_annotations": 0,
            "errors": [],
            "output_dir": str(run_dir),
            "output_location": actual_location,
            "coco_path": str(run_dir / "annotations.json") if save_coco else None,
        }

        if not remaining:
            logger.info("All images already processed.")
            self._print_summary(summary)
            return summary

        viz_alpha = self.out_cfg.get("viz_alpha", 0.5)
        combined_json = run_dir / "annotations.json"

        use_batched = isinstance(self.pipeline.detector, Sam3ImageDetector)
        batch_size = self.inference_batch_size if use_batched else 1

        total_remaining = len(remaining)
        processed_count = 0

        with tqdm(total=total_remaining, desc="Processing images") as pbar:
            for batch_start in range(0, total_remaining, batch_size):
                batch_paths = remaining[batch_start: batch_start + batch_size]

                if use_batched:
                    newly_done = self._run_image_batch(
                        batch_paths, active_classes,
                        run_dir, viz_dir, json_dir, viz_alpha,
                        save_coco, save_viz, summary,
                        progress_callback, skipped + processed_count, len(image_paths),
                        input_dir,
                    )
                else:
                    newly_done = self._run_single(
                        batch_paths[0], active_classes,
                        run_dir, viz_dir, json_dir, viz_alpha,
                        save_coco, save_viz, summary,
                        progress_callback, skipped + processed_count, len(image_paths),
                        input_dir,
                    )

                done_set.update(newly_done)
                self._save_checkpoint(checkpoint_path, done_set, len(image_paths))

                processed_count += len(batch_paths)
                pbar.update(len(batch_paths))
                gc.collect()

        # ── Finalise ──────────────────────────────────────────────────
        if save_coco:
            coco_summary = self._merge_per_image_jsons(json_dir, combined_json, active_classes)
            summary["coco_path"] = str(combined_json)
            summary["coco_summary"] = coco_summary

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if progress_callback:
            progress_callback(len(image_paths), len(image_paths), "Done")

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Output directory resolution
    # ------------------------------------------------------------------

    def _resolve_output_dir(
        self,
        input_dir: Path,
        output_dir_override: Optional[str | Path],
    ) -> tuple[Path, str]:
        """
        Returns (run_dir, location_label).
        location_label is "usb" or "local_fallback" — shown in GUI.
        """
        if output_dir_override is not None:
            d = Path(output_dir_override)
            d.mkdir(parents=True, exist_ok=True)
            return d, "custom"

        folder_name = f"{input_dir.parent.name}_{input_dir.name}"
        candidate = input_dir / folder_name

        try:
            candidate.mkdir(parents=True, exist_ok=True)
            _test = candidate / ".write_test"
            _test.touch()
            _test.unlink()
            return candidate, "usb"
        except OSError:
            fallback = Path(__file__).parent.parent / "outputs" / folder_name
            fallback.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"⚠ Cannot write to {candidate} (USB permission denied).\n"
                f"  Falling back to: {fallback.resolve()}\n"
                f"  Fix: sudo umount /mnt/usbssd && sudo mount -t exfat /dev/sda1 /mnt/usbssd "
                f"-o uid=1000,gid=1000,fmask=0133,dmask=0022,iocharset=utf8,errors=remount-ro"
            )
            return fallback, "local_fallback"

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    @staticmethod
    def _save_checkpoint(path: Path, done_set: set, total: int):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": sorted(done_set), "total": total}, indent=2))
        tmp.replace(path)  # atomic rename

    # ------------------------------------------------------------------
    # Batched inference (image backend)
    # ------------------------------------------------------------------

    def _run_image_batch(
        self, paths, active_classes,
        run_dir, viz_dir, json_dir, viz_alpha,
        save_coco, save_viz, summary,
        progress_callback, processed_so_far, total,
        input_dir=None,
    ) -> list[str]:
        """Returns list of successfully processed image keys (relative paths)."""
        images, loaded_paths = [], []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                loaded_paths.append(p)
            except Exception as e:
                summary["failed"] += 1
                rel = str(p.relative_to(input_dir)) if input_dir else p.name
                summary["errors"].append(f"{rel}: load error: {e}")

        if not images:
            return []

        if progress_callback:
            label = loaded_paths[0].name
            if len(loaded_paths) > 1:
                label += f" (+{len(loaded_paths)-1} more)"
            progress_callback(processed_so_far, total, f"Processing {label}")

        try:
            batch_results, raw_scores = self.pipeline.detector.detect_and_segment_batch(
                images, active_classes,
                confidence_threshold=self.pipeline.confidence_threshold,
                return_raw_scores=True,
            )
        except Exception as e:
            logger.error(f"Batch inference failed: {traceback.format_exc()}")
            for p in loaded_paths:
                summary["failed"] += 1
                rel = str(p.relative_to(input_dir)) if input_dir else p.name
                summary["errors"].append(f"{rel}: batch error: {e}")
            return []

        from .utils import apply_nms
        newly_done = []

        # Index-based loop so we can null out each slot immediately after use,
        # freeing mask numpy arrays and PIL images one by one instead of all at once.
        for idx in range(len(loaded_paths)):
            img_path = loaded_paths[idx]
            image = images[idx]
            rel_key = str(img_path.relative_to(input_dir)) if input_dir else img_path.name
            flat_stem = rel_key.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
            try:
                results = apply_nms(batch_results[idx], self.pipeline.nms_iou_threshold)
                if self.pipeline.sam_score_threshold > 0:
                    results = [r for r in results if r["sam_score"] >= self.pipeline.sam_score_threshold]

                img_w, img_h = image.size

                if save_coco:
                    per_writer = COCOWriter(active_classes)
                    per_writer.add_image_results(rel_key, results, img_w, img_h)
                    coco_dict = per_writer.to_dict()
                    # Store raw per-class max scores so training can build
                    # conservative negative samples (uncertain zone filtering).
                    coco_dict["confidence_scores"] = {
                        rel_key: {
                            cls: round(raw_scores[idx].get(cls, 0.0), 4)
                            for cls in active_classes
                        }
                    }
                    out_json = json_dir / f"{flat_stem}.json"
                    out_json.write_text(json.dumps(coco_dict, indent=2))

                if save_viz and results:
                    viz_img = visualize_results(image, results, active_classes, alpha=viz_alpha)
                    viz_img.save(viz_dir / f"{flat_stem}_annotated{img_path.suffix}")

                summary["processed"] += 1
                summary["total_annotations"] += len(results)
                newly_done.append(rel_key)

            except Exception as e:
                logger.error(f"Failed saving {rel_key}: {traceback.format_exc()}")
                summary["failed"] += 1
                summary["errors"].append(f"{rel_key}: {e}")
            finally:
                # Release mask arrays and PIL image immediately — don't wait for loop end
                batch_results[idx] = None
                images[idx] = None

        return newly_done

    # ------------------------------------------------------------------
    # Sequential (video backend)
    # ------------------------------------------------------------------

    def _run_single(
        self, img_path, active_classes,
        run_dir, viz_dir, json_dir, viz_alpha,
        save_coco, save_viz, summary,
        progress_callback, processed_so_far, total,
        input_dir=None,
    ) -> list[str]:
        rel_key = str(img_path.relative_to(input_dir)) if input_dir else img_path.name
        flat_stem = rel_key.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
        if progress_callback:
            progress_callback(processed_so_far, total, f"Processing {rel_key}")
        try:
            image, results = self.pipeline.process_image_path(img_path, active_classes)
            img_w, img_h = image.size

            if save_coco:
                per_writer = COCOWriter(active_classes)
                per_writer.add_image_results(rel_key, results, img_w, img_h)
                per_writer.save(json_dir / f"{flat_stem}.json")

            if save_viz and results:
                viz_img = visualize_results(image, results, active_classes, alpha=viz_alpha)
                viz_img.save(viz_dir / f"{flat_stem}_annotated{img_path.suffix}")

            summary["processed"] += 1
            summary["total_annotations"] += len(results)
            return [rel_key]

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {traceback.format_exc()}")
            summary["failed"] += 1
            summary["errors"].append(f"{rel_key}: {e}")
            return []

    # ------------------------------------------------------------------

    def run_auto(
        self,
        input_path: str | Path,
        active_classes: list[str],
        output_dir: Optional[str | Path] = None,
        save_viz: bool = True,
        save_coco: bool = True,
        progress_callback=None,
    ) -> list[dict]:
        """
        Smart entry point: auto-detects single-folder vs multi-folder layout.

        - If input contains sub-directories that have images, each sub-dir is
          processed independently and output is written into each sub-dir.
          output_dir is ignored in multi-folder mode so results stay next to
          their source data.
        - Otherwise falls back to plain run() on the input as-is.

        Returns a list of summary dicts (one per folder processed).
        Resume / checkpoint still works per-folder exactly as before.
        """
        input_path = Path(input_path)

        if input_path.is_dir():
            subdirs_with_images = sorted(
                d for d in input_path.iterdir()
                if d.is_dir() and get_image_paths(str(d))
            )
        else:
            subdirs_with_images = []

        if subdirs_with_images:
            # Precount for a unified cross-folder progress bar
            subdir_counts = [(d, len(get_image_paths(str(d)))) for d in subdirs_with_images]
            grand_total = sum(c for _, c in subdir_counts)

            print(f"\nMulti-folder mode: {len(subdirs_with_images)} sub-folders, {grand_total} images total")
            for d, n in subdir_counts:
                print(f"  {d.name}/  ({n} images)")
            print()

            summaries = []
            offset = 0
            for i, (subdir, count) in enumerate(subdir_counts):
                print(f"[{i + 1}/{len(subdirs_with_images)}] {subdir.name}")

                # Wrap callback so progress is relative to grand total
                if progress_callback is not None:
                    _offset = offset  # capture for closure

                    def _wrapped_cb(current, total, msg, _o=_offset):
                        progress_callback(_o + current, grand_total, msg)

                    cb = _wrapped_cb
                else:
                    cb = None

                summary = self.run(
                    input_path=subdir,
                    active_classes=active_classes,
                    output_dir=None,  # always write into each subdir
                    save_viz=save_viz,
                    save_coco=save_coco,
                    progress_callback=cb,
                )
                summaries.append(summary)
                offset += count

            return summaries

        # Single folder / single file — behave exactly as before
        return [self.run(input_path, active_classes, output_dir, save_viz, save_coco, progress_callback)]

    # ------------------------------------------------------------------
    # Merge per-image JSONs into one combined annotations.json
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_per_image_jsons(json_dir: Path, combined_path: Path, active_classes: list[str]) -> dict:
        """
        Reads all per-image JSON files from json_dir and writes a single
        merged annotations.json.  Done at the end of a run so the combined
        writer never accumulates in RAM during processing.
        Strips any legacy 'bbox' fields in the process.
        """
        writer = COCOWriter(active_classes)
        for json_file in sorted(json_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Skipping corrupt JSON {json_file.name}: {e}")
                continue
            for img_info in data.get("images", []):
                new_img_id = writer.add_image(
                    img_info["file_name"], img_info["width"], img_info["height"]
                )
                old_img_id = img_info["id"]
                for ann in data.get("annotations", []):
                    if ann["image_id"] != old_img_id:
                        continue
                    writer._annotations.append({
                        **{k: v for k, v in ann.items() if k not in ("id", "image_id", "bbox")},
                        "id": writer._ann_id,
                        "image_id": new_img_id,
                    })
                    writer._ann_id += 1
        writer.save(combined_path)
        return writer.summary()

    # ------------------------------------------------------------------

    def _print_summary(self, summary: dict):
        print(f"\n{'='*50}")
        print(f"Batch complete: {summary['processed']}/{summary['total_images']} images")
        if summary.get("skipped"):
            print(f"Skipped (already done): {summary['skipped']}")
        if summary.get("output_location") == "local_fallback":
            print(f"⚠ Output saved LOCALLY (USB not writable): {summary['output_dir']}")
        else:
            print(f"Output: {summary['output_dir']}")
        print(f"Total annotations: {summary['total_annotations']}")
        if summary.get("coco_summary"):
            for cls, count in sorted(summary["coco_summary"].get("classes", {}).items()):
                print(f"  {cls}: {count}")
        if summary["failed"]:
            print(f"Failures: {summary['failed']}")
            for e in summary["errors"][:5]:
                print(f"  - {e}")
        print(f"{'='*50}\n")
