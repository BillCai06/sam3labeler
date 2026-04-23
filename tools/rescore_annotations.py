#!/usr/bin/env python3
"""
Backfill confidence_scores into existing per-image annotation JSONs.

For each annotation JSON that was created before the batch processor started
saving raw per-class scores, this script runs inference at a very low threshold
and records the max detection confidence per class into the JSON file.

Usage:
    python tools/rescore_annotations.py
    python tools/rescore_annotations.py --outputs_dir /path/to/outputs --batch_size 16
    python tools/rescore_annotations.py --dry_run   # show how many files need updating
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rescore")


def find_annotation_jsons(outputs_dir: Path) -> list[Path]:
    """Return all per-image annotation JSONs under outputs_dir."""
    return sorted(outputs_dir.rglob("annotations/*.json"))


def needs_rescore(json_path: Path) -> bool:
    try:
        d = json.loads(json_path.read_text())
        return "confidence_scores" not in d
    except Exception:
        return False


def rescore_batch(
    json_paths: list[Path],
    detector,
    outputs_dir: Path,
    batch_size: int,
    dry_run: bool,
):
    from PIL import Image

    def find_image_file(filename: str, search_dirs: list[Path]) -> Path | None:
        for d in search_dirs:
            if not d.exists():
                continue
            p = d / filename
            if p.exists():
                return p
            for sub in d.iterdir():
                if sub.is_dir():
                    p2 = sub / filename
                    if p2.exists():
                        return p2
        return None

    # Build search dirs from outputs_dir
    def search_dirs_for(json_path: Path) -> list[Path]:
        ann_dir = json_path.parent
        run_dir = ann_dir.parent
        dirs = [run_dir, run_dir.parent, outputs_dir]
        if run_dir.parent.exists():
            dirs += [s for s in run_dir.parent.iterdir() if s.is_dir() and s != run_dir]
        return dirs

    import time
    from tqdm import tqdm

    updated = skipped = failed = 0
    total = len(json_paths)
    t0 = time.time()

    with tqdm(total=total, unit="img", dynamic_ncols=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for start in range(0, total, batch_size):
            chunk = json_paths[start: start + batch_size]

            records = []
            for jp in chunk:
                try:
                    coco = json.loads(jp.read_text())
                    images_list = coco.get("images", [])
                    if not images_list:
                        skipped += 1
                        pbar.update(1)
                        continue
                    filename = images_list[0]["file_name"]
                    img_path = find_image_file(filename, search_dirs_for(jp))
                    if img_path is None:
                        logger.warning(f"Image not found: {jp.name} ({filename})")
                        skipped += 1
                        pbar.update(1)
                        continue
                    categories = [c["name"] for c in coco.get("categories", [])]
                    if not categories:
                        skipped += 1
                        pbar.update(1)
                        continue
                    records.append((jp, coco, img_path, categories))
                except Exception as e:
                    logger.warning(f"Skip {jp.name}: {e}")
                    failed += 1
                    pbar.update(1)

            if not records:
                continue

            for jp, coco, img_path, categories in records:
                try:
                    img = Image.open(img_path).convert("RGB")
                    filename = coco["images"][0]["file_name"]

                    if dry_run:
                        updated += 1
                    else:
                        _, raw_scores = detector.detect_and_segment_batch(
                            [img], categories, scores_only=True,
                        )
                        coco["confidence_scores"] = {
                            filename: {
                                cls: round(raw_scores[0].get(cls, 0.0), 4)
                                for cls in categories
                            }
                        }
                        jp.write_text(json.dumps(coco, indent=2))
                        updated += 1

                    done = updated + skipped + failed
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    pbar.set_postfix(
                        updated=updated, skipped=skipped, failed=failed,
                        eta=f"{eta/60:.1f}min",
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Failed {jp.name}: {e}")
                    failed += 1
                    pbar.update(1)

    return updated, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Backfill confidence_scores into annotation JSONs")
    parser.add_argument("--outputs_dir", default=str(ROOT / "outputs"),
                        help="Root outputs directory to scan")
    parser.add_argument("--sam3_path", default=str(ROOT / "sam3"),
                        help="Path to SAM3 model (used for rescoring)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per inference call")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count files that need rescoring without running inference")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        logger.error(f"outputs_dir not found: {outputs_dir}")
        sys.exit(1)

    all_jsons = find_annotation_jsons(outputs_dir)
    to_rescore = [p for p in all_jsons if needs_rescore(p)]

    logger.info(f"Annotation JSONs found:      {len(all_jsons)}")
    logger.info(f"Already have scores:         {len(all_jsons) - len(to_rescore)}")
    logger.info(f"Need rescoring:              {len(to_rescore)}")

    if not to_rescore:
        logger.info("Nothing to do.")
        return

    if args.dry_run:
        logger.info("Dry run — no inference will run.")
        for p in to_rescore[:10]:
            logger.info(f"  {p}")
        if len(to_rescore) > 10:
            logger.info(f"  … and {len(to_rescore) - 10} more")
        return

    # Load detector
    from src.models.sam3_image_detector import Sam3ImageDetector
    detector = Sam3ImageDetector(sam3_local_path=args.sam3_path, device="cuda")
    detector.load()

    logger.info(f"\nRescoring {len(to_rescore)} files (batch_size={args.batch_size}) …\n")
    updated, skipped, failed = rescore_batch(
        to_rescore, detector, outputs_dir, args.batch_size, dry_run=False
    )

    logger.info(f"\nDone — updated={updated}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
