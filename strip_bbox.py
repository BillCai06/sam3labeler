#!/usr/bin/env python3
"""
Strip 'bbox' field from all COCO annotation JSON files under a root directory.

Usage:
    python strip_bbox.py /mnt/usbssd/husky_drone_dataset
    python strip_bbox.py /mnt/usbssd/husky_drone_dataset --dry-run
"""
import argparse
import json
from pathlib import Path


def strip_bbox(path: Path, dry_run: bool = False) -> int:
    """Return number of annotations modified."""
    with open(path) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    had_bbox = sum(1 for a in annotations if "bbox" in a)
    if had_bbox == 0:
        return 0

    for ann in annotations:
        ann.pop("bbox", None)

    if not dry_run:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    return had_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory to search for *.json files")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying")
    args = parser.parse_args()

    root = Path(args.root)
    json_files = list(root.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {root}")

    total_modified = 0
    for path in sorted(json_files):
        n = strip_bbox(path, dry_run=args.dry_run)
        if n:
            tag = "[dry-run] " if args.dry_run else ""
            print(f"  {tag}{path.relative_to(root)}  — stripped bbox from {n} annotations")
            total_modified += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {total_modified}/{len(json_files)} files.")


if __name__ == "__main__":
    main()
