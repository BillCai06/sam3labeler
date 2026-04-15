#!/usr/bin/env python3
"""
validate_dataset.py — 训练数据验证工具

用法:
    python3 validate_dataset.py path/to/annotations.json path/to/images_dir
    python3 validate_dataset.py outputs/run_20260401_200017/annotations.json outputs/

验证内容:
  - annotations.json 结构是否正确
  - 所有图片文件是否存在
  - polygon/bbox 格式是否有效
  - 各类别样本数量统计
  - 潜在问题警告（类别太少、polygon顶点数不足等）
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_image(filename: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = d / filename
        if p.exists():
            return p
        for sub in (d.iterdir() if d.exists() else []):
            if sub.is_dir():
                if (sub / filename).exists():
                    return sub / filename
    return None


def _merge_per_image_jsons(ann_dir: Path) -> dict:
    """Merge all per-image JSON files in ann_dir into a single COCO dict."""
    jsons = sorted(ann_dir.glob("*.json"))
    if not jsons:
        print(f"[FATAL] No .json files found in {ann_dir}")
        sys.exit(1)

    merged = {"info": {}, "licenses": [], "categories": [], "images": [], "annotations": []}
    # Collect ALL category IDs seen across all files
    all_cats: dict[int, dict] = {}
    global_img_id = 1
    global_ann_id = 1

    for jf in jsons:
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            print(f"  [WARN] Cannot parse {jf.name}: {e}")
            continue

        if not merged["info"]:
            merged["info"] = d.get("info", {})

        for cat in d.get("categories", []):
            all_cats.setdefault(cat["id"], cat)

        for img in d.get("images", []):
            new_img = dict(img)
            orig_img_id = new_img["id"]
            new_img["id"] = global_img_id

            for ann in d.get("annotations", []):
                if ann["image_id"] != orig_img_id:
                    continue
                new_ann = dict(ann)
                new_ann["id"]       = global_ann_id
                new_ann["image_id"] = global_img_id
                merged["annotations"].append(new_ann)
                global_ann_id += 1

            merged["images"].append(new_img)
            global_img_id += 1

    merged["categories"] = sorted(all_cats.values(), key=lambda c: c["id"])
    return merged


def validate(ann_path: str, images_dir: str):
    ann_path   = Path(ann_path)
    images_dir = Path(images_dir)
    search_dirs = [images_dir, ann_path if ann_path.is_dir() else ann_path.parent,
                   ann_path.parent if ann_path.is_dir() else ann_path.parent.parent]

    print(f"\n{'='*60}")
    print(f"  Dataset Validator")
    print(f"  annotations : {ann_path}")
    print(f"  images root : {images_dir}")

    errors   = []
    warnings = []

    # ── 1. Load JSON ────────────────────────────────────────────
    if ann_path.is_dir():
        jsons = list(ann_path.glob("*.json"))
        print(f"  format      : per-image JSON ({len(jsons)} files)")
        print(f"{'='*60}\n")
        data = _merge_per_image_jsons(ann_path)
    else:
        print(f"  format      : global annotations.json")
        print(f"{'='*60}\n")
        try:
            with open(ann_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[FATAL] Cannot parse annotations.json: {e}")
            sys.exit(1)

    # ── 2. Top-level keys ───────────────────────────────────────
    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")

    images      = data.get("images", [])
    annotations = data.get("annotations", [])
    categories  = data.get("categories", [])

    print(f"  Images:      {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories:  {len(categories)}")
    print()

    # ── 3. Categories ───────────────────────────────────────────
    print("── Categories ──────────────────────────────────────────")
    cat_ids = set()
    for cat in categories:
        for field in ("id", "name"):
            if field not in cat:
                errors.append(f"Category missing field '{field}': {cat}")
        cat_ids.add(cat.get("id"))

    # Count annotations per category
    ann_per_cat: Counter = Counter(a["category_id"] for a in annotations)
    id2name = {c["id"]: c["name"] for c in categories}

    for cat in sorted(categories, key=lambda c: c["id"]):
        cnt = ann_per_cat.get(cat["id"], 0)
        flag = ""
        if cnt == 0:
            flag = "  ⚠ NO ANNOTATIONS"
            warnings.append(f"Category '{cat['name']}' (id={cat['id']}) has 0 annotations")
        elif cnt < 10:
            flag = f"  ⚠ very few ({cnt})"
            warnings.append(f"Category '{cat['name']}' has only {cnt} annotations — may not train well")
        print(f"  id={cat['id']:3d}  {cat['name']:<20s}  {cnt:5d} annotations{flag}")
    print()

    # ── 4. Images ───────────────────────────────────────────────
    print("── Images ──────────────────────────────────────────────")
    img_ids = set()
    missing_files = []
    size_set = set()

    for img in images:
        for field in ("id", "file_name", "width", "height"):
            if field not in img:
                errors.append(f"Image entry missing field '{field}': {img}")
        img_ids.add(img.get("id"))
        size_set.add((img.get("width"), img.get("height")))

        found = find_image(img["file_name"], search_dirs)
        if found is None:
            missing_files.append(img["file_name"])

    print(f"  Found on disk: {len(images) - len(missing_files)} / {len(images)}")
    if missing_files:
        errors.append(f"{len(missing_files)} image files not found on disk")
        for f in missing_files[:5]:
            print(f"  [MISSING] {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files)-5} more")

    if len(size_set) == 1:
        w, h = next(iter(size_set))
        print(f"  Resolution:  all {w}×{h}")
    else:
        print(f"  Resolutions: {len(size_set)} unique sizes")
        for w, h in sorted(size_set)[:5]:
            print(f"    {w}×{h}")
    print()

    # ── 5. Annotations ──────────────────────────────────────────
    print("── Annotations ─────────────────────────────────────────")
    ann_ids = set()
    bad_bbox    = 0
    bad_seg     = 0
    rle_count   = 0
    poly_count  = 0
    empty_seg   = 0
    unknown_cats: Counter = Counter()
    anns_per_img: Counter = Counter(a["image_id"] for a in annotations)

    no_bbox_count = 0
    for ann in annotations:
        for field in ("id", "image_id", "category_id", "segmentation"):
            if field not in ann:
                errors.append(f"Annotation id={ann.get('id','?')} missing field '{field}'")
        if "bbox" not in ann:
            no_bbox_count += 1  # OK — will be derived from polygon

        ann_ids.add(ann.get("id"))

        # category check (warning only — training silently skips unknown categories)
        if ann.get("category_id") not in cat_ids:
            unknown_cats[ann["category_id"]] += 1

        # image check
        if ann.get("image_id") not in img_ids:
            errors.append(f"Annotation id={ann['id']}: image_id={ann['image_id']} not in images")

        # bbox check: [x, y, w, h] — w and h must be positive (optional field)
        bbox = ann.get("bbox")
        if bbox is not None:
            if len(bbox) != 4:
                bad_bbox += 1
            else:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    bad_bbox += 1

        # segmentation check
        seg = ann.get("segmentation", [])
        if isinstance(seg, dict):
            rle_count += 1          # RLE — not supported
        elif isinstance(seg, list):
            if len(seg) == 0:
                empty_seg += 1
            else:
                valid_poly = False
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        valid_poly = True
                        break
                if valid_poly:
                    poly_count += 1
                else:
                    bad_seg += 1
        else:
            bad_seg += 1

    print(f"  Polygon segmentations: {poly_count}")
    if unknown_cats:
        total_unknown = sum(unknown_cats.values())
        for cat_id, cnt in sorted(unknown_cats.items()):
            warnings.append(f"category_id={cat_id} not in categories list — {cnt} annotations skipped during training")
        print(f"  [WARN]  Unknown category IDs: {dict(unknown_cats)}  ({total_unknown} annotations will be skipped)")
    if no_bbox_count:
        print(f"  No bbox field (auto-derived from polygon): {no_bbox_count}  (OK)")
    if rle_count:
        errors.append(f"{rle_count} annotations use RLE segmentation — only polygon is supported")
        print(f"  [ERROR] RLE segmentations: {rle_count}  (NOT supported)")
    if bad_seg:
        warnings.append(f"{bad_seg} annotations have invalid/degenerate polygon (< 3 points)")
        print(f"  [WARN]  Degenerate polygons: {bad_seg}")
    if empty_seg:
        warnings.append(f"{empty_seg} annotations have empty segmentation — will use bbox fallback")
        print(f"  [WARN]  Empty segmentation: {empty_seg}  (bbox fallback used)")
    if bad_bbox:
        errors.append(f"{bad_bbox} annotations have invalid bbox (w<=0 or h<=0)")
        print(f"  [ERROR] Invalid bbox: {bad_bbox}")

    imgs_with_anns = len(anns_per_img)
    imgs_without   = len(images) - imgs_with_anns
    ann_counts     = list(anns_per_img.values())
    print(f"  Images with annotations:    {imgs_with_anns} / {len(images)}")
    if imgs_without:
        print(f"  Images WITHOUT annotations: {imgs_without}  (used as negatives, OK)")
    if ann_counts:
        print(f"  Annotations per image:  min={min(ann_counts)}  avg={sum(ann_counts)/len(ann_counts):.1f}  max={max(ann_counts)}")
    print()

    # ── 6. Duplicate ID check ───────────────────────────────────
    if len(img_ids) != len(images):
        errors.append(f"Duplicate image IDs detected ({len(images) - len(img_ids)} duplicates)")
    if len(ann_ids) != len(annotations):
        errors.append(f"Duplicate annotation IDs detected")

    # ── 7. Summary ──────────────────────────────────────────────
    print("── Summary ─────────────────────────────────────────────")
    if errors:
        print(f"\n  [ERRORS] {len(errors)} issue(s) that WILL break training:")
        for e in errors:
            print(f"    ✗ {e}")
    if warnings:
        print(f"\n  [WARNINGS] {len(warnings)} issue(s) that may reduce quality:")
        for w in warnings:
            print(f"    ⚠ {w}")
    if not errors and not warnings:
        print("  ✓ Dataset looks clean — ready for training!")
    elif not errors:
        print("\n  ✓ No blocking errors.  Dataset can be used (fix warnings if possible).")
    else:
        print("\n  ✗ Fix errors before training.")

    print(f"\n{'='*60}\n")
    return len(errors) == 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 validate_dataset.py <annotations.json> <images_dir>")
        sys.exit(1)
    ok = validate(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
