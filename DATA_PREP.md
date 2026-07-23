# Data Preparation Guide

This document covers the full data preparation workflow, from raw images to a dataset ready for SAM3 fine-tuning.

---

## Pipeline overview

```
Raw images
   │
   ▼
Step 1: Batch auto-labeling (run_batch.py)
   │  → outputs/<run_dir>/annotations/<image>.json
   │
   ▼
Step 2: Manual cleanup (Labeler WebUI)
   │  → delete wrong annotations / fix classes / add missing labels / merge overlaps
   │
   ▼
Step 3: Dataset validation (validate_dataset.py)
   │  → confirm no format errors, inspect class distribution
   │
   ▼
Step 4: Training (train_gui.py or train_sam3.py)
```

---

## Step 1: Batch auto-labeling

Run the current SAM3 model over the images to generate a first pass of annotations for manual cleanup.

```bash
# Basic usage (images in /path/to/images/, classes from config.yaml)
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --auto \
    --config config.yaml

# Specify classes explicitly
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --classes rock tree grass car person

# Adjust confidence thresholds (lower = more candidates for manual review; higher = cleaner but may miss detections)
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --auto --confidence 0.30 --sam-score 0.30
```

**Output directory structure:**
```
outputs/
└── run_20260415_143000/
    ├── annotations/
    │   ├── frame_000000.json   ← one JSON per image (same name as image)
    │   ├── frame_000001.json
    │   └── ...
    ├── annotations.json        ← merged global version (optional to use)
    ├── visualizations/         ← visualization images (for quick review)
    └── summary.json
```

**Suggested parameters:**

| Goal | confidence | sam-score | Effect |
|------|-----------|-----------|--------|
| Maximize coverage (more deletion later) | 0.25 | 0.25 | More candidates, heavier cleanup needed |
| Balanced (recommended starting point) | 0.35 | 0.35 | Default values |
| Maximize precision (may miss detections) | 0.50 | 0.50 | Cleaner, but may need to add missing labels |

---

## Step 2: Manual annotation cleanup (Labeler WebUI)

```bash
conda run -n qwen3vl2sam python3 run_batch.py \
    --labeler \
    --config config.yaml
# → opens http://localhost:7777 in your browser
```

### Interface operations

#### Browsing annotations

- Select a dataset and image from the list on the left
- All annotations for the current image are shown as colored masks
- Click a mask to select and highlight it

#### Deleting wrong annotations

- **Click** a mask → select it (highlighted outline)
- Press `Delete` → remove the selected annotation
- You can also click the delete button in the right-hand list

#### Fixing classes

- Select an annotation → pick a new class from the dropdown on the right → save

#### Adding missing labels (two methods)

**Method A: Draw box + auto-segment with SAM**
1. Select "Draw Box" mode in the toolbar
2. Draw a box around the target region
3. Pick a class and click "SAM" → the mask is generated automatically

**Method B: Click + auto-segment with SAM**
1. Select "Point" mode
2. Click the center of the target
3. SAM automatically segments the object under the point

#### Merging overlapping annotations of the same class

- Select multiple masks of the same class (Ctrl+Click or box-select)
- Click "Merge Class" → performs a Shapely union to merge them into a single polygon

#### Propagating across frames

- Select an annotation in the current frame
- Click "Propagate →" → the current frame's bbox is used as a prompt to re-infer on the next frame

#### Saving

- Click "Save" after every change (shortcut `Ctrl+S`)
- Saved to `annotations/<image_name>.json`, written **atomically** (writes to a `.tmp` file first, then renames)

### Cleanup strategy

**Priority order, highest first:**

1. **Delete low-confidence false detections**
   - Masks that are clearly the wrong class
   - Small fragmented pieces (area < 100 pixels)

2. **Fix class labels**
   - E.g. "tree" mislabeled as "branch" or similar near-duplicate classes
   - Double-check after the automatic `_CLASS_ALIASES` substitution runs (tree → trees)

3. **Merge same-class overlaps**
   - The same object detected multiple times
   - Use the "Merge Class" feature

4. **Add missing labels**
   - Add clearly missed targets using "Draw Box"

5. **Cross-frame consistency**
   - Use "Propagate" to quickly fill in sequential frames

### Minimum cleanup targets per class

| Class importance | Target annotation count | Estimated review effort |
|-----------|-----------|------------|
| Core classes | ≥ 200 | Review every annotation |
| Common classes | ≥ 50 | Spot-check 20% |
| Rare classes | ≥ 20 (below this, results suffer) | Review all |

---

## Step 3: Dataset validation

```bash
conda run -n qwen3vl2sam python3 validate_dataset.py \
    outputs/your_run/annotations \
    /path/to/images/

# For the merged global annotations.json format:
conda run -n qwen3vl2sam python3 validate_dataset.py \
    outputs/your_run/annotations.json \
    /path/to/images/
```

**Example of normal output:**
```
── Categories ──────────────────────────────────
  id=  8  rock         28769 annotations
  id= 22  car           4054 annotations
  id= 23  person        2386 annotations
  ...

── Summary ──────────────────────────────────────
  ✓ No blocking errors. Dataset can be used.
```

**Common errors and fixes:**

| Error message | Cause | Fix |
|---------|------|---------|
| `category_id=26 not in categories` | An annotation references an undefined class | Add the class to config.yaml, or fix it in the Labeler |
| `Image file not found` | Wrong image path | Confirm images and `annotations/` live in the same directory |
| `RLE segmentation not supported` | Segmentation uses RLE format | Convert to polygon format |
| `bbox w<=0 or h<=0` | Invalid bbox | Delete the annotation or redraw the box |
| `Duplicate annotation IDs` | ID collision (merge tool issue) | Usually harmless — training rebuilds IDs from the JSON |

**Warning handling guide:**

| Warning | Severity | Recommendation |
|-----|---------|------|
| Class has only 1–9 annotations | ⚠ High | Add more data or exclude the class from training |
| Class has 10–50 annotations | ⚠ Medium | Add more if possible — results will be affected |
| Class has 0 annotations | ℹ Low | Still useful as a pure negative class (teaches the model "not here") |

---

## Step 4: Start training

Once validation passes, start training directly:

```bash
# WebUI method (recommended)
conda run -n qwen3vl2sam python3 train_gui.py --port 7861
# → open http://localhost:7861 in your browser
# → click the "Phase 1 — Head Only" preset → Start Training

# CLI method (recommended parameters for ~1600 images)
conda run -n qwen3vl2sam python3 train_sam3.py \
    --outputs_dir outputs \
    --freeze_vision --freeze_text \
    --epochs 40 --lr 5e-4 --weight_decay 0.05 \
    --batch_size 2 --accum_steps 8 \
    --output_dir checkpoints/phase1
```

---

## Detailed data format specification

### Directory layout (per-image JSON format, recommended)

```
dataset_root/
├── frame_000000.jpg        ← image file
├── frame_000001.jpg
├── ...
└── annotations/            ← must be named "annotations"
    ├── frame_000000.json   ← filename must match the image (.json instead of .jpg/.png)
    ├── frame_000001.json
    └── ...
```

### Per-image JSON format

```json
{
  "info": {
    "description": "your dataset description",
    "version": "1.0"
  },
  "licenses": [],
  "categories": [
    {"id": 1, "name": "rock",   "supercategory": "object"},
    {"id": 2, "name": "tree",   "supercategory": "object"},
    {"id": 3, "name": "person", "supercategory": "object"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "frame_000000.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [
        [120, 45, 135, 42, 148, 50, 150, 68, 140, 75, 122, 70]
      ],
      "area": 1234.5,
      "iscrowd": 0
    }
  ]
}
```

### Field constraints

#### `categories`

```
id             integer, starting from 1; consecutive or not, either is fine
name           string, must match a name in the `classes` list in config.yaml
supercategory  string, use "object" for all entries
```

> The `categories` list must be identical across all JSON files (same id ↔ name mapping)

#### `images`

```
id            integer; always 1 in the per-image format
file_name     filename only, no path, e.g. "frame_000000.jpg"
width/height  pixel dimensions, must match the actual image
```

#### `annotations`

```
id           integer, unique within this file (starting from 1 is fine)
image_id     equal to images[0].id above; always 1 in the per-image format
category_id  must correspond to an id present in this file's categories
iscrowd      always 0
```

#### `segmentation` (polygon format)

```json
"segmentation": [
  [x1, y1, x2, y2, x3, y3, ..., xN, yN]
]
```

- Outer list is the list of polygons (usually just 1; complex shapes can have more)
- Inner list is the ordered vertex coordinates, **in pixel units**
- Each polygon needs **at least 3 points (6 numbers)**
- Coordinate range: `0 ≤ x < width`, `0 ≤ y < height`

#### `bbox` (optional)

```json
"bbox": [x, y, width, height]
```

- COCO standard: top-left corner `(x, y)` + width/height (pixels)
- **The Labeler's output can omit this field** — the training script computes it from the polygon automatically
- `width > 0`, `height > 0`

---

## Data volume reference

| Dataset size | Expected outcome | Recommended strategy |
|---------|---------|---------|
| < 200 images | Not really trainable | Collect more data |
| 200–500 images | Limited improvement | freeze_vision + freeze_text, head-only training |
| 500–2000 images | Noticeable improvement | freeze_vision + freeze_text, Phase 1 |
| 2000–5000 images | Good results | finetune_ratio=0.01, Phase 1+2 |
| > 5000 images | Close to full fine-tuning | finetune_ratio=0.05, full fine-tune |

Recommended annotation counts per class:

| Class type | Minimum | Recommended | Good |
|-----|-----|-----|-----|
| Primary classes (rock, car...) | 50 | 200 | 500+ |
| Secondary classes | 20 | 80 | 200 |
| Rare classes | 10 | 40 | 100 |

---

## Quick checklist

Confirm before training:

- [ ] `validate_dataset.py` runs with no ERROR
- [ ] All image files exist on disk
- [ ] Core classes have ≥ 50 annotations
- [ ] `name` values in `categories` match the `classes` list in `config.yaml`
- [ ] Every image has at least 1 annotation (fully empty images are allowed, but should be under 30% of the set)
- [ ] Segmentation polygons have ≥ 3 vertices (6 coordinate values)
- [ ] `image.width` / `image.height` match the actual image dimensions
