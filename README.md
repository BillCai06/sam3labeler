# Sam3Labeler

Auto-label image datasets using SAM3's text-grounded segmentation, then manually review and correct annotations in a built-in WebUI labeler. Outputs standard **COCO JSON** ready for training.

Designed for drone/outdoor footage with ~25 terrain and object classes. SAM3 detects and segments objects in one forward pass from plain text class names — no bounding-box annotations needed.

---

## How it works

```
Image + class list
       │
       ▼
 [SAM3 — CLIP text encoder + DETR detector]
 Class names → bounding boxes + masks in one pass
       │
       ▼
 [NMS — remove duplicate detections of the same object]
       │
       ▼
 [Class merge — union same-class masks into one region]  ← merge_same_class
       │
       ▼
 [Confidence + SAM score filter]
       │
       ▼
 COCO JSON  +  visualization images
       │
       ▼
 [Labeler WebUI]
 Manual review: delete wrong masks, draw new ones,
 run SAM on selected regions, dedup overlaps
```

---

## Requirements

- Linux (tested Ubuntu 22.04)
- NVIDIA GPU, CUDA 12.x
- ~6 GB VRAM (SAM3 image backend)
- conda or mamba

---

## Installation

### 1. Clone / enter the project

```bash
cd ~/qwen3vl2sam
```

### 2. Run the setup script

```bash
bash setup_env.sh
```

Creates a conda env named `qwen3vl2sam`, installs all dependencies, and installs SAM2 from Meta's GitHub. Takes 5–10 minutes on first run.

**Manual install:**

```bash
conda create -n qwen3vl2sam python=3.11 -y
conda activate qwen3vl2sam

# PyTorch — pick your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # CUDA 12.4

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3. Activate

```bash
conda activate qwen3vl2sam
```

---

## Configuration

Edit `config.yaml` before running.

### Classes

The default 25 classes cover outdoor/drone/trail footage. Edit to match your project:

```yaml
classes:
  - trail
  - grass
  - trees
  - branch
  - underbrush
  - mulch
  - log
  - rock
  - gravel
  - mud
  - water
  - snow
  - sky
  - fence
  - bush
  - pole
  - sign
  - concrete
  - asphalt
  - building
  - robot
  - car
  - person
  - animal
  - other
```

### Model

```yaml
models:
  detector:
    backend: image           # SAM3 image detector (default)
    sam3_local_path: sam3    # path to local SAM3 weights
    device: cuda
    score_threshold: 0.4     # raw DETR score cutoff
```

### Thresholds

```yaml
pipeline:
  confidence_threshold: 0.35   # post-NMS confidence filter
  sam_score_threshold:  0.35   # SAM mask quality filter
  nms_iou_threshold:    0.5    # NMS overlap threshold
  merge_same_class:     true   # union same-class masks into one region per class
```

Lower `confidence_threshold` (e.g. 0.05) to get more detections at the cost of more false positives. SAM3's DETR scores run 0.05–0.40 by design — don't expect 0.9+ scores.

**`merge_same_class`** — after NMS, any remaining detections of the same class are merged into a single mask using pixel-wise union. Useful for terrain/stuff categories (grass, trail, mud, sky) where multiple overlapping or adjacent detections should form one continuous annotation rather than many separate instances. Set to `false` for countable objects (car, person, robot) if you need individual instances.

### Output

```yaml
output:
  dir: outputs                 # runs saved as outputs/run_TIMESTAMP/
  save_visualization: true     # annotated images with colored overlays
  save_masks: false            # individual PNG mask files
```

---

## Usage

### Labeler — review and correct annotations

After auto-labeling, open the WebUI to fix mistakes:

```bash
python run_batch.py --labeler
# Opens at http://localhost:7777
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `D` / `→` | Next image (auto-saves) |
| `A` / `←` | Previous image (auto-saves) |
| `Del` | Delete selected annotation |
| `S` | Run SAM on drawn bbox |
| `Q` | Dedup overlapping masks |
| `Y` / `N` | Accept / reject proposed annotations |
| `Ctrl+Z` | Undo (50 steps) |
| `Space` | Toggle annotation overlay |
| `F` | Fit image to canvas |
| `Esc` | Cancel / deselect |
| `Ctrl+Scroll` | Zoom |
| `Mid-drag` | Pan |
| `Scroll` (Draw mode) | Adjust SAM threshold |
| `Scroll` (Point mode) | Adjust click radius |

**Selecting stacked annotations:** Click the same spot repeatedly to cycle through overlapping masks.

**Dedup:** Removes lower-confidence duplicates when two annotations overlap above the IoU threshold (adjustable slider, default 0.70). Works across different classes — e.g. the same shrub detected as both `bush` and `tree`.

**SAM modes:**
- **Draw mode** — drag a bbox, then click a class or press `S` to run SAM on that region
- **Point mode** — click anywhere on an object; SAM auto-segments the area under the cursor

**Custom port:**
```bash
python run_batch.py --labeler --port 8888
```

**Network access:** The server binds to `0.0.0.0`, so other machines on the LAN can open `http://<host-ip>:7777`. See [`LABELER_NETWORK.md`](LABELER_NETWORK.md) for firewall and multi-user setup.

---

### CLI batch

```bash
# Detect specific classes in a folder
python run_batch.py -i /path/to/images/ -c robot car person

# Override thresholds at runtime
python run_batch.py -i ./frames/ -c trail grass trees \
  --confidence 0.2 --sam-score 0.3

# Auto mode: read classes from config.yaml, process each subfolder separately
python run_batch.py --auto -i /path/to/dataset/

# Single image
python run_batch.py -i ./test.jpg -c rock log

# Skip visualizations (faster)
python run_batch.py -i ./frames/ -c car person --no-viz
```

---

### Gradio GUI

```bash
python run_batch.py --gui
# Opens at http://localhost:7860
```

- **Single Image tab** — upload one image, pick classes, click Run
- **Batch Folder tab** — paste a folder path, select classes, stream progress

---

### FastAPI backend

```bash
python run_batch.py --api
# API at http://localhost:8000  |  Docs at http://localhost:8000/docs
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/process` | Single image (multipart: `file` + `classes`) |
| `POST` | `/batch` | Start batch job |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs/{job_id}/download` | Download COCO JSON |

```bash
# Single image
curl -X POST http://localhost:8000/process \
  -F "file=@image.jpg" -F "classes=car,person,robot"

# Start batch job
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/data/frames", "classes": ["car", "person"]}'
```

---

## Output format

```
outputs/run_20260318_140904/
├── annotations.json           ← combined COCO JSON
├── annotations/               ← per-image COCO JSONs (used by labeler)
│   ├── frame001.json
│   └── frame002.json
├── summary.json               ← stats: count, class breakdown, errors
└── visualizations/
    ├── frame001_annotated.jpg
    └── frame002_annotated.jpg
```

### COCO JSON

Standard COCO instance segmentation — compatible with CVAT, Label Studio, Roboflow, Detectron2, MMDetection, Ultralytics:

```json
{
  "categories": [{"id": 1, "name": "robot", "supercategory": "object"}],
  "images":     [{"id": 1, "file_name": "frame001.jpg", "width": 3840, "height": 2160}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "segmentation": [[x1, y1, x2, y2, ...]],
    "bbox": [x, y, w, h],
    "area": 14520.0,
    "iscrowd": 0,
    "attributes": {
      "detection_confidence": 0.38,
      "sam_score": 0.41
    }
  }]
}
```

---

## Project structure

```
qwen3vl2sam/
├── config.yaml                    # Classes, thresholds, model settings
├── run_batch.py                   # Unified entry point (batch / GUI / API / labeler)
├── requirements.txt
├── setup_env.sh
│
├── src/
│   ├── models/
│   │   ├── sam3_image_detector.py # SAM3 single-image detect+segment (active backend)
│   │   ├── sam3_video_detector.py # SAM3 video/batch detector variant
│   │   └── sam_segmentor.py       # SAM3/SAM2 mask-from-bbox wrapper
│   ├── pipeline.py                # detect → NMS → segment orchestration
│   ├── batch_processor.py         # folder-level batch loop + COCO output
│   ├── coco_writer.py             # COCO JSON assembly
│   └── utils.py                   # config loader, NMS, class-mask merge, color palette, mask_to_polygon
│
├── labeler/
│   ├── server.py                  # FastAPI backend for the labeler
│   └── static/
│       └── index.html             # Full WebUI — single file, no build step
│
├── gui/
│   └── app.py                     # Gradio interface
│
├── api/
│   ├── main.py                    # FastAPI batch API
│   └── schemas.py
│
├── sam3/                          # SAM3 model weights (local)
├── checkpoints/                   # SAM1/SAM2 checkpoints (if used)
└── outputs/                       # All run outputs
```

---

## Tips

**Nothing detected?** Lower `confidence_threshold` to 0.05–0.10. SAM3's DETR scores are low by design (0.05–0.40 range).

**Too many false positives?** Raise `confidence_threshold` to 0.50+ and use the labeler's `Q` (dedup) to clean up stacked masks.

**Multiple fragmented masks for the same terrain class?** Enable `merge_same_class: true` in config. This unions all detections of the same class into one annotation per class before writing to COCO — ideal for ground-cover classes like grass, trail, or mud.

**Masks are wrong but objects found?** Run SAM on a manually drawn bbox in the labeler — crop-based inference is more accurate than full-image detection.

**Large/4K images are slow or OOM?** The labeler crops to your drawn bbox before passing to SAM, so only a small region is processed. Batch mode processes the full image — reduce `inference_batch_size` in config if you hit OOM.

**Short class names work best.** SAM3 uses CLIP embeddings — single nouns like `rock`, `log`, `bush` outperform descriptive phrases. Use the labeler to manually fix edge cases rather than tuning class names.

---

## Troubleshooting

**`sm_120 is not compatible` (Blackwell GPU — RTX 5090, RTX PRO 6000, etc.)**
```bash
pip install torch==2.11.0+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall
```

**`ImportError: No module named 'sam2'`**
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**`CUDA out of memory`**
- Reduce `inference_batch_size` in config (try 4 or 8)
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already set by run_batch.py)

**SAM3 loads with `UNEXPECTED` / `MISSING` keys**
Weight remapping is handled automatically in `sam_segmentor.py`. If you still see it, make sure you're on the latest version of the file.
