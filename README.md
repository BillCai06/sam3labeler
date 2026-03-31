# Sam3Labler

Auto-labeling pipeline: detect objects by class name → segment them → export **COCO JSON** ready for training or labeling review.

Two detector backends are available:

| Backend | How it works | VRAM |
|---|---|---|
| `sam3_video` **(default)** | SAM3's built-in CLIP+DETR detector → masks in one pass | ~6 GB |
| `qwen` | Qwen VL bounding boxes → SAM3 masks | ~18 GB |

Designed for batch processing large image datasets with up to ~20 custom classes per job. Not all classes need to be active per image — you select which ones to look for each run.

---

## How it works

### sam3_video backend (default)

```
Image + active class list
        │
        ▼
  [Sam3VideoModel — CLIP text encoder + DETR detector]
  Text class names → bounding boxes + masks in one forward pass
        │
        ▼
  [NMS + confidence filter]
        │
        ▼
  COCO JSON  +  visualized images
```

### qwen backend

```
Image + active class list
        │
        ▼
  [Qwen VL Detector]
  Structured JSON prompt → bounding boxes per class
        │
        ▼
  [NMS + confidence filter]
        │
        ▼
  [SAM3 Segmentor]
  set_image() once → predict(bbox) per instance
        │
        ▼
  COCO JSON  +  visualized images
```

The `sam3_video` backend runs SAM3's CLIP text encoder and DETR detection head; detection and segmentation share a single model and a single image encoding. The `qwen` backend uses a separate VLM for detection and SAM3 for segmentation; it is more accurate for complex scenes and descriptive class names but requires ~3× more VRAM.

---

## Requirements

- Linux (tested on Ubuntu 20.04/22.04)
- NVIDIA GPU with CUDA 12.x
- conda or mamba

**VRAM guide:**

| Configuration | VRAM needed |
|---|---|
| SAM3 video (sam3_video backend) | ~6 GB |
| Qwen2.5-VL-7B (bfloat16) + SAM3 | ~18 GB |
| Qwen2.5-VL-7B (4-bit) + SAM3 | ~9 GB |
| Qwen2.5-VL-3B (bfloat16) + SAM3 | ~10 GB |

The default `sam3_video` backend runs on a single 8 GB card. Switch to `backend: "qwen"` for better accuracy on complex scenes (requires 18+ GB).

**Blackwell GPU (RTX PRO 6000, RTX 5090, etc.):** PyTorch 2.7+ with CUDA 12.8 is required for sm_120 support. If you have an older PyTorch, see [Troubleshooting](#troubleshooting).

---

## Installation

### 1. Clone / enter the project directory

```bash
cd ~/qwen3vl2sam
```

### 2. Run the setup script

```bash
bash setup_env.sh
```

This creates a conda environment named `qwen3vl2sam`, installs all Python dependencies, and installs SAM2 from Meta's GitHub. It takes 5–10 minutes on first run.

If you prefer to do it manually:

```bash
conda create -n qwen3vl2sam python=3.11 -y
conda activate qwen3vl2sam

# PyTorch — pick the URL for your CUDA version
# CUDA 12.8 (Blackwell / sm_120 support, RTX 5000-series, RTX PRO 6000, etc.):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.4:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Main dependencies
pip install -r requirements.txt

# SAM2 (not on PyPI, install from source)
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3. Activate the environment

```bash
conda activate qwen3vl2sam
```

### 4. (Optional) SAM1 fallback

Only needed if you cannot use SAM2:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then in `config.yaml` set `models.segmentor.backend: "sam1"`.

### 5. (Optional) 4-bit quantization for Qwen

Required if your GPU has less than ~18 GB VRAM:

```bash
pip install bitsandbytes
```

Then in `config.yaml` uncomment:
```yaml
models:
  detector:
    quantization: "4bit"
```

---

## Configuration

Edit `config.yaml` before your first run.

### Detector backend

```yaml
models:
  detector:
    backend: "sam3_video"   # default — SAM3 built-in detector, no Qwen needed
    # backend: "qwen"       # use Qwen VL for detection + SAM3 for masks
```

#### sam3_video settings

```yaml
models:
  detector:
    backend: "sam3_video"
    sam3_local_path: "sam3"   # path to local SAM3 weights directory
    score_threshold: 0.05     # raw DETR score; keep low — pipeline threshold does final filtering
    new_det_thresh:  0.05     # SAM3 tracker new-object threshold
```

#### qwen settings

```yaml
models:
  detector:
    backend: "qwen"
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct"   # or Qwen3-VL-7B-Instruct
    device: "cuda"
    dtype: "bfloat16"
    max_new_tokens: 2048
    # quantization: "4bit"   # saves ~7 GB VRAM (requires bitsandbytes)
```

The first run with a new Qwen model auto-downloads it from HuggingFace (~15 GB for 7B). Set `HF_HOME` to control where it's cached.

### Set your classes

Edit the `classes` list to match your labeling project (up to ~20):

```yaml
classes:
  - person
  - hard_hat
  - forklift
  - pallet
  - fire_extinguisher
  - safety_cone
  # ... etc
```

You don't need to detect all classes in every image — the GUI and CLI let you select the active subset per run.

### Key thresholds

Thresholds differ by backend because SAM3's DETR scores run 0.02–0.25 while Qwen scores run 0.5–0.95:

```yaml
pipeline:
  # sam3_video backend:
  confidence_threshold: 0.03   # lower = more detections (more FP)
  nms_iou_threshold: 0.5
  sam_score_threshold: 0.0     # SAM3 reuses detection score; keep at 0

  # qwen backend (suggested overrides):
  # confidence_threshold: 0.25
  # sam_score_threshold: 0.5
```

### Output directory

```yaml
output:
  dir: "outputs"               # all runs saved here as outputs/run_TIMESTAMP/
  save_visualization: true     # annotated images with colored mask overlays
  save_masks: false            # individual PNG mask files per instance
```

---

## Usage

### Gradio GUI (recommended)

```bash
python run_batch.py --gui
# Opens at http://localhost:7860
```

**Single Image tab** — upload one image, check the classes you want to detect, click Run. Outputs the annotated image and a downloadable COCO JSON.

**Batch Folder tab** — paste a folder path, select classes, click Start. Streams progress, produces COCO JSON + visualizations in `outputs/run_TIMESTAMP/`.

**Help tab** — tips on class naming, threshold tuning, and GPU memory.

Custom config file:
```bash
python run_batch.py --gui --config /path/to/my_config.yaml
```

### CLI batch

```bash
# Basic
python run_batch.py -i /path/to/images/ -c car person truck

# With options
python run_batch.py \
  -i ./dataset/frames/ \
  -c car person bicycle traffic_light \
  --output ./labeled/ \
  --confidence 0.3 \
  --sam-score 0.6 \
  --no-viz          # skip saving visualization images

# Single image
python run_batch.py -i ./test.jpg -c dog cat
```

### FastAPI backend

```bash
python run_batch.py --api
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/process` | Single image (multipart: `file` + `classes`) |
| `POST` | `/batch` | Start batch job (JSON body) |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs/{job_id}/download` | Download COCO JSON |

Example — process one image:
```bash
curl -X POST http://localhost:8000/process \
  -F "file=@image.jpg" \
  -F "classes=car,person,truck"
```

Example — start a batch job:
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/data/frames", "classes": ["car", "person"]}'

# Returns {"job_id": "a1b2c3d4", ...}

curl http://localhost:8000/jobs/a1b2c3d4
```

---

## Output format

Each batch run produces:

```
outputs/run_20250327_143021/
├── annotations.json        ← COCO instance segmentation JSON
├── summary.json            ← stats: processed count, class breakdown, errors
└── visualizations/
    ├── frame001_annotated.jpg
    ├── frame002_annotated.jpg
    └── ...
```

### COCO JSON structure

Standard COCO instance segmentation format — compatible with CVAT, Label Studio, Roboflow, and all major training frameworks (Detectron2, MMDetection, Ultralytics, etc.):

```json
{
  "categories": [
    {"id": 1, "name": "car", "supercategory": "object"},
    ...
  ],
  "images": [
    {"id": 1, "file_name": "frame001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, w, h],
      "area": 14520.0,
      "iscrowd": 0,
      "attributes": {
        "detection_confidence": 0.87,
        "sam_score": 0.94
      }
    }
  ]
}
```

The `attributes` field carries `detection_confidence` (from Qwen) and `sam_score` (from SAM2) — useful for filtering borderline annotations during review.

---

## Tips

**Class naming — `sam3_video` backend.**
SAM3 uses CLIP embeddings, so short standard nouns work best:
- ✓ `"person"`, `"forklift"`, `"fire extinguisher"`
- Descriptive modifiers (`"red"`, `"wooden"`) usually don't help and can hurt recall

**Class naming — `qwen` backend.**
Qwen performs much better with specific, visual descriptions:
- ✓ `"red fire extinguisher"` vs ✗ `"fire extinguisher"`
- ✓ `"wooden pallet"` vs ✗ `"pallet"`
- ✓ `"yellow construction helmet"` vs ✗ `"helmet"`

**Use fewer active classes per run.** If you have 20 classes, split them across multiple runs of related classes. This reduces false positives on both backends.

**Low recall with `sam3_video`?** Lower `confidence_threshold` to 0.01–0.02. SAM3's detector scores are low by nature; the threshold only needs to cut obvious noise.

**Low recall with `qwen`?** Lower `confidence_threshold` to 0.1–0.15.

**Too many false positives?**
- `sam3_video`: raise `confidence_threshold` to 0.05–0.10
- `qwen`: raise `confidence_threshold` to 0.4–0.5, or tighten class names

**Masks leaking into background?** Raise `sam_score_threshold` (qwen backend only). Low SAM scores usually indicate ambiguous boundaries.

---

## Project structure

```
qwen3vl2sam/
├── config.yaml                       # All settings: models, thresholds, classes
├── requirements.txt
├── setup_env.sh                      # Automated environment setup
├── run_batch.py                      # Unified CLI (batch / GUI / API modes)
│
├── src/
│   ├── models/
│   │   ├── qwen_detector.py          # Qwen VL wrapper — structured JSON bbox output
│   │   ├── sam3_video_detector.py    # SAM3 combined detect+segment (sam3_video backend)
│   │   └── sam_segmentor.py          # SAM3/SAM2/SAM1 wrapper — mask from bbox prompt
│   ├── pipeline.py                   # detect → NMS → segment orchestration
│   ├── batch_processor.py            # folder-level batch loop + output saving
│   ├── coco_writer.py                # COCO JSON assembly
│   └── utils.py                      # visualization, NMS, color palette, I/O helpers
│
├── gui/
│   └── app.py                        # Gradio interface (single image + batch tabs)
│
├── api/
│   ├── main.py                       # FastAPI app with background job queue
│   └── schemas.py                    # Pydantic request/response models
│
├── sam3/                             # Local SAM3 weights (downloaded separately)
├── checkpoints/                      # SAM1 checkpoint goes here (if using SAM1)
└── outputs/                          # All batch run outputs land here
```

---

## Troubleshooting

**Blackwell GPU warning: `sm_120 is not compatible with the current PyTorch installation`**
Your GPU (RTX PRO 6000, RTX 5090, B200, etc.) requires PyTorch 2.7+ with CUDA 12.8.
The `--force-reinstall` flag is needed because pip won't upgrade across build variants otherwise:
```bash
pip install torch==2.11.0+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall
```

**SAM3 loads with all-`UNEXPECTED` / all-`MISSING` keys**
The SAM3 checkpoint on disk is `Sam3VideoModel` (keys prefixed `tracker_model.*`) but the
segmentor code loads it as `Sam3TrackerModel`.  This is fixed automatically by the
`_load_sam3` weight-remapping in `sam_segmentor.py` — no manual action needed.
If you see this warning anyway, make sure you have the latest version of this file.

**`sam3_video` backend detects nothing**
SAM3's DETR scores run 0.02–0.25, much lower than Qwen's 0.5–0.95. Make sure your thresholds are set for the right backend:
```yaml
pipeline:
  confidence_threshold: 0.03   # use 0.25 for qwen backend
  sam_score_threshold:  0.0    # use 0.5  for qwen backend
```

**`ImportError: No module named 'sam2'`**
SAM2 must be installed from source, not PyPI:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**`CUDA out of memory` with qwen backend**
Enable 4-bit quantization in `config.yaml` and install bitsandbytes:
```bash
pip install bitsandbytes
```
```yaml
models:
  detector:
    backend: "qwen"
    quantization: "4bit"
```
Or switch to `backend: "sam3_video"` which uses ~6 GB.

**`Could not load model Qwen/Qwen3-VL-7B-Instruct`**
Qwen3-VL may not be available yet on HuggingFace for your transformers version. Fall back to:
```yaml
models:
  detector:
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
```

**Qwen returns empty `[]` or garbled JSON**
- Check that `qwen-vl-utils` is installed: `pip install qwen-vl-utils`
- The pipeline has multi-stage JSON extraction with regex fallback, but very corrupted output may still fail — try a lower `max_new_tokens` (e.g. 512) if you only have a few classes

**SAM `reset_predictor()` AttributeError**
You have an older SAM2 version. Update:
```bash
pip install --upgrade git+https://github.com/facebookresearch/sam2.git
```
