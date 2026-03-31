# Deprecated Components

These components were removed in favour of the SAM3-only pipeline. SAM3 (`sam3_video` backend) handles detection and segmentation in a single pass and was found to outperform the two-stage Qwen + SAM approach for this use case.

---

## Removed: Qwen VL Detector (`src/models/qwen_detector.py`)

**What it did:** Used `Qwen2.5-VL-7B-Instruct` (or `Qwen3-VL`) as a bounding-box detector. Sent the image and a list of class names to the model, parsed the JSON response containing normalised `[x1, y1, x2, y2]` bounding boxes, then passed those boxes to SAM for segmentation.

**Why removed:**
- Required ~18 GB VRAM (7 GB with 4-bit quantisation via `bitsandbytes`)
- Slow inference — LLM autoregressive generation for every image
- SAM3's built-in CLIP+DETR detector runs at ~6 GB VRAM and is significantly faster
- Two-stage pipeline added latency and a JSON-parsing failure mode

**Key implementation details (for reference):**
- `QwenDetector.detect(image, classes)` → `list[{"class", "bbox", "confidence"}]`
- Handled Qwen's native 0–1000 bbox range and clamped to [0, 1]
- Fuzzy class matching (case-insensitive)
- 4-bit quantisation via `BitsAndBytesConfig`
- Dependencies: `qwen-vl-utils>=0.0.8`, `transformers Qwen2_5_VLForConditionalGeneration`

---

## Removed: SAM2 Backend (`sam_segmentor.py → _load_sam2 / _predict_sam2`)

**What it did:** Wrapped `SAM2ImagePredictor` from the `sam2` package. Called `predictor.set_image(image)` once per image to compute embeddings, then `predictor.predict(point_coords, point_labels, box)` per instance.

**Why removed:**
- Requires a custom PyTorch build with CUDA `sm_120` support for Blackwell GPUs (`conda install pytorch-nightly`)
- Heavier install (`pip install git+https://github.com/facebookresearch/sam2.git`)
- SAM3 is a drop-in improvement that works on all CUDA GPUs

**Key implementation details:**
- Used dual prompt: centre point (foreground) + bounding box
- HuggingFace model: `facebook/sam2-hiera-large`
- Required `torch.inference_mode()` context

---

## Removed: SAM1 Backend (`sam_segmentor.py → _load_sam1 / _predict_sam1`)

**What it did:** Wrapped `SamPredictor` from the `segment-anything` package. Standard Meta SAM1 with `vit_h`, `vit_l`, or `vit_b` backbones loaded from a local `.pth` checkpoint.

**Why removed:**
- Requires manual checkpoint download (~2.5 GB for `vit_h`)
- Oldest and least accurate of the three SAM variants
- No text-guided detection — still needed Qwen or another detector upstream
- SAM3 supersedes SAM1 in every dimension

**Key implementation details:**
- Checkpoint: `checkpoints/sam_vit_h.pth` (manual download from `dl.fbaipublicfiles.com`)
- `sam_model_registry[model_type](checkpoint=path)` loading pattern
- Box input shape: `(4,)` — no batch dimension (unlike SAM2's `(1, 4)`)

---

## Removed: Config Sections

The following `config.yaml` keys were removed:

```yaml
models:
  detector:
    # Qwen VL settings
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
    device: "cuda"
    dtype: "bfloat16"
    max_new_tokens: 2048
    # quantization: "4bit"

  segmentor:
    # SAM2 fallback
    sam2_model_id: "facebook/sam2-hiera-large"
    # SAM1 fallback
    sam1_checkpoint: "checkpoints/sam_vit_h.pth"
    sam1_model_type: "vit_h"
    multimask_output: false
```

---

## Removed: Dependencies

```
qwen-vl-utils>=0.0.8          # Qwen VL input processing
# bitsandbytes>=0.43.0        # 4-bit quantisation for Qwen
# pip install git+https://github.com/facebookresearch/sam2.git
# pip install segment-anything
```
