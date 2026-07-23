# SAM3 Fine-tuning Notes

## Environment

- Hardware: 8× NVIDIA H200 (140 GB each)
- Conda environment: `qwen3vl2sam`
- Python: 3.11
- Training script: `train_sam3.py`

---

## Data

- Data directory: `outputs/20260320_125939_dataset/`
- Format: COCO JSON, covering drone and husky image classes
- Image sources: `drone_frames/`, `husky_frames/`
- Mix of auto-labeled and manually reviewed annotations; low-confidence auto-labels filtered with `--min_prelabel_conf 0.5`
- Train/val split: 85% / 15% (`--val_split 0.15`)
- Negative sample ratio: 25% (`--neg_ratio 0.25`)

---

## Training strategy: two phases

SAM3 fine-tuning is split into two phases, because unfreezing the whole model at once tends to destroy the pretrained features.

### Phase 1 — Head Only (frozen backbone)

Only the detection head (decoder) is trained; the vision encoder and text encoder are both fully frozen.

**Goal:** let the head quickly adapt to the new data distribution with low risk, without damaging the pretrained features.

**Key parameters:**

| Parameter | Value | Notes |
|------|-----|------|
| `--freeze_vision` | ✓ | Freeze the vision encoder |
| `--freeze_text` | ✓ | Freeze the text encoder |
| `--finetune_ratio` | `0.0` | Backbone fully frozen |
| `--lr` | `6e-4` | Head learning rate |
| `--batch_size` | `64` | Single GPU — frozen backbone leaves plenty of VRAM headroom |
| `--accum_steps` | `1` | Effective batch = 64 |
| `--epochs` | `40` | Head-only convergence needs more epochs |
| `--warmup_steps` | `300` | Linear warmup |
| `--weight_decay` | `0.05` | |

**Run command (single GPU):**

```bash
python train_sam3.py \
  --sam3_path sam3 \
  --outputs_dir outputs \
  --output_dir checkpoints/phase1 \
  --freeze_vision \
  --freeze_text \
  --finetune_ratio 0.0 \
  --epochs 40 \
  --lr 6e-4 \
  --batch_size 64 \
  --accum_steps 1 \
  --warmup_steps 300 \
  --weight_decay 0.05 \
  --neg_ratio 0.25 \
  --val_split 0.15 \
  --mask_loss_weight 3.0 \
  --dice_loss_weight 3.0
```

**Actual results (across runs):**

| Run | Epochs | LR | Best val_loss | Best mask_IoU |
|-----|--------|----|--------------|--------------|
| `phase1_h200` | 39/40 | 6e-4 | 2.536 | **0.529** |
| `phase1_h200_lr6e4` | 40/40 | 6e-4 | 3.122 | 0.392 |
| `phase1_h200_v2` | 22/40 | 3e-4 | 3.051 | 0.404 |

> **Best Phase 1 checkpoint: `checkpoints/phase1_h200/best`**

**Observations:**
- val loss decreases steadily (3.48 → 3.12) with no clear overfitting
- mask_iou is noticeably noisy in the head-only phase (±0.1), so it's not reliable for model selection on its own
- dice_loss keeps decreasing (0.43 → 0.34) — soft mask quality keeps improving, but that doesn't always track hard IoU (threshold 0.5) in lockstep
- **Model selection criterion: val_loss** (IoU is too noisy — with a small val set it can get locked to a single-epoch spike)

---

### Phase 2 — Light Backbone Fine-tune (light vision unfreeze)

Loads weights from the best Phase 1 checkpoint, unfreezes the vision encoder, and fine-tunes it with a very low LR. The text encoder stays frozen.

**Goal:** let the vision encoder adapt to the visual characteristics of the new data, pushing past the head-only IoU ceiling.

**Key parameters:**

| Parameter | Value | Notes |
|------|-----|------|
| `--freeze_text` | ✓ | Text encoder stays frozen |
| `--freeze_vision` | ✗ | Vision encoder unfrozen |
| `--finetune_ratio` | `0.05` | Backbone LR = 5% of head LR |
| `--lr` | `2e-4` | Head LR; backbone LR = 1e-5 |
| `--batch_size` | `4` | Backbone now has gradients, so VRAM usage jumps a lot |
| `--accum_steps` | `8` | Effective batch = 4×8×4 GPUs = 128 |
| `--epochs` | `20` | Backbone converges faster |
| `--warmup_steps` | `30` | Continuing from trained weights, so warmup is short |

> **Note:** Phase 2 must not use `--resume` — use `--sam3_path` instead.
> Reason: `--resume` restores the Phase 1 optimizer state (which has only 1 param group), but Phase 2 has 2 param groups once the backbone is unfrozen, so loading it raises a mismatch error.
> `--sam3_path` loads only the model weights and reinitializes the optimizer.

**Run command (4 GPUs):**

```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29602 train_sam3.py \
  --sam3_path checkpoints/phase1_h200/best \
  --outputs_dir outputs \
  --output_dir checkpoints/phase2 \
  --freeze_text \
  --finetune_ratio 0.05 \
  --epochs 20 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --batch_size 4 \
  --accum_steps 8 \
  --warmup_steps 30 \
  --neg_ratio 0.25 \
  --val_split 0.15 \
  --min_prelabel_conf 0.5 \
  --mask_loss_weight 3.0 \
  --dice_loss_weight 3.0 \
  --save_interval 5 \
  --log_interval 10 \
  --num_workers 4
```

**Actual results (across runs):**

| Run | From | finetune_ratio | LR | Best val_loss | Best mask_IoU |
|-----|------|----------------|----|--------------|--------------|
| `phase2_h200_ft3` | phase1_h200 | 0.05 | 2e-4 | **2.530** | **0.527** |
| `phase2_h200_ft5` | phase1_h200 | 0.05 | 1.5e-4 | 2.579 | 0.528 |
| `phase2_h200_ft2` | phase1_h200 | 0.02 | 1e-4 | 2.575 | 0.520 |
| `phase2` | phase1_h200_lr6e4 | 0.01 | 1e-4 | 3.053 | 0.380 |

> **Best Phase 2 checkpoint: `checkpoints/phase2_h200_ft3/best`**

**Observations:**
- Starting Phase 2 from a good Phase 1 checkpoint matters a lot (`phase1_h200` IoU 0.529 → `phase2_h200_ft3` IoU 0.527)
- Starting from a weak Phase 1 checkpoint (`phase1_h200_lr6e4` IoU 0.392 → `phase2` IoU 0.380), Phase 2 can't make up the gap
- `finetune_ratio=0.05` outperforms `0.01` and `0.02`
- Phase 2 IoU is more stable than Phase 1 (features are stronger once the backbone is unfrozen)

---

## VRAM notes

| Phase | GPUs | batch_size | Backbone gradients | Per-GPU VRAM |
|------|--------|-----------|--------------|------------|
| Phase 1 | 1 | 64 | ✗ | ~40 GB |
| Phase 2 | 4 | 4 | ✓ | ~120 GB |

VRAM usage jumps sharply once the Phase 2 backbone is unfrozen. On H200 (140 GB), `batch_size=16` OOMs — you need to drop to `batch_size=4`.

---

## Model selection criteria

- **Phase 1**: use `val_loss` (mask_iou is too noisy and can get locked to an early spike)
- **Phase 2**: both `val_loss` and `mask_iou` are usable and generally agree

---

## Using a trained model

```python
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained(
    "checkpoints/phase2_h200_ft3/best",
    torch_dtype=torch.bfloat16,
)
processor = Sam3Processor.from_pretrained("checkpoints/phase2_h200_ft3/best")
```

In the inference pipeline:

```python
detector = Sam3ImageDetector(
    sam3_local_path="checkpoints/phase2_h200_ft3/best"
)
```

---

## Loss weight notes

```
cls_loss_weight      = 1.0   # Focal loss, classification
box_loss_weight      = 1.0   # L1 loss, bounding box regression
giou_loss_weight     = 1.0   # GIoU loss, box shape
mask_loss_weight     = 3.0   # BCE loss, per-pixel mask
dice_loss_weight     = 3.0   # Dice loss, overall mask shape
presence_loss_weight = 1.0   # whether the object is present
```

All of the above are the script's own defaults — neither run command above overrides `cls`, `box`, `giou`, or `presence`, and `mask`/`dice` are only passed explicitly for clarity. The mask-related losses (BCE + Dice) already default higher than the rest, which emphasizes segmentation quality over box/class accuracy.
