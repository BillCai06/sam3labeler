# SAM3 Fine-tuning 训练说明

## 环境

- 硬件：8× NVIDIA H200 (140 GB each)
- Conda 环境：`qwen3vl2sam`
- Python：3.11
- 训练脚本：`train_sam3.py`

---

## 数据

- 数据目录：`outputs/20260320_125939_dataset/`
- 格式：COCO JSON，包含 drone 和 husky 两类图像
- 图像来源：`drone_frames/`、`husky_frames/`
- 自动标注 + 人工审核混合；使用 `--min_prelabel_conf 0.5` 过滤低置信度自动标注
- 训练/验证比：85% / 15%（`--val_split 0.15`）
- 负样本比例：25%（`--neg_ratio 0.25`）

---

## 训练策略：两阶段

SAM3 的 fine-tuning 分两个阶段，原因是直接解冻整个模型容易破坏预训练特征。

### Phase 1 — Head Only（冻结 Backbone）

只训练检测头（decoder），vision encoder 和 text encoder 全部冻结。

**目标：** 让 head 快速适应新数据分布，风险低，不会破坏预训练特征。

**关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--freeze_vision` | ✓ | 冻结视觉编码器 |
| `--freeze_text` | ✓ | 冻结文本编码器 |
| `--finetune_ratio` | `0.0` | backbone 完全冻结 |
| `--lr` | `6e-4` | head 学习率 |
| `--batch_size` | `64` | 单 GPU，冻结 backbone 显存充足 |
| `--accum_steps` | `1` | effective batch = 64 |
| `--epochs` | `40` | head-only 收敛需要较多 epoch |
| `--warmup_steps` | `300` | 线性 warmup |
| `--weight_decay` | `0.05` | |

**运行命令（单 GPU）：**

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

**实际结果（各次实验）：**

| Run | Epochs | LR | Best val_loss | Best mask_IoU |
|-----|--------|----|--------------|--------------|
| `phase1_h200` | 39/40 | 6e-4 | 2.536 | **0.529** |
| `phase1_h200_lr6e4` | 40/40 | 6e-4 | 3.122 | 0.392 |
| `phase1_h200_v2` | 22/40 | 3e-4 | 3.051 | 0.404 |

> **最佳 Phase 1 checkpoint：`checkpoints/phase1_h200/best`**

**观察：**
- val loss 稳定下降（3.48 → 3.12），无明显过拟合
- mask_iou 在 head-only 阶段有明显噪声（±0.1），不适合直接用 IoU 选模型
- dice_loss 持续下降（0.43 → 0.34），soft mask 质量在改善，但 hard IoU（阈值 0.5）未必同步
- **选模型标准：val_loss**（IoU 噪声太大，val set 小时容易被单 epoch 峰值锁死）

---

### Phase 2 — Light Backbone Fine-tune（轻度解冻 Vision）

从 Phase 1 最佳 checkpoint 加载模型权重，解冻 vision encoder，用极低 LR 微调。text encoder 保持冻结。

**目标：** 让 vision encoder 适应新数据的视觉特征，突破 head-only 的 IoU 上限。

**关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--freeze_text` | ✓ | text encoder 继续冻结 |
| `--freeze_vision` | ✗ | vision encoder 解冻 |
| `--finetune_ratio` | `0.05` | backbone LR = 5% of head LR |
| `--lr` | `2e-4` | head LR；backbone LR = 1e-5 |
| `--batch_size` | `4` | backbone 有梯度，显存需求大幅增加 |
| `--accum_steps` | `8` | effective batch = 4×8×4GPU = 128 |
| `--epochs` | `20` | backbone 收敛快 |
| `--warmup_steps` | `30` | 从已训练权重继续，warmup 短 |

> **注意：** Phase 2 不能用 `--resume`，必须用 `--sam3_path`。  
> 原因：`--resume` 会加载 Phase 1 的 optimizer 状态（仅 1 个 param group），而 Phase 2 解冻 backbone 后有 2 个 param group，加载会报 mismatch 错误。  
> 用 `--sam3_path` 加载模型权重，optimizer 重新初始化。

**运行命令（4 GPU）：**

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

**实际结果（各次实验）：**

| Run | From | finetune_ratio | LR | Best val_loss | Best mask_IoU |
|-----|------|----------------|----|--------------|--------------|
| `phase2_h200_ft3` | phase1_h200 | 0.05 | 2e-4 | **2.530** | **0.527** |
| `phase2_h200_ft5` | phase1_h200 | 0.05 | 1.5e-4 | 2.579 | 0.528 |
| `phase2_h200_ft2` | phase1_h200 | 0.02 | 1e-4 | 2.575 | 0.520 |
| `phase2` | phase1_h200_lr6e4 | 0.01 | 1e-4 | 3.053 | 0.380 |

> **最佳 Phase 2 checkpoint：`checkpoints/phase2_h200_ft3/best`**

**观察：**
- Phase 2 从好的 Phase 1 起点出发至关重要（`phase1_h200` IoU 0.529 → `phase2_h200_ft3` IoU 0.527）
- 从差的 Phase 1 起点出发（`phase1_h200_lr6e4` IoU 0.392 → `phase2` IoU 0.380），Phase 2 无法弥补
- `finetune_ratio=0.05` 效果好于 `0.01` 和 `0.02`
- Phase 2 的 IoU 比 Phase 1 更稳定（backbone 解冻后特征更强）

---

## 显存注意事项

| 阶段 | GPU 数 | batch_size | backbone 梯度 | 每 GPU 显存 |
|------|--------|-----------|--------------|------------|
| Phase 1 | 1 | 64 | ✗ | ~40 GB |
| Phase 2 | 4 | 4 | ✓ | ~120 GB |

Phase 2 backbone 解冻后显存需求暴增。在 H200 (140 GB) 上，`batch_size=16` OOM，需降至 `batch_size=4`。

---

## 模型选择标准

- **Phase 1**：用 `val_loss`（mask_iou 噪声太大，容易被早期峰值锁死）
- **Phase 2**：`val_loss` 和 `mask_iou` 均可参考，通常一致

---

## 使用训练好的模型

```python
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained(
    "checkpoints/phase2_h200_ft3/best",
    torch_dtype=torch.bfloat16,
)
processor = Sam3Processor.from_pretrained("checkpoints/phase2_h200_ft3/best")
```

在推理 pipeline 中：

```python
detector = Sam3ImageDetector(
    sam3_local_path="checkpoints/phase2_h200_ft3/best"
)
```

---

## Loss 权重说明

```
cls_loss_weight     = 1.0   # Focal loss，分类
box_loss_weight     = 5.0   # L1 loss，bounding box 回归
giou_loss_weight    = 2.0   # GIoU loss，box 形状
mask_loss_weight    = 3.0   # BCE loss，mask 像素级
dice_loss_weight    = 3.0   # Dice loss，mask 整体形状
presence_loss_weight = 1.0  # 目标是否存在
```

Mask 相关 loss（BCE + Dice）权重设为 3.0（高于默认），强调 segmentation 质量。
