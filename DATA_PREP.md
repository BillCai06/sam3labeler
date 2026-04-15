# 数据准备指南

本文档涵盖从原始图片到可用于 SAM3 fine-tuning 的完整数据准备流程。

---

## 全流程总览

```
原始图片
   │
   ▼
Step 1: 批量自动标注 (run_batch.py)
   │  → outputs/<run_dir>/annotations/<image>.json
   │
   ▼
Step 2: 人工清洗 (Labeler WebUI)
   │  → 删错误标注 / 改类别 / 补漏标 / 合并重叠
   │
   ▼
Step 3: 数据验证 (validate_dataset.py)
   │  → 确认无格式错误，查看类别分布
   │
   ▼
Step 4: 训练 (train_gui.py 或 train_sam3.py)
```

---

## Step 1：批量自动标注

用当前 SAM3 模型对图片做初始推理，生成待清洗的标注。

```bash
# 基本用法（图片在 /path/to/images/，类别在 config.yaml 里）
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --auto \
    --config config.yaml

# 指定类别
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --classes rock tree grass car person

# 调整置信度阈值（降低 = 更多候选，留给人工清洗；提高 = 更干净但可能漏检）
conda run -n qwen3vl2sam python3 run_batch.py \
    --input /path/to/images/ \
    --auto --confidence 0.30 --sam-score 0.30
```

**输出目录结构：**
```
outputs/
└── run_20260415_143000/
    ├── annotations/
    │   ├── frame_000000.json   ← 每张图一个 JSON（和图片同名）
    │   ├── frame_000001.json
    │   └── ...
    ├── annotations.json        ← 全局合并版（可选用）
    ├── visualizations/         ← 可视化图（便于快速检查）
    └── summary.json
```

**建议参数：**

| 场景 | confidence | sam-score | 效果 |
|------|-----------|-----------|------|
| 追求覆盖率（后期多删） | 0.25 | 0.25 | 候选多，需要大量清洗 |
| 平衡（推荐起点） | 0.35 | 0.35 | 默认值 |
| 追求精度（可能漏检） | 0.50 | 0.50 | 较干净，需补漏 |

---

## Step 2：人工标注清洗（Labeler WebUI）

```bash
conda run -n qwen3vl2sam python3 run_batch.py \
    --labeler \
    --config config.yaml
# → 浏览器打开 http://localhost:7777
```

### 界面操作

#### 浏览标注

- 左侧列表选择数据集和图片
- 当前图片的所有标注以彩色 mask 显示
- 点击 mask 可选中并高亮

#### 删除错误标注

- **单击** mask → 选中（高亮边框）
- 按 `Delete` 键 → 删除选中标注
- 也可在右侧列表点击删除按钮

#### 修改类别

- 选中标注 → 右侧下拉菜单选新类别 → 保存

#### 补充漏标（两种方式）

**方式 A：画框 + SAM 自动分割**
1. 在工具栏选"Draw Box"模式
2. 在图片上框出目标区域
3. 选择类别，点击"SAM"按钮 → 自动生成 mask

**方式 B：点击 + SAM 自动分割**
1. 选"Point"模式
2. 点击目标中心
3. SAM 自动识别点所在的对象

#### 合并同类重叠标注

- 选中多个同类 mask（Ctrl+Click 或框选）
- 点击"Merge Class"→ 用 Shapely 做 union 合并成一个多边形

#### 跨帧传播

- 选中当前帧的标注
- 点击"Propagate →"→ 将当前帧的 bbox 作为 prompt 在下一帧重新推理

#### 保存

- 每次改动后点"Save"（快捷键 `Ctrl+S`）
- 保存到 `annotations/<image_name>.json`，**原子写入**（先写 .tmp 再 rename）

### 清洗策略

**优先级从高到低：**

1. **删除置信度极低的错误检测**（FP）
   - 明显不是目标类别的 mask
   - 破碎的小碎片（area < 100 pixels）

2. **修正类别标签**
   - "tree" 误标为 "branch" 等近似类别
   - 运行了 `_CLASS_ALIASES`（tree → trees）的自动替换后检查

3. **合并同类重叠**
   - 同一个物体被检测到多次
   - 使用"Merge Class"功能

4. **补漏**
   - 明显漏掉的目标用"Draw Box"补上

5. **跨帧一致性**
   - 用"Propagate"快速补全序列帧

### 每类最低清洗目标

| 类别重要性 | 目标标注数 | 清洗时间估计 |
|-----------|-----------|------------|
| 核心类别 | ≥ 200 条 | 重点检查每条 |
| 普通类别 | ≥ 50 条 | 抽查 20% |
| 稀有类别 | ≥ 20 条（低于此值效果差） | 全检 |

---

## Step 3：数据验证

```bash
conda run -n qwen3vl2sam python3 validate_dataset.py \
    outputs/your_run/annotations \
    /path/to/images/

# 如果是全局 annotations.json 格式：
conda run -n qwen3vl2sam python3 validate_dataset.py \
    outputs/your_run/annotations.json \
    /path/to/images/
```

**正常输出示例：**
```
── Categories ──────────────────────────────────
  id=  8  rock         28769 annotations
  id= 22  car           4054 annotations
  id= 23  person        2386 annotations
  ...

── Summary ──────────────────────────────────────
  ✓ No blocking errors. Dataset can be used.
```

**常见错误和修复：**

| 错误信息 | 原因 | 修复方法 |
|---------|------|---------|
| `category_id=26 not in categories` | 标注了未定义的类 | 在 config.yaml 补上该类，或在 Labeler 改类别 |
| `Image file not found` | 图片路径不对 | 确认图片和 annotations/ 在同一目录 |
| `RLE segmentation not supported` | 用了 RLE 格式 | 需转换为 polygon 格式 |
| `bbox w<=0 or h<=0` | 无效 bbox | 删除该标注或重新画框 |
| `Duplicate annotation IDs` | ID 冲突（merge 工具问题） | 通常无影响，training 按 JSON 重建 ID |

**警告处理建议：**

| 警告 | 严重程度 | 建议 |
|-----|---------|------|
| 类别只有 1-9 条标注 | ⚠ 高 | 补充数据或从训练中排除该类 |
| 类别有 10-50 条标注 | ⚠ 中 | 尽量增加，效果会受影响 |
| 某类 0 条标注 | ℹ 低 | 作为纯负样本类仍有用（让模型学会"这里没有"） |

---

## Step 4：开始训练

验证通过后，直接训练：

```bash
# WebUI 方式（推荐）
conda run -n qwen3vl2sam python3 train_gui.py --port 7861
# → 浏览器打开 http://localhost:7861
# → 点击 "Phase 1 — Head Only" 预设 → Start Training

# CLI 方式（1600张图推荐参数）
conda run -n qwen3vl2sam python3 train_sam3.py \
    --outputs_dir outputs \
    --freeze_vision --freeze_text \
    --epochs 40 --lr 5e-4 --weight_decay 0.05 \
    --batch_size 2 --accum_steps 8 \
    --output_dir checkpoints/phase1
```

---

## 数据格式详细规范

### 目录结构（Per-image JSON 格式，推荐）

```
dataset_root/
├── frame_000000.jpg        ← 图片文件
├── frame_000001.jpg
├── ...
└── annotations/            ← 必须命名为 "annotations"
    ├── frame_000000.json   ← 文件名必须和图片同名（.json 替换 .jpg/.png）
    ├── frame_000001.json
    └── ...
```

### 每个 JSON 文件格式

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

### 字段约束

#### `categories`

```
id           整数，从 1 开始，连续或不连续均可
name         字符串，和 config.yaml 里的 classes 列表一致
supercategory  字符串，统一写 "object" 即可
```

> 所有 JSON 文件里的 categories 列表必须相同（id 和 name 的对应关系一致）

#### `images`

```
id           整数，per-image 格式固定为 1
file_name    仅文件名，不含路径，例如 "frame_000000.jpg"
width/height 像素尺寸，必须和实际图片匹配
```

#### `annotations`

```
id           整数，在本文件内唯一（从 1 开始即可）
image_id     等于上面 images[0].id，per-image 格式固定为 1
category_id  必须在本文件的 categories 中有对应 id
iscrowd      固定为 0
```

#### `segmentation`（polygon 格式）

```json
"segmentation": [
  [x1, y1, x2, y2, x3, y3, ..., xN, yN]
]
```

- 外层是多边形列表（一般只有 1 个，复杂形状可以多个）
- 内层是顺序顶点坐标，**坐标单位是像素**
- 每个多边形**至少 3 个点（6 个数）**
- 坐标范围：`0 ≤ x < width`，`0 ≤ y < height`

#### `bbox`（可选）

```json
"bbox": [x, y, width, height]
```

- COCO 标准：左上角 `(x, y)` + 宽高（像素）
- **Labeler 输出可以不包含此字段** — 训练脚本自动从 polygon 计算
- `width > 0`，`height > 0`

---

## 数据量参考

| 数据规模 | 效果预期 | 推荐策略 |
|---------|---------|---------|
| < 200 张 | 基本无法训练 | 扩充数据 |
| 200–500 张 | 有限改善 | freeze_vision + freeze_text，只训练头部 |
| 500–2000 张 | 明显改善 | freeze_vision + freeze_text，Phase 1 |
| 2000–5000 张 | 良好效果 | finetune_ratio=0.01，Phase 1+2 |
| > 5000 张 | 接近全量训练 | finetune_ratio=0.05，全量 fine-tune |

每个类别建议标注数：

| 类型 | 最低 | 推荐 | 优秀 |
|-----|-----|-----|-----|
| 主要类别（rock, car...）| 50 | 200 | 500+ |
| 次要类别 | 20 | 80 | 200 |
| 稀有类别 | 10 | 40 | 100 |

---

## 快速检查清单

训练前确认：

- [ ] `validate_dataset.py` 运行无 ERROR
- [ ] 所有图片文件存在于磁盘
- [ ] 核心类别标注数 ≥ 50
- [ ] `categories` 里的 `name` 和 `config.yaml` 的 `classes` 一致
- [ ] 每张图片至少有 1 条标注（纯空图可以有，但不应超过 30%）
- [ ] segmentation polygon 顶点数 ≥ 3（6 个坐标值）
- [ ] `image.width` / `image.height` 和实际图片尺寸一致
