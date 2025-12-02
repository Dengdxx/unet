# U-Net 细胞分割

**环境**: RTX 4070 (8GB) | PyTorch 2.9.0+cu130 | Python 3.12 | Cell Tracking Challenge

本项目实现了一个基于 U-Net 架构的细胞分割系统。它旨在处理 Cell Tracking Challenge 数据集，提供了从数据加载、模型训练到推理可视化的完整工作流。

---

## 目录结构

```
unet/                          # 项目根目录
├── README.md                  # 项目流程指南
├── model/                     # 模型权重 (只读*)
│   └── <dataset>_<timestamp>/
│       ├── best.ckpt          # 最佳模型权重
│       ├── last.ckpt          # 最后epoch权重
│       ├── history.json       # 详细训练历史(Loss/Dice/LR/Time...)
│       └── figures/
│           └── training_history.png  # 6合1训练曲线
├── dataset/                   # 数据集 (只读*)
│   ├── DIC-C2DH-HeLa/
│   │   ├── 01/, 02/           # 原始图像序列
│   │   ├── 01_ST/, 02_ST/     # 银标准标注
│   │   └── 01_GT/, 02_GT/     # 人工标注
│   └── PhC-C2DH-U373/
├── inference_results/         # 推理输出
│   └── inference_samples.png  # 批量预测可视化
└── code/                      # 核心源代码 (可读写)
    ├── train_unet.py          # 训练脚本
    ├── inference.py           # 推理脚本
    ├── test_modules.py        # 模块测试
    ├── requirements.txt       # 依赖清单
    └── modules/               # 核心模块
        ├── datasets/          # 数据加载 (CellSegDataset)
        ├── models/            # U-Net架构 (UNet, ConvBlock...)
        └── utils/             # 损失/指标/可视化 (DiceLoss, visualization...)
```

**权限说明**: 
- `code/` - 可读写,存放所有源代码
- `model/` - 只读*,训练产生的模型权重
- `dataset/` - 只读*,原始数据集
- `inference_results/` - 可读写,推理可视化输出
- *发布后设为只读,开发阶段可写

---

## 快速启动

### 1. 环境准备

首先，激活您的 conda 环境并安装必要的依赖项。

```bash
# 激活环境
conda activate pytorch

# 进入代码目录并安装依赖
cd code
pip install -r requirements.txt
```

### 2. 模块测试

在开始长时间训练之前，建议运行测试脚本以确保所有模块正常工作。

```bash
python test_modules.py
```

### 3. 训练模型

使用默认配置启动训练。训练过程大约需要 200 秒（80 epochs）。

```bash
python train_unet.py --dataset-name DIC-C2DH-HeLa --epochs 80
```

主要参数：
- `--dataset-name`: 数据集名称 (默认为 `DIC-C2DH-HeLa`)。
- `--epochs`: 训练轮数。
- `--batch-size`: 批大小。
- `--output-dir`: 模型保存路径。

### 4. 模型推理

加载训练好的最佳模型进行推理和可视化。

```bash
# 请将 <run_name> 替换为实际生成的运行目录名
python inference.py --checkpoint ../model/<run_name>/best.ckpt --num-vis 8
```

---

##  核心参数说明

| 参数 | 默认值 | 说明 | 调优建议 |
|------|------|------|---------|
| `--dataset-name` | DIC-C2DH-HeLa | 数据集 | 可切换为 `PhC-C2DH-U373` |
| `--sequences` | 01 02 | 训练序列 | 单序列训练可用 `--sequences 01` |
| `--mask-source` | ST | 标注源 | `GT`=人工标注(更准/帧少), `ST`=银标准 |
| `--resize` | 512 | 图像尺寸 | 显存不足时可降至 384 |
| `--batch-size` | 4 | 批大小 | 2→~3.8GB, 1→~2.5GB 显存占用 |
| `--epochs` | 80 | 训练轮数 | 快速调试可用 1 |
| `--lr` | 1e-3 | 学习率 | 精细调试可尝试 5e-4 |
| `--dice-weight` | 1.0 | Dice权重 | 提升边界分割精度: 1.5-2.0 |

---

## 输出与结果

### 训练输出
训练完成后，结果将保存在 `model/<run_name>/` 目录下：
- **best.ckpt**: 验证集损失最低的模型权重。
- **history.json**: 包含所有训练指标的详细 JSON 记录。
- **figures/training_history.png**: 包含 Loss、Dice、过拟合监控等信息的训练曲线图。

### 推理输出
推理结果将保存在 `inference_results/` 目录下：
- **inference_samples.png**: 包含原图、GT、预测概率图和叠加结果的对比图。

---

## 典型使用场景

**显存不足 (≤4GB)**
```bash
python train_unet.py --batch-size 2 --resize 384 --epochs 100
```

**单序列+人工标注 (更高精度)**
```bash
python train_unet.py --sequences 01 --mask-source GT --epochs 100
```

**跨数据集验证**
```bash
python train_unet.py --dataset-name PhC-C2DH-U373 --epochs 80
```

**超参搜索**
```bash
for lr in 1e-3 5e-4 1e-4; do
  python train_unet.py --lr $lr --epochs 80 --run-name exp_lr$lr
done
```

---

## 技术亮点

- **高性能**: 采用 AMP 混合精度训练 (2-3×加速) 和 Tensor Cores 优化 (TF32)。
- **复合损失**: 结合 BCEWithLogitsLoss 和 DiceLoss，兼顾像素分类准确率和区域重叠度。
- **数据增强**: 实现了同步的几何变换 (翻转、旋转、仿射) 和光度扰动，提升模型泛化能力。
- **完善的可视化**: 自动生成详尽的训练监控曲线和高质量的推理对比图。
- **健壮性**: 包含全局种子设置、最佳模型追踪和断点续训支持。

---

## 性能基准 (RTX 4070)

| 配置 | 速度 | 显存占用 | Val Dice |
|------|------|------|----------|
| 512²×4 + AMP | 2.5s/epoch | 6.2GB | 0.88-0.92 |
| 512²×2 + AMP | 1.3s/epoch | 3.8GB | 0.86-0.90 |
| 384²×4 + AMP | 1.5s/epoch | 4.5GB | 0.85-0.88 |

---

**参考资源**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
