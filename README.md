# U-Net 细胞分割

基于 PyTorch 实现的 U-Net，用于 Cell Tracking Challenge 数据集的细胞分割。

## 环境

- Python 3.12 / PyTorch 2.9 / CUDA 13.0
- RTX 4070 (8GB)

## 快速开始

```bash
conda activate pytorch
cd code
pip install -r requirements.txt

# 测试模块
python test_modules.py

# 训练 (约200秒/80epochs)
python train_unet.py --dataset-name DIC-C2DH-HeLa

# 推理
python inference.py --checkpoint ../model/<run_name>/best.ckpt
```

## 目录

```
├── code/                # 源码
│   ├── train_unet.py    # 训练
│   ├── inference.py     # 推理
│   └── modules/         # 模型/数据/工具
├── dataset/             # Cell Tracking Challenge 数据
├── model/               # 训练输出 (权重+曲线)
└── inference_results/   # 推理可视化
```

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset-name` | DIC-C2DH-HeLa | 数据集 |
| `--mask-source` | ST | ST=银标准, GT=人工标注 |
| `--resize` | 512 | 输入尺寸 |
| `--batch-size` | 4 | 批大小 (2→3.8GB显存) |
| `--epochs` | 80 | 轮数 |
| `--lr` | 1e-3 | 学习率 |

## 显存不足?

```bash
python train_unet.py --batch-size 2 --resize 384
```

## 输出

训练完成后在 `model/<run_name>/` 下生成:
- `best.ckpt` - 最佳模型
- `history.json` - 训练日志
- `figures/training_history.png` - 训练曲线

## 性能

512×512, batch=4, AMP: ~2.5s/epoch, 6.2GB显存, Dice 0.88-0.92

