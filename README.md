# U-Net 细胞分割

**环境**: RTX 4070 (8GB) | PyTorch 2.9.0+cu130 | Python 3.12 | Cell Tracking Challenge

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
        ├── datasets/          # 数据加载
        ├── models/            # U-Net架构
        └── utils/             # 损失/指标/可视化
```

**权限说明**: 
- `code/` - 可读写,存放所有源代码
- `model/` - 只读*,训练产生的模型权重
- `dataset/` - 只读*,原始数据集
- `inference_results/` - 可读写,推理可视化输出
- *发布后设为只读,开发阶段可写

---

## 快速启动

```bash
# 1. 激活环境并安装依赖
conda activate pytorch
cd code && pip install -r requirements.txt

# 2. 测试模块
python test_modules.py

# 3. 训练 (80 epochs × 2.5s ≈ 200秒, 显存~6.2GB)
python train_unet.py --dataset-name DIC-C2DH-HeLa --epochs 80

# 4. 推理
python inference.py --checkpoint ../model/<run_name>/best.ckpt --num-vis 8
```

**输出文件**: 
- `model/<run_name>/` → best.ckpt + history.json + figures/training_history.png
- `inference_results/` → inference_samples.png

**history.json详细记录**:
- Loss/Dice曲线数据 (train/val/test)
- 学习率衰减曲线
- 每个epoch耗时
- 最佳epoch与指标
- 完整超参数配置

---

##  核心参数

| 参数 | 默认 | 说明 | 调优建议 |
|------|------|------|---------|
| `--dataset-name` | DIC-C2DH-HeLa | 数据集 | PhC-C2DH-U373可选 |
| `--sequences` | 01 02 | 训练序列 | 单序列用 `--sequences 01` |
| `--mask-source` | ST | 标注源 | GT=人工标注(更准/帧少) |
| `--resize` | 512 | 图像尺寸 | 显存不足降至384 |
| `--batch-size` | 4 | 批大小 | 2→~3.8GB, 1→~2.5GB |
| `--epochs` | 80 | 训练轮数 | 快速调试用1 |
| `--lr` | 1e-3 | 学习率 | 精调试5e-4 |
| `--dice-weight` | 1.0 | Dice权重 | 提升边界: 1.5-2.0 |

---

## 典型场景

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

- **性能**: AMP混合精度(2-3×加速) + TF32(Tensor Cores) + CUDA优化
- **损失**: BCE + Dice复合损失 (分类+重叠度)
- **增强**: 几何变换(翻转/旋转) + 光度扰动(亮度/对比度)
- **可视化**: 训练曲线(Loss/Dice) + 批量推理对比图
- **稳定性**: 全局seed + 最佳模型追踪 + 断点续训支持

---

## 性能基准

| 配置 | 速度 | 显存 | Val Dice |
|------|------|------|----------|
| 512²×4 + AMP | 2.5s/epoch | 6.2GB | 0.88-0.92 |
| 512²×2 + AMP | 1.3s/epoch | 3.8GB | 0.86-0.90 |
| 384²×4 + AMP | 1.5s/epoch | 4.5GB | 0.85-0.88 |

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| OOM显存溢出 | `--batch-size 2` 或 `--resize 384` |
| 导入错误 | 确认在 `code/` 目录运行 |
| 中文乱码 | 已自动处理(Noto Sans CJK) |
| 数据集缺失 | 检查 `../dataset/DIC-C2DH-HeLa/01/` 存在 |

---

**运行前记得**: `conda activate pytorch`  
**参考资源**: [U-Net论文](https://arxiv.org/abs/1505.04597)

