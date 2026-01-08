"""Quick sanity check"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

print("=" * 60)
print("U-Net 细胞分割项目 - 模块测试")
print("=" * 60)

# 1. 测试模型
print("\n[1/4] 测试U-Net模型...")
try:
    from modules.models import UNet
    
    model = UNet(in_channels=1, out_channels=1, base_channels=32, depth=4)
    dummy_input = torch.randn(2, 1, 512, 512)
    output = model(dummy_input)
    
    assert output.shape == (2, 1, 512, 512), f"输出形状错误: {output.shape}"
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载成功 | 参数量: {param_count:,}")
except Exception as e:
    print(f"✗ 模型测试失败: {e}")
    sys.exit(1)

# 2. 测试损失函数和指标
print("\n[2/4] 测试损失函数和指标...")
try:
    from modules.utils import DiceLoss, dice_coefficient
    
    dice_loss = DiceLoss()
    logits = torch.randn(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss = dice_loss(logits, targets)
    dice = dice_coefficient(logits, targets, threshold=0.5)
    
    assert 0 <= loss.item() <= 2, f"Dice Loss范围异常: {loss.item()}"
    assert 0 <= dice.item() <= 1, f"Dice系数范围异常: {dice.item()}"
    
    print(f"✓ 损失函数正常 | DiceLoss: {loss.item():.4f} | Dice: {dice.item():.4f}")
except Exception as e:
    print(f"✗ 损失函数测试失败: {e}")
    sys.exit(1)

# 3. 测试可视化工具
print("\n[3/4] 测试可视化工具...")
try:
    from modules.utils import setup_cn_font
    
    setup_cn_font()
    print("✓ 中文字体配置成功")
except Exception as e:
    print(f"⚠ 中文字体配置失败(不影响训练): {e}")

# 4. 测试数据集(如果数据存在)
print("\n[4/4] 测试数据集加载...")
data_root = Path("../dataset")
dataset_name = "DIC-C2DH-HeLa"
dataset_path = data_root / dataset_name / "01"

if dataset_path.exists():
    try:
        from modules.datasets import build_cell_dataloaders
        
        train_loader, val_loader, test_loader = build_cell_dataloaders(
            data_root=str(data_root),
            dataset_name=dataset_name,
            resize_to=(512, 512),
            batch_size=2,
            num_workers=0,  # 测试时不用多进程
            val_ratio=0.2,
            test_ratio=0.2,
            sequences=("01",),
        )
        
        # 测试一个batch
        batch = next(iter(train_loader))
        assert batch['image'].shape[0] <= 2, "Batch size错误"
        assert batch['image'].shape[1:] == (1, 512, 512), "图像尺寸错误"
        assert batch['mask'].shape[1:] == (1, 512, 512), "掩码尺寸错误"
        
        print(f"✓ 数据集加载成功 | 训练集: {len(train_loader.dataset)} | "
              f"验证集: {len(val_loader.dataset)} | 测试集: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        sys.exit(1)
else:
    print(f"⚠ 数据集不存在: {dataset_path}")
    print(f"  请将数据放置到 {data_root}/{dataset_name}/ 目录下")

print("\n" + "=" * 60)
print("✓ 所有核心模块测试通过!")
print("=" * 60)
print("\n可以开始训练:")
print("  conda activate pytorch")
print("  python train_unet.py --dataset-name DIC-C2DH-HeLa --epochs 80")
