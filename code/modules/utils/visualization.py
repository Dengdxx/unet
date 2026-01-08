"""matplotlib可视化工具"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端,避免进程卡死
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager, rcParams


def setup_cn_font() -> None:
    """配置中文字体"""
    font_name = next(
        (f.name for f in font_manager.fontManager.ttflist if 'noto sans cjk' in f.name.lower()),
        'SimHei',
    )
    rcParams['font.sans-serif'] = [font_name, 'sans-serif']
    rcParams['axes.unicode_minus'] = False


def plot_training_history(history: Dict[str, List[float]], save_dir: Path) -> None:
    """绘制训练曲线(Loss/Dice/LR/Time)"""
    setup_cn_font()
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建3x2子图布局
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Loss曲线 - 左上
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    
    # Dice曲线 - 右上
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='训练Dice', linewidth=2)
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='验证Dice', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice', fontsize=12)
    axes[0, 1].set_title('Dice系数曲线', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    
    # Loss对数坐标 - 左下
    axes[1, 0].semilogy(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[1, 0].semilogy(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1, 0].set_title('损失曲线(对数)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, linestyle='--', alpha=0.3)
    
    # 过拟合监控: val-train差值 - 右下
    loss_gap = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
    dice_gap = [t - v for t, v in zip(history['train_dice'], history['val_dice'])]
    
    ax_loss_gap = axes[1, 1]
    ax_dice_gap = ax_loss_gap.twinx()
    
    l1 = ax_loss_gap.plot(epochs, loss_gap, 'r-', label='Loss差(Val-Train)', linewidth=2)
    ax_loss_gap.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_loss_gap.set_xlabel('Epoch', fontsize=12)
    ax_loss_gap.set_ylabel('Loss Gap', fontsize=12, color='r')
    ax_loss_gap.tick_params(axis='y', labelcolor='r')
    
    l2 = ax_dice_gap.plot(epochs, dice_gap, 'b-', label='Dice差(Train-Val)', linewidth=2)
    ax_dice_gap.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_dice_gap.set_ylabel('Dice Gap', fontsize=12, color='b')
    ax_dice_gap.tick_params(axis='y', labelcolor='b')
    
    # 合并图例
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_loss_gap.legend(lines, labels, fontsize=11, loc='upper left')
    ax_loss_gap.set_title('过拟合监控', fontsize=14, fontweight='bold')
    ax_loss_gap.grid(True, linestyle='--', alpha=0.3)
    
    # 学习率变化 - 第三行左
    if 'learning_rates' in history and history['learning_rates']:
        axes[2, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[2, 0].set_xlabel('Epoch', fontsize=12)
        axes[2, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[2, 0].set_title('学习率衰减', fontsize=14, fontweight='bold')
        axes[2, 0].grid(True, linestyle='--', alpha=0.3)
        axes[2, 0].set_yscale('log')
    else:
        axes[2, 0].text(0.5, 0.5, '无学习率数据', ha='center', va='center', fontsize=14)
        axes[2, 0].axis('off')
    
    # 每个epoch耗时 - 第三行右
    if 'epoch_times' in history and history['epoch_times']:
        axes[2, 1].plot(epochs, history['epoch_times'], 'm-', linewidth=2)
        axes[2, 1].axhline(np.mean(history['epoch_times']), color='r', linestyle='--', 
                          label=f"平均: {np.mean(history['epoch_times']):.2f}s", linewidth=2)
        axes[2, 1].set_xlabel('Epoch', fontsize=12)
        axes[2, 1].set_ylabel('Time (seconds)', fontsize=12)
        axes[2, 1].set_title('训练耗时', fontsize=14, fontweight='bold')
        axes[2, 1].legend(fontsize=11)
        axes[2, 1].grid(True, linestyle='--', alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, '无耗时数据', ha='center', va='center', fontsize=14)
        axes[2, 1].axis('off')
    
    # 添加整体统计信息
    stats_text = f"最佳Epoch: {history.get('best_epoch', 'N/A')} | "
    stats_text += f"最佳Val Loss: {history.get('best_val_loss', 0):.4f}"
    if history.get('test_loss') is not None:
        stats_text += f" | 测试Loss: {history['test_loss']:.4f} | 测试Dice: {history['test_dice']:.4f}"
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_dir / 'training_history.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')  # 确保关闭所有图形


def visualize_predictions(
    model: nn.Module,
    dataset,
    device: torch.device,
    save_dir: Path,
    threshold: float = 0.5,
    num_samples: int = 4,
) -> None:
    """推理结果可视化: 原图/GT/概率图/叠加"""
    setup_cn_font()
    model.eval()
    
    # 均匀采样索引
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask_gt = sample['mask'][0].cpu().numpy()
            
            # 推理
            logits = model(image)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > threshold).astype(float)
            
            # 反归一化图像
            image_np = sample['image'][0].cpu().numpy()
            if hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
                mean = dataset.mean[0].item()
                std = dataset.std[0].item()
                image_np = (image_np * std) + mean
            
            # 第1列: 原图
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('输入图像' if i == 0 else '', fontsize=12)
            axes[i, 0].axis('off')
            
            # 第2列: GT掩码
            axes[i, 1].imshow(mask_gt, cmap='jet')
            axes[i, 1].set_title('真实标签' if i == 0 else '', fontsize=12)
            axes[i, 1].axis('off')
            
            # 第3列: 概率图
            im = axes[i, 2].imshow(prob, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title('预测概率' if i == 0 else '', fontsize=12)
            axes[i, 2].axis('off')
            if i == 0:
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # 第4列: 预测叠加
            axes[i, 3].imshow(image_np, cmap='gray')
            axes[i, 3].imshow(pred, cmap='Reds', alpha=0.5)
            axes[i, 3].contour(mask_gt, colors='lime', linewidths=1.5, levels=[0.5])
            axes[i, 3].set_title('预测叠加(红)+GT轮廓(绿)' if i == 0 else '', fontsize=12)
            axes[i, 3].axis('off')
            
            # 计算该样本的Dice
            intersection = np.sum(pred * mask_gt)
            union = np.sum(pred) + np.sum(mask_gt)
            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            axes[i, 0].text(
                5, 25, f'Dice: {dice:.3f}',
                color='yellow', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            )
    
    plt.tight_layout()
    plt.savefig(save_dir / 'inference_samples.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')  # 确保关闭所有图形
