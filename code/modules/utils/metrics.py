"""损失函数与评估指标

本模块提供了用于图像分割任务的损失函数和评估指标。
包含 DiceLoss 类和 dice_coefficient 函数。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice 损失函数。

    Dice Loss 直接优化 Dice 系数（也称为 F1 Score），特别适用于样本不平衡的分割任务。
    它可以缓解背景像素过多导致模型倾向于预测背景的问题。

    Attributes:
        smooth (float): 平滑因子，用于防止除零错误。
    """
    
    def __init__(self, smooth: float = 1e-5) -> None:
        """初始化 DiceLoss。

        Args:
            smooth (float): 平滑因子，默认为 1e-5。
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算 Dice 损失。

        Args:
            logits (torch.Tensor): 模型的原始输出（未经过 Sigmoid），形状为 (N, C, H, W)。
            targets (torch.Tensor): 真实标签（Ground Truth），形状应与 logits 相同，取值为 0 或 1。

        Returns:
            torch.Tensor: 计算得到的 Dice 损失值（标量）。
        """
        probs = torch.sigmoid(logits)
        # 空间维度求和,batch维度求平均
        dims = (1, 2, 3)
        numerator = 2 * torch.sum(probs * targets, dim=dims) + self.smooth
        denominator = torch.sum(probs + targets, dim=dims) + self.smooth
        dice = numerator / denominator
        return 1.0 - dice.mean()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """计算 Dice 系数用于评估。

    Dice 系数是用于衡量两个集合相似度的统计量，通常用于评估图像分割的准确性。
    此函数计算预测值和真实值之间的 Dice 系数。

    Args:
        logits (torch.Tensor): 模型的原始输出（未经过 Sigmoid），形状为 (N, C, H, W)。
        targets (torch.Tensor): 真实标签（Ground Truth），形状应与 logits 相同。
        threshold (float): 二值化阈值，默认为 0.5。概率大于此值的像素被预测为正类。
        eps (float): 平滑项，防止除零错误，默认为 1e-5。

    Returns:
        torch.Tensor: 计算得到的平均 Dice 系数（标量）。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = torch.sum(preds * targets, dim=(1, 2, 3))
    union = torch.sum(preds, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()
