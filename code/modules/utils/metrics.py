"""Dice损失与指标"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss = 1 - Dice，缓解类别不平衡"""
    
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (N,C,H,W) 未过sigmoid; targets: {0,1}"""
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
    """计算Dice系数(评估用)"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = torch.sum(preds * targets, dim=(1, 2, 3))
    union = torch.sum(preds, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()
