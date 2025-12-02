"""U-Net 模型架构

本模块实现了经典的 U-Net 架构，适用于生物医学图像分割任务。
包含基本的卷积块（ConvBlock）、下采样块（DownBlock）、上采样块（UpBlock）以及完整的 U-Net 模型类。
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积块 - U-Net的基础构建单元。

    包含两个 3x3 卷积层，每个卷积层后接 BatchNorm 和 GELU 激活函数。
    这种结构有助于提取特征并保持数值稳定性。

    Attributes:
        block (nn.Sequential): 包含卷积、BN 和激活函数的序列模块。
        dropout (nn.Module): Dropout 层，用于防止过拟合。
    """
    
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        """初始化 ConvBlock。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            dropout (float): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ConvBlock 的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, in_channels, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (N, out_channels, H, W)。
        """
        return self.dropout(self.block(x))


class DownBlock(nn.Module):
    """下采样块。

    包含一个 2x2 最大池化层，后接一个 ConvBlock。
    用于在编码器路径中降低特征图的空间分辨率并增加通道数。

    Attributes:
        pool (nn.MaxPool2d): 最大池化层。
        conv (ConvBlock): 双卷积块。
    """

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        """初始化 DownBlock。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            dropout (float): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DownBlock 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 下采样后的特征图。
        """
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """上采样块。

    用于解码器路径，将特征图分辨率恢复，并与编码器路径的对应特征图进行拼接（Skip Connection）。
    支持双线性插值或转置卷积进行上采样。

    Attributes:
        up (nn.Module): 上采样层（Upsample 或 ConvTranspose2d）。
        conv (ConvBlock): 双卷积块，用于处理拼接后的特征。
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """初始化 UpBlock。

        Args:
            in_channels (int): 输入通道数（来自上一层上采样）。
            skip_channels (int): 跳跃连接的通道数（来自编码器对应层）。
            out_channels (int): 输出通道数。
            bilinear (bool): 是否使用双线性插值进行上采样。True 使用 bilinear，False 使用 ConvTranspose2d。
            dropout (float): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels // 2 + skip_channels
        self.conv = ConvBlock(conv_in, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """UpBlock 的前向传播。

        Args:
            x (torch.Tensor): 来自下层的输入张量。
            skip (torch.Tensor): 来自编码器的跳跃连接张量。

        Returns:
            torch.Tensor: 上采样并融合后的特征图。
        """
        x = self.up(x)
        # 处理尺寸不匹配(可能由输入非2^n尺寸导致)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)  # 跳跃连接 - U-Net的灵魂
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 模型。

    一种全卷积神经网络，由编码器（下采样路径）和解码器（上采样路径）组成，
    并利用跳跃连接来保留高分辨率特征。广泛用于生物医学图像分割。

    Attributes:
        inc (ConvBlock): 输入层的卷积块。
        down_blocks (nn.ModuleList): 编码器的下采样块列表。
        bottleneck (ConvBlock): 瓶颈层卷积块。
        up_blocks (nn.ModuleList): 解码器的上采样块列表。
        out_conv (nn.Conv2d): 输出层 1x1 卷积。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """初始化 U-Net 模型。

        Args:
            in_channels (int): 输入图像的通道数，默认为 1（灰度图）。
            out_channels (int): 输出掩码的通道数，默认为 1（二值分割）。
            base_channels (int): 基础通道数（第一层的滤波器数量），默认为 32。
            depth (int): 网络的深度（下采样次数），默认为 4。必须 >= 1。
            bilinear (bool): 是否在解码器中使用双线性插值上采样，默认为 True。
            dropout (float): Dropout 概率，默认为 0.1。

        Raises:
            ValueError: 如果 depth < 1。
        """
        super().__init__()
        if depth < 1:
            raise ValueError("depth 必须 >= 1")

        factors = [2**i for i in range(depth)]
        channels: Sequence[int] = [base_channels * f for f in factors]

        self.inc = ConvBlock(in_channels, channels[0], dropout=dropout)
        self.down_blocks = nn.ModuleList()
        for idx in range(1, depth):
            self.down_blocks.append(DownBlock(channels[idx - 1], channels[idx], dropout=dropout))

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        in_ch = channels[-1] * 2
        for skip_ch in reversed_channels:
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    skip_ch,
                    skip_ch,
                    bilinear=bilinear,
                    dropout=dropout,
                )
            )
            in_ch = skip_ch

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """初始化模型权重。

        使用 Kaiming Normal 初始化卷积层权重，偏置设为 0。
        BatchNorm 层权重设为 1，偏置设为 0。

        Args:
            module (nn.Module): 要初始化的模块。
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """U-Net 的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 (N, in_channels, H, W)。

        Returns:
            torch.Tensor: 输出 logits 张量，形状为 (N, out_channels, H, W)。
        """
        shortcuts: List[torch.Tensor] = []
        x = self.inc(x)
        shortcuts.append(x)
        for down in self.down_blocks:
            x = down(x)
            shortcuts.append(x)

        x = self.bottleneck(x)

        for skip, up in zip(reversed(shortcuts), self.up_blocks):
            x = up(x, skip)

        return self.out_conv(x)
