"""U-Net 模型定义"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-GELU x2"""
    
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
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
        return self.dropout(self.block(x))


class DownBlock(nn.Module):
    """MaxPool + ConvBlock"""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample + Concat(skip) + ConvBlock"""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels // 2 + skip_channels
        self.conv = ConvBlock(conv_in, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 尺寸对齐
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)  # 跳跃连接 - U-Net的灵魂
        return self.conv(x)


class UNet(nn.Module):
    """U-Net: Encoder-Decoder with Skip Connections"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.1,
    ) -> None:
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
        # 跳过最后一层channel，因为它是bottleneck的输入，不作为上采样的skip connection
        reversed_channels = list(reversed(channels[:-1]))
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
        """Kaiming初始化"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) -> (N,out_channels,H,W)"""
        shortcuts: List[torch.Tensor] = []
        x = self.inc(x)
        shortcuts.append(x)
        for down in self.down_blocks:
            x = down(x)
            shortcuts.append(x)

        x = self.bottleneck(x)

        # 同样跳过最后一个shortcut (bottleneck的输入)
        for skip, up in zip(reversed(shortcuts[:-1]), self.up_blocks):
            x = up(x, skip)

        return self.out_conv(x)
