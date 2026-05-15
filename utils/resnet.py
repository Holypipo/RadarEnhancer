"""
utils/resnet.py
---------------
RadarResNet-v3: Residual Channel-Attention Network for radar super-resolution.

Kept in a separate module so that the rest of the pipeline can import
enhancement.py without requiring PyTorch to be installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel re-calibration."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _ResBlock(nn.Module):
    """Residual block with channel attention (RCAB)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = _ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.ca(self.conv(x))
        return x + res


class RadarResNet(nn.Module):
    """
    Residual Channel-Attention Network for 2× radar super-resolution.

    Input  : (B, 1, H,   W)    — normalised dBZ patch
    Output : (B, 1, H*2, W*2)  — super-resolved patch
    """

    def __init__(self, scale: int = 2, n_blocks: int = 8, channels: int = 64):
        super().__init__()
        self.scale = scale

        self.conv_in = nn.Conv2d(1, channels, 3, padding=1)

        self.body = nn.Sequential(*[_ResBlock(channels) for _ in range(n_blocks)])
        self.conv_body = nn.Conv2d(channels, channels, 3, padding=1)

        # PixelShuffle for scale=2; bicubic upsample otherwise
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=scale, mode="bicubic", align_corners=False
            )

        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        res  = self.conv_body(self.body(feat))
        feat = feat + res           # global residual
        return self.conv_out(self.upsample(feat))
