"""
CBAM: Convolutional Block Attention Module
Paper: Woo et al., ECCV 2018 (arXiv:1807.06521)

Standalone implementation for documentation purposes.
The actual registered CBAM in ultralytics uses the built-in conv.py version
(ChannelAttention + SpatialAttention already present in ultralytics).
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Sequentially applies channel attention (WHAT to focus on) then spatial
    attention (WHERE to focus), refining feature maps for better detection.

    Placed after C2f blocks at P4 and P5 backbone levels in AquaDebris-Net
    to suppress cluttered underwater backgrounds and amplify debris features.
    """

    def __init__(self, c1, kernel_size=7, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(c1, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))
