"""
CARAFE: Content-Aware ReAssembly of FEatures
Paper: Wang et al., ICCV 2019 (arXiv:1905.02188)

Replaces nn.Upsample (nearest-neighbor) in the YOLOv8 neck with a
content-aware upsampling module. For each output pixel, CARAFE predicts
a small reassembly kernel from the local feature content and uses it to
compute that pixel as a weighted sum of nearby input pixels.

This preserves fine details of small debris during feature upsampling,
improving detection of small plastic fragments and debris pieces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """
    Content-Aware ReAssembly of FEatures upsampling module.

    Replaces nn.Upsample in the FPN/PAN neck of YOLOv8.

    Args:
        c1 (int): Input channel count. Output channels == c1 (no change).
        scale_factor (int): Upsampling scale. Default 2 (same as nn.Upsample).
        k_up (int): Size of the reassembly kernel. Default 5.
        k_enc (int): Kernel size for the kernel encoder conv. Default 3.
        compressed_channels (int): Intermediate channel compression. Default 64.
    """

    def __init__(self, c1, scale_factor=2, k_up=5, k_enc=3, compressed_channels=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.k_up = k_up

        # Step 1: Channel compressor — reduce to compressed_channels
        self.comp = nn.Conv2d(c1, compressed_channels, 1, bias=False)
        self.comp_bn = nn.BatchNorm2d(compressed_channels)

        # Step 2: Kernel encoder — predict (scale*k_up)^2 kernel weights per location
        self.enc = nn.Conv2d(
            compressed_channels,
            (scale_factor * k_up) ** 2,
            k_enc,
            padding=k_enc // 2,
            bias=False,
        )
        self.enc_bn = nn.BatchNorm2d((scale_factor * k_up) ** 2)

        # PixelShuffle rearranges (B, (s*k)^2, H, W) → (B, k^2, H*s, W*s)
        self.pix_shf = nn.PixelShuffle(scale_factor)

        # Upsampler for content, unfold for local patch extraction
        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, padding=k_up // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = h * self.scale_factor
        w_ = w * self.scale_factor

        # Predict per-location reassembly kernels from compressed features
        W = F.relu(self.comp_bn(self.comp(x)), inplace=True)  # (B, C_comp, H, W)
        W = self.enc_bn(self.enc(W))                           # (B, (s*k)^2, H, W)
        W = self.pix_shf(W)                                    # (B, k^2, H*s, W*s)
        W = F.softmax(W, dim=1)                                # normalise kernels

        # Extract local patches from nearest-neighbour upsampled input
        x_up = self.upsmp(x)                                   # (B, C, H*s, W*s)
        x_unfold = self.unfold(x_up)                           # (B, C*k^2, H*s*W*s)
        x_unfold = x_unfold.view(b, c, self.k_up ** 2, h_, w_)  # (B,C,k^2,H*s,W*s)

        # Weighted reassembly: sum over kernel positions
        W = W.unsqueeze(1)                                     # (B, 1, k^2, H*s, W*s)
        out = (x_unfold * W).sum(dim=2)                        # (B, C, H*s, W*s)

        return out
