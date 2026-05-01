"""
DCNv2 Bottleneck and C2f_DCN module for AquaDebris-Net.
Paper: Zhu et al., "Deformable ConvNets v2", CVPR 2019 (arXiv:1811.11168)

Replaces the standard 3x3 conv inside Bottleneck blocks with Deformable
Convolution v2. The network learns offsets that shift sampling positions
to match irregular debris shapes (crushed bottles, tangled fishing nets, etc.)
"""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from ultralytics.nn.modules.conv import Conv


class BottleneckDCN(nn.Module):
    """
    Bottleneck block where the second 3x3 conv is replaced by DCNv2.

    Standard Bottleneck:  1x1 Conv → 3x3 Conv → add
    BottleneckDCN:        1x1 Conv → DCNv2     → add

    DCNv2 learns 9 spatial offsets (dx, dy per sampling point) and
    9 modulation scalars, allowing the receptive field to deform to fit
    the actual shape of the detected object.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)

        # Predict offsets (18 channels: 9 x-offsets + 9 y-offsets)
        # and modulation masks (9 channels), total 27
        self.offset_conv = nn.Conv2d(c_, 3 * 3 * 3, 3, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.dcn = DeformConv2d(c_, c2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        feat = self.cv1(x)

        offset_mask = self.offset_conv(feat)
        offset = offset_mask[:, :18, :, :]                           # 2 * 9 offsets
        mask = torch.sigmoid(offset_mask[:, 18:, :, :])             # 9 modulation masks

        out = self.dcn(feat, offset, mask)
        out = self.act(self.bn(out))

        return out + identity if self.add else out


class C2f_DCN(nn.Module):
    """
    C2f module with BottleneckDCN instead of standard Bottleneck.

    Drop-in replacement for C2f in the YOLOv8 backbone at P4 and P5 levels.
    Deformable convolutions adapt their sampling grid to the irregular
    geometry of underwater debris, improving localization accuracy.

    Constructor is compatible with both:
    - Direct call: C2f_DCN(c1, c2, n, shortcut) — when in base_modules path
    - Else path:   C2f_DCN(c2_scaled, bool_shortcut) — auto-detects and adjusts
    """

    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        # Handle the case where ultralytics passes args as (c2_scaled, shortcut)
        # from the 'else: c2=ch[f]' branch (c2 ends up being bool or wrong type)
        if isinstance(c2, bool):
            shortcut = c2
            c2 = c1
        elif c2 is None:
            c2 = c1

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BottleneckDCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
