"""
patch_ultralytics.py — One-time setup script for AquaDebris-Net.

Injects custom modules (C2f_DCN, CARAFE, CBAM) into the installed
ultralytics package so they can be referenced by name in YAML model configs.

Run ONCE before training:
    python patch_ultralytics.py

What this script does:
1. Appends BottleneckDCN, C2f_DCN, CARAFE classes to ultralytics block.py
2. Exports them from ultralytics/nn/modules/__init__.py
3. Imports them in tasks.py and adds C2f_DCN to parse_model's
   base_modules + repeat_modules frozensets (so channels are handled correctly)
4. Adds CBAM to tasks.py globals so it's found by name in YAML configs
   (CBAM already exists in conv.py, just needs to be in parse_model scope)
"""

import sys
import importlib
import ultralytics


def get_ultralytics_path():
    import os
    return os.path.dirname(ultralytics.__file__)


def patch_block_py(base_path):
    """Append BottleneckDCN, C2f_DCN, CARAFE to block.py if not already there."""
    block_path = f"{base_path}/nn/modules/block.py"
    content = open(block_path, encoding="utf-8").read()

    if "class BottleneckDCN" in content:
        print("  block.py: already patched, skipping.")
        return

    code_to_append = '''

# =============================================================================
# AquaDebris-Net Custom Modules — appended by patch_ultralytics.py
# =============================================================================

from torchvision.ops import DeformConv2d as _DeformConv2d
import torch.nn.functional as _F


class BottleneckDCN(nn.Module):
    """Bottleneck with Deformable Convolution v2 (Zhu et al., CVPR 2019).
    Replaces the second 3x3 conv with DCNv2 for shape-adaptive feature sampling."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.offset_conv = nn.Conv2d(c_, 27, 3, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        self.dcn = _DeformConv2d(c_, c2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        feat = self.cv1(x)
        om = self.offset_conv(feat)
        out = self.dcn(feat, om[:, :18], torch.sigmoid(om[:, 18:]))
        out = self.act(self.bn(out))
        return out + identity if self.add else out


class C2f_DCN(nn.Module):
    """C2f with BottleneckDCN for deformable feature extraction.
    Drop-in replacement for C2f at P4/P5 backbone levels."""

    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        if isinstance(c2, bool):
            shortcut, c2 = c2, c1
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


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (Wang et al., ICCV 2019).
    Replaces nn.Upsample in the FPN/PAN neck for content-aware upsampling."""

    def __init__(self, c1, scale_factor=2, k_up=5, k_enc=3, compressed_channels=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.k_up = k_up
        self.comp = nn.Conv2d(c1, compressed_channels, 1, bias=False)
        self.comp_bn = nn.BatchNorm2d(compressed_channels)
        self.enc = nn.Conv2d(compressed_channels, (scale_factor * k_up) ** 2,
                             k_enc, padding=k_enc // 2, bias=False)
        self.enc_bn = nn.BatchNorm2d((scale_factor * k_up) ** 2)
        self.pix_shf = nn.PixelShuffle(scale_factor)
        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, padding=k_up // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        h_, w_ = h * self.scale_factor, w * self.scale_factor
        W = _F.relu(self.comp_bn(self.comp(x)), inplace=True)
        W = self.enc_bn(self.enc(W))
        W = _F.softmax(self.pix_shf(W), dim=1)
        x_unfold = self.unfold(self.upsmp(x)).view(b, c, self.k_up ** 2, h_, w_)
        return (x_unfold * W.unsqueeze(1)).sum(dim=2)
'''
    with open(block_path, "a", encoding="utf-8") as f:
        f.write(code_to_append)
    print("  block.py: appended BottleneckDCN, C2f_DCN, CARAFE.")


def patch_modules_init(base_path):
    """Add C2f_DCN, BottleneckDCN, CARAFE to nn/modules/__init__.py exports."""
    init_path = f"{base_path}/nn/modules/__init__.py"
    content = open(init_path, encoding="utf-8").read()

    if "C2f_DCN" in content:
        print("  nn/modules/__init__.py: already patched, skipping.")
        return

    # Add to block imports
    old_import = "    RepNCSPELAN4,"
    new_import = "    RepNCSPELAN4,\n    BottleneckDCN,\n    C2f_DCN,\n    CARAFE,"
    content = content.replace(old_import, new_import, 1)

    # Add to __all__
    old_all = '    "RepNCSPELAN4",'
    new_all = '    "RepNCSPELAN4",\n    "BottleneckDCN",\n    "C2f_DCN",\n    "CARAFE",'
    content = content.replace(old_all, new_all, 1)

    with open(init_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  nn/modules/__init__.py: added C2f_DCN, CARAFE exports.")


def patch_tasks_py(base_path):
    """
    Modify tasks.py to:
    1. Import CBAM (from conv), C2f_DCN, CARAFE (from block)
    2. Add C2f_DCN to base_modules frozenset
    3. Add C2f_DCN to repeat_modules frozenset
    """
    tasks_path = f"{base_path}/nn/tasks.py"
    content = open(tasks_path, encoding="utf-8").read()

    if "C2f_DCN" in content:
        print("  tasks.py: already patched, skipping.")
        return

    # 1. Add imports after the existing nn.modules import block
    old_import_end = "from ultralytics.utils import DEFAULT_CFG_DICT"
    new_import_block = (
        "from ultralytics.nn.modules.block import BottleneckDCN, C2f_DCN, CARAFE  "
        "# AquaDebris-Net\n"
        "from ultralytics.nn.modules.conv import CBAM  # AquaDebris-Net\n"
        "from ultralytics.utils import DEFAULT_CFG_DICT"
    )
    content = content.replace(old_import_end, new_import_block, 1)

    # 2. Add C2f_DCN to base_modules frozenset (after A2C2f entry)
    #    The base_modules set ends with: A2C2f,\n        }\n    )\n    repeat_modules
    old_base = "            A2C2f,\n        }\n    )\n    repeat_modules = frozenset("
    new_base = "            A2C2f,\n            C2f_DCN,  # AquaDebris-Net\n        }\n    )\n    repeat_modules = frozenset("
    content = content.replace(old_base, new_base, 1)

    # 3. Add C2f_DCN to repeat_modules frozenset
    #    repeat_modules ends with: A2C2f,\n        }\n    )\n    for i
    old_repeat = "            A2C2f,\n        }\n    )\n    for i, (f, n, m, args)"
    new_repeat = "            A2C2f,\n            C2f_DCN,  # AquaDebris-Net\n        }\n    )\n    for i, (f, n, m, args)"
    content = content.replace(old_repeat, new_repeat, 1)

    # 4. Add elif branch for CBAM/CARAFE so they receive width-scaled channels
    old_else = (
        "        elif m in frozenset({TorchVision, Index}):\n"
        "            c2 = args[0]\n"
        "            c1 = ch[f]\n"
        "            args = [*args[1:]]\n"
        "        else:\n"
        "            c2 = ch[f]"
    )
    new_else = (
        "        elif m in frozenset({TorchVision, Index}):\n"
        "            c2 = args[0]\n"
        "            c1 = ch[f]\n"
        "            args = [*args[1:]]\n"
        "        elif m in {CBAM, CARAFE}:  # AquaDebris-Net: channel-preserving modules\n"
        "            c2 = ch[f]\n"
        "            args = [c2]            # pass actual (width-scaled) channel count\n"
        "        else:\n"
        "            c2 = ch[f]"
    )
    content = content.replace(old_else, new_else, 1)

    with open(tasks_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  tasks.py: imported C2f_DCN/CARAFE/CBAM, added C2f_DCN to base/repeat_modules, CBAM/CARAFE elif.")


def verify():
    """Reload ultralytics and check our modules are accessible."""
    # Force reload
    mods_to_reload = [k for k in sys.modules if k.startswith("ultralytics")]
    for m in mods_to_reload:
        sys.modules.pop(m, None)

    try:
        from ultralytics.nn.modules.block import C2f_DCN, CARAFE, BottleneckDCN
        from ultralytics.nn.modules.conv import CBAM
        print("  Verification PASSED: C2f_DCN, CARAFE, BottleneckDCN, CBAM all importable.")
    except ImportError as e:
        print(f"  Verification FAILED: {e}")
        sys.exit(1)

    try:
        import ultralytics.nn.tasks as t
        assert hasattr(t, "C2f_DCN"), "C2f_DCN not in tasks globals"
        assert hasattr(t, "CARAFE"), "CARAFE not in tasks globals"
        assert hasattr(t, "CBAM"), "CBAM not in tasks globals"
        print("  Verification PASSED: modules visible in tasks.py globals.")
    except AssertionError as e:
        print(f"  Verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=== AquaDebris-Net: Patching ultralytics ===")
    base = get_ultralytics_path()
    print(f"Ultralytics path: {base}")

    patch_block_py(base)
    patch_modules_init(base)
    patch_tasks_py(base)
    verify()

    print("\nDone! Custom modules are ready.")
    print("You can now run any training script.")
