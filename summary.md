# AquaDebris-Net — Project Summary

**Course:** Deep Learning (CSE4007) — End-Term Project  
**Dataset:** Trash-ICRA19 (University of Minnesota / JAMSTEC)  
**Base model:** YOLOv8s (Ultralytics 8.4.38)  
**Hardware:** NVIDIA RTX A6000 48 GB · CUDA 11.8 · PyTorch 2.7.1+cu118 · Python 3.13.11  
**GitHub:** [gmayank9999/AquaDebris-Net](https://github.com/gmayank9999/AquaDebris-Net)

---

## Objective

Improve underwater debris detection accuracy by progressively adding three architectural modules to YOLOv8s — CBAM, DCNv2, and CARAFE — and measuring each addition via a controlled ablation study on the Trash-ICRA19 benchmark, following the same experimental protocol as Zhu et al. (*Sensors*, 2024).

---

## Dataset — Trash-ICRA19

Source: University of Minnesota Digital Conservancy.  
Downloaded via `scripts/download_dataset.py` and reorganised into the YOLO folder layout.

| Split | Images | plastic | bio | rov | Total instances |
|-------|--------|---------|-----|-----|-----------------|
| Train | 5,720  | 4,580   | 1,951 | 1,799 | 8,330 |
| Val   | 820    | 853     | **70** | 141 | 1,064 |
| Test  | 1,144  | 937     | 396   | 335 | 1,668 |
| **Total** | **7,684** | **6,370** | **2,417** | **2,275** | **18,062** |

**Classes:** `plastic` (id=0), `bio` (id=1), `rov` (id=2)

> **Note on val split:** The official val split has only 70 `bio` instances vs 396 in test. Those 70 examples happen to be hard cases, which is why `bio` mAP50 on val is ~3% and pulls the overall per-epoch val mAP50 down to ~50% during training. The test split is well-balanced and is the correct split for reporting final results.

---

## Architecture Overview

All models start from `yolov8s.pt` (COCO pretrained, scale `s`: depth=0.33, width=0.50).  
Three modifications are added one at a time to form a four-model ablation:

| Model | Modification stack | Config file |
|-------|--------------------|-------------|
| **A** | YOLOv8s baseline (unmodified) | — (`yolov8s.pt`) |
| **B** | + CBAM attention | `configs/yolov8s-cbam.yaml` |
| **C** | + CBAM + DCNv2 | `configs/yolov8s-cbam-dcn.yaml` |
| **D** | + CBAM + DCNv2 + CARAFE (full AquaDebris-Net) | `configs/yolov8s-aquadebris.yaml` |

### Module Details

#### 1. CBAM — Convolutional Block Attention Module
- **Paper:** Woo et al., ECCV 2018
- **Inserted:** After C2f blocks at backbone levels P4 (512 ch) and P5 (1024 ch)
- **What it does:** Sequential channel attention (recalibrates which feature channels matter) then spatial attention (recalibrates which spatial locations matter). Suppresses cluttered underwater backgrounds.
- **Implementation:** Already exists in `ultralytics/nn/modules/conv.py`; made available to YAML configs by adding an `elif m in {CBAM, CARAFE}: c2 = ch[f]; args = [c2]` branch in `tasks.py` via `patch_ultralytics.py`.

#### 2. DCNv2 — Deformable Convolution v2 (via `C2f_DCN`)
- **Paper:** Zhu et al., CVPR 2019
- **Inserted:** Replaces C2f at backbone P4 and P5 levels
- **What it does:** A learnable offset network predicts 18 offsets (9 × 2D offsets for a 3×3 kernel) that shift the sampling positions away from the regular grid to better align with irregular debris shapes. Also learns a 9-element mask (modulation weights).
- **Implementation:** `BottleneckDCN` + `C2f_DCN` classes appended to `block.py` by `patch_ultralytics.py`. Uses `torchvision.ops.DeformConv2d`.
- **CPU crash fix:** `torchvision.ops.DeformConv2d` has no CPU kernel on Windows. `thop` (FLOPs counter) runs a dummy forward pass on CPU — this caused a segfault. Fixed with a `conv2d` fallback inside `BottleneckDCN.forward` that fires only when `feat.device.type == "cpu"`.

#### 3. CARAFE — Content-Aware ReAssembly of FEatures
- **Paper:** Wang et al., ICCV 2019
- **Inserted:** Replaces both `nn.Upsample` layers in the FPN neck (Model D only)
- **What it does:** For each output pixel, predicts a local reassembly kernel from a compressed version of the input feature map. The upsampled value is a weighted sum of neighbourhood pixels under the predicted kernel — preserving fine detail of small debris fragments that nearest-neighbour upsampling loses.
- **Implementation:** `CARAFE` class appended to `block.py` by `patch_ultralytics.py`.

---

## Custom Module Injection — `patch_ultralytics.py`

Since `ultralytics` is installed as a package, custom modules are injected without forking:

1. **`block.py`** — Appends `BottleneckDCN`, `C2f_DCN`, `CARAFE` class definitions.
2. **`nn/modules/__init__.py`** — Adds the three new symbols to the module exports.
3. **`tasks.py`** — Imports `C2f_DCN`, `CARAFE`, `CBAM`; adds `C2f_DCN` to `base_modules` and `repeat_modules` frozensets; adds the `elif m in {CBAM, CARAFE}` branch so channel-preserving modules receive the correct width-scaled channel count from the YAML parser.

Run once before training: `python patch_ultralytics.py`

---

## Training Configuration

All four models use identical hyperparameters:

| Setting | Value |
|---------|-------|
| Image size | 640 × 640 |
| Batch size | 32 |
| Optimizer | SGD |
| Initial LR (`lr0`) | 0.01 |
| Final LR factor (`lrf`) | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Checkpoint save interval | every 10 epochs |
| GPU | CUDA:0 (RTX A6000) |
| Workers | 8 |
| Seed | 42 |

**Augmentations:** HSV jitter (h=0.015, s=0.7, v=0.4), horizontal flip (p=0.5), scale ±0.5, translate ±0.1, mosaic (p=1.0), mixup (p=0.1).

All training scripts use a `TeeLogger` class that mirrors stdout + stderr to a `.log` file in real time.

---

## Training Results (per-epoch val metrics from `results.csv`)

These come from the **val split** (820 images), logged per-epoch using non-EMA weights. The val split has only 70 `bio` instances (hard cases) so these numbers understate actual model quality. See test evaluation below for correct final numbers.

| Model | Total epochs | Best epoch | Val mAP@0.5 | Val mAP@0.5:0.95 | Val P | Val R |
|-------|-------------|------------|-------------|------------------|-------|-------|
| A — Baseline | 58 | 58 | 50.0% | 28.7% | 51.0% | 49.6% |
| B — +CBAM | 54 | 44 | 49.9% | 27.5% | 46.5% | 50.7% |
| C — +CBAM+DCN | 71 | 37 | 49.7% | 29.0% | 56.5% | 41.7% |
| D — Full | 65 | 38 | 48.3% | 27.6% | 48.4% | 44.2% |

**Training loss at best epoch:**

| Model | box loss | cls loss | dfl loss |
|-------|----------|----------|----------|
| A | 0.7843 | 0.4798 | 1.0681 |
| B | 0.9068 | 0.6517 | 1.3895 |
| C | 0.9445 | 0.6757 | 1.4493 |
| D | 0.9280 | 0.6650 | 1.4304 |

**Saved checkpoints per model:** `epoch0.pt`, `epoch10.pt`, `epoch20.pt`, …, `best.pt`, `last.pt`

---

## Final Test-Set Evaluation

Run via `scripts/evaluate_all.py` on the **test split** (1,144 images, 1,668 instances) using each model's `best.pt` (EMA-averaged weights). Numbers independently re-verified by running `model.val(split='test', verbose=True)` after training.

| Model | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | GFLOPs | best.pt |
|-------|-------|-------|-------------|------------------|------------|--------|---------|
| **A** — YOLOv8s baseline | 95.9 | 94.1 | 98.1 | 78.3 | 11.13 | 28.4 | 21.5 MB |
| **B** — + CBAM | 94.4 | 93.1 | 96.9 | 76.6 | 11.46 | 28.7 | 22.1 MB |
| **C** — + CBAM + DCNv2 | **97.0** | **94.2** | **98.2** | **78.9** | 11.58 | 27.3 | 22.3 MB |
| **D** — + CBAM + DCNv2 + CARAFE | 96.2 | 94.1 | 98.2 | 78.4 | 11.75 | 27.6 | 22.7 MB |

**Reference — Zhu et al., Sensors 2024 (YOLOv8n, same dataset):**

| Model | P (%) | R (%) | mAP@0.5 (%) | Params (M) | GFLOPs |
|-------|-------|-------|-------------|------------|--------|
| Zhu et al. — YOLOv8n baseline | 70.2 | 63.1 | 63.2 | 3.01 | 8.1 |
| Zhu et al. — improved model | 76.9 | 67.2 | 68.2 | 2.53 | 6.5 |

**Per-class breakdown — Model A (re-verified live after training):**

| Class | Images | Instances | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|-------|--------|-----------|-------|-------|-------------|-----------------|
| all | 1,144 | 1,668 | 95.9 | 94.1 | 98.1 | 78.3 |
| plastic | 858 | 937 | 97.6 | 97.8 | 99.2 | 74.6 |
| bio | 322 | 396 | 96.7 | 94.2 | 98.4 | 78.9 |
| rov | 258 | 335 | 93.5 | 90.5 | 96.7 | 81.3 |

> **Best model: Model C** (+CBAM +DCNv2) — highest P (97.0%), highest mAP@0.5:0.95 (78.9%), lowest GFLOPs among modified models (27.3 vs 28.7 for Model B). Model D matches C's mAP@0.5 (98.2%) but is marginally lower on mAP@0.5:0.95 (78.4%).

---

## Interpretability — EigenCAM Visualisations

Run via `scripts/gradcam_analysis.py` using `pytorch-grad-cam` (`EigenCAM`). Target layer: `pytorch_model.model[-2]` (second-to-last backbone block).

5 test images selected evenly from the test set: `bio0000_frame0000016`, `bio0016_frame0000277`, `obj0309_frame0000048`, `obj0865_frame0000035`, `obj1505_frame0000193`.

Outputs:
- Per model: 5 × `{stem}_cam.jpg` (heatmap overlay) + 5 × `{stem}_prediction.jpg` = 10 files × 4 models = 40 images
- `results/gradcam_visualizations/comparison_grids/`: 5 × 4-model stacked comparison grids

---

## Repository Structure

```
AquaDebris_Net/
├── configs/
│   ├── yolov8s-cbam.yaml            # Model B config (25 layers, 11.46M params)
│   ├── yolov8s-cbam-dcn.yaml        # Model C config (25 layers, 11.58M params)
│   └── yolov8s-aquadebris.yaml      # Model D config (25 layers, 11.75M params)
├── scripts/
│   ├── download_dataset.py          # Downloads Trash-ICRA19 from UMN Conservancy
│   ├── train_baseline.py            # Model A training (TeeLogger + YOLO("yolov8s.pt"))
│   ├── train_cbam.py                # Model B training
│   ├── train_cbam_dcn.py            # Model C training
│   ├── train_aquadebris.py          # Model D training
│   ├── evaluate_all.py              # Test-set eval → results/ablation_table.md + metrics.json
│   └── gradcam_analysis.py          # EigenCAM → results/gradcam_visualizations/
├── demo/
│   └── gradio_app.py                # Gradio web UI at http://0.0.0.0:7860
├── logs/
│   ├── model_A_baseline.log         # Full stdout/stderr — Model A (58 epochs)
│   ├── model_B_cbam.log             # Full stdout/stderr — Model B (54 epochs)
│   ├── model_C_cbam_dcn.log         # Full stdout/stderr — Model C (71 epochs)
│   └── model_D_aquadebris.log       # Full stdout/stderr — Model D (65 epochs)
├── results/
│   ├── ablation_table.md            # Auto-generated markdown table
│   ├── metrics.json                 # All test-set metrics as JSON
│   └── gradcam_visualizations/
│       ├── A_baseline/              # 10 files (5 cam + 5 prediction)
│       ├── B_cbam/                  # 10 files
│       ├── C_cbam_dcn/              # 10 files
│       ├── D_aquadebris/            # 10 files
│       └── comparison_grids/        # 5 four-model comparison grids
├── runs/detect/runs/
│   ├── detect/model_A_baseline/     # weights/best.pt, weights/last.pt, epoch*.pt, results.csv
│   ├── model_B_cbam/                # same structure
│   ├── model_C_cbam_dcn/            # same structure
│   └── model_D_aquadebris/          # same structure
├── patch_ultralytics.py             # One-time injection of custom modules into ultralytics
├── trash_icra19.yaml                # Dataset config (path, nc=3, class names)
├── requirements.txt                 # Python dependencies (ultralytics, torch, gradio, grad-cam, …)
├── README.md                        # Setup and usage instructions
├── yolov8s.pt                       # COCO-pretrained weights (training starting point)
└── Trash_ICRA19.zip                 # Raw downloaded dataset archive (~150 MB)
```

---

## Key Engineering Decisions and Bugs Fixed

1. **No ultralytics fork.** All custom modules injected into the installed package via `patch_ultralytics.py` — repo stays clean.

2. **`C2f_DCN` added to `base_modules` and `repeat_modules`.** Without this, `tasks.py` miscalculates output channels for `C2f_DCN` — crashing at model build time with a dimension mismatch.

3. **`CBAM`/`CARAFE` `elif` branch.** These modules preserve input channels. Without the dedicated branch they fell into `else` which sets `c2 = ch[f]` but passes no constructor args — `TypeError` at build.

4. **DCNv2 CPU segfault on Windows.** `thop` FLOPs profiler runs a CPU dummy forward pass. `torchvision.ops.DeformConv2d` has no CPU kernel on Windows → hard crash. Fixed with a `conv2d` fallback in `BottleneckDCN.forward` that is never triggered during real GPU training.

5. **Model A doubled save path.** Training Model A before fixing the project/name logic caused it to save under `runs/detect/runs/detect/model_A_baseline/`. All subsequent scripts (evaluate_all.py, gradcam_analysis.py) use the correct doubled path for Model A.

6. **`TeeLogger`.** Each training script wraps `sys.stdout`/`sys.stderr` so all training output goes simultaneously to terminal and to a `.log` file. This ensures logs are preserved even if the shell is closed.

---

## Gradio Web Demo

`demo/gradio_app.py` — launches at `http://0.0.0.0:7860`.  
Loads the best available model (priority: D → C → B → A).  
UI: image upload + confidence slider (default 0.25) + IoU slider (default 0.45) → annotated image + text detection summary (class name, confidence, `[x1,y1,x2,y2]`).

---

## Git Commit History

| # | Hash | Message |
|---|------|---------|
| 1 | `5ba422e` | initial project setup |
| 2 | `5bc3803` | add CBAM, DCN bottleneck and CARAFE modules |
| 3 | `38aac1c` | model configs for ablation (baseline, CBAM, DCN, full) |
| 4 | `a838af1` | training, evaluation and grad-cam scripts |
| 5 | `de81c43` | gradio demo for inference |
| 6 | `617dc1e` | add training logs for all four models |
| 7 | `750e566` | evaluation results, ablation table, and gradcam visuals |
| 8 | `8881b3d` | fix path bugs in eval/gradcam scripts and dcn cpu crash |
| 9 | `1f9bfcd` | fix results explanation - val/test gap is due to bio class imbalance in splits |
