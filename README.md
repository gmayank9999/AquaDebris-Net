# AquaDebris-Net

**Deep Learning (CSE4007) End-Term Project**

Underwater debris detection using a modified YOLOv8s with three progressive architectural improvements: CBAM attention, DCNv2 deformable convolutions, and CARAFE content-aware upsampling.

Follows the experimental setup of Zhu et al., *"YOLOv8-C2f-Faster-EMA"*, Sensors 2024 — same dataset, same classes, same metrics — but proposes complementary modifications focused on **accuracy** rather than efficiency.

---

## Modifications

| # | Module | Where | Why |
|---|--------|--------|-----|
| 1 | **CBAM** (Woo et al., ECCV 2018) | After C2f @ P4 and P5 backbone | Suppresses cluttered underwater backgrounds; attends to debris features |
| 2 | **DCNv2** (Zhu et al., CVPR 2019) | Inside C2f Bottlenecks @ P4 and P5 | Deforms sampling grid to match irregular debris shapes |
| 3 | **CARAFE** (Wang et al., ICCV 2019) | Replaces nn.Upsample in FPN neck | Content-aware upsampling preserves small debris details |

## Ablation Study

| Model | Modification | mAP@0.5 |
|-------|-------------|---------|
| A | YOLOv8s baseline | TBD |
| B | + CBAM | TBD |
| C | + CBAM + DCNv2 | TBD |
| D | + CBAM + DCNv2 + CARAFE | TBD |

*(Fill in after training)*

---

## Dataset

**Trash-ICRA19** — University of Minnesota / JAMSTEC deep-sea ROV footage.

- 7,684 annotated images · 3 classes: `plastic`, `bio`, `rov`
- Official splits: Train 5,720 · Val 820 · Test 1,144
- Download: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare dataset
python scripts/download_dataset.py

# 3. Patch ultralytics with custom modules (run once)
python patch_ultralytics.py

# 4. Train all 4 models sequentially
python scripts/train_baseline.py
python scripts/train_cbam.py
python scripts/train_cbam_dcn.py
python scripts/train_aquadebris.py

# 5. Evaluate on test set
python scripts/evaluate_all.py

# 6. Grad-CAM visualization
python scripts/gradcam_analysis.py

# 7. Run web demo
python demo/gradio_app.py
```

---

## Hardware

Tested on NVIDIA RTX A6000 (48 GB VRAM). Training time per model: ~1–2 hours.

---

## Repository Structure

```
AquaDebris-Net/
├── patch_ultralytics.py         # One-time setup: injects custom modules
├── trash_icra19.yaml            # Dataset config
├── requirements.txt
├── configs/
│   ├── yolov8s-cbam.yaml        # Model B
│   ├── yolov8s-cbam-dcn.yaml    # Model C
│   └── yolov8s-aquadebris.yaml  # Model D (full)
├── ultralytics_custom/
│   ├── cbam.py                  # CBAM implementation
│   ├── dcn_bottleneck.py        # BottleneckDCN + C2f_DCN
│   └── carafe.py                # CARAFE upsampling
├── scripts/
│   ├── download_dataset.py
│   ├── train_baseline.py        # Model A
│   ├── train_cbam.py            # Model B
│   ├── train_cbam_dcn.py        # Model C
│   ├── train_aquadebris.py      # Model D
│   ├── evaluate_all.py
│   └── gradcam_analysis.py
├── demo/
│   └── gradio_app.py
└── results/
    ├── ablation_table.md        # Auto-generated after eval
    ├── confusion_matrices/
    ├── pr_curves/
    └── gradcam_visualizations/
```

---

## Reference Paper

> Jin Zhu et al., *"YOLOv8-C2f-Faster-EMA: An Improved Underwater Trash Detection Model Based on YOLOv8"*, Sensors (MDPI), Vol. 24, No. 8, April 2024.  
> DOI: [10.3390/s24082483](https://doi.org/10.3390/s24082483)  
> Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11054227/

---

## License

Academic use only. Dataset licensed under JAMSTEC terms.
