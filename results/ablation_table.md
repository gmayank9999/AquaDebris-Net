# AquaDebris-Net � Ablation Study Results

Evaluation on Trash-ICRA19 **test set** (1,144 images).
Following the experimental setup of Zhu et al. (Sensors, 2024).

| Model | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| Zhu et al. YOLOv8n baseline | 70.2 | 63.1 | 63.2 | � | 3.01 | 8.1 |
| Zhu et al. improved model | 76.9 | 67.2 | 68.2 | � | 2.53 | 6.5 |
| **Model A — YOLOv8s baseline** | 95.9 | 94.1 | 98.1 | 78.3 | 11.13 | 28.4 |
| **Model B — + CBAM** | 94.4 | 93.1 | 96.9 | 76.6 | 11.46 | 28.7 |
| **Model C — + CBAM + DCNv2** | 97.0 | 94.2 | 98.2 | 78.9 | 11.58 | 27.3 |
| **Model D — + CBAM + DCNv2 + CARAFE** | 96.2 | 94.1 | 98.2 | 78.4 | 11.75 | 27.6 |