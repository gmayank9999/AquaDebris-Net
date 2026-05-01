"""
Train Model B: YOLOv8s + CBAM.

Adds Convolutional Block Attention Module (Woo et al., ECCV 2018) after the
C2f blocks at P4 and P5 backbone levels. CBAM performs sequential channel
attention (WHAT to focus on) and spatial attention (WHERE to focus).

Requires patch_ultralytics.py to have been run first.

Run:
    python patch_ultralytics.py   # once
    python scripts/train_cbam.py
"""

from ultralytics import YOLO

COMMON = dict(
    data="trash_icra19.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    patience=20,
    device=0,
    workers=8,
    seed=42,
    verbose=True,
)

if __name__ == "__main__":
    # Load custom architecture
    model = YOLO("configs/yolov8s-cbam.yaml")
    # Transfer pretrained COCO weights for all matching layers
    model.load("yolov8s.pt")

    results = model.train(
        project="runs/detect",
        name="model_B_cbam",
        exist_ok=True,
        **COMMON,
    )
    print("\n=== Model B (CBAM) Training Complete ===")
    print(f"Best weights: runs/detect/model_B_cbam/weights/best.pt")
