"""
Train Model A: YOLOv8s baseline (no modifications).

This is the control group. All results from Models B, C, D are
compared against this baseline. Uses standard YOLOv8s pretrained
on COCO and fine-tuned on Trash-ICRA19.

Run:
    python scripts/train_baseline.py
"""

import sys
import os
from ultralytics import YOLO


class TeeLogger:
    """Tees stdout+stderr to a log file and the original streams simultaneously."""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._log.close()


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
    save_period=10,
    device=0,
    workers=8,
    seed=42,
    pretrained=True,
    verbose=True,
)

if __name__ == "__main__":
    logger = TeeLogger("logs/model_A_baseline.log")
    try:
        model = YOLO("yolov8s.pt")
        results = model.train(
            project="runs",
            name="model_A_baseline",
            exist_ok=True,
            **COMMON,
        )
        print("\n=== Model A (Baseline) Training Complete ===")
        print(f"Best weights: runs/detect/model_A_baseline/weights/best.pt")
    finally:
        logger.close()
