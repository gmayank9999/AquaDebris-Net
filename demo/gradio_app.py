"""
Gradio web demo for AquaDebris-Net.

Loads the best trained model (Model D — full AquaDebris-Net) and
provides a simple web interface to upload an underwater image and
see detected objects with bounding boxes.

Run:
    python demo/gradio_app.py

Opens a local web server. The interface can also be shared publicly
via Gradio's built-in tunnel (share=True).
"""

import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Try best model first, fall back to alternatives
MODEL_PRIORITY = [
    "runs/detect/model_D_aquadebris/weights/best.pt",
    "runs/detect/model_C_cbam_dcn/weights/best.pt",
    "runs/detect/model_B_cbam/weights/best.pt",
    "runs/detect/model_A_baseline/weights/best.pt",
]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

CLASS_NAMES = {0: "plastic", 1: "bio", 2: "rov"}
CLASS_COLORS = {0: (255, 80, 80), 1: (80, 255, 80), 2: (80, 80, 255)}  # RGB


def load_model():
    from ultralytics import YOLO
    for path in MODEL_PRIORITY:
        if os.path.exists(path):
            print(f"Loading model: {path}")
            return YOLO(path), path
    raise FileNotFoundError(
        "No trained model found. Run one of the training scripts first.\n"
        "Quickest: python scripts/train_baseline.py"
    )


def detect(image: Image.Image, conf: float, iou: float):
    """Run detection on a PIL image and return annotated PIL image."""
    model, _ = _model_cache

    results = model(image, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    annotated = results[0].plot()                          # BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)


def get_detections_text(image: Image.Image, conf: float, iou: float) -> str:
    """Return human-readable detection summary."""
    model, model_path = _model_cache
    results = model(image, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return "No objects detected."

    lines = [f"Model: {Path(model_path).parts[-4]}", ""]
    for box in boxes:
        cls = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        lines.append(f"• {name:8s}  conf={conf_val:.2f}  bbox=[{x1},{y1},{x2},{y2}]")

    lines.append(f"\nTotal detections: {len(boxes)}")
    return "\n".join(lines)


def run_demo(image, conf_slider, iou_slider):
    if image is None:
        return None, "Please upload an image."
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    annotated = detect(pil, conf_slider, iou_slider)
    summary = get_detections_text(pil, conf_slider, iou_slider)
    return annotated, summary


if __name__ == "__main__":
    # Load model once at startup
    _model_cache = load_model()

    demo = gr.Interface(
        fn=run_demo,
        inputs=[
            gr.Image(type="numpy", label="Upload underwater image"),
            gr.Slider(0.1, 0.9, value=CONF_THRESHOLD, step=0.05,
                      label="Confidence threshold"),
            gr.Slider(0.1, 0.9, value=IOU_THRESHOLD, step=0.05,
                      label="IoU threshold (NMS)"),
        ],
        outputs=[
            gr.Image(type="pil", label="Detection result"),
            gr.Textbox(label="Detection summary", lines=10),
        ],
        title="AquaDebris-Net: Underwater Trash Detection",
        description=(
            "Upload an underwater image to detect objects.\n"
            "Classes: **plastic** (trash), **bio** (marine life), **rov** (robot).\n"
            "Based on YOLOv8s with CBAM + DCNv2 + CARAFE modifications."
        ),
        examples=[],
        allow_flagging="never",
    )

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
