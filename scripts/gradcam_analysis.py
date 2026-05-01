"""
Grad-CAM interpretability analysis for all 4 AquaDebris-Net models.

Uses EigenCAM (fast, suitable for YOLO detection heads) from the grad-cam
library to produce class activation maps. Generates a comparison grid
showing how attention changes as modifications are added.

Run after training all 4 models:
    python scripts/gradcam_analysis.py

Outputs saved to: results/gradcam_visualizations/
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

try:
    from pytorch_grad_cam import EigenCAM, GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    try:
        from grad_cam import GradCAM
        HAS_GRADCAM = False
        print("Using fallback grad-cam. For best results: pip install grad-cam")
    except ImportError:
        HAS_GRADCAM = False
        print("Warning: grad-cam library not found. Install with: pip install grad-cam")


MODELS = [
    ("A_baseline",    "runs/detect/model_A_baseline/weights/best.pt"),
    ("B_cbam",        "runs/detect/model_B_cbam/weights/best.pt"),
    ("C_cbam_dcn",    "runs/detect/model_C_cbam_dcn/weights/best.pt"),
    ("D_aquadebris",  "runs/detect/model_D_aquadebris/weights/best.pt"),
]

OUTPUT_DIR = Path("results/gradcam_visualizations")
IMG_SIZE = 640


def find_test_images(n=5):
    """Grab n sample images from the test set."""
    test_dir = Path("Trash-ICRA19/images/test")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return []
    imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    # Pick evenly spaced samples
    step = max(1, len(imgs) // n)
    return [str(imgs[i]) for i in range(0, min(len(imgs), n * step), step)][:n]


def preprocess(img_path, size=IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (size, size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f32 = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0)
    return img_bgr, img_rgb, img_f32, tensor


def run_gradcam_ultralytics(model_path, img_paths, model_tag):
    """Use Ultralytics built-in activation maps (no extra library needed)."""
    model = YOLO(model_path)
    out_dir = OUTPUT_DIR / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        stem = Path(img_path).stem
        # Ultralytics predict with save
        results = model.predict(
            img_path,
            imgsz=IMG_SIZE,
            conf=0.25,
            save=False,
            verbose=False,
        )
        # Save annotated prediction
        annotated = results[0].plot()
        cv2.imwrite(str(out_dir / f"{stem}_prediction.jpg"), annotated)

    print(f"  Predictions saved to {out_dir}/")


def run_eigencam(model_path, img_paths, model_tag):
    """Run EigenCAM on the last backbone conv layer for each image."""
    if not HAS_GRADCAM:
        run_gradcam_ultralytics(model_path, img_paths, model_tag)
        return

    model = YOLO(model_path)
    pytorch_model = model.model
    pytorch_model.eval()

    out_dir = OUTPUT_DIR / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Target the last C2f block in the backbone (before SPPF)
    # For standard YOLOv8s: model.model.model[8] is the last C2f
    # For our custom models with CBAM, SPPF is at index 11 → last C2f is index 9
    try:
        target_layer = [pytorch_model.model[-2]]  # second to last backbone block
    except Exception:
        target_layer = [list(pytorch_model.modules())[-5]]

    try:
        cam = EigenCAM(pytorch_model, target_layer, use_cuda=torch.cuda.is_available())
    except Exception as e:
        print(f"  EigenCAM init failed ({e}), falling back to predictions only.")
        run_gradcam_ultralytics(model_path, img_paths, model_tag)
        return

    for img_path in img_paths:
        stem = Path(img_path).stem
        img_bgr, img_rgb, img_f32, tensor = preprocess(img_path)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        try:
            grayscale_cam = cam(input_tensor=tensor)[0]
            cam_image = show_cam_on_image(img_f32, grayscale_cam, use_rgb=True)
            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_cam.jpg"), cam_bgr)
        except Exception as e:
            print(f"  CAM generation failed for {stem}: {e}")

        # Also save plain prediction
        results = YOLO(model_path).predict(img_path, imgsz=IMG_SIZE, conf=0.25,
                                           save=False, verbose=False)
        cv2.imwrite(str(out_dir / f"{stem}_prediction.jpg"), results[0].plot())

    print(f"  CAM images saved to {out_dir}/")


def make_comparison_grid(img_paths):
    """
    Create side-by-side comparison grids showing CAM heatmaps for all 4 models
    on the same image. Layout: rows = models, cols = images.
    """
    grid_dir = OUTPUT_DIR / "comparison_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    tags = [m[0] for m in MODELS]

    for img_path in img_paths:
        stem = Path(img_path).stem
        rows = []
        for tag in tags:
            cam_file = OUTPUT_DIR / tag / f"{stem}_cam.jpg"
            pred_file = OUTPUT_DIR / tag / f"{stem}_prediction.jpg"
            if cam_file.exists():
                row_img = cv2.imread(str(cam_file))
            elif pred_file.exists():
                row_img = cv2.imread(str(pred_file))
            else:
                row_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            # Add model label
            row_img = cv2.resize(row_img, (IMG_SIZE, IMG_SIZE))
            cv2.putText(row_img, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            rows.append(row_img)

        if rows:
            grid = np.vstack(rows)
            out_path = grid_dir / f"{stem}_comparison.jpg"
            cv2.imwrite(str(out_path), grid)

    print(f"Comparison grids saved to {grid_dir}/")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_images = find_test_images(n=5)

    if not test_images:
        print("No test images found. Make sure dataset is downloaded.")
        exit(1)

    print(f"Using {len(test_images)} test images for Grad-CAM analysis.")

    for tag, weight_path in MODELS:
        if not os.path.exists(weight_path):
            print(f"\n[SKIP] {tag}: weights not found.")
            continue
        print(f"\nRunning EigenCAM for {tag} ...")
        run_eigencam(weight_path, test_images, tag)

    print("\nGenerating comparison grids ...")
    make_comparison_grid(test_images)

    print("\nGrad-CAM analysis complete.")
    print(f"Results: {OUTPUT_DIR}/")
