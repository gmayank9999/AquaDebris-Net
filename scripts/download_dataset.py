"""
Download and prepare the Trash-ICRA19 dataset.

Source: University of Minnesota Digital Conservancy
URL: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649

Run this script once before training:
    python scripts/download_dataset.py
"""

import os
import zipfile
import shutil
import urllib.request

DATASET_URL = (
    "https://conservancy.umn.edu/bitstreams/"
    "7a1d2b70-e088-4327-aaca-6cd53e06ad20/download"
)
ZIP_PATH = "Trash_ICRA19.zip"
EXTRACT_DIR = "Trash-ICRA19-raw"
FINAL_DIR = "Trash-ICRA19"


def download_with_progress(url, dest):
    print(f"Downloading dataset to {dest} ...")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def reorganize(raw_dir, final_dir):
    """
    Walk raw extracted folder and copy images + labels into
    the expected YOLO layout:
        final_dir/images/{train,val,test}/
        final_dir/labels/{train,val,test}/
    """
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(final_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(final_dir, "labels", split), exist_ok=True)

    for root, dirs, files in os.walk(raw_dir):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(root, raw_dir).replace("\\", "/").lower()

            # Determine split from path
            split = None
            for s in splits:
                if s in rel:
                    split = s
                    break
            if split is None:
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp"):
                dst = os.path.join(final_dir, "images", split, fname)
                shutil.copy2(src, dst)
            elif ext == ".txt":
                dst = os.path.join(final_dir, "labels", split, fname)
                shutil.copy2(src, dst)

    # Print counts
    for split in splits:
        imgs = len(os.listdir(os.path.join(final_dir, "images", split)))
        lbls = len(os.listdir(os.path.join(final_dir, "labels", split)))
        print(f"  {split}: {imgs} images, {lbls} labels")


def verify_dataset(final_dir):
    """Quick sanity check — every image should have a label file."""
    issues = 0
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(final_dir, "images", split)
        lbl_dir = os.path.join(final_dir, "labels", split)
        for fname in os.listdir(img_dir):
            stem = os.path.splitext(fname)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl):
                print(f"  WARNING: no label for {fname}")
                issues += 1
    if issues == 0:
        print("  All images have corresponding label files.")
    else:
        print(f"  {issues} images missing labels (may be unannotated background images).")


if __name__ == "__main__":
    if not os.path.exists(FINAL_DIR + "/images/train"):
        if not os.path.exists(ZIP_PATH):
            download_with_progress(DATASET_URL, ZIP_PATH)
        else:
            print(f"Found existing zip: {ZIP_PATH}")

        print("Extracting ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT_DIR)

        print("Reorganizing into YOLO format ...")
        reorganize(EXTRACT_DIR, FINAL_DIR)

        print("Verifying ...")
        verify_dataset(FINAL_DIR)

        # Clean up
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        print("Done! Dataset ready at:", FINAL_DIR)
    else:
        print("Dataset already exists. Verifying ...")
        verify_dataset(FINAL_DIR)
