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
import requests

DATASET_URL = (
    "https://conservancy.umn.edu/bitstreams/"
    "0239b06a-512e-49c3-80aa-ba33371e11de/download"
)
ITEM_PAGE = "https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649"
ZIP_PATH = "Trash_ICRA19.zip"
EXTRACT_DIR = "Trash-ICRA19-raw"
FINAL_DIR = "Trash-ICRA19"


def download_with_progress(url, dest):
    print(f"Downloading dataset to {dest} ...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    # Visit item page first to get session cookies
    session.get(ITEM_PAGE)
    r = session.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded*100/total:.1f}%  ({downloaded>>20} MB / {total>>20} MB)", end="", flush=True)
            else:
                print(f"\r  {downloaded>>20} MB downloaded", end="", flush=True)
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
    train_imgs = os.path.join(FINAL_DIR, "images", "train")
    has_images = os.path.exists(train_imgs) and len(os.listdir(train_imgs)) > 0
    if not has_images:
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
