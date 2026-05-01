"""
Evaluate all 4 models on the TEST set and print the ablation table.

Run AFTER all 4 models have been trained:
    python scripts/evaluate_all.py

Outputs:
  - Console: ablation table with P, R, mAP@0.5, mAP@0.5:0.95, Params, GFLOPs
  - results/ablation_table.md: same table in Markdown format
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO

MODELS = [
    ("Model A — YOLOv8s baseline",    "runs/detect/model_A_baseline/weights/best.pt"),
    ("Model B — + CBAM",              "runs/detect/model_B_cbam/weights/best.pt"),
    ("Model C — + CBAM + DCNv2",      "runs/detect/model_C_cbam_dcn/weights/best.pt"),
    ("Model D — + CBAM + DCNv2 + CARAFE", "runs/detect/model_D_aquadebris/weights/best.pt"),
]

PAPER_ROWS = [
    ("Zhu et al. YOLOv8n baseline", 70.2, 63.1, 63.2, None, 3.01, 8.1),
    ("Zhu et al. improved model",   76.9, 67.2, 68.2, None, 2.53, 6.5),
]

DATA_YAML = "trash_icra19.yaml"


def eval_model(name, weight_path):
    if not os.path.exists(weight_path):
        print(f"  [SKIP] {name}: weights not found at {weight_path}")
        return None

    model = YOLO(weight_path)
    metrics = model.val(data=DATA_YAML, split="test", verbose=False)

    # Parameter and FLOP count
    info = model.info(verbose=False)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    # GFLOPs from model info (index 1 of the tuple returned)
    try:
        gflops = info[1] / 1e9 if info and len(info) > 1 else 0
    except Exception:
        gflops = 0

    return {
        "name": name,
        "P":    round(metrics.box.mp * 100, 1),
        "R":    round(metrics.box.mr * 100, 1),
        "mAP50":   round(metrics.box.map50 * 100, 1),
        "mAP5095": round(metrics.box.map * 100, 1),
        "params": round(params_m, 2),
        "gflops": round(gflops, 1),
    }


def print_table(rows):
    header = (
        f"{'Model':<45} {'P':>7} {'R':>7} {'mAP@.5':>8} "
        f"{'mAP@.5:.95':>11} {'Params(M)':>10} {'GFLOPs':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        if r is None:
            continue
        map95 = f"{r['mAP5095']:.1f}" if r.get("mAP5095") else "  —  "
        print(
            f"{r['name']:<45} {r['P']:>7.1f} {r['R']:>7.1f} "
            f"{r['mAP50']:>8.1f} {map95:>11} "
            f"{r['params']:>10.2f} {r['gflops']:>8.1f}"
        )
    print(sep)


def write_markdown(rows):
    Path("results").mkdir(exist_ok=True)
    lines = [
        "# AquaDebris-Net — Ablation Study Results",
        "",
        "Evaluation on Trash-ICRA19 **test set** (1,144 images).",
        "Following the experimental setup of Zhu et al. (Sensors, 2024).",
        "",
        "| Model | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | GFLOPs |",
        "|---|---|---|---|---|---|---|",
    ]
    # Paper reference rows
    for name, P, R, map50, map5095, params, gflops in PAPER_ROWS:
        m95 = f"{map5095:.1f}" if map5095 else "—"
        lines.append(f"| {name} | {P} | {R} | {map50} | {m95} | {params} | {gflops} |")

    # Our model rows
    for r in rows:
        if r is None:
            continue
        m95 = f"{r['mAP5095']:.1f}" if r.get("mAP5095") else "—"
        lines.append(
            f"| **{r['name']}** | {r['P']} | {r['R']} | "
            f"{r['mAP50']} | {m95} | {r['params']} | {r['gflops']} |"
        )

    with open("results/ablation_table.md", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved: results/ablation_table.md")


if __name__ == "__main__":
    print("=== AquaDebris-Net Evaluation on Test Set ===")
    results = []
    for name, path in MODELS:
        print(f"\nEvaluating {name} ...")
        r = eval_model(name, path)
        results.append(r)

    print_table(results)
    write_markdown(results)

    # Also save as JSON for later use
    valid = [r for r in results if r]
    with open("results/metrics.json", "w") as f:
        json.dump(valid, f, indent=2)
    print("Saved: results/metrics.json")
