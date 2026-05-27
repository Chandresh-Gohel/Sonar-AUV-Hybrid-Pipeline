"""
=============================================================
  Source Model Retraining

  Retrains yolov5n (preprocessed) and yolov8s (raw)
  on source domain only — Kenai Channel excluded.

  Run once:
    python retrain_source.py

  Trains yolov5n first, then yolov8s sequentially.

  Output:
    runs/detect/yolov5n_source/weights/best.pt
    runs/detect/yolov8s_source/weights/best.pt
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
from ultralytics import YOLO
from config import BASE_DIR

OUTPUT_DIR  = os.path.join(BASE_DIR, "runs", "detect")
DEVICE      = 0 if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 512
EPOCHS      = 70
BATCH       = 256
PATIENCE    = 15

MODELS = [
    # {
    #     "name":         "yolov5n_source",
    #     "base_weights": "yolov5nu.pt",
    #     "yaml":         os.path.join(BASE_DIR, "sonar_dataset_source_pre.yaml"),
    #     "note":         "preprocessed pipeline — MOG2+TVG",
    # },
    {
        "name":         "yolov26s_source",
        "base_weights": "yolo26s.pt",
        "yaml":         os.path.join(BASE_DIR, "sonar_dataset_source_raw.yaml"),
        "note":         "raw pipeline",
    },
]


def train(cfg):
    print(f"\n{'='*55}")
    print(f"  Training : {cfg['name']}")
    print(f"  Data     : {cfg['yaml']}")
    print(f"  Note     : {cfg['note']}")
    print(f"{'='*55}")

    if not Path(cfg["yaml"]).exists():
        print(f"  [SKIP] YAML not found: {cfg['yaml']}")
        print(f"  Run filter_dataset.py first.")
        return None

    model   = YOLO(cfg["base_weights"])
    model.train(
        data       = cfg["yaml"],
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH,
        device     = DEVICE,
        patience   = PATIENCE,
        name       = cfg["name"],
        project    = OUTPUT_DIR,
        exist_ok   = True,
        plots      = True,
        save       = True,
        val        = True,
        verbose    = True,
    )

    best_pt = Path(OUTPUT_DIR) / cfg["name"] / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n  [DONE] {cfg['name']} -> {best_pt}")
        return str(best_pt)

    print(f"  [WARN] best.pt not found at {best_pt}")
    return None


if __name__ == "__main__":
    print("=" * 55)
    print("  Source Model Retraining")
    print("  yolov5n (pre) then yolov8s (raw)")
    print("=" * 55)

    results = {}
    for cfg in MODELS:
        pt = train(cfg)
        if pt:
            results[cfg["name"]] = pt

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)
    for name, path in results.items():
        print(f"  {name:<25}: {path}")
    print(f"\n  Next: python pseudo_label.py")
    print("=" * 55)