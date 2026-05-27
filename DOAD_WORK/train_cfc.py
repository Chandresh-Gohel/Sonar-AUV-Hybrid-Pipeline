"""
=============================================================
  Train on CFC Source Dataset — All 4 Configurations

  Run:
    python train_cfc.py --mode yolov5n_mog
    python train_cfc.py --mode yolov8s_raw
    python train_cfc.py --mode yolov26s_mog
    python train_cfc.py --mode yolov26s_raw

  Output:
    runs/detect/yolov5n_cfc_mog2tvg/weights/best.pt
    runs/detect/yolov8s_cfc_raw/weights/best.pt
    runs/detect/yolov26s_cfc_mog2tvg/weights/best.pt
    runs/detect/yolov26s_cfc_raw/weights/best.pt
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, torch
from pathlib import Path
from ultralytics import YOLO
from utils import get_logger

from config import BASE_DIR, LOG_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

DEVICE   = 0 if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

CONFIGS = {
    "yolov5n_mog": {
        "weights":  "yolov5nu.pt",
        "yaml":     os.path.join(BASE_DIR, "cfc_yolo_mog2tvg.yaml"),
        "name":     "yolov5n_cfc_mog2tvg",
        "epochs":   75,
        "batch":    48,
        "patience": 10,
        "note":     "YOLOv5n on MOG2+TVG preprocessed CFC source",
    },
    "yolov8s_raw": {
        "weights":  "yolov8s.pt",
        "yaml":     os.path.join(BASE_DIR, "cfc_yolo_raw.yaml"),
        "name":     "yolov8s_cfc_raw",
        "epochs":   75,
        "batch":    48,
        "patience": 10,
        "note":     "YOLOv8s on raw Ch3 CFC source",
    },
    "yolov26s_mog": {
        "weights":  "yolo26s.pt",
        "yaml":     os.path.join(BASE_DIR, "cfc_yolo_mog2tvg.yaml"),
        "name":     "yolov26s_cfc_mog2tvg",
        "epochs":   75,
        "batch":    256,
        "patience": 10,
        "note":     "YOLOv26s on MOG2+TVG preprocessed CFC source",
    },
    "yolov26s_raw": {
        "weights":  "yolo26s.pt",
        "yaml":     os.path.join(BASE_DIR, "cfc_yolo_raw.yaml"),
        "name":     "yolov26s_cfc_raw",
        "epochs":   75,
        "batch":    256,
        "patience": 10,
        "note":     "YOLOv26s on raw Ch3 CFC source",
    },
}


def run(mode, logger):
    cfg = CONFIGS[mode]
    logger.info(f"\n{'='*55}")
    logger.info(f"  Training : {cfg['name']}")
    logger.info(f"  Data     : {cfg['yaml']}")
    logger.info(f"  Epochs   : {cfg['epochs']} | Batch: {cfg['batch']}")
    logger.info(f"  Note     : {cfg['note']}")
    logger.info(f"{'='*55}")

    if not Path(cfg["yaml"]).exists():
        logger.error(f"YAML not found: {cfg['yaml']}")
        logger.error("Run build_yolo_dataset.py first.")
        return None

    model = YOLO(cfg["weights"])
    model.train(
        data     = cfg["yaml"],
        epochs   = cfg["epochs"],
        imgsz    = IMG_SIZE,
        batch    = cfg["batch"],
        device   = DEVICE,
        patience = cfg["patience"],
        name     = cfg["name"],
        project  = OUTPUT_DIR,
        exist_ok = True,
        plots    = True,
        save     = True,
        val      = True,
        verbose  = True,
    )

    best_pt = Path(OUTPUT_DIR) / cfg["name"] / "weights" / "best.pt"
    if best_pt.exists():
        logger.info(f"\n  [DONE] {cfg['name']} -> {best_pt}")
        return str(best_pt)
    logger.warning(f"  best.pt not found at {best_pt}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("train_cfc", LOG_DIR)
    run(args.mode, logger)
