"""
=============================================================
  DAOD Fine-tuning — All 4 Configurations

  Run:
    python finetune.py --mode yolov5n_mog
    python finetune.py --mode yolov8s_raw
    python finetune.py --mode yolov26s_mog
    python finetune.py --mode yolov26s_raw

  Output:
    daod/weights/yolov5n_cfc_mog2tvg_daod/weights/best.pt
    daod/weights/yolov8s_cfc_raw_daod/weights/best.pt
    daod/weights/yolov26s_cfc_mog2tvg_daod/weights/best.pt
    daod/weights/yolov26s_cfc_raw_daod/weights/best.pt
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, torch
from pathlib import Path
from ultralytics import YOLO
from utils import get_logger

from config import BASE_DIR, LOG_DIR
WEIGHTS_DIR = os.path.join(BASE_DIR, "daod", "weights")

DEVICE   = 0 if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
EPOCHS   = 30
BATCH    = 48
PATIENCE = 10
LR0      = 0.001
FREEZE   = 10

CONFIGS = {
    "yolov5n_mog": {
        "source_weights": os.path.join(BASE_DIR, "runs", "detect",
                                       "yolov5n_cfc_mog2tvg", "weights", "best.pt"),
        "ft_name":        "yolov5n_cfc_mog2tvg_daod",
        "yaml":           os.path.join(BASE_DIR, "daod", "mixed_yolov5n_mog_source.yaml"),
    },
    "yolov8s_raw": {
        "source_weights": os.path.join(BASE_DIR, "runs", "detect",
                                       "yolov8s_cfc_raw", "weights", "best.pt"),
        "ft_name":        "yolov8s_cfc_raw_daod",
        "yaml":           os.path.join(BASE_DIR, "daod", "mixed_yolov8s_raw_source.yaml"),
    },
    "yolov26s_mog": {
        "source_weights": os.path.join(BASE_DIR, "runs", "detect",
                                       "yolov26s_cfc_mog2tvg", "weights", "best.pt"),
        "ft_name":        "yolov26s_cfc_mog2tvg_daod",
        "yaml":           os.path.join(BASE_DIR, "daod", "mixed_yolov26s_mog_source.yaml"),
    },
    "yolov26s_raw": {
        "source_weights": os.path.join(BASE_DIR, "runs", "detect",
                                       "yolov26s_cfc_raw", "weights", "best.pt"),
        "ft_name":        "yolov26s_cfc_raw_daod",
        "yaml":           os.path.join(BASE_DIR, "daod", "mixed_yolov26s_raw_source.yaml"),
    },
}


def run(mode, logger):
    cfg = CONFIGS[mode]
    logger.info(f"\n{'='*55}")
    logger.info(f"  Fine-tuning : {cfg['ft_name']}")
    logger.info(f"  From        : {cfg['source_weights']}")
    logger.info(f"  Data        : {cfg['yaml']}")
    logger.info(f"  Epochs      : {EPOCHS} | Freeze: {FREEZE} layers")
    logger.info(f"{'='*55}")

    if not Path(cfg["source_weights"]).exists():
        logger.error(f"Source weights not found: {cfg['source_weights']}")
        logger.error("Run train_cfc.py first.")
        return None
    if not Path(cfg["yaml"]).exists():
        logger.error(f"Mixed YAML not found: {cfg['yaml']}")
        logger.error("Run dataset_mixer.py first.")
        return None

    model = YOLO(cfg["source_weights"])
    model.train(
        data     = cfg["yaml"],
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        patience = PATIENCE,
        name     = cfg["ft_name"],
        project  = WEIGHTS_DIR,
        exist_ok = True,
        lr0      = LR0,
        freeze   = FREEZE,
        plots    = True,
        save     = True,
        val      = True,
        verbose  = True,
    )

    best_pt = Path(WEIGHTS_DIR) / cfg["ft_name"] / "weights" / "best.pt"
    if best_pt.exists():
        logger.info(f"\n  [DONE] {cfg['ft_name']} -> {best_pt}")
        return str(best_pt)
    logger.warning(f"  best.pt not found at {best_pt}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("finetune", LOG_DIR)
    run(args.mode, logger)
