"""
=============================================================
  Pseudo-Label Generation — All 4 Configurations

  Run:
    python pseudo_label.py --mode yolov5n_mog
    python pseudo_label.py --mode yolov8s_raw
    python pseudo_label.py --mode yolov26s_mog
    python pseudo_label.py --mode yolov26s_raw

  Output:
    daod/pseudo_labels/raw/yolov5n_mog_source/
    daod/pseudo_labels/raw/yolov8s_raw_source/
    daod/pseudo_labels/raw/yolov26s_mog_source/
    daod/pseudo_labels/raw/yolov26s_raw_source/
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random, argparse
import numpy as np
import cv2, torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
from utils import get_sequence_key, xyxy_to_yolo, write_yolo_labels, get_image_size, save_json, get_logger
from preprocessing import MOG2TVGPreprocessor

from config import BASE_DIR, LOG_DIR
CHANNEL_TRAIN_DIR = os.path.join(BASE_DIR, "cfc_channel_train")
PSEUDO_DIR        = os.path.join(BASE_DIR, "daod", "pseudo_labels", "raw")

MAX_IMAGES     = 5000
CONF_THRESHOLD = 0.3
IMG_SIZE       = 512
DEVICE         = 0 if torch.cuda.is_available() else "cpu"
SEED           = 42

MODELS = {
    "yolov5n_mog": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov5n_cfc_mog2tvg", "weights", "best.pt"),
        "pipeline": "mog2tvg",
    },
    "yolov8s_raw": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov8s_cfc_raw", "weights", "best.pt"),
        "pipeline": "raw",
    },
    "yolov26s_mog": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_mog2tvg", "weights", "best.pt"),
        "pipeline": "mog2tvg",
    },
    "yolov26s_raw": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_raw", "weights", "best.pt"),
        "pipeline": "raw",
    },
}


def prepare_input(img_path, pipeline, preprocessor):
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None
    ch3 = frame[:, :, 2]
    if pipeline == "mog2tvg":
        return preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))
    return cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)


def run(mode, logger):
    cfg        = MODELS[mode]
    model_name = f"{mode}_source"
    out_dir    = Path(PSEUDO_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Model    : {model_name}")
    logger.info(f"  Weights  : {cfg['weights']}")
    logger.info(f"  Pipeline : {cfg['pipeline']}")
    logger.info(f"  Conf     : {CONF_THRESHOLD}")
    logger.info(f"{'='*55}")

    if not Path(cfg["weights"]).exists():
        logger.error(f"Weights not found: {cfg['weights']}")
        logger.error("Run train_cfc.py first.")
        return

    model        = YOLO(cfg["weights"])
    preprocessor = MOG2TVGPreprocessor()

    all_images = sorted(list(Path(CHANNEL_TRAIN_DIR).glob("*.jpg")) +
                        list(Path(CHANNEL_TRAIN_DIR).glob("*.png")))
    random.seed(SEED)
    if len(all_images) > MAX_IMAGES:
        all_images = random.sample(all_images, MAX_IMAGES)
    logger.info(f"  Images    : {len(all_images)}")

    seq_groups = defaultdict(list)
    for img_path in all_images:
        seq_key, frame_num = get_sequence_key(img_path.name)
        seq_groups[seq_key].append((frame_num, img_path))
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])
    logger.info(f"  Sequences : {len(seq_groups)}")

    total = 0; with_dets = 0; total_dets = 0; conf_list = []

    for seq_key, frames in tqdm(seq_groups.items(), desc=f"  {model_name}"):
        if cfg["pipeline"] == "mog2tvg":
            preprocessor.reset()
        for frame_num, img_path in frames:
            total += 1
            img_w, img_h = get_image_size(img_path)
            if img_w is None:
                continue
            processed = prepare_input(img_path, cfg["pipeline"], preprocessor)
            if processed is None:
                continue
            results    = model.predict(source=processed, conf=CONF_THRESHOLD,
                                       imgsz=IMG_SIZE, verbose=False, device=DEVICE)
            result     = results[0]
            pred_count = len(result.boxes) if result.boxes is not None else 0
            label_path = out_dir / (img_path.stem + ".txt")
            if pred_count == 0:
                label_path.write_text("")
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            yolo_labels = []
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                xc, yc, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                yolo_labels.append((0,
                    float(np.clip(xc,0,1)), float(np.clip(yc,0,1)),
                    float(np.clip(w, 0,1)), float(np.clip(h, 0,1))))
                conf_list.append(float(c))
            write_yolo_labels(str(label_path), yolo_labels)
            with_dets  += 1
            total_dets += pred_count

    det_rate = round(100 * with_dets / max(total, 1), 1)
    avg_conf = round(float(np.mean(conf_list)) if conf_list else 0.0, 4)
    stats = {"model": model_name, "total_images": total,
             "with_detections": with_dets, "detection_rate_%": det_rate,
             "total_pseudo_labels": total_dets, "avg_confidence": avg_conf}
    save_json(str(out_dir / "stats.json"), stats)

    logger.info(f"\n  Total images     : {total}")
    logger.info(f"  With detections  : {with_dets} ({det_rate}%)")
    logger.info(f"  Total labels     : {total_dets}")
    logger.info(f"  Avg confidence   : {avg_conf}")
    logger.info(f"  Output           : {out_dir}")
    logger.info(f"\n  Next: python temporal_filter.py --mode {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("pseudo_label", LOG_DIR)
    run(args.mode, logger)
