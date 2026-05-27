"""
=============================================================
  DAOD Evaluation — All 4 Configurations (Quick MAE only)

  Faster than evaluate_daod_full.py — no model.val() call.
  Use this for quick counting metrics during development.
  Use evaluate_daod_full.py for final thesis numbers.

  Run:
    python evaluate_daod.py --mode yolov5n_mog
    python evaluate_daod.py --mode yolov8s_raw
    python evaluate_daod.py --mode yolov26s_mog
    python evaluate_daod.py --mode yolov26s_raw

  Output:
    daod/results/daod_eval_yolov5n_mog.csv
    daod/results/daod_eval_yolov8s_raw.csv
    daod/results/daod_eval_yolov26s_mog.csv
    daod/results/daod_eval_yolov26s_raw.csv
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, argparse
import numpy as np
import cv2, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from utils import get_sequence_key, save_csv, save_json, get_logger
from preprocessing import MOG2TVGPreprocessor

from config import BASE_DIR, LOG_DIR
CHANNEL_TEST_DIR  = os.path.join(BASE_DIR, "cfc_channel_test")
CHANNEL_TEST_JSON = os.path.join(BASE_DIR, "cfc_channel_test.json")
RESULTS_DIR       = os.path.join(BASE_DIR, "daod", "results")

CONF_THRESHOLD = 0.4
IMG_SIZE       = 512
DEVICE         = 0 if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "yolov5n_mog": {
        "source":   os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov5n_cfc_mog2tvg", "weights", "best.pt"),
        "daod":     os.path.join(BASE_DIR, "daod", "weights",
                                 "yolov5n_cfc_mog2tvg_daod", "weights", "best.pt"),
        "pipeline": "mog2tvg",
    },
    "yolov8s_raw": {
        "source":   os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov8s_cfc_raw", "weights", "best.pt"),
        "daod":     os.path.join(BASE_DIR, "daod", "weights",
                                 "yolov8s_cfc_raw_daod", "weights", "best.pt"),
        "pipeline": "raw",
    },
    "yolov26s_mog": {
        "source":   os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_mog2tvg", "weights", "best.pt"),
        "daod":     os.path.join(BASE_DIR, "daod", "weights",
                                 "yolov26s_cfc_mog2tvg_daod", "weights", "best.pt"),
        "pipeline": "mog2tvg",
    },
    "yolov26s_raw": {
        "source":   os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_raw", "weights", "best.pt"),
        "daod":     os.path.join(BASE_DIR, "daod", "weights",
                                 "yolov26s_cfc_raw_daod", "weights", "best.pt"),
        "pipeline": "raw",
    },
}


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)
    id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in data["images"]}
    gt_counts  = defaultdict(int)
    for ann in data["annotations"]:
        bbox = ann["bbox"]
        if bbox[0] == -1 and bbox[1] == -1:
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        stem = id_to_stem.get(ann["image_id"], "")
        if stem:
            gt_counts[stem] += 1
    for img in data["images"]:
        stem = Path(img["file_name"]).stem
        if stem not in gt_counts:
            gt_counts[stem] = 0
    return dict(gt_counts)


def prepare_input(img_path, pipeline, preprocessor):
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None
    ch3 = frame[:, :, 2]
    if pipeline == "mog2tvg":
        return preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))
    return cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)


def evaluate_config(config_name, weight_path, pipeline, gt_counts, logger):
    logger.info(f"\n  {'='*50}")
    logger.info(f"  Config   : {config_name}")
    logger.info(f"  Weights  : {weight_path}")
    logger.info(f"  Pipeline : {pipeline}")

    if not Path(weight_path).exists():
        logger.warning(f"  [SKIP] Not found: {weight_path}")
        return None

    model        = YOLO(weight_path)
    preprocessor = MOG2TVGPreprocessor()

    all_images = sorted(list(Path(CHANNEL_TEST_DIR).glob("*.jpg")) +
                        list(Path(CHANNEL_TEST_DIR).glob("*.png")))
    if not all_images:
        logger.error(f"  No images in {CHANNEL_TEST_DIR}")
        return None
    logger.info(f"  Test images: {len(all_images)}")

    seq_groups = defaultdict(list)
    for img_path in all_images:
        seq_key, frame_num = get_sequence_key(img_path.name)
        seq_groups[seq_key].append((frame_num, img_path))
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])

    mae_errors = []; nmae_errors = []
    total_gt = 0; total_pred = 0
    missed = 0; false_pos = 0; total_frames = 0

    for seq_key, frames in tqdm(seq_groups.items(), desc=f"  {config_name}"):
        if pipeline == "mog2tvg":
            preprocessor.reset()
        for frame_num, img_path in frames:
            stem     = img_path.stem
            gt_count = gt_counts.get(stem, 0)
            processed = prepare_input(img_path, pipeline, preprocessor)
            if processed is None:
                continue
            results    = model.predict(source=processed, conf=CONF_THRESHOLD,
                                       imgsz=IMG_SIZE, verbose=False, device=DEVICE)
            pred_count = len(results[0].boxes) if results[0].boxes is not None else 0
            mae_errors.append(abs(pred_count - gt_count))
            if gt_count > 0:
                nmae_errors.append(abs(pred_count - gt_count) / gt_count)
            elif pred_count == 0:
                nmae_errors.append(0.0)
            total_gt   += gt_count; total_pred += pred_count; total_frames += 1
            if gt_count > 0 and pred_count == 0: missed    += 1
            if gt_count == 0 and pred_count > 0: false_pos += 1

    mae        = round(float(np.mean(mae_errors)),  4) if mae_errors  else 0.0
    nmae       = round(float(np.mean(nmae_errors)), 4) if nmae_errors else 0.0
    recall     = round(total_pred / max(total_gt, 1), 4)
    missed_pct = round(100 * missed    / max(total_frames, 1), 2)
    fp_pct     = round(100 * false_pos / max(total_frames, 1), 2)

    result = {"config": config_name, "MAE": mae, "nMAE": nmae,
              "Recall": recall, "missed_frame_%": missed_pct,
              "false_pos_%": fp_pct, "total_frames": total_frames,
              "total_gt_fish": total_gt, "total_pred_fish": total_pred}

    logger.info(f"\n  MAE          : {mae}")
    logger.info(f"  nMAE         : {nmae}")
    logger.info(f"  Recall       : {recall}")
    logger.info(f"  Missed_%     : {missed_pct}%")
    logger.info(f"  FalsePos_%   : {fp_pct}%")
    return result


def print_delta(source_r, daod_r, logger):
    logger.info(f"\n  {'='*55}")
    logger.info(f"  IMPROVEMENT: Source Only -> After DAOD")
    logger.info(f"  {'Metric':<20} {'Source':>10} {'DAOD':>10} {'Delta':>12}")
    logger.info(f"  {'-'*54}")
    for metric, direction in [("MAE","lower"),("nMAE","lower"),
                               ("Recall","higher"),("missed_frame_%","lower")]:
        s = source_r.get(metric, 0); d = daod_r.get(metric, 0)
        delta  = d - s
        better = "IMPROVED" if (delta < 0 if direction == "lower" else delta > 0) else "worse"
        logger.info(f"  {metric:<20} {s:>10} {d:>10} {delta:>+10.4f} {better}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("evaluate_daod", LOG_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cfg = CONFIGS[args.mode]
    logger.info(f"\n{'='*55}")
    logger.info(f"  DAOD Evaluation — {args.mode}")
    logger.info(f"  Channel test: {CHANNEL_TEST_DIR}")
    logger.info(f"{'='*55}")

    gt_counts = load_gt(CHANNEL_TEST_JSON)
    logger.info(f"  GT frames : {len(gt_counts)}")
    logger.info(f"  With fish : {sum(1 for v in gt_counts.values() if v > 0)}")
    logger.info(f"  Total fish: {sum(gt_counts.values())}")

    r_source = evaluate_config(f"{args.mode}_source_only",
                               cfg["source"], cfg["pipeline"], gt_counts, logger)
    r_daod   = evaluate_config(f"{args.mode}_daod",
                               cfg["daod"],   cfg["pipeline"], gt_counts, logger)

    if r_source and r_daod:
        print_delta(r_source, r_daod, logger)

    results = [r for r in [r_source, r_daod] if r]
    if results:
        out = os.path.join(RESULTS_DIR, f"daod_eval_{args.mode}.csv")
        save_csv(out, results)
        logger.info(f"\n  [SAVED] {out}")
    logger.info("\n[DONE]")
