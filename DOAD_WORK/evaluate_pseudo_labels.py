"""
=============================================================
  Pseudo-Label Quality Evaluator — All 4 Configurations

  Run:
    python evaluate_pseudo_labels.py --mode yolov5n_mog
    python evaluate_pseudo_labels.py --mode yolov8s_raw
    python evaluate_pseudo_labels.py --mode yolov26s_mog
    python evaluate_pseudo_labels.py --mode yolov26s_raw

  Output:
    daod/results/pseudo_label_quality_yolov5n_mog_source.csv
    daod/results/pseudo_label_quality_yolov8s_raw_source.csv
    daod/results/pseudo_label_quality_yolov26s_mog_source.csv
    daod/results/pseudo_label_quality_yolov26s_raw_source.csv
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils import get_logger, save_csv, save_json

from config import BASE_DIR, LOG_DIR
CHANNEL_TRAIN_JSON = os.path.join(BASE_DIR, "cfc_channel_train.json")
PSEUDO_RAW_DIR     = os.path.join(BASE_DIR, "daod", "pseudo_labels", "raw")
PSEUDO_FILT_DIR    = os.path.join(BASE_DIR, "daod", "pseudo_labels", "filtered")
RESULTS_DIR        = os.path.join(BASE_DIR, "daod", "results")

# All 4 valid modes
VALID_MODES = ["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"]


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    gt_counts  = defaultdict(int)
    for ann in data["annotations"]:
        bbox = ann["bbox"]
        if bbox[0] == -1 and bbox[1] == -1:
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        fname = id_to_file.get(ann["image_id"], "")
        if fname:
            gt_counts[fname] += 1
    for img in data["images"]:
        if img["file_name"] not in gt_counts:
            gt_counts[img["file_name"]] = 0
    return dict(gt_counts)


def load_pseudo_counts(labels_dir):
    counts = {}
    for txt_file in Path(labels_dir).glob("*.txt"):
        with open(txt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        counts[txt_file.stem] = len(lines)
    return counts


def match_filenames(gt_counts, pseudo_counts):
    gt_by_stem = {Path(fname).stem: count for fname, count in gt_counts.items()}
    matched = []; matched_stems = set()
    unmatched_pred = 0
    for stem, pred_count in pseudo_counts.items():
        if stem in gt_by_stem:
            matched.append((gt_by_stem[stem], pred_count))
            matched_stems.add(stem)
        else:
            unmatched_pred += 1
    unmatched_gt = len(gt_by_stem) - len(matched_stems)
    return matched, unmatched_gt, unmatched_pred


def compute_metrics(matched_pairs):
    if not matched_pairs:
        return {}
    gt_counts   = [p[0] for p in matched_pairs]
    pred_counts = [p[1] for p in matched_pairs]
    errors      = [abs(p - g) for g, p in matched_pairs]
    mae         = float(np.mean(errors))
    nmae_errors = [abs(p-g)/g for g, p in matched_pairs if g > 0]
    nmae        = float(np.mean(nmae_errors)) if nmae_errors else 0.0
    pred_pos    = [(g, p) for g, p in matched_pairs if p > 0]
    precision   = sum(1 for g, p in pred_pos if g > 0) / max(len(pred_pos), 1)
    gt_pos      = [(g, p) for g, p in matched_pairs if g > 0]
    recall      = sum(1 for g, p in gt_pos if p > 0) / max(len(gt_pos), 1)
    frames_with_pred = sum(1 for p in pred_counts if p > 0)
    return {
        "matched_frames":   len(matched_pairs),
        "frames_with_gt":   sum(1 for g in gt_counts if g > 0),
        "frames_with_pred": frames_with_pred,
        "total_gt_fish":    sum(gt_counts),
        "total_pred_fish":  sum(pred_counts),
        "MAE":              round(mae,       4),
        "nMAE":             round(nmae,      4),
        "frame_precision":  round(precision, 4),
        "frame_recall":     round(recall,    4),
        "detection_rate_%": round(100 * frames_with_pred / max(len(matched_pairs), 1), 1),
    }


def evaluate(model_name, labels_dir, gt_counts, stage, logger):
    pseudo_counts = load_pseudo_counts(labels_dir)
    if not pseudo_counts:
        logger.warning(f"  No labels found in {labels_dir}")
        return None
    matched, unmatched_gt, unmatched_pred = match_filenames(gt_counts, pseudo_counts)
    metrics = compute_metrics(matched)
    logger.info(f"\n  [{stage}] {model_name}")
    logger.info(f"    Matched frames   : {metrics.get('matched_frames', 0)}")
    logger.info(f"    Total GT fish    : {metrics.get('total_gt_fish', 0)}")
    logger.info(f"    Total pred fish  : {metrics.get('total_pred_fish', 0)}")
    logger.info(f"    MAE              : {metrics.get('MAE', 0)}")
    logger.info(f"    Frame Precision  : {metrics.get('frame_precision', 0)}")
    logger.info(f"    Frame Recall     : {metrics.get('frame_recall', 0)}")
    logger.info(f"    Detection rate   : {metrics.get('detection_rate_%', 0)}%")
    logger.info(f"    Unmatched GT     : {unmatched_gt}")
    logger.info(f"    Unmatched pred   : {unmatched_pred}")
    metrics["model"] = model_name
    metrics["stage"] = stage
    return metrics


def run(mode, logger):
    model_name = f"{mode}_source"
    logger.info(f"\n{'='*55}")
    logger.info(f"  Pseudo-Label Quality Evaluation")
    logger.info(f"  Model : {model_name}")
    logger.info(f"{'='*55}")

    gt_counts = load_gt(CHANNEL_TRAIN_JSON)
    logger.info(f"  GT images    : {len(gt_counts)}")
    logger.info(f"  GT with fish : {sum(1 for v in gt_counts.values() if v > 0)}")
    logger.info(f"  Total GT fish: {sum(gt_counts.values())}")

    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw_dir  = Path(PSEUDO_RAW_DIR)  / model_name
    filt_dir = Path(PSEUDO_FILT_DIR) / model_name

    if raw_dir.exists():
        m = evaluate(model_name, str(raw_dir),  gt_counts, "RAW",      logger)
        if m: results.append(m)
    else:
        logger.warning(f"  Raw labels not found: {raw_dir}")
        logger.warning(f"  Run pseudo_label.py --mode {mode} first.")

    if filt_dir.exists():
        m = evaluate(model_name, str(filt_dir), gt_counts, "FILTERED", logger)
        if m: results.append(m)
    else:
        logger.info(f"  Filtered labels not found yet.")
        logger.info(f"  Run temporal_filter.py --mode {mode} first.")

    if results:
        out = os.path.join(RESULTS_DIR, f"pseudo_label_quality_{model_name}.csv")
        save_csv(out, results)
        logger.info(f"\n  Results saved: {out}")

    if len(results) == 2:
        raw_r  = results[0]; filt_r = results[1]
        logger.info(f"\n{'='*55}")
        logger.info(f"  RAW vs FILTERED — {model_name}")
        logger.info(f"{'='*55}")
        logger.info(f"  {'Metric':<25} {'Raw':>10} {'Filtered':>10}")
        logger.info(f"  {'-'*47}")
        for key in ["frames_with_pred","total_pred_fish","MAE",
                    "nMAE","frame_precision","frame_recall","detection_rate_%"]:
            logger.info(f"  {key:<25} {str(raw_r.get(key,''))!s:>10} "
                        f"{str(filt_r.get(key,''))!s:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    args   = parser.parse_args()
    logger = get_logger("eval_pseudo", LOG_DIR)
    run(args.mode, logger)
