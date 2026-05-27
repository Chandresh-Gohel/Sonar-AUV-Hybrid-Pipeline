"""
=============================================================
  Temporal Pseudo-Label Filter — All 4 Configurations

  Run:
    python temporal_filter.py --mode yolov5n_mog
    python temporal_filter.py --mode yolov8s_raw
    python temporal_filter.py --mode yolov26s_mog
    python temporal_filter.py --mode yolov26s_raw

  Output:
    daod/pseudo_labels/filtered/yolov5n_mog_source/
    daod/pseudo_labels/filtered/yolov8s_raw_source/
    daod/pseudo_labels/filtered/yolov26s_mog_source/
    daod/pseudo_labels/filtered/yolov26s_raw_source/
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from kalman_centroid import KalmanCentroidTrack, TemporalPseudoLabelFilter
from utils import (get_sequence_key, read_yolo_labels, write_yolo_labels,
                   xyxy_to_yolo, yolo_to_xyxy, get_image_size, save_json, get_logger)

from config import BASE_DIR, LOG_DIR
CHANNEL_TRAIN_DIR = os.path.join(BASE_DIR, "cfc_channel_train")
PSEUDO_RAW_DIR    = os.path.join(BASE_DIR, "daod", "pseudo_labels", "raw")
PSEUDO_FILT_DIR   = os.path.join(BASE_DIR, "daod", "pseudo_labels", "filtered")

MAX_AGE        = 45
MIN_HITS       = 1
MAX_DISTANCE   = 50.0
MIN_TRACK_HITS = 2


def run(mode, logger):
    model_name = f"{mode}_source"
    raw_dir    = Path(PSEUDO_RAW_DIR)  / model_name
    filt_dir   = Path(PSEUDO_FILT_DIR) / model_name
    filt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Model      : {model_name}")
    logger.info(f"  Raw labels : {raw_dir}")
    logger.info(f"  Filtered   : {filt_dir}")
    logger.info(f"  min_track_hits: {MIN_TRACK_HITS}")
    logger.info(f"{'='*55}")

    if not raw_dir.exists():
        logger.error(f"Raw labels not found: {raw_dir}")
        logger.error("Run pseudo_label.py first.")
        return

    label_files = sorted(raw_dir.glob("*.txt"))
    logger.info(f"  Label files: {len(label_files)}")

    seq_groups = defaultdict(list)
    for lbl_path in label_files:
        seq_key, frame_num = get_sequence_key(lbl_path.name)
        img_path = Path(CHANNEL_TRAIN_DIR) / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = Path(CHANNEL_TRAIN_DIR) / (lbl_path.stem + ".png")
        seq_groups[seq_key].append((frame_num, lbl_path, img_path))
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])
    logger.info(f"  Sequences  : {len(seq_groups)}")

    KalmanCentroidTrack.count = 0
    total_raw = 0; total_kept = 0; total_seq = 0

    for seq_key, frames in tqdm(seq_groups.items(), desc=f"  {model_name}"):
        total_seq += 1
        seq_filter = TemporalPseudoLabelFilter(
            max_age=MAX_AGE, min_hits=MIN_HITS,
            max_distance=MAX_DISTANCE, min_track_hits=MIN_TRACK_HITS)

        for frame_idx, (frame_num, lbl_path, img_path) in enumerate(frames):
            img_w, img_h = get_image_size(img_path) if img_path.exists() else (512, 512)
            if img_w is None:
                img_w, img_h = 512, 512
            raw_labels = read_yolo_labels(str(lbl_path))
            dets = []
            for cls, xc, yc, w, h in raw_labels:
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                dets.append([x1, y1, x2, y2, 1.0])
            seq_filter.update(dets, frame_idx, img_path=str(img_path))
            total_raw += len(raw_labels)

        confirmed = seq_filter.get_confirmed_labels()
        by_frame  = defaultdict(list)
        for lbl in confirmed:
            by_frame[lbl["frame_idx"]].append(lbl)

        for frame_idx, (frame_num, lbl_path, img_path) in enumerate(frames):
            out_path   = filt_dir / lbl_path.name
            frame_lbls = by_frame.get(frame_idx, [])
            if frame_lbls:
                img_w, img_h = get_image_size(img_path) if img_path.exists() else (512, 512)
                if img_w is None:
                    img_w, img_h = 512, 512
                yolo_labels = []
                for lbl in frame_lbls:
                    xc, yc, w, h = xyxy_to_yolo(lbl["x1"], lbl["y1"],
                                                  lbl["x2"], lbl["y2"], img_w, img_h)
                    yolo_labels.append((0,
                        float(np.clip(xc,0,1)), float(np.clip(yc,0,1)),
                        float(np.clip(w, 0,1)), float(np.clip(h, 0,1))))
                write_yolo_labels(str(out_path), yolo_labels)
                total_kept += len(yolo_labels)
            else:
                out_path.write_text("")

    rejection = round(100 * (total_raw - total_kept) / max(total_raw, 1), 1)
    result = {"model": model_name, "total_sequences": total_seq,
              "total_raw": total_raw, "total_kept": total_kept,
              "total_rejected": total_raw - total_kept, "rejection_%": rejection}
    save_json(str(filt_dir / "filter_stats.json"), result)

    logger.info(f"\n  Raw labels   : {total_raw}")
    logger.info(f"  Kept labels  : {total_kept}")
    logger.info(f"  Rejected     : {total_raw - total_kept} ({rejection}%)")
    logger.info(f"  Output       : {filt_dir}")
    logger.info(f"\n  Next: python dataset_mixer.py --mode {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("temporal_filter", LOG_DIR)
    run(args.mode, logger)
