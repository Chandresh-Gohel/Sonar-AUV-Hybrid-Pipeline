"""
=============================================================
  CFC Source Data Preprocessor

  Converts full CFC source dataset (Baseline++ preprocessed)
  to correct format for model training.

  Two modes:
    --mode mog2tvg  -> for yolov5n training
                       Ch3 extraction -> MOG2+TVG (history=200)
    --mode raw      -> for yolov8s training
                       Ch3 extraction only -> BGR

  Processes images in sequence order for proper MOG2 warmup.
  Sequences shorter than MIN_SEQ_LENGTH skipped for mog2tvg.

  Run:
    python preprocess_cfc.py --mode mog2tvg --split train
    python preprocess_cfc.py --mode mog2tvg --split val
    python preprocess_cfc.py --mode raw     --split train
    python preprocess_cfc.py --mode raw     --split val

  Input:
    cfc_train/ or cfc_val/  (Baseline++ images)

  Output:
    cfc_source_mog2tvg/train/ or cfc_source_mog2tvg/val/
    cfc_source_raw/train/     or cfc_source_raw/val/
=============================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from preprocessing import MOG2TVGPreprocessor
from utils import get_sequence_key, get_logger

# ── Config ─────────────────────────────────────────────────
from config import BASE_DIR, LOG_DIR

INPUT_DIRS = {
    "train": os.path.join(BASE_DIR, "cfc_train"),
    "val":   os.path.join(BASE_DIR, "cfc_val"),
}

OUTPUT_DIRS = {
    "mog2tvg": {
        "train": os.path.join(BASE_DIR, "cfc_source_mog2tvg", "train"),
        "val":   os.path.join(BASE_DIR, "cfc_source_mog2tvg", "val"),
    },
    "raw": {
        "train": os.path.join(BASE_DIR, "cfc_source_raw", "train"),
        "val":   os.path.join(BASE_DIR, "cfc_source_raw", "val"),
    },
}

MOG2_HISTORY  = 200   # full history for long sequences
MIN_SEQ_LEN   = 30    # skip very short sequences for mog2tvg


def get_seq_key_cfc(filename):
    """
    Parse CFC source filename.
    Example: 2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004_285_885_0.jpg
    Key: 2018-05-26_080004_285_885  Frame: 0
    """
    name  = Path(filename).stem
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6}_\d+_\d+)_(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, 0


def process_frame_mog2tvg(frame_bgr, preprocessor):
    """Extract Ch3 from Baseline++ then apply MOG2+TVG."""
    ch3 = frame_bgr[:, :, 2]
    return preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))


def process_frame_raw(frame_bgr):
    """Extract Ch3 from Baseline++ only."""
    ch3 = frame_bgr[:, :, 2]
    return cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)


def run(mode, split, logger):
    input_dir  = Path(INPUT_DIRS[split])
    output_dir = Path(OUTPUT_DIRS[mode][split])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Mode   : {mode}")
    logger.info(f"  Split  : {split}")
    logger.info(f"  Input  : {input_dir}")
    logger.info(f"  Output : {output_dir}")
    logger.info(f"{'='*55}")

    if not input_dir.exists():
        logger.error(f"Input not found: {input_dir}")
        logger.error(f"Extract cfc_{split}.zip first.")
        return

    # Get all images
    all_images = sorted(list(input_dir.glob("*.jpg")) +
                        list(input_dir.glob("*.png")))
    logger.info(f"  Total images: {len(all_images)}")

    # Group into sequences
    seq_groups = defaultdict(list)
    for img_path in all_images:
        seq_key, frame_num = get_seq_key_cfc(img_path.name)
        seq_groups[seq_key].append((frame_num, img_path))
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])

    logger.info(f"  Sequences   : {len(seq_groups)}")

    # Filter short sequences for mog2tvg
    if mode == "mog2tvg":
        before = len(seq_groups)
        seq_groups = {k: v for k, v in seq_groups.items()
                      if len(v) >= MIN_SEQ_LEN}
        skipped = before - len(seq_groups)
        logger.info(f"  Skipped     : {skipped} sequences (< {MIN_SEQ_LEN} frames)")
        logger.info(f"  Processing  : {len(seq_groups)} sequences")

    preprocessor = MOG2TVGPreprocessor(mog2_history=MOG2_HISTORY) \
                   if mode == "mog2tvg" else None

    processed = 0
    failed    = 0

    for seq_key, frames in tqdm(seq_groups.items(),
                                desc=f"  {mode}/{split}"):
        # Reset MOG2 per sequence
        if mode == "mog2tvg" and preprocessor:
            preprocessor.reset()

        for frame_num, img_path in frames:
            out_path = output_dir / img_path.name

            # Skip if already processed
            if out_path.exists():
                processed += 1
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                failed += 1
                continue

            try:
                if mode == "mog2tvg":
                    result = process_frame_mog2tvg(frame, preprocessor)
                else:
                    result = process_frame_raw(frame)

                cv2.imwrite(str(out_path), result)
                processed += 1
            except Exception as e:
                logger.warning(f"  Failed {img_path.name}: {e}")
                failed += 1

    logger.info(f"\n  Processed : {processed}")
    logger.info(f"  Failed    : {failed}")
    logger.info(f"  Output    : {output_dir}")
    logger.info(f"\n  Next: python convert_cfc_labels.py --split {split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["mog2tvg", "raw"], required=True)
    parser.add_argument("--split", choices=["train", "val"],   required=True)
    args   = parser.parse_args()
    logger = get_logger("preprocess_cfc", LOG_DIR)
    run(args.mode, args.split, logger)
