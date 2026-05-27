"""
=============================================================
  Build YOLO Dataset Structure with Sequence-Diverse Sampling

  Samples images evenly across sequences to maximize
  diversity while keeping dataset size manageable.

  Dataset sizes per pipeline:
    mog2tvg (YOLOv5n + YOLOv26s_mog) -> MAX_IMAGES = 15000
    raw     (YOLOv8s + YOLOv26s_raw) -> MAX_IMAGES = 30000

  Sampling strategy:
    - Distribute MAX_IMAGES evenly across all sequences
    - From each sequence pick evenly spaced frames
    - Ensures model sees all sonar environments/lighting
    - Better generalization than random sampling

  Run:
    python build_yolo_dataset.py --mode mog2tvg
    python build_yolo_dataset.py --mode raw

  Output:
    cfc_yolo_mog2tvg/images/train|val + labels/train|val
    cfc_yolo_mog2tvg.yaml
    cfc_yolo_raw/...
    cfc_yolo_raw.yaml
=============================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import get_logger, write_yaml, get_sequence_key

from config import BASE_DIR, LOG_DIR

CONFIGS = {
    "mog2tvg": {
        "images": {
            "train": os.path.join(BASE_DIR, "cfc_source_mog2tvg", "train"),
            "val":   os.path.join(BASE_DIR, "cfc_source_mog2tvg", "val"),
        },
        "output_dir":  os.path.join(BASE_DIR, "cfc_yolo_mog2tvg"),
        "yaml":        os.path.join(BASE_DIR, "cfc_yolo_mog2tvg.yaml"),
        "max_train":   15000,   # used by YOLOv5n_mog and YOLOv26s_mog
        "max_val":     3000,    # ~20% of train
    },
    "raw": {
        "images": {
            "train": os.path.join(BASE_DIR, "cfc_source_raw", "train"),
            "val":   os.path.join(BASE_DIR, "cfc_source_raw", "val"),
        },
        "output_dir":  os.path.join(BASE_DIR, "cfc_yolo_raw"),
        "yaml":        os.path.join(BASE_DIR, "cfc_yolo_raw.yaml"),
        "max_train":   30000,   # used by YOLOv8s_raw and YOLOv26s_raw
        "max_val":     6000,
    },
}

# Shared labels
LABELS = {
    "train": os.path.join(BASE_DIR, "cfc_source_labels", "train"),
    "val":   os.path.join(BASE_DIR, "cfc_source_labels", "val"),
}


def get_seq_key_cfc(filename):
    """Parse CFC source filename to sequence key."""
    name  = Path(filename).stem
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6}_\d+_\d+)_(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, 0


def sample_diverse(images, max_images, logger):
    """
    Sample max_images from images list with sequence diversity.

    Strategy:
      1. Group images into sequences
      2. Calculate frames_per_sequence = max_images / n_sequences
      3. From each sequence pick evenly spaced frames
      4. If sequence shorter than quota, take all frames

    This ensures all 482 sequences are represented rather than
    randomly picking which could miss entire sequences.
    """
    if len(images) <= max_images:
        logger.info(f"    Using all {len(images)} images (below max)")
        return images

    # Group by sequence
    seq_groups = defaultdict(list)
    for img_path in images:
        seq_key, frame_num = get_seq_key_cfc(img_path.name)
        seq_groups[seq_key].append((frame_num, img_path))

    # Sort frames within each sequence
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])

    n_seqs           = len(seq_groups)
    frames_per_seq   = max(1, max_images // n_seqs)
    leftover         = max_images - frames_per_seq * n_seqs

    logger.info(f"    Sequences        : {n_seqs}")
    logger.info(f"    Frames per seq   : {frames_per_seq}")
    logger.info(f"    Leftover budget  : {leftover}")

    selected = []

    for seq_key, frames in seq_groups.items():
        n = len(frames)

        if n <= frames_per_seq:
            # Take all frames from short sequences
            selected.extend([fp for _, fp in frames])
        else:
            # Evenly spaced indices across sequence
            indices = np.linspace(0, n - 1, frames_per_seq, dtype=int)
            selected.extend([frames[i][1] for i in indices])

    # Use leftover budget on longest sequences
    if leftover > 0 and len(selected) < max_images:
        # Find sequences that had more frames than quota
        rich_seqs = [(k, v) for k, v in seq_groups.items()
                     if len(v) > frames_per_seq]
        rich_seqs.sort(key=lambda x: len(x[1]), reverse=True)

        selected_stems = set(p.stem for p in selected)
        extra = []
        for seq_key, frames in rich_seqs:
            if len(extra) >= leftover:
                break
            for _, fp in frames:
                if fp.stem not in selected_stems:
                    extra.append(fp)
                    selected_stems.add(fp.stem)
                    if len(extra) >= leftover:
                        break
        selected.extend(extra)

    # Final shuffle for training diversity
    np.random.seed(42)
    np.random.shuffle(selected)

    logger.info(f"    Selected         : {len(selected)}")
    return selected[:max_images]


def build_split(split, cfg, logger):
    img_src  = Path(cfg["images"][split])
    lbl_src  = Path(LABELS[split])
    out_dir  = Path(cfg["output_dir"])
    img_dst  = out_dir / "images" / split
    lbl_dst  = out_dir / "labels" / split
    max_imgs = cfg[f"max_{split}"]

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    if not img_src.exists():
        logger.warning(f"  Images not found: {img_src}")
        logger.warning(f"  Run preprocess_cfc.py --mode {split} first")
        return 0

    if not lbl_src.exists():
        logger.warning(f"  Labels not found: {lbl_src}")
        logger.warning(f"  Run convert_cfc_labels.py --split {split} first")
        return 0

    # Get all available images
    all_images = sorted(list(img_src.glob("*.jpg")) +
                        list(img_src.glob("*.png")))
    logger.info(f"\n  [{split.upper()}] Available: {len(all_images)}")
    logger.info(f"  [{split.upper()}] Max target: {max_imgs}")

    # Sample with sequence diversity
    selected = sample_diverse(all_images, max_imgs, logger)

    # Copy selected images + labels
    copied_imgs  = 0
    copied_lbls  = 0
    missing_lbls = 0

    for img_path in tqdm(selected, desc=f"  Copying {split}"):
        # Copy image
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        copied_imgs += 1

        # Copy label
        lbl_path = lbl_src / (img_path.stem + ".txt")
        dst_lbl  = lbl_dst / (img_path.stem + ".txt")
        if lbl_path.exists():
            if not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)
            copied_lbls += 1
        else:
            dst_lbl.write_text("")
            missing_lbls += 1

    logger.info(f"  Images copied : {copied_imgs}")
    logger.info(f"  Labels copied : {copied_lbls}")
    logger.info(f"  Empty labels  : {missing_lbls}")

    return copied_imgs


def run(mode, logger):
    cfg     = CONFIGS[mode]
    out_dir = Path(cfg["output_dir"])

    logger.info(f"\n{'='*55}")
    logger.info(f"  Mode        : {mode}")
    logger.info(f"  Max train   : {cfg['max_train']}")
    logger.info(f"  Max val     : {cfg['max_val']}")
    logger.info(f"  Output      : {out_dir}")
    logger.info(f"{'='*55}")

    train_count = build_split("train", cfg, logger)
    val_count   = build_split("val",   cfg, logger)

    # Write YAML
    write_yaml(
        yaml_path   = cfg["yaml"],
        dataset_dir = str(out_dir),
        train_rel   = "images/train",
        val_rel     = "images/val",
        class_names = {0: "fish"},
        nc          = 1,
    )

    logger.info(f"\n{'='*55}")
    logger.info(f"  DONE — {mode}")
    logger.info(f"  Train : {train_count} images")
    logger.info(f"  Val   : {val_count} images")
    logger.info(f"  YAML  : {cfg['yaml']}")
    logger.info(f"  Next  : python train_cfc.py --mode yolov5n_mog  (or yolov8s_raw / yolov26s_mog / yolov26s_raw)")
    logger.info(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mog2tvg", "raw"], required=True)
    args   = parser.parse_args()
    logger = get_logger("build_dataset", LOG_DIR)
    run(args.mode, logger)
