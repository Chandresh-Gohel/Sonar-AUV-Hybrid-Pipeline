"""
=============================================================
  CFC COCO JSON -> YOLO Label Converter

  Converts cfc_train.json / cfc_val.json COCO annotations
  to per-image YOLO .txt label files.

  Skips invalid annotations (bbox [-1,-1,0,0] placeholders).
  Writes empty .txt for images with no valid fish annotations.

  Run:
    python convert_cfc_labels.py --split train
    python convert_cfc_labels.py --split val

  Output:
    cfc_source_labels/train/*.txt
    cfc_source_labels/val/*.txt

  Same label files used for both mog2tvg and raw
  since preprocessing doesn't change bboxes.
=============================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import get_logger

# ── Config ─────────────────────────────────────────────────
from config import BASE_DIR, LOG_DIR

JSON_PATHS = {
    "train": os.path.join(BASE_DIR, "cfc_train.json"),
    "val":   os.path.join(BASE_DIR, "cfc_val.json"),
}

OUTPUT_LABEL_DIRS = {
    "train": os.path.join(BASE_DIR, "cfc_source_labels", "train"),
    "val":   os.path.join(BASE_DIR, "cfc_source_labels", "val"),
}

# Reference image dirs to know which images exist
# Labels written for preprocessed images (same stems)
REF_IMAGE_DIRS = {
    "mog2tvg": {
        "train": os.path.join(BASE_DIR, "cfc_source_mog2tvg", "train"),
        "val":   os.path.join(BASE_DIR, "cfc_source_mog2tvg", "val"),
    },
    "raw": {
        "train": os.path.join(BASE_DIR, "cfc_source_raw", "train"),
        "val":   os.path.join(BASE_DIR, "cfc_source_raw", "val"),
    },
}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert COCO bbox [x, y, w, h] (top-left + size)
    to YOLO [xc, yc, w, h] normalized.
    """
    x, y, w, h = bbox
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return (
        float(np.clip(xc, 0, 1)),
        float(np.clip(yc, 0, 1)),
        float(np.clip(wn, 0, 1)),
        float(np.clip(hn, 0, 1)),
    )


def is_valid_bbox(bbox):
    """Skip placeholder annotations."""
    x, y, w, h = bbox
    if x == -1 and y == -1:
        return False
    if w <= 0 or h <= 0:
        return False
    return True


def run(split, logger):
    json_path  = JSON_PATHS[split]
    out_dir    = Path(OUTPUT_LABEL_DIRS[split])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Split    : {split}")
    logger.info(f"  JSON     : {json_path}")
    logger.info(f"  Output   : {out_dir}")
    logger.info(f"{'='*55}")

    if not Path(json_path).exists():
        logger.error(f"JSON not found: {json_path}")
        return

    # Load COCO JSON
    logger.info("  Loading JSON...")
    with open(json_path) as f:
        data = json.load(f)

    # Build image info lookup
    img_info = {img["id"]: img for img in data["images"]}
    logger.info(f"  Images      : {len(img_info)}")
    logger.info(f"  Annotations : {len(data['annotations'])}")

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    valid_anns = 0
    invalid_anns = 0

    for ann in data["annotations"]:
        if is_valid_bbox(ann["bbox"]):
            ann_by_img[ann["image_id"]].append(ann)
            valid_anns += 1
        else:
            invalid_anns += 1

    logger.info(f"  Valid anns  : {valid_anns}")
    logger.info(f"  Invalid anns: {invalid_anns} (skipped)")

    # Write YOLO labels for each image
    written   = 0
    empty     = 0
    skipped   = 0

    for img_id, img in tqdm(img_info.items(), desc=f"  {split}"):
        stem     = Path(img["file_name"]).stem
        out_path = out_dir / (stem + ".txt")

        # Skip if already written
        if out_path.exists():
            written += 1
            continue

        img_w = img["width"]
        img_h = img["height"]
        anns  = ann_by_img.get(img_id, [])

        if not anns:
            # No fish — write empty label
            out_path.write_text("")
            empty += 1
            continue

        # Write YOLO labels
        lines = []
        for ann in anns:
            xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_path.write_text("\n".join(lines) + "\n")
        written += 1

    logger.info(f"\n  Written (with fish) : {written}")
    logger.info(f"  Written (empty)     : {empty}")
    logger.info(f"  Output              : {out_dir}")

    # Verify counts
    total_files = len(list(out_dir.glob("*.txt")))
    logger.info(f"  Total label files   : {total_files}")
    logger.info(f"\n  Next: python build_yolo_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args   = parser.parse_args()
    logger = get_logger("convert_labels", LOG_DIR)
    run(args.split, logger)
