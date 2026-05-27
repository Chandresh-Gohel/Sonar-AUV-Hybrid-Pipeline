"""
=============================================================
  Shared Utilities
  bbox conversion, YOLO label read/write, sequence builder,
  logging helpers — used across all DAOD pipeline scripts
=============================================================
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2


# ── Logging ────────────────────────────────────────────────
def get_logger(name, log_dir=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt    = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                                datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── BBox Conversions ───────────────────────────────────────
def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert [x1,y1,x2,y2] pixel coords to YOLO normalized format.
    Returns (xc, yc, w, h) all normalized to [0,1].
    """
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    Convert YOLO normalized format to [x1,y1,x2,y2] pixel coords.
    """
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return x1, y1, x2, y2


def bbox_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_bbox(x1, y1, x2, y2, img_w, img_h):
    """Clip bbox to image boundaries."""
    return (
        max(0, min(x1, img_w)),
        max(0, min(y1, img_h)),
        max(0, min(x2, img_w)),
        max(0, min(y2, img_h)),
    )


# ── YOLO Label IO ──────────────────────────────────────────
def read_yolo_labels(label_path):
    """
    Read YOLO label file.
    Returns list of (class_id, xc, yc, w, h) — all normalized.
    Returns empty list if file doesn't exist or is empty.
    """
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                labels.append((int(cls), xc, yc, w, h))
    return labels


def write_yolo_labels(label_path, labels, class_id=0):
    """
    Write YOLO label file.
    labels: list of (xc, yc, w, h) normalized, or (class_id, xc, yc, w, h)
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for label in labels:
            if len(label) == 4:
                xc, yc, w, h = label
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            elif len(label) == 5:
                cls, xc, yc, w, h = label
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def count_labels(label_path):
    """Count number of fish annotations in a label file."""
    return len(read_yolo_labels(label_path))


def get_image_size(img_path):
    """Get (width, height) of image without loading full image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]


# ── Sequence Builder ───────────────────────────────────────
def get_sequence_key(filename):
    """
    Parse CFC filename to extract sequence key and frame number.
    Pattern: ..._YYYY-MM-DD_HHMMSS_start_end_framenum.jpg
    Example: elwha_..._2018-07-09_190000_490_941_213.jpg
             → key: "2018-07-09_190000_490_941", frame: 213
    """
    name    = Path(filename).stem
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{6}_\d+_\d+)_(\d+)$'
    match   = re.search(pattern, name)
    if match:
        return match.group(1), int(match.group(2))
    # fallback — treat whole name as one sequence
    return name, 0


def build_sequences(images_dir, extensions=(".jpg", ".png")):
    """
    Group images in a directory into sequences based on CFC filename pattern.
    Returns OrderedDict: {seq_key: [sorted list of image paths]}
    """
    all_files = []
    for ext in extensions:
        all_files.extend(Path(images_dir).glob(f"*{ext}"))

    sequences = defaultdict(list)
    for fp in all_files:
        seq_key, frame_num = get_sequence_key(fp.name)
        sequences[seq_key].append((frame_num, fp))

    sorted_seqs = {}
    for key, frames in sequences.items():
        frames.sort(key=lambda x: x[0])
        sorted_seqs[key] = [fp for _, fp in frames]

    return sorted_seqs


def get_label_path(img_path, labels_dir=None):
    """
    Get corresponding label path for an image.
    If labels_dir given, use that. Otherwise infer by replacing 'images' with 'labels'.
    """
    img_path = Path(img_path)
    if labels_dir:
        return Path(labels_dir) / (img_path.stem + ".txt")
    # Infer from standard YOLO structure
    label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
    return label_path


# ── CSV Helpers ────────────────────────────────────────────
def save_csv(filepath, rows, fieldnames=None):
    """Save list of dicts to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = fieldnames or list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(filepath):
    """Load CSV to list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return list(csv.DictReader(f))


def save_json(filepath, data):
    """Save dict/list to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load JSON to dict/list."""
    with open(filepath, "r") as f:
        return json.load(f)


# ── YOLO Dataset YAML Writer ───────────────────────────────
def write_yaml(yaml_path, dataset_dir, train_rel, val_rel,
               class_names=None, nc=1):
    """
    Write a YOLO dataset YAML file.
    """
    if class_names is None:
        class_names = {0: "fish"}
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write(f"train: {train_rel}\n")
        f.write(f"val: {val_rel}\n\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for idx, name in class_names.items():
            f.write(f"  {idx}: {name}\n")
    print(f"[YAML] Written: {yaml_path}")


# ── Image Helpers ──────────────────────────────────────────
def draw_boxes_on_image(img, boxes_xyxy, color=(0, 255, 0),
                        thickness=2, labels=None):
    """
    Draw bounding boxes on image.
    boxes_xyxy: list of [x1,y1,x2,y2]
    labels: optional list of strings per box
    """
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            cv2.putText(img, labels[i], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def save_debug_image(img_path, pseudo_labels_xyxy, out_dir,
                     gt_labels_xyxy=None):
    """
    Save debug visualization of pseudo-labels on image.
    pseudo: green, gt: blue
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    if gt_labels_xyxy:
        draw_boxes_on_image(img, gt_labels_xyxy,
                            color=(255, 0, 0), labels=["GT"]*len(gt_labels_xyxy))
    draw_boxes_on_image(img, pseudo_labels_xyxy,
                        color=(0, 255, 0), labels=["PL"]*len(pseudo_labels_xyxy))
    out_path = Path(out_dir) / Path(img_path).name
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ── Stats Helpers ──────────────────────────────────────────
def compute_mae(gt_counts, pred_counts):
    """MAE between two lists of counts (per frame)."""
    errors = [abs(p - g) for p, g in zip(pred_counts, gt_counts)]
    return float(np.mean(errors)) if errors else 0.0


def compute_nmae(gt_counts, pred_counts):
    """Normalized MAE."""
    errors = []
    for p, g in zip(pred_counts, gt_counts):
        if g > 0:
            errors.append(abs(p - g) / g)
        elif p == 0:
            errors.append(0.0)
    return float(np.mean(errors)) if errors else 0.0


def print_table(rows, title="Results"):
    """Print a list of dicts as a formatted table."""
    if not rows:
        print(f"[{title}] No results.")
        return
    keys   = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k,""))) for r in rows))
              for k in keys}
    header = " | ".join(f"{k:<{widths[k]}}" for k in keys)
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print("="*len(header))
    print(header)
    print("─"*len(header))
    for row in rows:
        print(" | ".join(f"{str(row.get(k,'')):<{widths[k]}}" for k in keys))
    print("="*len(header))


# ── Pseudo-label format helpers ────────────────────────────
def detections_to_xyxy_conf(results):
    """
    Extract [x1,y1,x2,y2,conf] from ultralytics YOLO results object.
    Returns np.array (N,5) or empty array.
    """
    if results.boxes is None or len(results.boxes) == 0:
        return np.empty((0, 5))
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy().reshape(-1, 1)
    return np.hstack([boxes, confs])


if __name__ == "__main__":
    print("=== Utils Test ===\n")

    # Test bbox conversions
    x1, y1, x2, y2 = 100, 150, 200, 250
    img_w, img_h   = 512, 512
    xc, yc, w, h   = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
    print(f"xyxy → yolo : {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}")
    rx1, ry1, rx2, ry2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
    print(f"yolo → xyxy : {rx1:.1f} {ry1:.1f} {rx2:.1f} {ry2:.1f}")
    assert abs(rx1 - x1) < 1e-3, "Conversion mismatch!"
    print("Conversion round-trip: OK\n")

    # Test sequence builder (mock filenames)
    mock_files = [
        "elwha_Elwha_2018_OM_ARIS_2018_07_09_2018-07-09_190000_490_941_213.jpg",
        "elwha_Elwha_2018_OM_ARIS_2018_07_09_2018-07-09_190000_490_941_216.jpg",
        "elwha_Elwha_2018_OM_ARIS_2018_07_09_2018-07-09_190001_490_941_001.jpg",
    ]
    for f in mock_files:
        key, frame = get_sequence_key(f)
        print(f"File: {f[-30:]} → key: {key}, frame: {frame}")

    print("\n[OK] utils.py all tests passed")
