"""
=============================================================
  Source Domain Dataset Filter

  Removes Kenai Channel images from train/val split
  to create a clean source model for DAOD.

  Run TWICE — once for raw, once for preprocessed:

    python filter_dataset.py --mode raw
    python filter_dataset.py --mode pre

  Output:
    mode=raw -> sonar_yolo_dataset_source_raw/
                sonar_dataset_source_raw.yaml    (for yolov8s)

    mode=pre -> sonar_yolo_dataset_source_pre/
                sonar_dataset_source_pre.yaml    (for yolov5n)
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from config import BASE_DIR

DATASETS = {
    "raw": {
        "input_images_train": os.path.join(BASE_DIR, "sonar_yolo_dataset", "images", "train"),
        "input_labels_train": os.path.join(BASE_DIR, "sonar_yolo_dataset", "labels", "train"),
        "input_images_val":   os.path.join(BASE_DIR, "sonar_yolo_dataset", "images", "val"),
        "input_labels_val":   os.path.join(BASE_DIR, "sonar_yolo_dataset", "labels", "val"),
        "output_dir":         os.path.join(BASE_DIR, "sonar_yolo_dataset_source_raw"),
        "output_yaml":        os.path.join(BASE_DIR, "sonar_dataset_source_raw.yaml"),
        "for_model":          "yolov26s",
    },
    "pre": {
        "input_images_train": os.path.join(BASE_DIR, "sonar_yolo_dataset_pre_pro", "images", "train"),
        "input_labels_train": os.path.join(BASE_DIR, "sonar_yolo_dataset_pre_pro", "labels", "train"),
        "input_images_val":   os.path.join(BASE_DIR, "sonar_yolo_dataset_pre_pro", "images", "val"),
        "input_labels_val":   os.path.join(BASE_DIR, "sonar_yolo_dataset_pre_pro", "labels", "val"),
        "output_dir":         os.path.join(BASE_DIR, "sonar_yolo_dataset_source_pre"),
        "output_yaml":        os.path.join(BASE_DIR, "sonar_dataset_source_pre.yaml"),
        "for_model":          "yolov5n",
    },
}

CHANNEL_KEYWORD = "kenai-channel"


# ── Helpers ────────────────────────────────────────────────
def is_channel(filename):
    return CHANNEL_KEYWORD in filename.lower()


def write_yaml(yaml_path, dataset_dir):
    with open(yaml_path, "w") as f:
        f.write(f"# Source domain — Kenai Channel excluded\n\n")
        f.write(f"path: {dataset_dir}\n\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names:\n")
        f.write(f"  0: fish\n")
    print(f"[YAML] {yaml_path}")


def filter_split(input_images, input_labels,
                 output_images, output_labels, split):
    output_images = Path(output_images)
    output_labels = Path(output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(list(Path(input_images).glob("*.jpg")) +
                      list(Path(input_images).glob("*.png")))

    kept = 0
    removed = 0
    location_kept = defaultdict(int)

    for img_path in all_imgs:
        location = img_path.name.split("_")[0].lower()

        if is_channel(img_path.name):
            removed += 1
            continue

        shutil.copy2(img_path, output_images / img_path.name)

        lbl_src = Path(input_labels) / (img_path.stem + ".txt")
        lbl_dst = output_labels / (img_path.stem + ".txt")
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")

        kept += 1
        location_kept[location] += 1

    print(f"\n  [{split.upper()}] Total: {kept+removed} | Kept: {kept} | Removed: {removed}")
    for loc, count in sorted(location_kept.items()):
        print(f"    {loc:<20}: {count}")

    return kept, removed


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "pre"], required=True,
                        help="raw=yolov26s | pre=yolov5n")
    args = parser.parse_args()
    cfg  = DATASETS[args.mode]

    print("=" * 55)
    print(f"  Filter Dataset  mode={args.mode} | for {cfg['for_model']}")
    print("=" * 55)

    if not Path(cfg["input_images_train"]).exists():
        print(f"[ERROR] Not found: {cfg['input_images_train']}")
        exit(1)

    tr_kept,  tr_rm  = filter_split(
        cfg["input_images_train"], cfg["input_labels_train"],
        os.path.join(cfg["output_dir"], "images", "train"),
        os.path.join(cfg["output_dir"], "labels", "train"),
        "train"
    )
    val_kept, val_rm = filter_split(
        cfg["input_images_val"], cfg["input_labels_val"],
        os.path.join(cfg["output_dir"], "images", "val"),
        os.path.join(cfg["output_dir"], "labels", "val"),
        "val"
    )

    write_yaml(cfg["output_yaml"], cfg["output_dir"])

    print("\n" + "=" * 55)
    print(f"  Train : {tr_kept} kept | {tr_rm} removed")
    print(f"  Val   : {val_kept} kept | {val_rm} removed")
    print(f"  YAML  : {cfg['output_yaml']}")
    print(f"  Next  : python retrain_source.py --mode {args.mode}")
    print("=" * 55)