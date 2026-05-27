"""
=============================================================
  Dataset Mixer — All 4 Configurations

  Run:
    python dataset_mixer.py --mode yolov5n_mog
    python dataset_mixer.py --mode yolov8s_raw
    python dataset_mixer.py --mode yolov26s_mog
    python dataset_mixer.py --mode yolov26s_raw

  Output:
    daod/mixed_dataset/yolov5n_mog_source/
    daod/mixed_dataset/yolov8s_raw_source/
    daod/mixed_dataset/yolov26s_mog_source/
    daod/mixed_dataset/yolov26s_raw_source/
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shutil, random, argparse, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils import get_logger, save_csv, write_yaml, get_sequence_key
from preprocessing import MOG2TVGPreprocessor

from config import BASE_DIR, LOG_DIR
CHANNEL_TRAIN_DIR = os.path.join(BASE_DIR, "cfc_channel_train")
PSEUDO_FILT_DIR   = os.path.join(BASE_DIR, "daod", "pseudo_labels", "filtered")
MIXED_DIR         = os.path.join(BASE_DIR, "daod", "mixed_dataset")

SOURCE_RATIO = 0.5
SEED         = 42

SOURCE = {
    "yolov5n_mog": {
        "images_train": os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "images", "train"),
        "labels_train": os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "labels", "train"),
        "images_val":   os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "images", "val"),
        "labels_val":   os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "labels", "val"),
        "pipeline":     "mog2tvg",
    },
    "yolov8s_raw": {
        "images_train": os.path.join(BASE_DIR, "cfc_yolo_raw", "images", "train"),
        "labels_train": os.path.join(BASE_DIR, "cfc_yolo_raw", "labels", "train"),
        "images_val":   os.path.join(BASE_DIR, "cfc_yolo_raw", "images", "val"),
        "labels_val":   os.path.join(BASE_DIR, "cfc_yolo_raw", "labels", "val"),
        "pipeline":     "raw",
    },
    "yolov26s_mog": {
        "images_train": os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "images", "train"),
        "labels_train": os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "labels", "train"),
        "images_val":   os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "images", "val"),
        "labels_val":   os.path.join(BASE_DIR, "cfc_yolo_mog2tvg", "labels", "val"),
        "pipeline":     "mog2tvg",
    },
    "yolov26s_raw": {
        "images_train": os.path.join(BASE_DIR, "cfc_yolo_raw", "images", "train"),
        "labels_train": os.path.join(BASE_DIR, "cfc_yolo_raw", "labels", "train"),
        "images_val":   os.path.join(BASE_DIR, "cfc_yolo_raw", "images", "val"),
        "labels_val":   os.path.join(BASE_DIR, "cfc_yolo_raw", "labels", "val"),
        "pipeline":     "raw",
    },
}


def run(mode, logger):
    model_name = f"{mode}_source"
    src        = SOURCE[mode]
    pipeline   = src["pipeline"]
    filt_dir   = Path(PSEUDO_FILT_DIR) / model_name
    out_dir    = Path(MIXED_DIR) / model_name
    yaml_path  = Path(BASE_DIR) / "daod" / f"mixed_{model_name}.yaml"

    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Model        : {model_name}")
    logger.info(f"  Pipeline     : {pipeline}")
    logger.info(f"  Source ratio : {SOURCE_RATIO}")
    logger.info(f"{'='*55}")

    if not filt_dir.exists():
        logger.error(f"Filtered labels not found: {filt_dir}")
        logger.error("Run temporal_filter.py first.")
        return

    random.seed(SEED)
    preprocessor = MOG2TVGPreprocessor() if pipeline == "mog2tvg" else None

    source_imgs = sorted(list(Path(src["images_train"]).glob("*.jpg")) +
                         list(Path(src["images_train"]).glob("*.png")))
    target_pairs = []
    for lbl_file in sorted(filt_dir.glob("*.txt")):
        if lbl_file.stat().st_size == 0:
            continue
        img_path = Path(CHANNEL_TRAIN_DIR) / (lbl_file.stem + ".jpg")
        if not img_path.exists():
            img_path = Path(CHANNEL_TRAIN_DIR) / (lbl_file.stem + ".png")
        if img_path.exists():
            target_pairs.append((img_path, lbl_file))

    n_target = len(target_pairs)
    n_source = int(n_target * SOURCE_RATIO / (1 - SOURCE_RATIO)) if SOURCE_RATIO < 1.0 else len(source_imgs)
    n_source = min(n_source, len(source_imgs))
    selected_source = random.sample(source_imgs, n_source)

    logger.info(f"  Source selected  : {n_source}")
    logger.info(f"  Target selected  : {n_target}")
    logger.info(f"  Total train      : {n_source + n_target}")

    # Copy source
    for img_path in selected_source:
        shutil.copy2(img_path, out_dir / "images" / "train" / img_path.name)
        lbl_src = Path(src["labels_train"]) / (img_path.stem + ".txt")
        lbl_dst = out_dir / "labels" / "train" / (img_path.stem + ".txt")
        shutil.copy2(lbl_src, lbl_dst) if lbl_src.exists() else lbl_dst.write_text("")

    # Copy target — convert to correct format
    seq_groups = defaultdict(list)
    for img_path, lbl_path in target_pairs:
        seq_key, frame_num = get_sequence_key(img_path.name)
        seq_groups[seq_key].append((frame_num, img_path, lbl_path))
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[0])

    for seq_key, frames in seq_groups.items():
        if pipeline == "mog2tvg" and preprocessor:
            preprocessor.reset()
        for frame_num, img_path, lbl_path in frames:
            new_name = f"ch_{img_path.name}"
            frame    = cv2.imread(str(img_path))
            if frame is None:
                continue
            ch3 = frame[:, :, 2]
            if pipeline == "mog2tvg":
                converted = preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))
            else:
                converted = cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_dir / "images" / "train" / new_name), converted)
            shutil.copy2(lbl_path, out_dir / "labels" / "train" / (Path(new_name).stem + ".txt"))

    # Copy source val
    for img_path in sorted(list(Path(src["images_val"]).glob("*.jpg")) +
                           list(Path(src["images_val"]).glob("*.png"))):
        shutil.copy2(img_path, out_dir / "images" / "val" / img_path.name)
        lbl_src = Path(src["labels_val"]) / (img_path.stem + ".txt")
        lbl_dst = out_dir / "labels" / "val" / (img_path.stem + ".txt")
        shutil.copy2(lbl_src, lbl_dst) if lbl_src.exists() else lbl_dst.write_text("")

    write_yaml(str(yaml_path), str(out_dir), "images/train", "images/val",
               class_names={0: "fish"}, nc=1)

    logger.info(f"\n  Mixed dataset : {out_dir}")
    logger.info(f"  YAML          : {yaml_path}")
    logger.info(f"  Next: python finetune.py --mode {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("dataset_mixer", LOG_DIR)
    run(args.mode, logger)
