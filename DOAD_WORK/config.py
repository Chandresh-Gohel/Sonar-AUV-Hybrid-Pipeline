"""
=============================================================
  DAOD Pipeline Configuration
  All paths and hyperparameters in one place.
  Edit this file before running any pipeline script.
=============================================================
"""

import os

# ── Base Directories ───────────────────────────────────────
# Set the SONAR_BASE_DIR environment variable or change this path directly.
BASE_DIR        = os.environ.get("SONAR_BASE_DIR", r"./data")
DAOD_DIR        = os.path.join(BASE_DIR, "daod")

# ── Source Domain (Kenai Left Bank) ───────────────────────
# Already trained — from paper 1369
SOURCE_DIR      = os.path.join(BASE_DIR, "sonar_yolo_dataset_pre_pro")
SOURCE_TRAIN_IMAGES = os.path.join(SOURCE_DIR, "images", "train")
SOURCE_TRAIN_LABELS = os.path.join(SOURCE_DIR, "labels", "train")
SOURCE_VAL_IMAGES   = os.path.join(SOURCE_DIR, "images", "val")
SOURCE_VAL_LABELS   = os.path.join(SOURCE_DIR, "labels", "val")
SOURCE_YAML         = os.path.join(BASE_DIR, "sonar_dataset_pre_pro.yaml")

# ── Target Domain (Kenai Channel) ─────────────────────────
# Raw sonar frames — your own preprocessing applied
TARGET_RAW_DIR       = os.path.join(BASE_DIR, "cfc_channel_raw")
TARGET_TRAIN_RAW     = os.path.join(TARGET_RAW_DIR, "train")   # unlabeled
TARGET_TEST_RAW      = os.path.join(TARGET_RAW_DIR, "test")    # for evaluation

# After preprocessing applied
TARGET_PREPROCESSED_DIR   = os.path.join(DAOD_DIR, "target_preprocessed")
TARGET_TRAIN_PRE          = os.path.join(TARGET_PREPROCESSED_DIR, "train")
TARGET_TEST_PRE           = os.path.join(TARGET_PREPROCESSED_DIR, "test")

# ── Pseudo Labels ─────────────────────────────────────────
PSEUDO_LABEL_DIR          = os.path.join(DAOD_DIR, "pseudo_labels")
PSEUDO_LABEL_RAW_DIR      = os.path.join(PSEUDO_LABEL_DIR, "raw")       # before filtering
PSEUDO_LABEL_FILTERED_DIR = os.path.join(PSEUDO_LABEL_DIR, "filtered")  # after temporal filter

# ── Mixed Dataset (source + filtered target) ──────────────
MIXED_DATASET_DIR         = os.path.join(DAOD_DIR, "mixed_dataset")
MIXED_TRAIN_IMAGES        = os.path.join(MIXED_DATASET_DIR, "images", "train")
MIXED_TRAIN_LABELS        = os.path.join(MIXED_DATASET_DIR, "labels", "train")
MIXED_VAL_IMAGES          = os.path.join(MIXED_DATASET_DIR, "images", "val")
MIXED_VAL_LABELS          = os.path.join(MIXED_DATASET_DIR, "labels", "val")
MIXED_YAML                = os.path.join(DAOD_DIR, "mixed_dataset.yaml")

# ── Model Weights ─────────────────────────────────────────
# Burn-in model (from paper 1369) — used to generate pseudo-labels
BURNIN_WEIGHTS = {
    "yolov5s_pre": os.path.join(BASE_DIR, "runs", "detect",
                                "yolov5s_pre_pro_run_100e", "weights", "best.pt"),
    "yolov8s_pre": os.path.join(BASE_DIR, "runs", "detect",
                                "yolov8s_pre_pro_run_100e", "weights", "best.pt"),
}

# After DAOD fine-tuning
DAOD_WEIGHTS_DIR = os.path.join(DAOD_DIR, "weights")

# ── Output Directories ────────────────────────────────────
RESULTS_DIR      = os.path.join(DAOD_DIR, "results")
EXPORT_DIR       = os.path.join(DAOD_DIR, "export")      # ONNX / quantized models
LOG_DIR          = os.path.join(DAOD_DIR, "logs")

# ── Preprocessing Parameters ──────────────────────────────
# TVG (Time Varying Gain)
TVG_SLOPE       = 0.0005    # range map scale
TVG_OFFSET      = 1e-3      # minimum range
TVG_MAX_DB      = 50.0      # clip TVG gain at 50 dB
TVG_ATTN        = 0.04      # water attenuation coefficient

# CLAHE
CLAHE_CLIP      = 2.0
CLAHE_GRID      = (8, 8)

# Gaussian blur before TVG
BLUR_KERNEL     = (5, 5)

# ── Detection Parameters ──────────────────────────────────
CONF_THRESHOLD_BURNIN    = 0.4    # for burn-in evaluation
CONF_THRESHOLD_PSEUDO    = 0.6    # higher threshold for pseudo-label generation
                                   # (only high-confidence detections become labels)
YOLO_IMGSZ               = 512
DEVICE                   = "0"    # GPU 0

# ── Kalman Centroid Tracker Parameters ────────────────────
# For pseudo-label temporal filtering
TRACKER_MAX_AGE      = 45     # frames track survives without detection (3s at 15fps)
TRACKER_MIN_HITS     = 2      # consecutive hits before track confirmed
TRACKER_MAX_DISTANCE = 50.0   # max centroid distance for matching (pixels)
MIN_TRACK_HITS       = 3      # minimum track length for pseudo-label acceptance

# For video inference (gmrt)
VIDEO_TRACKER_MAX_AGE      = 75
VIDEO_TRACKER_MIN_HITS     = 2
VIDEO_TRACKER_MAX_DISTANCE = 50.0

# ── Fine-tuning Parameters ────────────────────────────────
FINETUNE_EPOCHS     = 50
FINETUNE_BATCH      = 8
FINETUNE_LR         = 0.001
FINETUNE_PATIENCE   = 10      # early stopping patience
SOURCE_RATIO        = 0.5     # ratio of source images in mixed dataset
                               # 0.5 = equal source and target

# ── Evaluation ────────────────────────────────────────────
EVAL_CONF           = 0.4
EVAL_BATCH          = 1

# ── Export / Quantization ─────────────────────────────────
ONNX_OPSET         = 12      # ONNX opset — Vitis AI compatible
QUANTIZE_CALIB_DIR = os.path.join(TARGET_TEST_PRE)  # calibration images for INT8

# ── Create output dirs on import ──────────────────────────
_DIRS_TO_CREATE = [
    DAOD_DIR, TARGET_PREPROCESSED_DIR, TARGET_TRAIN_PRE, TARGET_TEST_PRE,
    PSEUDO_LABEL_DIR, PSEUDO_LABEL_RAW_DIR, PSEUDO_LABEL_FILTERED_DIR,
    MIXED_DATASET_DIR, MIXED_TRAIN_IMAGES, MIXED_TRAIN_LABELS,
    MIXED_VAL_IMAGES, MIXED_VAL_LABELS,
    DAOD_WEIGHTS_DIR, RESULTS_DIR, EXPORT_DIR, LOG_DIR,
]

def create_dirs():
    for d in _DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    print(f"[CONFIG] All output directories ready under {DAOD_DIR}")


if __name__ == "__main__":
    create_dirs()
    print("\n--- DAOD Configuration ---")
    print(f"Base dir        : {BASE_DIR}")
    print(f"Source domain   : {SOURCE_DIR}")
    print(f"Target raw      : {TARGET_RAW_DIR}")
    print(f"Pseudo labels   : {PSEUDO_LABEL_DIR}")
    print(f"Mixed dataset   : {MIXED_DATASET_DIR}")
    print(f"Results         : {RESULTS_DIR}")
    print(f"\nConf (burnin)   : {CONF_THRESHOLD_BURNIN}")
    print(f"Conf (pseudo)   : {CONF_THRESHOLD_PSEUDO}")
    print(f"Min track hits  : {MIN_TRACK_HITS}")
    print(f"Finetune epochs : {FINETUNE_EPOCHS}")
