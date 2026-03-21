# Sonar-AUV-Hybrid-Pipeline

This repository provides tools to prepare and evaluate the **Caltech Fish Counting Dataset (2022)** for **YOLO-based fish detection** in sonar images. The original dataset is designed for detection, tracking, and counting fish in sonar videos, but this project focuses on **static frame object detection** using YOLO.

We include scripts for:
- Converting original annotations (likely CSV or similar) → YOLO `.txt` label format
- Preprocessing sonar images (e.g., contrast enhancement, noise reduction)
- Filtering misaligned image-label pairs
- Training/evaluation on raw vs. preprocessed data

## Dataset

- **Source**: [Caltech Fish Counting Dataset 2022](https://data.caltech.edu/records/1y23m-j8r69)
- **Paper**: [The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting (ECCV 2022)](https://arxiv.org/abs/2207.09295)
- **Official repo / guide**: [visipedia/caltech-fish-counting](https://github.com/visipedia/caltech-fish-counting)
- **Used subset**: `tiny_dataset.tar.gz` (~1.4 GB) – small version for experimentation

The dataset contains sonar video frames (ARIS sonar) with fish annotations (bounding boxes). This project treats frames as independent images for object detection (class: fish / class 0).

## Features / Scripts

| Script                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `convert_csv_to_YOLO.py`      | Converts original CSV annotations to YOLO `.txt` label files (normalized xywh format) |
| `create_csv.py`               | Filters image-label pairs (removes non-aligning / corrupted entries) and creates cleaned CSV |
| `pre_pro_data.py`             | Generates preprocessed images (e.g., denoising, contrast adjustment for sonar) |
| `evaluate_yolo_raw.py`        | Evaluates YOLO model performance on **raw (original)** sonar images         |
| `evaluate_yolo_pre_pro.py`    | Evaluates the same YOLO model on **preprocessed** images – shows improvement |
| `sonar_dataset_pre_pro.yaml`  | YOLO dataset configuration file for the **preprocessed** dataset            |

Typical workflow:
1. Download & extract `tiny_dataset.tar.gz`
2. Run filtering → cleaning (`create_csv.py`)
3. Convert annotations to YOLO format (`convert_csv_to_YOLO.py`)
4. Generate preprocessed images (`pre_pro_data.py`)
5. Create dataset YAML for raw and preprocessed versions
6. Train/evaluate YOLO (e.g. YOLOv8/YOLOv11 via Ultralytics)
7. Compare metrics: raw vs. preprocessed

## Installation

```bash
# Recommended: Python 3.9–3.11
git clone https://github.com/YOUR-USERNAME/sonar-yolo-fish-detection.git
cd sonar-yolo-fish-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies (adjust versions as needed)
pip install -r requirements.txt
