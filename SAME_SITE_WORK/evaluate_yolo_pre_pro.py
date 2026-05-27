from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import os
import cv2  # Added for drawing
import shutil

# --- Configuration ---
MODEL_PATHS = {
    "yolov5nu": r'runs\detect\yolov5n_pre_pro_run_100e\weights\best.pt',
    "yolov5su": r'runs\detect\yolov5s_pre_pro_run_100e\weights\best.pt',
    "yolov8n": r'runs\detect\yolov8n_pre_pro_run_100e\weights\best.pt',
    "yolov8s": r'runs\detect\yolov8s_pre_pro_run_100e\weights\best.pt',
    "yolo11n": r'runs\detect\yolo11n_pre_pro_run_100e\weights\best.pt',
    "yolo11s": r'runs\detect\yolo11s_pre_pro_run_100e\weights\best.pt',
    "yolo26n": r'runs\detect\yolo26n_pre_pro_run_100e\weights\best.pt',
    "yolo26s": r'runs\detect\yolo26s_pre_pro_run_100e\weights\best.pt',
}
VAL_CSV_PATH = 'cfc_test_pre_pro.csv'
VAL_IMG_DIR = r'sonar_yolo_dataset_pre_pro\images\val'
# Assumes labels are in standard YOLO structure: .../images/val -> .../labels/val
VAL_LABEL_DIR = VAL_IMG_DIR.replace('images', 'labels') 

CONF_THRESHOLD = 0.4 
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512
VIS_OUTPUT_ROOT = r'runs\inference_vis' # Root folder for saved visualizations
# ---

def get_gt_counts(csv_path):
    """Gets the ground-truth counts from our original CSV."""
    print(f"Reading ground truth counts from {csv_path}...")
    df = pd.read_csv(csv_path)
    def get_yolo_name(image_name):
        return image_name.replace('/', '_').replace('\\', '_')
    df['yolo_name'] = df['image_path'].apply(get_yolo_name)
    gt_counts = df.groupby('yolo_name').size()
    return gt_counts.to_dict()

def save_visuals_and_logs(img_path, results, gt_count_csv, output_dir_img, output_dir_txt):
    """
    Draws GT (Green) and Pred (Red) boxes on image and saves it.
    Saves a text file with counts.
    """
    # 1. Load Image
    img = cv2.imread(str(img_path))
    if img is None: 
        return
    h, w, _ = img.shape
    
    # 2. Determine GT Label Path (Standard YOLO assumption)
    # Replaces 'images' with 'labels' and extension with .txt
    label_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
    
    # 3. Draw Ground Truth Boxes (GREEN)
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    cls, xc, yc, bw, bh = parts
                    
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Green
    
    # 4. Draw Predicted Boxes (RED)
    pred_count = len(results[0].boxes)
    for box in results[0].boxes:
        # xyxy coordinates
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        conf = box.conf[0].item()
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red

    # 5. Save Image
    save_img_name = img_path.name
    cv2.imwrite(str(Path(output_dir_img) / save_img_name), img)

    # 6. Save Text Log
    save_txt_name = img_path.stem + ".txt"
    with open(Path(output_dir_txt) / save_txt_name, 'w') as f:
        f.write(f"Image: {img_path.name}\n")
        f.write(f"Ground Truth Count (CSV): {gt_count_csv}\n")
        f.write(f"Predicted Count: {pred_count}\n")


def calculate_mae_and_visualize(model, gt_counts, img_paths, model_name):
    """Calculates MAE and saves visualizations."""
    print(f"  Calculating MAE and generating visualizations for {model_name}...")
    
    # Setup Output Directories
    vis_dir = Path(VIS_OUTPUT_ROOT) / model_name
    img_out_dir = vis_dir / "images"
    txt_out_dir = vis_dir / "labels" # or "logs"
    
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    txt_out_dir.mkdir(parents=True, exist_ok=True)
    
    errors, norm_errors = [], []
    
    for img_path in tqdm(img_paths, desc="  Processing", leave=False):
        img_name = img_path.name
        gt_count = gt_counts.get(img_name, 0)
        
        results = model.predict(
            img_path,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=DEVICE,
            imgsz=IMG_SIZE
        )
        pred_count = len(results[0].boxes)
        
        # --- Visualization & Logging Call ---
        save_visuals_and_logs(img_path, results, gt_count, img_out_dir, txt_out_dir)
        # ------------------------------------

        errors.append(abs(pred_count - gt_count))
        if gt_count > 0:
            norm_errors.append(abs(pred_count - gt_count)/gt_count)
        elif pred_count == 0:
            norm_errors.append(0.0)
        
    return np.mean(errors), np.mean(norm_errors) if errors else -1

def get_model_stats(model_name, model_path, img_paths, gt_counts):
    """
    Loads a model and runs all benchmarks.
    """
    print(f"\n--- Benchmarking Model: {model_name} ---")
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Skipping.")
        return None
        
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}. Error: {e}")
        return None

    # 1. Get mAP, Precision, and Speed from model.val()
    print("  Running model.val() ...")
    val_results = model.val(
        data='sonar_dataset_pre_pro.yaml',
        imgsz=IMG_SIZE,
        batch=1, 
        split='val',
        device=DEVICE,
        verbose=False,
        plots=False
    )
    
    # 2. Get MAE (Counting) & Generate Visuals
    # PASSED model_name here
    mae, nmae = calculate_mae_and_visualize(model, gt_counts, img_paths, model_name)

    # 3. Get Model Params and Size
    # Check if base pt exists, else use loaded model
    base_pt_path = model_name + '.pt'
    if os.path.exists(base_pt_path):
        base_model = YOLO(base_pt_path)
        params_m = sum(p.numel() for p in base_model.parameters()) / 1e6
    else:
        # Fallback to current model params if base not found
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # 4. Get FPS
    inference_time_ms = val_results.speed['inference']
    fps_gpu = 1000.0 / inference_time_ms

    return {
        "Model": model_name,
        "Params (M)": params_m,
        "Size (MB)": size_mb,
        "mAP@0.5:0.95": val_results.box.map,
        "mAP@0.5": val_results.box.map50,
        "Precision": val_results.box.p[0], 
        "MAE (Count)": mae,
        "nMAE (Count)": nmae,
        "FPS (GPU)": fps_gpu,
    }

if __name__ == "__main__":
    print("--- Starting Full Model Benchmark Evaluation ---")
    print(f"--- Visualizations will be saved to: {VIS_OUTPUT_ROOT} ---")
    
    gt_counts = get_gt_counts(VAL_CSV_PATH)
    
    img_paths = list(Path(VAL_IMG_DIR).glob('*.jpg'))
    if not img_paths:
        img_paths = list(Path(VAL_IMG_DIR).glob('*.png'))
        
    if not img_paths:
        print(f"Error: No images found in {VAL_IMG_DIR}. Exiting.")
        exit()

    all_stats = []
    
    for model_name, model_path in MODEL_PATHS.items():
        stats = get_model_stats(model_name, model_path, img_paths, gt_counts)
        if stats:
            all_stats.append(stats)

    # --- Print Final Comparison Table ---
    print("\n\n--- FINAL THESIS BENCHMARK TABLE ---")
    
    if all_stats:
        header = all_stats[0].keys()
        print(f"{'Model':<26} | {'Params(M)':<9} | {'Size(MB)':<9} | {'mAP@.5:.95':<10} | {'mAP@.5':<10} | {'Precision':<10} | {'MAE':<10} | {'nMAE':<10} | {'FPS(GPU)':<10}")
        print("-" * 125)
        
        for stats in all_stats:
            map_val = f"{stats['mAP@0.5:0.95']:.3f}" if stats['mAP@0.5:0.95'] != -1 else "N/A"
            map50_val = f"{stats['mAP@0.5']:.3f}" if stats['mAP@0.5'] != -1 else "N/A"
            prec_val = f"{stats['Precision']:.3f}" if stats['Precision'] != -1 else "N/A"
            mae_val = f"{stats['MAE (Count)']:.3f}" if stats['MAE (Count)'] != -1 else "N/A"
            nmae_val = f"{stats['nMAE (Count)']:.3f}" if stats['nMAE (Count)'] != -1 else "N/A"
            fps_gpu_val = f"{stats['FPS (GPU)']:.1f}" if stats['FPS (GPU)'] != -1 else "N/A"

            print(f"{stats['Model']:<26} | "
                  f"{stats['Params (M)']:<9.2f} | "
                  f"{stats['Size (MB)']:<9.2f} | "
                  f"{map_val:<10} | "
                  f"{map50_val:<10} | "
                  f"{prec_val:<10} | "
                  f"{mae_val:<10} | "
                  f"{nmae_val:<10} | "
                  f"{fps_gpu_val:<10} | ")
                  
        print("-" * 125)