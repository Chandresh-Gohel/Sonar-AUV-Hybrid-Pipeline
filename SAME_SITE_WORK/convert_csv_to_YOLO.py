import pandas as pd
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from PIL import Image

# --- Configuration ---
# Your original clean CSVs
TRAIN_CSV      = 'train_annotations.csv'
TEST_CSV       = 'test_annotations.csv'

# The base directory where your images are
# Override with environment variable or edit directly.
IMAGE_BASE_DIR = os.environ.get("SONAR_IMAGE_BASE_DIR", "./data_preprocessed/raw")

# The new directory for our YOLO dataset
YOLO_DATA_DIR = Path('sonar_yolo_dataset_pre_pro')
IMG_SIZE = (512, 512) # We will resize images to this
# --- End Configuration ---

def process_csv(csv_path, dataset_type):
    """
    Reads a CSV, copies images, and creates YOLO .txt label files.
    dataset_type: 'train' or 'val'
    """
    
    print(f"Processing {dataset_type} data from {csv_path}...")
    
    # Create new directories
    img_output_dir = YOLO_DATA_DIR / 'images' / dataset_type
    label_output_dir = YOLO_DATA_DIR / 'labels' / dataset_type
    
    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Cannot find {csv_path}. Please make sure it's in the same folder.")
        return
    image_groups = df.groupby('image_path')
    
    for image_name, group in tqdm(image_groups, desc=f"Converting {dataset_type}"):
        
        original_img_path = Path(IMAGE_BASE_DIR) / image_name
        
        # Create a unique, flat filename for YOLO
        new_img_name = image_name.replace('/', '_').replace('\\', '_')
        new_img_path = img_output_dir / new_img_name
        
        try:
            with Image.open(original_img_path) as img:
                img_orig_w, img_orig_h = img.size
                img = img.resize(IMG_SIZE)
                img = img.convert("RGB") # Ensure 3 channels
                img.save(new_img_path)
        except FileNotFoundError:
            # This check is now redundant since your CSV is filtered, but good to have
            print(f"Warning: Image not found at {original_img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Failed to process {original_img_path}. Error: {e}. Skipping.")
            continue
            
        label_path = label_output_dir / (new_img_path.stem + '.txt')
        
        yolo_labels = []
        for _, row in group.iterrows():
            # YOLO format: [class_id] [x_center] [y_center] [width] [height]
            class_id = 0 # 0-indexed for 'fish'
            
            x_min, y_min = row['x_min'], row['y_min']
            x_max, y_max = row['x_max'], row['y_max']
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            x_center = x_min + box_w / 2
            y_center = y_min + box_h / 2
            
            # Normalize by original image size
            x_center_norm = x_center / img_orig_w
            y_center_norm = y_center / img_orig_h
            box_w_norm = box_w / img_orig_w
            box_h_norm = box_h / img_orig_h
            
            yolo_labels.append(f"{class_id} {x_center_norm} {y_center_norm} {box_w_norm} {box_h_norm}")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
            
    print(f"Finished processing {dataset_type} data.")

if __name__ == "__main__":
    
    if YOLO_DATA_DIR.exists():
        print(f"Removing old dataset directory: {YOLO_DATA_DIR}")
        shutil.rmtree(YOLO_DATA_DIR)
        
    process_csv(TRAIN_CSV, 'train')
    # process_csv(VAL_CSV, 'val')
    process_csv(TEST_CSV, 'test')
    
    print("\n--- SUCCESS! ---")
    print("YOLO dataset created at:", YOLO_DATA_DIR.resolve())