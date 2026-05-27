import pandas as pd
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- 1. CRITICAL: YOU MUST UPDATE THESE CONSTANTS ---

# --- Train/Test Split ---
# What percentage of images should go into the training set?
TRAIN_SPLIT_RATIO = 0.8  # 80% for training, 20% for testing

# --- Image Filename Format ---
# Look at your image files (e.g., in .../raw/kanai-train/dir1/images/)
# and set these to match the filenames.
#
# Example: if your images are named '000001.jpg', '000002.jpg', etc.
FRAME_PADDING = 0  # 6 digits (e.g., '000001')
IMAGE_EXTENSION = '.jpg' # '.jpg' or '.png'
# ---

def convert_and_filter_mot(annotations_dir, images_dir, output_train_csv, output_test_csv):
    """
    Parses MOT-style directories, filters for existing images,
    labels EVERYTHING as a single 'fish' class, and splits
    the results into train and test CSVs.
    """
    
    gt_files = list(Path(annotations_dir).rglob('gt_tiny.txt'))
    
    if not gt_files:
        print(f"Error: No 'gt_tiny.txt' files found in {annotations_dir}")
        print("Note: This script looks for 'gt_tiny.txt', not 'gt_tiny.txt'.")
        return

    print(f"Found {len(gt_files)} 'gt_tiny.txt' files. Processing and filtering...")
    
    all_valid_annotations = []
    image_base_path = Path(images_dir)
    
    total_annotations_read = 0
    total_frames_found = 0
    total_frames_missing = 0
    total_fish_count = 0

    for gt_path in gt_files:
        try:
            relative_ann_dir = gt_path.parent.relative_to(annotations_dir)
        except ValueError:
            print(f"Warning: Skipping {gt_path} as it's not in a standard subfolder.")
            continue
            
        # This is the path we will check for images
        # e.g., 'raw/kanai-train/dir1'
        image_check_dir = image_base_path / relative_ann_dir
        image_csv_dir = relative_ann_dir
        
        if not image_check_dir.is_dir():
            print(f"Warning: Corresponding image folder not found, skipping: {image_check_dir}")
            continue

        try:
            df = pd.read_csv(
                gt_path, 
                header=None, 
                names=['frame_num', 'track_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
            )
            total_annotations_read += len(df)
        except Exception as e:
            print(f"Warning: Could not read {gt_path}. Skipping. Error: {e}")
            continue
            
        # Group by frame to process one image at a time
        grouped = df.groupby('frame_num')
        
        for frame_num, boxes in grouped:
            frame_str = str(frame_num).zfill(FRAME_PADDING)
            image_name_for_csv = str(image_csv_dir / (frame_str + IMAGE_EXTENSION))
            full_img_check_path = image_base_path / image_name_for_csv
            # The Critical Check: Does this image file exist?
            if full_img_check_path.exists():
                total_frames_found += 1
                
                # --- NEW SINGLE-CLASS LOGIC ---
                # We are now ignoring school/fry rules.
                # Every valid bounding box is just a "fish".
                for _, row in boxes.iterrows():
                    total_fish_count += 1
                    x_min = row['bb_left'] - 1
                    y_min = row['bb_top'] - 1
                    x_max = x_min + row['bb_width']
                    y_max = y_min + row['bb_height']
                    all_valid_annotations.append({
                        'image_path': image_name_for_csv,
                        'bb_left': row['bb_left'],
                        'bb_top': row['bb_top'],
                        'bb_width': row['bb_width'],
                        'bb_height': row['bb_height'],
                        'class_id': 'fish', # Hard-coded single class ## Class_For Fish
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                    })
                # --- END NEW LOGIC ---
            else:
                # Discard annotations for this frame, image file is missing
                total_frames_missing += 1
    # --- Save the final CSVs ---
    if not all_valid_annotations:
        print("Error: No valid, matching annotations were found.")
        print("Please check your FRAME_PADDING and IMAGE_EXTENSION constants.")
        return
        
    master_df = pd.DataFrame(all_valid_annotations)
    
    # --- Split into Train and Test ---
    # We must split by image, not by annotation
    unique_images = master_df['image_path'].unique()
    np.random.shuffle(unique_images)
    
    split_index = int(len(unique_images) * TRAIN_SPLIT_RATIO)
    train_images = unique_images[:split_index]
    test_images = unique_images[split_index:]
    
    train_df = master_df[master_df['image_path'].isin(train_images)]
    test_df = master_df[master_df['image_path'].isin(test_images)]
    
    # Save the files
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)
    
    print("\n--- Success! ---")
    print(f"Total annotations read: {total_annotations_read}")
    print(f"Frames with images: {total_frames_found}")
    print(f"Frames skipped (no image): {total_frames_missing}")
    print("\n--- Labeling Stats ---")
    print(f"Total 'fish' boxes generated: {total_fish_count}")
    print("\n--- Split Stats ---")
    print(f"Total unique images: {len(unique_images)}")
    print(f"Train images: {len(train_images)} ({len(train_df)} annos)")
    print(f"Test images:  {len(test_images)} ({len(test_df)} annos)")
    print(f"\nFiltered annotations saved to:")
    print(f"TRAIN: {output_train_csv}")
    print(f"TEST:  {output_test_csv}")


if __name__ == "__main__":
    # --- How to run this script ---
    #
    # 1. Update constants at the top (PADDING, EXTENSION).
    #
    # 2. Run from your command line:
    #
    # python create_single_class_dataset.py \
    #   --annotations_dir annotations-tiny \
    #   --images_dir raw \
    #   --output_train_csv train_annotations.csv \
    #   --output_test_csv test_annotations.csv
    #
    # -----------------------------------------------------------------
    
    parser = argparse.ArgumentParser(description="Filter MOT data, apply single 'fish' label, and split into train/test CSVs.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Base annotation directory (e.g., 'annotations-tiny')")
    parser.add_argument("--images_dir", type=str, required=True, help="Base image directory (e.g., 'raw')")
    parser.add_argument("--output_train_csv", type=str, required=True, help="Path to save the training CSV (e.g., 'train_annotations.csv')")
    parser.add_argument("--output_test_csv", type=str, required=True, help="Path to save the testing CSV (e.g., 'test_annotations.csv')")
    
    args = parser.parse_args()
    
    # --- Dummy file creation (from your script) for testing ---
    if not os.path.exists(args.annotations_dir) and not os.path.exists(args.images_dir):
        print("--- WARNING: Creating dummy files for testing script ---")
        
        from PIL import Image
        
        # Path for gt_tiny.txt
        dummy_ann_path = Path(args.annotations_dir) / 'kanai-train' / 'dir1' / 'gt'
        os.makedirs(dummy_ann_path, exist_ok=True)
        
        # Path for images
        dummy_img_path = Path(args.images_dir) / 'kanai-train' / 'dir1'
        os.makedirs(dummy_img_path, exist_ok=True)
        
        # Create dummy gt_tiny.txt
        with open(dummy_ann_path / 'gt_tiny.txt', 'w') as f:
            # Frame 70 (School) > 10 boxes -> will become 12 'fish' boxes
            for i in range(12):
                f.write(f"70,{i},50,50,20,20,-1,-1,-1,-1\n")
            # Frame 71 (Fry) -> will become 1 'fish' box
            f.write("71,1,50,50,10,10,-1,-1,-1,-1\n")
            # Frame 72 (Fish) -> will become 1 'fish' box
            f.write("72,1,50,50,100,100,-1,-1,-1,-1\n")
            # Frame 73 (Missing image)
            f.write("73,1,50,50,100,100,-1,-1,-1,-1\n")
            
        # Create dummy image files
        for frame_num in [70, 71, 72]:
            frame_str = str(frame_num).zfill(FRAME_PADDING)
            image_name = f"{frame_str}{IMAGE_EXTENSION}"
            Image.new('RGB', (100, 100)).save(dummy_img_path / image_name)
        
        print("Dummy files created. Running script...")
    
    elif not os.path.exists(args.annotations_dir):
        print(f"Error: Annotation directory not found: {args.annotations_dir}")
        exit()
    elif not os.path.exists(args.images_dir):
        print(f"Error: Image directory not found: {args.images_dir}")
        exit()
        
    convert_and_filter_mot(args.annotations_dir, args.images_dir, args.output_train_csv, args.output_test_csv)