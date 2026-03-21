import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# -------------------------------
# CONFIG
# -------------------------------
# --- 1. SET YOUR PATHS ---
# The root folder containing all data (e.g., ...\raw)
ROOT_INPUT_DIR = r"D:\SONAR_PROJECT\_SONAR_RUN_3\tiny_dataset\raw"

# Where to save the mirrored output structure
ROOT_OUTPUT_DIR = r"D:\SONAR_PROJECT\_SONAR_RUN_3\tiny_dataset_pre_pro\raw"

# --- 2. SET FILE EXTENSION ---
FILE_EXTENSION = "jpg"

# --- 3. MOG2 PARAMETERS ---
MOG2_HISTORY = 200  # How many frames to use for the background model

# -------------------------------
# 1. Setup & Find Sequences
# -------------------------------
print("Starting recursive baseline++ generation...")
root_in = Path(ROOT_INPUT_DIR)
root_out = Path(ROOT_OUTPUT_DIR)
root_out.mkdir(parents=True, exist_ok=True)

if not root_in.exists():
    print(f"Error: Input directory not found: {root_in}")
    sys.exit()

# Recursively find all images
print(f"Scanning for '*{FILE_EXTENSION}' files in {root_in}...")
all_images = list(root_in.rglob(f"*.{FILE_EXTENSION}"))

if not all_images:
    print(f"Error: No '*{FILE_EXTENSION}' files found anywhere inside {root_in}")
    sys.exit()

# Get a unique set of all parent directories that contain images
sequence_folders = sorted(list(set([img_file.parent for img_file in all_images])))

print(f"Found {len(all_images)} images across {len(sequence_folders)} unique sequences (folders).")

# -------------------------------
# 2. Main Loop (Per Sequence Folder)
# -------------------------------
for sequence_folder in sequence_folders:
    
    # Create the matching output folder structure
    # 'relative_to' correctly maps the subfolder path
    relative_path = sequence_folder.relative_to(root_in)
    output_folder = root_out / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all images *just for this folder*
    image_files = sorted(sequence_folder.glob(f"*.{FILE_EXTENSION}"))
    
    if not image_files:
        continue # Should not happen, but good to check

    print(f"\n--- Processing sequence: {relative_path} ({len(image_files)} images) ---")

    # --- Initialize TVG (from your script) ---
    try:
        first_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
        h, w = first_img.shape
    except Exception as e:
        print(f"Error reading first image in {sequence_folder}: {e}")
        continue
    
    # Create the TVG map once per sequence
    x = np.arange(w, dtype=np.float32)
    range_map = np.tile(x, (h, 1))
    R = range_map * 0.0005 + 1e-3
    tvg = np.clip(20 * np.log10(R) + 0.04 * R, 0, 50)
    tvg_lin = 10 ** (tvg / 20.0)

    # --- Initialize Background Subtractor (CRITICAL!) ---
    # Create a NEW subtractor for EACH sequence
    backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, detectShadows=False)
    
    # Reset the previous frame for each new sequence
    prev_frame_tvg = None

    # --- Inner Loop (Per Image) ---
    for img_path in tqdm(image_files, desc=f"Processing {relative_path.name}"):
        
        frame_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if frame_gray is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        # Step A: Apply TVG Correction
        frame_tvg = np.clip(frame_gray.astype(np.float32) * tvg_lin, 0, 255).astype(np.uint8)

        # Step B: Generate 3 Channels
        
        # Channel 1: Foreground Mask
        fg_mask = backSub.apply(frame_tvg)

        # Channel 2: Background Model
        bg_model = backSub.getBackgroundImage()

        # Channel 3: Motion Mask (Frame Differencing)
        if prev_frame_tvg is not None:
            motion_mask = cv2.absdiff(frame_tvg, prev_frame_tvg)
        else:
            motion_mask = np.zeros_like(frame_tvg)
            
        prev_frame_tvg = frame_tvg.copy()

        # Step C: Merge and Save
        # B = Foreground, G = Background, R = Motion
        # Using os.path.basename to get just the filename
        output_name = os.path.basename(img_path)
        output_path = output_folder / output_name
        
        try:
            output_image = cv2.merge([fg_mask, bg_model, motion_mask])
            cv2.imwrite(str(output_path), output_image)
        except cv2.error as e:
            print(f"Error merging/saving {img_path}: {e}")
            print(f"fg_mask shape: {fg_mask.shape}, bg_model shape: {bg_model.shape}, motion_mask shape: {motion_mask.shape}")
            # This can happen if the first frame's bg_model isn't ready
            if bg_model is None:
                print("Skipping frame, background model not yet initialized.")
                # Create a black background model as a placeholder if needed
                bg_model = np.zeros_like(fg_mask)
                motion_mask = np.zeros_like(fg_mask) # also ensure motion mask exists
                output_image = cv2.merge([fg_mask, bg_model, motion_mask])
                cv2.imwrite(str(output_path), output_image) # Try saving again
            

print("\n--- All Sequences Processed! ---")
print(f"Your 'preprocessed' images are ready in: {ROOT_OUTPUT_DIR}")
print("The folder structure from 'raw' has been mirrored in the output directory.")