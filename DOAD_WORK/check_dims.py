"""Check if COCO JSON dimensions match actual image file dimensions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import cv2
from pathlib import Path
from config import BASE_DIR

GT_JSON  = os.path.join(BASE_DIR, "cfc_channel_test.json")
TEST_DIR = os.path.join(BASE_DIR, "cfc_channel_test")

gt = json.load(open(GT_JSON))

mismatches = 0
checked = 0
for img in gt["images"][:500]:  # Check first 500
    path = Path(TEST_DIR) / img["file_name"]
    if not path.exists():
        continue
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    
    actual_h, actual_w = frame.shape[:2]
    json_w = img["width"]
    json_h = img["height"]
    
    if actual_w != json_w or actual_h != json_h:
        mismatches += 1
        if mismatches <= 3:
            print(f"  MISMATCH: {img['file_name']}: JSON=({json_w},{json_h}), actual=({actual_w},{actual_h})")
    checked += 1

print(f"Checked {checked} images, {mismatches} mismatches")

# Also check what MOG2+TVG preprocessing does to image dimensions
from preprocessing import MOG2TVGPreprocessor

preprocessor = MOG2TVGPreprocessor()
preprocessor.reset()

# Find first valid sequence
for img in gt["images"][:5]:
    path = Path(TEST_DIR) / img["file_name"]
    if not path.exists():
        continue
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    
    # Extract Ch3
    ch3 = frame[:, :, 2]
    processed = preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))
    
    print(f"\n  Image: {img['file_name']}")
    print(f"    Original:  {frame.shape}")
    print(f"    Ch3:       {ch3.shape}")
    print(f"    Processed: {processed.shape}")
    break
