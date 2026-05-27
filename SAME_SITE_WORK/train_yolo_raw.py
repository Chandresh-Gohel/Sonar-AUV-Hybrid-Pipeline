from ultralytics import YOLO
import torch
import os

# --- Configuration ---
DATASET_CONFIG = 'sonar_dataset.yaml'
EPOCHS =  100
IMG_SIZE = 512
BATCH_SIZE = 8 # Adjust if you get Out-of-Memory, e.g., 4 or 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 10 # Early stopping: stops if no improvement after 10 epochs
# ---

def train_yolo(model_name, run_name):
    """Trains a specific YOLO model with early stopping and plotting."""
    print(f"\n--- Starting YOLO training for {model_name} ({EPOCHS} Epochs) ---")
    
    # Load the pre-trained lightweight model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=run_name,
        patience=PATIENCE, # Enable early stopping
        workers=3,
        plots=True # This automatically generates your loss graphs
    )
    
    print(f"\n--- Training complete for {model_name} ---")
    print(f"Model and plots saved in 'runs/detect/{run_name}'")
    return results

if __name__ == "__main__":
    # Check if dataset config exists
    if not os.path.exists(DATASET_CONFIG):
        print(f"ERROR: {DATASET_CONFIG} not found.")
        print("Please run convert_csv_to_yolo.py and create the .yaml file first.")
    else:
        print(f"Using device: {DEVICE}")
        
        # 1. Train YOLOv5n
        train_yolo(
            model_name='yolov5n.pt', 
            run_name='yolov5n_100e_run'
        )
        
        # 2. Train YOLOv8n
        train_yolo(
            model_name='yolov8n.pt', 
            run_name='yolov8n_100e_run'
        )
        # 3. Train YOLOv5s
        train_yolo(
            model_name='yolov5s.pt', 
            run_name='yolov5s_100e_run'
        )
        
        # 4. Train YOLOv8s
        train_yolo(
            model_name='yolov8s.pt', 
            run_name='yolov8s_100e_run'
        )

        # 5. Train YOLOv11s
        train_yolo(
            model_name='yolo11s.pt', 
            run_name='yolo11s_100e_run'
        )
        # 6. Train YOLOv11n
        train_yolo(
            model_name='yolo11n.pt', 
            run_name='yolo11n_100e_run'
        )
        # 7. Train YOLOv26n
        train_yolo(
            model_name='yolo26n.pt', 
            run_name='yolo26n_100e_run'
        )
        # 8. Train YOLOv26s
        train_yolo(
            model_name='yolo26s.pt', 
            run_name='yolo26s_100e_run'
        )
        print("\n\n--- All YOLO training finished! ---")