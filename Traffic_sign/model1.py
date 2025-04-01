from ultralytics import YOLO
import torch
import os
import glob
import pandas as pd
from datetime import datetime

# =========================
# Configuration
# =========================
class Config:
    MODEL_TYPE = "yolov8l"
    PRETRAINED_WEIGHTS = f"{MODEL_TYPE}.pt"
    
    # Training
    EPOCHS = 100
    PATIENCE = 20
    IMAGE_SIZE = 1280
    BATCH_SIZE = 16
    OPTIMIZER = "AdamW"
    LR0 = 1e-3
    LRF = 0.01
    WORKERS = 8

    # Augmentation
    AUGMENT = True
    MOSAIC = 0.8
    MIXUP = 0.2
    FLIP_LR = 0.3
    PERSPECTIVE = 0.0005
    HSV_H = 0.015

    # Regularization
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY = 0.0005

# =========================
# Training Function
# =========================
def train_model():
    """
    Multi-GPU training using YOLOv8.
    """
    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} GPU(s) on this node.")

    if gpu_count > 1:
        device_list = ",".join(str(i) for i in range(gpu_count))  # "0,1,2,3"
        print(f"✅ Using Multi-GPU: {device_list}")
    elif gpu_count == 1:
        device_list = "0"  # Single GPU
        print("✅ Using single GPU (device=0).")
    else:
        device_list = "cpu"
        print("❌ No GPU found. Using CPU.")

    # Initialize YOLO model
    model = YOLO(Config.PRETRAINED_WEIGHTS)

    try:
        results = model.train(
            data="dataset.yaml",
            cache="disk",  # Prevent OOM issues
            epochs=Config.EPOCHS,
            imgsz=1280,
            batch=16,  # Adjust batch size based on available memory
            device=device_list,  # Set multi-GPU device list
            patience=Config.PATIENCE,
            optimizer=Config.OPTIMIZER,
            lr0=Config.LR0,
            lrf=Config.LRF,
            workers=Config.WORKERS,
            augment=Config.AUGMENT,
            mosaic=Config.MOSAIC,
            mixup=Config.MIXUP,
            fliplr=Config.FLIP_LR,
            perspective=Config.PERSPECTIVE,
            hsv_h=Config.HSV_H,
            dropout=Config.DROPOUT,
            weight_decay=Config.WEIGHT_DECAY,
            close_mosaic=10,
            amp=True,
            plots=True,
            save_json=True,
            exist_ok=True,
            resume=False
        )
        return results
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

# =========================
# Find Best Model Function
# =========================
def find_best_model():
    """Find the best model based on mAP50-95 using `results.csv`."""
    exp_dirs = sorted(glob.glob("runs/train/exp*"), key=os.path.getmtime, reverse=True)
    if not exp_dirs:
        raise FileNotFoundError("❌ No training runs found.")

    best_model_path = None
    best_mAP = 0.0

    for exp_dir in exp_dirs:
        results_file = os.path.join(exp_dir, "results.csv")
        model_path = os.path.join(exp_dir, "weights/best.pt")

        if os.path.exists(results_file) and os.path.exists(model_path):
            try:
                df = pd.read_csv(results_file)
                last_row = df.iloc[-1]  # Get last epoch's results
                current_mAP = last_row.get("metrics/mAP50-95(B)", 0)

                if current_mAP > best_mAP:
                    best_mAP = current_mAP
                    best_model_path = model_path
            except Exception as e:
                print(f"⚠️ Error reading {results_file}: {str(e)}")

    if not best_model_path:
        raise ValueError("❌ No valid models found.")

    print(f"🏆 Best Model: {best_model_path} | 📈 mAP50-95: {best_mAP:.3f}")
    return best_model_path

# =========================
# Export Best Model Function
# =========================
def export_best_model(best_model_path):
    """Export the best model to ONNX and TensorRT (if supported)."""
    best_model = YOLO(best_model_path)
    export_formats = ["onnx", "engine"]  # e.g. ONNX and TensorRT
    for fmt in export_formats:
        try:
            best_model.export(format=fmt, imgsz=Config.IMAGE_SIZE)
            print(f"✅ Exported best model to {fmt.upper()}")
        except Exception as e:
            print(f"⚠️ Export to {fmt} failed: {str(e)}")

# =========================
# Script Entry Point
# =========================
if __name__ == "__main__":
    # 1. Verify dataset
    if not os.path.exists("dataset.yaml"):
        raise FileNotFoundError("❌ dataset.yaml not found! Check your dataset path.")

    # 2. Train
    start_time = datetime.now()
    print(f"🚀 Starting training at {start_time}")
    train_results = train_model()
    
    if train_results:
        # 3. Find best model
        best_model_path = find_best_model()
        
        # 4. Export the best model
        export_best_model(best_model_path)
    
    # 5. Print total runtime
    total_duration = datetime.now() - start_time
    print(f"⏱️ Total duration: {total_duration}")
