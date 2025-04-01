
from ultralytics import YOLO
import torch
import shutil
import os
import glob
import pandas as pd

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")

# Ensure dataset.yaml exists
if not os.path.exists("dataset.yaml"):
    raise FileNotFoundError("❌ dataset.yaml not found! Ensure the correct path.")

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model with early stopping
model.train(
    data="dataset.yaml",
    epochs=50,
    batch=16 if device == "cuda" else 4,
    imgsz=640,
    device=device,
    patience=5,
    optimizer="AdamW",
    workers=8 if device == "cuda" else 0
)

# ✅ Find the latest YOLO experiment folder
exp_dirs = sorted(glob.glob("runs/train/exp*"), key=os.path.getmtime, reverse=True)

if not exp_dirs:
    raise FileNotFoundError("❌ No training runs found. Training might have failed.")

best_model_path = None
best_mAP = 0.0  # Track highest mAP_50-95

# ✅ Iterate through experiments and find the best model
for exp_dir in exp_dirs:
    results_csv = os.path.join(exp_dir, "results.csv")
    best_pt = os.path.join(exp_dir, "weights/best.pt")

    # ✅ Ensure required files exist before processing
    if os.path.exists(results_csv) and os.path.exists(best_pt):
        df = pd.read_csv(results_csv)