import os
import torch
from ultralytics import YOLO

# Get the correct absolute path dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "experiment", "weights", "best.pt")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# Load trained model
model = YOLO(MODEL_PATH)

# Force CPU usage
DEVICE = "cpu"
print(f"🔍 Running inference on: {DEVICE}")

# Run inference
results = model.predict(
    source=TEST_IMAGES_DIR,
    save=True,
    project=OUTPUT_DIR,
    name="inference_results",
    conf=0.4,
    iou=0.5,
    device=DEVICE
)

print(f"✅ Inference complete. Results saved in {OUTPUT_DIR}")
