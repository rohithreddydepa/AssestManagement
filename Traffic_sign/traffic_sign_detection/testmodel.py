import cv2
import torch
from ultralytics import YOLO
import os

# =========================
# Configuration
# =========================
MODEL_PATH = "best.pt"  # Change if needed
TEST_IMAGE_DIR = "test_images/"  # Folder containing test images
OUTPUT_DIR = "test_results/"  # Folder to save results

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

# =========================
# Function to Run Inference on All Images
# =========================
def process_all_images():
    """Runs YOLOv8 inference on all images in the directory and saves results."""
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print("❌ No images found in", TEST_IMAGE_DIR)
        return

    print(f"🚀 Processing {len(image_files)} images...\n")

    for image_file in image_files:
        image_path = os.path.join(TEST_IMAGE_DIR, image_file)

        # Load image
        img = cv2.imread(image_path)

        # Run inference
        results = model(image_path)  # YOLOv8 inference
        result = results[0]  # First result (single image)

        # Draw bounding boxes on the image
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save output image
        save_path = os.path.join(OUTPUT_DIR, f"pred_{image_file}")
        cv2.imwrite(save_path, img)

        print(f"✅ Processed {image_file} | Saved to {save_path}")

# =========================
# Run Model Tests
# =========================
if __name__ == "__main__":
    print("🚀 Running YOLOv8 Model on All Images...\n")
    process_all_images()
    print("\n✅ Processing complete! Check results in", OUTPUT_DIR)
