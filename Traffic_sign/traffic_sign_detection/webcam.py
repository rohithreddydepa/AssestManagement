import cv2
import torch
from ultralytics import YOLO

# =========================
# Configuration
# =========================
MODEL_PATH = "best.pt"  # Path to your trained YOLOv8 model

# Load YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

# =========================
# Function to Run Inference on Webcam
# =========================
def test_webcam():
    """Run YOLOv8 on live webcam feed."""
    cap = cv2.VideoCapture(0)  # Open webcam (0 = default webcam)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Run YOLOv8 inference
        results = model(frame)

        # Render results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam closed.")

# =========================
# Run Model Tests
# =========================
if __name__ == "__main__":
    print("🚀 Running YOLOv8 on Webcam...\n")
    test_webcam()
