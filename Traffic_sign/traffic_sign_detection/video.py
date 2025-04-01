import cv2
import torch
from ultralytics import YOLO
import os

# =========================
# Configuration
# =========================
MODEL_PATH = "best.pt"  # Path to your trained YOLOv8 model
TEST_VIDEO = "test_video.mp4"  # Input video file
OUTPUT_DIR = "test_results/"  # Output directory
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output_video.mp4")  # Output video file

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

# =========================
# Function to Run Inference on a Video
# =========================
def process_video():
    """Runs YOLOv8 inference on a video and saves the output with bounding boxes."""
    cap = cv2.VideoCapture(TEST_VIDEO)  # Open video file

    if not cap.isOpened():
        print("❌ Failed to open video:", TEST_VIDEO)
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    print(f"🚀 Processing video: {TEST_VIDEO}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run YOLOv8 inference
        results = model(frame)
        result = results[0]

        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Display the video in a window (optional)
        cv2.imshow("YOLOv8 Video Detection", frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Video processing complete! Output saved to {OUTPUT_VIDEO}")

# =========================
# Run Video Processing
# =========================
if __name__ == "__main__":
    print("🚀 Running YOLOv8 on Video...\n")
    process_video()
    print("\n✅ Video processing finished!")
