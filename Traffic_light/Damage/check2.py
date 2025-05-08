from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/yolov8n_gtsdb4/weights/best.pt")

# Run detection on the image with lower confidence threshold
results = model.predict(
    source="test_image.jpg",
    conf=0.1,        # lower this to catch weak detections
    iou=0.4,         # optional: increase to include overlapping boxes
    save=True,
    save_txt=True,
    verbose=True
)

print("✅ Classes:", model.names)
