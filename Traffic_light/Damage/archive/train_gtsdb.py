from ultralytics import YOLO

# Load a YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="gtsdb.yaml", 
    epochs=50, 
    imgsz=640, 
    project="runs", 
    name="yolov8n_gtsdb"
)
