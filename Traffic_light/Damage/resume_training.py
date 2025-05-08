from ultralytics import YOLO

# Correct path based on your file structure
model = YOLO("runs/yolov8n_gtsdb4/weights/last.pt")
model.train(resume=True)
