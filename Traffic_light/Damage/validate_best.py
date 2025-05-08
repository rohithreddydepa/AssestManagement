from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/yolov8n_gtsdb4/weights/best.pt")

# Validate the model on the dataset
metrics = model.val(data="gtsdb.yaml")

# Print results
print("Validation Results:")
print(metrics)
