import cv2
import numpy as np
import os
import urllib.request

# Download YOLO files if not already present
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}")

# URLs for YOLOv3 files
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
config_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
labels_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Paths to save the files
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"

# Download files
download_file(weights_url, weights_path)
download_file(config_url, config_path)
download_file(labels_url, labels_path)

# Load labels
with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(weights_path, config_path)

folder_path = "test"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Process images
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    print(f"Processing {image_name}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)

    # Collect relevant boxes
boxes = []
confidences = []

for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and classes[class_id] == "traffic light":
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Apply NMS to keep only best non-overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

if len(indexes) > 0:
    for idx in indexes:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        x, y, w, h = boxes[i]
        label = f"Traffic Light: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

print("Done. Saved to 'output' folder.")