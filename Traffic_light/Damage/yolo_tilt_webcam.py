import cv2
import numpy as np
from ultralytics import YOLO
from tilt_angle_model import predict_tilt


model = YOLO("yolov8n.pt")

# Define traffic sign labels only
traffic_sign_labels = {
    'stop', 'stop sign', 'speed_limit_20', 'speed_limit_30', 'speed_limit_50',
    'speed_limit_60', 'speed_limit_70', 'speed_limit_80', 'speed_limit_100',
    'speed_limit_120', 'no_entry', 'no_overtaking', 'no_trucks',
    'priority_road', 'yield', 'caution', 'dangerous_curve_left',
    'dangerous_curve_right', 'children', 'pedestrian_crossing', 'roundabout',
    'turn_left', 'turn_right', 'go_straight', 'keep_right', 'keep_left'
}


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]

        
        cropped = frame[y1:y2, x1:x2]

        if label in traffic_sign_labels:
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            
            tilt_angle = predict_tilt(cropped)
            if tilt_angle is not None:
                tilt_text = f"Tilt: {int(round(tilt_angle))} degrees"
            else:
                tilt_text = "Tilt: ?"
            cv2.putText(frame, tilt_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        else:
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    
    cv2.imshow("YOLO Traffic Sign Tilt Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
