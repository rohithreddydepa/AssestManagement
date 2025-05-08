import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("tilt_angle_model.h5", compile=False)

IMG_SIZE = 64  # Input size for your model

def preprocess_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # OPTIONAL: Select a fixed ROI (Region of Interest) where the sign appears
    # Crop center square for now — you can adjust this logic
    h, w, _ = frame.shape
    roi = frame[h//2 - 100:h//2 + 100, w//2 - 100:w//2 + 100]

    # Make prediction on the ROI
    input_image = preprocess_frame(roi)
    prediction = model.predict(input_image)[0][0]

    # Display the ROI and prediction
    cv2.putText(frame, f"Tilt: {prediction:.2f} deg", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (0, 255, 0), 2)

    cv2.imshow("Webcam Tilt Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
