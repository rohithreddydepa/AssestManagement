import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model (skip loading training configs)
model = load_model("tilt_angle_model.h5", compile=False)

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (64, 64))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

def predict_tilt(img_path):
    preprocessed = preprocess_image(img_path)
    angle = model.predict(preprocessed)[0][0]
    print(f"Predicted tilt angle: {angle:.2f} degrees")

# Example usage
predict_tilt("test_image.jpg")
