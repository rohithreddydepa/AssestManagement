import numpy as np
from keras.models import load_model
import tensorflow as tf


model = load_model("tilt_angle_model.h5", compile=False)

def predict_tilt(image):
    try:
        
        resized = tf.image.resize(image, (64, 64)) / 255.0
        resized = tf.reshape(resized, (1, 64, 64, 3))

        
        prediction = model.predict(resized, verbose=0)

        
        return int(round(float(prediction[0][0])))
    except Exception as e:
        print("Tilt prediction error:", e)
        return None
