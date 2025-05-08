from tensorflow.keras.models import load_model

model = load_model("tilt_angle_model.h5", compile=False)
model.summary()
