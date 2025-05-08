from tensorflow.keras.models import load_model

# Load without compiling (skip the mse error)
model = load_model("tilt_angle_model.h5", compile=False)

# Show model summary
model.summary()

# Print the expected input shape
print("Expected input shape:", model.input_shape)
