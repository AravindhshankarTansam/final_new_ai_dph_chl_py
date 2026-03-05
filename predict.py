import cv2
import joblib
import numpy as np
from config import MODEL_PATH
from feature_extraction import extract_features

# Load the trained model
model = joblib.load(MODEL_PATH)

# Path to test image
image_path = r"C:\Users\HW-07\Desktop\test_7.jpg"

# Read the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at path: {image_path}")

# Extract features
features = extract_features(img)

# Predict chlorine concentration
prediction = model.predict(features.reshape(1, -1))[0]

# Round the result
ppm_value = round(float(prediction), 2)

# Output
result = {"chlorine_ppm": ppm_value}
print(result)