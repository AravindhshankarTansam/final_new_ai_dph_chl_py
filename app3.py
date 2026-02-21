import os
import re
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
from feature_extractor import extract_features

app = Flask(__name__)

# ================= LOAD MODEL =================
model = joblib.load("ppm_classifier_gpu.pkl")
le = joblib.load("label_encoder.pkl")
EXPECTED_FEATURES = model.n_features_in_

# ================= HAND DETECTION =================
def is_hand_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    skin_ratio = skin_pixels / total_pixels

    return skin_ratio > 0.08


# ================= LABEL → DECIMAL PPM =================
def convert_label_to_ppm(label):
    nums = re.findall(r"\d*\.?\d+", label)

    if len(nums) == 1:
        return float(nums[0])
    elif len(nums) == 2:
        low, high = map(float, nums)
        return (low + high) / 2
    else:
        return None


# ================= API ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():
    temp_path = "temp.jpg"

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        file.save(temp_path)

        # 🚨 Hand Detection
        if is_hand_present(temp_path):
            return jsonify({
                "success": False,
                "error": "Human hand detected. Please capture only the test tube."
            }), 400

        # 🔥 Feature Extraction
        feat = extract_features(temp_path)

        if feat is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        feat = np.array(feat, dtype=np.float32)

        if len(feat) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Feature mismatch. Expected {EXPECTED_FEATURES}, got {len(feat)}"
            }), 500

        feat = feat.reshape(1, -1)

        # 🔥 Prediction
        pred = model.predict(feat)
        label = le.inverse_transform(pred)[0]

        # 🔥 Convert to numeric PPM
        ppm_value = convert_label_to_ppm(label)

        if ppm_value is None:
            return jsonify({"error": f"Invalid PPM label: {label}"}), 500

        return jsonify({
            "success": True,
            "range": label,
            "ppm": round(ppm_value, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)