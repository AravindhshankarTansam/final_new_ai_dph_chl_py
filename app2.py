import re
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template_string
from feature_extractor import extract_features

app = Flask(__name__)

# ================= LOAD =================
model = joblib.load("ppm_classifier_gpu.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES = model.n_features_in_

# ================= UI =================
HTML = """
💧 Liquid PPM Classifier


{{result}}
"""

# ================= HAND DETECTION (OpenCV ONLY) =================
def is_hand_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin color range (works well for hands)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Calculate skin area ratio
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]

    skin_ratio = skin_pixels / total_pixels

    # 🚨 If too much skin → likely hand
    return skin_ratio > 0.08   # threshold tuned for hands

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        file = request.files["image"]

        if file:
            path = "temp.jpg"
            file.save(path)

            # 🚨 HAND CHECK
            if is_hand_present(path):
                result = "❌ Human hand detected. Please capture only the test tube."
                return render_template_string(HTML, result=result)

            feat = extract_features(path)

            if feat is None:
                result = "❌ Feature extraction failed"
                return render_template_string(HTML, result=result)

            feat = np.array(feat, dtype=np.float32)

            # ✅ FEATURE COUNT CHECK
            if len(feat) != EXPECTED_FEATURES:
                result = f"❌ Feature mismatch. Expected {EXPECTED_FEATURES}, got {len(feat)}"
                return render_template_string(HTML, result=result)

            feat = feat.reshape(1, -1)

            # ================= PREDICTION =================
            pred = model.predict(feat)
            label = le.inverse_transform(pred)[0]

            # ================= LABEL → DECIMAL PPM =================
            nums = re.findall(r"\d*\.?\d+", label)

            if len(nums) == 1:
                ppm_value = float(nums[0])          # 0_PPM → 0.0
            elif len(nums) == 2:
                low, high = map(float, nums)        # 0.5_to_1 → 0.75
                ppm_value = (low + high) / 2
            else:
                result = f"❌ Invalid PPM label: {label}"
                return render_template_string(HTML, result=result)

            result = f"✅ Predicted PPM: {ppm_value:.2f}"

    return render_template_string(HTML, result=result)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)


