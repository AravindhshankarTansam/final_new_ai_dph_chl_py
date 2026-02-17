import numpy as np
import joblib
from flask import Flask, request, jsonify
from feature_extractor import extract_features

app = Flask(__name__)

# ================= LOAD =================
model = joblib.load("ppm_classifier_gpu.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES = model.n_features_in_

# ================= CLASS → PPM VALUE =================
CLASS_TO_VALUE = {
    "0": 0.0,
    "0.0": 0.0,
    "0.1_to_0.5": 0.25,
    "0.5_to_1": 0.75,
    "1_to_1.5": 1.25,
    "1.5_to_2": 1.75,
    "2_to_2.5": 2.25,
    "2.5_to_3": 2.75,
    "3_to_3.5": 3.25,
    "3.5_to_4": 3.75,
    "above_4": 4.5
}


@app.route("/")
def index():
    return jsonify({"message": "Chlorine Detection API is running"}), 200


# ================= API ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        path = "temp.jpg"
        file.save(path)

        # 🔥 Feature extraction
        feat = extract_features(path)

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

        # 🔥 Normalize
        label_clean = label.replace("_PPM", "").strip()

        if label_clean in ["0", "0.0", "0_to_0.1"]:
            ppm_value = 0.0
        else:
            ppm_value = CLASS_TO_VALUE.get(label_clean, "Unknown")

        return jsonify({
            "success": True,
            "range": label,
            "ppm": ppm_value
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
