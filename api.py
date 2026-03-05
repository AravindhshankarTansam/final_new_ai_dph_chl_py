from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import joblib

from feature_extraction import extract_features
from config import *

app = Flask(__name__)

model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        image_base64 = data["image"]

        # decode base64
        image_bytes = base64.b64decode(image_base64)

        np_arr = np.frombuffer(image_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # debug image
        cv2.imwrite("debug_api_image.jpg", img)

        features = extract_features(img)

        prediction = model.predict(features.reshape(1,-1))

        ppm = round(float(prediction[0]),2)

        return jsonify({
            "chlorine_ppm": ppm
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)