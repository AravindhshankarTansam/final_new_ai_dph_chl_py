from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import joblib


from feature_extraction import extract_features
from config import MODEL_PATH

app = FastAPI()

# ======================
# Load Model
# ======================

model = joblib.load(MODEL_PATH)
print("Model Loaded Successfully")

# ======================
# Home Route
# ======================

@app.get("/")
def home():
    return {"message": "Chlorine Prediction API Running"}

# ======================
# Prediction Route
# ======================
@app.post("/predict")
async def predict(request: Request):

    try:
        form = await request.form()

        if not form:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"}
            )

        # Get first uploaded file regardless of field name
        file = list(form.values())[0]

        contents = await file.read()

        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        # Extract features
        features = extract_features(img)

        if features is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Feature extraction failed"}
            )

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        chlorine_ppm = round(float(prediction), 2)

        return {"ppm": chlorine_ppm}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )