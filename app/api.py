import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import prediction
from preprocessing import CLASS_NAMES

MODEL_PATH = ROOT / "models" / "flowers_model.pth"

app = FastAPI(title="Flower Classifier API")
START_TIME = time.time()


@app.on_event("startup")
def warm_model():
    # load once at startup so the first real request isn't slow
    if MODEL_PATH.exists():
        try:
            prediction._get_model(MODEL_PATH)
        except Exception:
            pass


@app.get("/")
def root():
    return {"service": "Flower Classifier API", "classes": CLASS_NAMES,
            "model_loaded": MODEL_PATH.exists()}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(),
            "uptime_seconds": round(time.time() - START_TIME, 1)}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="model not available")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    try:
        data = await file.read()
        result = prediction.predict(data, model_path=MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")
    return JSONResponse(result)
