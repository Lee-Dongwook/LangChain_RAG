"""CIFAKE 판별 REST API (FastAPI).

실행:
    uvicorn app.api:app --reload
문서:
    http://127.0.0.1:8000/docs  (Swagger UI)
호출:
    curl -F "file=@some_image.png" http://127.0.0.1:8000/predict
"""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from detector.inference import load_model, predict_pil

CKPT = os.environ.get("CKPT", "models/cifake_resnet18.pt")

app = FastAPI(title="CIFAKE Detector", version="0.1.0")

# 모델은 프로세스 시작 시 한 번만 로드해 요청마다 재사용한다.
model, device = load_model(CKPT)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "ckpt": CKPT}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    label, confidence, probs = predict_pil(model, image, device)
    return {
        "filename": file.filename,
        "label": label,
        "confidence": confidence,
        "probs": probs,
    }
