"""
FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="delfosA1C8 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = Path("models/gradient_boosting.pkl")
    if model_path.exists():
        model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    edad: float = 55
    imc: float = 28.5
    tas: float = 135
    tad: float = 85
    perimetro_abdominal: float = 95
    peso: float = 70
    talla: float = 165
    riesgo_dm: float = 0.5
    puntaje_total: float = 10

class PredictionResponse(BaseModel):
    glucosa_predicha: float
    categoria: str
    confianza: float

@app.get("/")
def root():
    return {"status": "active", "model": "delfosA1C8"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = np.array([[
        request.edad, request.talla, request.peso, request.imc,
        request.tas, request.tad, request.perimetro_abdominal,
        request.puntaje_total, request.riesgo_dm
    ]])
    
    glucose = float(model.predict(features)[0])
    
    if glucose < 100:
        category = "normal"
    elif glucose <= 126:
        category = "prediabetes"
    else:
        category = "diabetes"
    
    return PredictionResponse(
        glucosa_predicha=round(glucose, 2),
        categoria=category,
        confianza=0.87
    )
