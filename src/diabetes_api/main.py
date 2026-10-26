"""Application FastAPI qui sert le modèle entraîné.

Lancement : uvicorn diabetes_api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from . import __version__
from .data import FEATURES
from .schemas import ModelInfo, PatientFeatures, Prediction
from .train import DEFAULT_MODEL_PATH

# Seuils de lecture de la probabilité, choisis pour la démonstration.
MODERATE_RISK = 0.3
HIGH_RISK = 0.6

state: dict = {}


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


@asynccontextmanager
async def lifespan(_: FastAPI):
    path = model_path()
    if path.exists():
        state.update(joblib.load(path))
    yield
    state.clear()


app = FastAPI(
    title="Diabetes Risk API",
    version=__version__,
    description="Estimation du risque de diabète à partir de 8 mesures cliniques (dataset Pima).",
    lifespan=lifespan,
)


def _loaded() -> dict:
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Modèle non chargé : lancer python -m diabetes_api.train")
    return state


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "élevé"
    if probability >= MODERATE_RISK:
        return "modéré"
    return "faible"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> dict:
    return _loaded()["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures) -> Prediction:
    loaded = _loaded()
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    probability = float(loaded["pipeline"].predict_proba(row)[0, 1])
    return Prediction(
        probability=round(probability, 4),
        risk=risk_level(probability),
        model=loaded["metadata"]["model"],
    )
