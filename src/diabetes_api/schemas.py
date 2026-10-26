"""Schémas d'entrée et de sortie de l'API, avec des bornes physiologiques plausibles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 138,
                "blood_pressure": 72,
                "skin_thickness": 30,
                "insulin": 120,
                "bmi": 33.6,
                "diabetes_pedigree": 0.62,
                "age": 45,
            }
        },
    )

    pregnancies: int = Field(ge=0, le=20, description="Nombre de grossesses")
    glucose: float = Field(gt=0, le=300, description="Glycémie à 2 h (test de tolérance au glucose), mg/dL")
    blood_pressure: float | None = Field(default=None, gt=0, le=200, description="Pression diastolique, mm Hg")
    skin_thickness: float | None = Field(default=None, gt=0, le=100, description="Pli cutané du triceps, mm")
    insulin: float | None = Field(default=None, gt=0, le=1000, description="Insuline sérique à 2 h, µU/mL")
    bmi: float = Field(gt=10, le=80, description="Indice de masse corporelle, kg/m²")
    diabetes_pedigree: float = Field(ge=0, le=3, description="Score d'antécédents familiaux")
    age: int = Field(ge=18, le=120, description="Âge, en années")


class Prediction(BaseModel):
    probability: float = Field(description="Probabilité estimée de diabète (0 à 1)")
    risk: Literal["faible", "modéré", "élevé"]
    model: str
    disclaimer: str = "Démonstration technique : ce résultat ne constitue pas un avis médical."


class ModelInfo(BaseModel):
    model: str
    version: str
    trained_at: str
    features: list[str]
    test: dict[str, float]
