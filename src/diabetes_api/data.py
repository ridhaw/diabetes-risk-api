"""Téléchargement et préparation du dataset Pima Indians Diabetes."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]
TARGET = "outcome"

# Dans ce dataset, un 0 sur ces mesures signifie « non mesuré », pas une vraie valeur.
ZERO_MEANS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def download(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "pima_diabetes.csv"
    if not target.exists():
        print("Téléchargement du dataset Pima Indians Diabetes…")
        urllib.request.urlretrieve(DATA_URL, target)
    return target


def load(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Renvoie le dataset avec des noms de colonnes explicites et les zéros manquants en NaN."""
    df = pd.read_csv(download(data_dir), header=None, names=FEATURES + [TARGET])
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    return df
