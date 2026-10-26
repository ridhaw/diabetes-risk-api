"""Entraîne et compare plusieurs modèles, puis exporte le meilleur.

Sélection par validation croisée (5 plis, ROC AUC) sur le train ; le jeu de test
(20 %, stratifié) ne sert qu'à mesurer le modèle retenu.

Usage : python -m diabetes_api.train
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .data import FEATURES, TARGET, load

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"


def candidates(seed: int) -> dict[str, Pipeline]:
    def pipe(clf):
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)]
        )

    return {
        "logistic_regression": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=seed
            )
        ),
        "gradient_boosting": pipe(
            GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=seed)
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    args = parser.parse_args(argv)

    df = load()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=args.seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = {}
    for name, model in candidates(args.seed).items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"roc_auc_mean": round(scores.mean(), 4), "roc_auc_std": round(scores.std(), 4)}
        print(f"  {name:<20} CV ROC AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_scores, key=lambda n: cv_scores[n]["roc_auc_mean"])
    model = candidates(args.seed)[best_name].fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    predicted = (proba >= 0.5).astype(int)
    test = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "f1": round(f1_score(y_test, predicted), 4),
    }
    print(f"Modèle retenu : {best_name} — test {test}")

    metadata = {
        "model": best_name,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "cross_validation": cv_scores,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "metadata": metadata}, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Modèle exporté : {args.output}")
    return metadata


if __name__ == "__main__":
    main()
