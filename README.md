# Diabetes Risk API — du modèle à la production

[![CI](https://github.com/ridhaw/diabetes-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/ridhaw/diabetes-risk-api/actions/workflows/ci.yml)

API REST qui estime le risque de diabète à partir de 8 mesures cliniques. Le projet couvre
toute la chaîne MLOps : téléchargement des données, entraînement et sélection du modèle,
service par **FastAPI**, validation des entrées, conteneur **Docker** et intégration continue
**GitHub Actions**.

> ⚠️ Démonstration technique. Les prédictions ne constituent pas un avis médical.

## Données

[Pima Indians Diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
— 768 patientes, 8 variables (grossesses, glycémie, pression artérielle, pli cutané, insuline,
IMC, antécédents familiaux, âge). Téléchargé automatiquement au premier entraînement.

Dans ce dataset, un `0` sur la glycémie, la pression, le pli cutané, l'insuline ou l'IMC
signifie « non mesuré » : ces zéros sont convertis en valeurs manquantes puis imputés par
la médiane.

## Résultats

Sélection par validation croisée (5 plis, ROC AUC) sur 80 % des données :

| Modèle | ROC AUC (validation croisée) |
|---|---|
| **Régression logistique** | **0,844 ± 0,016** |
| Random Forest | 0,832 ± 0,024 |
| Gradient Boosting | 0,825 ± 0,016 |

Modèle retenu, mesuré une seule fois sur le jeu de test (20 %, 154 patientes) :

| ROC AUC | Accuracy | Précision | Rappel | F1 |
|---|---|---|---|---|
| 0,813 | 73,4 % | 60,3 % | 70,4 % | 0,650 |

Les classes sont pondérées (`class_weight="balanced"`) : dans un contexte de dépistage, on
préfère détecter davantage de cas (rappel) quitte à générer plus de faux positifs.

## API

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | état du service et du modèle |
| `GET` | `/model-info` | modèle, version, date d'entraînement, métriques de test |
| `POST` | `/predict` | probabilité de diabète et niveau de risque |
| `GET` | `/docs` | documentation interactive (Swagger) |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "pregnancies": 2, "glucose": 138, "blood_pressure": 72, "skin_thickness": 30,
  "insulin": 120, "bmi": 33.6, "diabetes_pedigree": 0.62, "age": 45
}'
```

```json
{"probability": 0.6507, "risk": "élevé", "model": "logistic_regression",
 "disclaimer": "Démonstration technique : ce résultat ne constitue pas un avis médical."}
```

Les entrées sont validées par Pydantic : valeurs hors bornes physiologiques ou champs
inconnus → `422`. La pression, le pli cutané et l'insuline sont facultatifs.

## Lancer

**Avec Docker** (le modèle est entraîné pendant la construction de l'image) :

```bash
docker build -t diabetes-risk-api .
docker run -p 8000:8000 diabetes-risk-api
```

**En local :**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
python -m diabetes_api.train
uvicorn diabetes_api.main:app --reload
```

## Tests et intégration continue

`pytest` lance 16 tests : entraînement complet, routes de l'API, validation des entrées,
cohérence du modèle (un profil à risque obtient une probabilité plus élevée qu'un profil
sain) et seuils de risque.

À chaque push, GitHub Actions exécute les tests, construit l'image Docker puis démarre le
conteneur et interroge `/health` et `/predict`.

## Structure

```
src/diabetes_api/
  data.py      téléchargement et nettoyage du dataset
  train.py     comparaison des modèles, sélection, export
  schemas.py   schémas Pydantic d'entrée et de sortie
  main.py      application FastAPI
tests/         tests pytest
Dockerfile     image multi-étapes, exécution sans root
.github/       pipeline CI
```

## Stack

Python · scikit-learn · FastAPI · Pydantic · Uvicorn · Docker · GitHub Actions · pytest
