# Étape 1 : installation et entraînement du modèle
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install . \
 && PYTHONPATH=/install/lib/python3.12/site-packages python -m diabetes_api.train \
      --output /app/models/model.joblib --metrics /app/results/metrics.json

# Étape 2 : image d'exécution légère, sans root
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /app/models ./models
ENV MODEL_PATH=/app/models/model.joblib
RUN useradd --create-home api
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "diabetes_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
