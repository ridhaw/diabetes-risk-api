import pytest

HEALTHY = {
    "pregnancies": 0, "glucose": 85, "blood_pressure": 66, "skin_thickness": 29,
    "insulin": 60, "bmi": 22.0, "diabetes_pedigree": 0.2, "age": 25,
}
AT_RISK = {
    "pregnancies": 8, "glucose": 196, "blood_pressure": 80, "skin_thickness": 35,
    "insulin": 300, "bmi": 41.0, "diabetes_pedigree": 1.2, "age": 55,
}


def test_health_reports_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_exposes_test_metrics(client):
    info = client.get("/model-info").json()
    assert info["features"][0] == "pregnancies"
    assert 0.5 < info["test"]["roc_auc"] <= 1.0


def test_predict_returns_probability_and_risk(client):
    body = client.post("/predict", json=HEALTHY).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk"] in {"faible", "modéré", "élevé"}
    assert "avis médical" in body["disclaimer"]


def test_higher_glucose_and_bmi_increase_risk(client):
    low = client.post("/predict", json=HEALTHY).json()["probability"]
    high = client.post("/predict", json=AT_RISK).json()["probability"]
    assert high > low


def test_optional_measurements_can_be_omitted(client):
    partial = {k: v for k, v in HEALTHY.items() if k not in {"insulin", "skin_thickness"}}
    assert client.post("/predict", json=partial).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [("glucose", 0), ("bmi", 5), ("age", 10), ("pregnancies", -1)],
)
def test_rejects_implausible_values(client, field, value):
    assert client.post("/predict", json={**HEALTHY, field: value}).status_code == 422


def test_rejects_unknown_fields(client):
    assert client.post("/predict", json={**HEALTHY, "weight": 70}).status_code == 422
