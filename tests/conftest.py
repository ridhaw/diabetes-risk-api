import pytest
from fastapi.testclient import TestClient

from diabetes_api import train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    metadata = train.main(["--output", str(out / "model.joblib"), "--metrics", str(out / "metrics.json")])
    return out / "model.joblib", metadata


@pytest.fixture
def client(trained_model, monkeypatch):
    from diabetes_api.main import app

    monkeypatch.setenv("MODEL_PATH", str(trained_model[0]))
    with TestClient(app) as c:
        yield c
