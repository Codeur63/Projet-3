"""
Tests de l'API FastAPI FinaScore
   - Vérifier que l'API démarre correctement
   - Vérifier que le modèle est chargé
   - Tester une prédiction individuelle
   - Tester une prédiction batch
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_FEATURES = {
    "age": 35,
    "pays": "cameroun",
    "secteur_activite": "commerce",
    "zone": "urbain",
    "revenu_mensuel_xaf": 250000,
    "anciennete_emploi": 4,
    "ratio_endettement": 0.35,
    "historique_credit": 650,
    "nb_credits_actifs": 1,
    "mobile_money_score": 72,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# Test pour le root de l'API
def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


# Test pour le Check de l'API
def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None


def test_model_info_endpoint(client):
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "model_path" in data
    assert "expected_columns_count" in data
    assert "expected_columns" in data
    assert "production_warning" in data

    assert data["expected_columns_count"] > 0
    assert isinstance(data["expected_columns"], list)


def test_predict_endpoint(client):
    payload = {"features": SAMPLE_FEATURES}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability_default" in data
    assert "threshold" in data
    assert "decision_label" in data
    assert "warning" in data

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability_default"] <= 1.0
    assert data["threshold"] == 0.5


def test_predict_batch_endpoint(client):
    payload = {
        "records": [
            SAMPLE_FEATURES,
            {
                "age": 48,
                "pays": "gabon",
                "secteur_activite": "transport",
                "zone": "semi_urbaine",
                "revenu_mensuel_xaf": 180000,
                "anciennete_emploi": 2,
                "ratio_endettement": 0.65,
                "historique_credit": 580,
                "nb_credits_actifs": 2,
                "mobile_money_score": 55,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "n_records" in data
    assert "predictions" in data
    assert "warning" in data

    assert data["n_records"] == 2
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert 0.0 <= prediction["probability_default"] <= 1.0
        assert prediction["threshold"] == 0.5


def test_predict_batch_empty_payload(client):
    payload = {"records": []}

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 400


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "finascore_api_requests_total" in response.text
