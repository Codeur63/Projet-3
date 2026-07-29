"""
API FastAPI pour le scoring crédit FinaScore SA
    Endpoints :
    - GET /health
    - GET /model/info
    - POST /predict
    - POST /predict/batch
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.cache import cache_service
from api.model_loader import model_service
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="FinaScore Credit Scoring API",
    description=("API de scoring crédit basée sur un modèle ML. " "Le modèle actuel est servi en mode staging/démonstration car le seuil " "de performance production n'est pas atteint."),
    version="0.1.0",
)

model = None


def ensure_model_loaded():
    if model_service.model is None:
        model_service.load()
        cache_service.connect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarage de L'API Finascaore")
    try:
        model_service.load()
        cache_service.connect()
        print("Modèle et Cache chargé ... ")
    except Exception as e:
        print(f"Erreur Critique avec le LifeSpan(Démarrqge API) : {e}")

    yield

    print("Arrêt de l'API")


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "FinaScore Credit Scoring API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/cache/status", tags=["Cache"])
def cache_status():
    return {"status": "ok", "model_loaded": model_service.model is not None, "cache": cache_service.client}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    try:
        ensure_model_loaded()
    except Exception:
        print("Erreur de chargement du modèle: {e}")

    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    try:
        ensure_model_loaded()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {e}")

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    promotion_decision = None
    registry_alias = None

    if model_service.registry_decision:
        promotion_decision = model_service.registry_decision.get("promotion_decision", {}).get("decision")

        registry_alias = model_service.registry_decision.get("registry_info", {}).get("alias")

    return {
        "model_name": model_service.model_name,
        "model_path": str(model_service.model_path),
        "expected_columns_count": len(model_service.expected_columns),
        "expected_columns": model_service.expected_columns,
        "promotion_decision": promotion_decision,
        "registry_alias": registry_alias,
        "production_warning": ("Modèle en staging/démonstration. " "Ne pas utiliser pour une décision automatique de crédit réelle " "tant que le performance gate AUC >= 0.80 n'est pas validé."),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """Prédit le risque de défaut pour un demandeur avec cache Redis."""

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        cache_key = cache_service.make_prediction_key(request.features)
        cached_result = cache_service.get(cache_key)

        if cached_result is not None:
            cached_result["cached"] = True
            return cached_result

        result = model_service.predict_one(request.features)

        result["warning"] = "Modèle non promu en production. Résultat à utiliser uniquement " "pour démonstration ou aide à l'analyse."

        result["cached"] = False

        cache_service.set(cache_key, result)

        return result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Prédit le risque de défaut pour plusieurs demandeurs."""

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="Le batch est vide.")

    try:
        predictions = model_service.predict_batch(request.records)

        return {
            "n_records": len(predictions),
            "predictions": predictions,
            "warning": ("Modèle non promu en production. Résultats à utiliser uniquement " "pour démonstration ou aide à l'analyse."),
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
