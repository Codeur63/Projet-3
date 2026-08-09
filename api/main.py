"""
API FastAPI pour le scoring crédit FinaScore SA
    Endpoints :
    - GET /health
    - GET /model/info
    - POST /predict
    - POST /predict/batch
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.cache import cache_service
from api.metrics import (
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    LAST_DEFAULT_PROBABILITY,
    PREDICTION_COUNT,
    prometheus_metrics_response,
)
from api.model_loader import model_service
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger("finascore.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarage de l'API FinaScore...")
    try:
        model_service.load()
        cache_service.connect()
        logger.info("Modèle et Cache chargés avec succès.")
    except Exception as e:
        logger.critical(f"Erreur Critique lors du chargement des ressources : {e}")
    yield
    logger.info("Arrêt de l'API FinaScore...")


app = FastAPI(
    title="FinaScore Credit Scoring API",
    description=("API de scoring crédit basée sur un modèle ML. Le modèle actuel est servi en mode staging/démonstration car le seuil de performance production n'est pas atteint."),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)

    API_REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        http_status=status_code,
    ).inc()

    API_REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint,
    ).observe(latency)

    return response


# Exposition des métrics de Prometheus
@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return prometheus_metrics_response()


@app.get("/", tags=["Root"])
def root():
    return {
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    try:
        is_loaded = model_service.model is not None
    except Exception as e:
        logger.critical(f"Erreur Critique lors du chargement des ressources : {e}")

    return {
        "status": "ok" if is_loaded else "not",
        "model_loaded": is_loaded,
        "model_name": getattr(model_service, "model_name", "Unknown"),
    }


@app.get("/cache/status", tags=["Cache"])
def cache_status():
    return {
        "status": "Redis ok",
        "model_loaded": model_service.model is not None,
        "cache": getattr(cache_service, "client", None) is not None,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle Non disponible")

    promotion_decision = None
    registry_alias = None
    metric = None

    if model_service.registry_decision:
        promotion_decision = model_service.registry_decision.get("promotion_decision", {}).get("decision")

        registry_alias = model_service.registry_decision.get("registry_info", {}).get("alias")
        metric = model_service.registry_decision.get("metrics", {}).get("auc_roc")

    return {
        "model_name": model_service.model_name,
        "model_path": str(model_service.model_path),
        "model_metric": metric,
        "expected_columns_count": len(model_service.expected_columns),
        "expected_columns": model_service.expected_columns,
        "promotion_decision": promotion_decision,
        "registry_alias": registry_alias,
        "production_warning": "NOT_PROMOTE",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle Non disponible")

    try:
        cache_key = cache_service.make_prediction_key(request.features)
        cached_result = cache_service.get(cache_key)

        if cached_result is not None:
            cached_result["cached"] = True
            return cached_result
    except Exception as e:
        logger.warning(f"Échec de lecture dans le cache Redis : {e}")
        cache_key = None

    try:
        result = model_service.predict_one(request.features)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as error:
        logger.error(f"Erreur lors de la prédiction : {error}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du modèle de prédiction.",
        )

    PREDICTION_COUNT.labels(prediction=str(result["prediction"])).inc()

    LAST_DEFAULT_PROBABILITY.set(result["probability_default"])

    result["warning"] = "Modèle non promu en production Metrique faible"

    result["cached"] = False

    if cache_key:
        try:
            cache_service.set(cache_key, result)
        except Exception as e:
            logger.warning(f"Échec d'écriture dans le cache Redis : {e}")

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Prédit le risque de défaut pour plusieurs demandeurs."""

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="Le batch est vide.")

    try:
        predictions = model_service.predict_batch(request.records)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as error:
        logger.error(f"Erreur lors de la prédiction batch : {error}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du modèle lors de l'inférence par lot.",
        )

    for prediction in predictions:
        PREDICTION_COUNT.labels(prediction=str(prediction["prediction"])).inc()

        LAST_DEFAULT_PROBABILITY.set(prediction["probability_default"])

    return {
        "n_records": len(predictions),
        "predictions": predictions,
        "warning": ("Modèle non promu en production. Métric Faible faible"),
    }
