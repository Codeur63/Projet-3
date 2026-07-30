"""
api/metrics.py - Métriques Prometheus pour l'API FinaScore
"""

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

API_REQUEST_COUNT = Counter(
    "finascore_api_requests_total",
    "Nombre total de requêtes reçues par l'API",
    ["method", "endpoint", "http_status"],
)

API_REQUEST_LATENCY = Histogram(
    "finascore_api_request_latency_seconds",
    "Temps de réponse des endpoints API",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "finascore_predictions_total",
    "Nombre total de prédictions effectuées",
    ["prediction"],
)

MODEL_LOADED_GAUGE = Gauge(
    "finascore_model_loaded",
    "Indique si le modèle est chargé : 1 oui, 0 non",
)

LAST_DEFAULT_PROBABILITY = Gauge(
    "finascore_last_default_probability",
    "Dernière probabilité de défaut retournée par le modèle",
)


def prometheus_metrics_response():
    """Retourne les métriques Prometheus."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
