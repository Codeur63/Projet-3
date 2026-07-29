"""
api/schemas.py - Schémas Pydantic pour l'API FinaScore
"""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Requête pour une prédiction individuelle."""

    features: dict[str, Any] = Field(
        ...,
        description="Dictionnaire contenant les variables du demandeur.",
        examples=[
            {
                "age": 35,
                "pays": "cameroun",
                "secteur_activite": "commerce",
                "zone": "urbaine",
                "revenu_mensuel_xaf": 250000,
                "anciennete_emploi": 4,
                "ratio_endettement": 0.35,
                "historique_credit": 650,
                "nb_credits_actifs": 1,
                "mobile_money_score": 72,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Réponse d'une prédiction individuelle."""

    prediction: int
    probability_default: float
    threshold: float
    decision_label: str
    warning: str | None = None


class BatchPredictionRequest(BaseModel):
    """Requête pour prédiction batch."""

    records: list[dict[str, Any]] = Field(
        ...,
        description="Liste de dictionnaires de variables demandeur.",
    )


class BatchPredictionResponse(BaseModel):
    """Réponse d'une prédiction batch."""

    n_records: int
    predictions: list[dict[str, Any]]
    warning: str | None = None


class HealthResponse(BaseModel):
    """Réponse health check."""

    status: str
    model_loaded: bool
    model_name: str | None


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle chargé."""

    model_name: str
    model_path: str
    expected_columns_count: int
    expected_columns: list[str]
    promotion_decision: str | None
    registry_alias: str | None
    production_warning: str
    
class PredictionResponse(BaseModel):
    """Réponse d'une prédiction individuelle."""

    prediction: int
    probability_default: float
    threshold: float
    decision_label: str
    warning: str | None = None
    cached: bool = False


class CacheStatusResponse(BaseModel):
    """Statut du cache Redis."""

    redis_url: str
    connected: bool
    error: str | None
    ttl_seconds: int    