from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
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
    prediction: int
    probability_default: float
    threshold: float
    decision_label: str
    warning: str | None = None
    cached: bool = False


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description="Liste de dictionnaires de variables demandeur.",
    )


class BatchPredictionResponse(BaseModel):
    n_records: int
    predictions: list[dict[str, Any]]
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    model_metric: float | None
    expected_columns_count: int
    expected_columns: list[str]
    promotion_decision: str | None
    registry_alias: str | None
    production_warning: str


class CacheStatusResponse(BaseModel):
    redis_url: str
    connected: bool
    error: str | None
    ttl_seconds: int
