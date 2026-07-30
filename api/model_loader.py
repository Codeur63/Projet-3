"""
Chargement du modèle de scoring crédit
Le modèle est chargé une seule fois au démarrage de l'API.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

logger = logging.getLogger("finascore.model_loader")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models/optuna"
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_MODEL_PATH = MODELS_DIR / "xgboost_optuna.pkl"
REGISTRY_DECISION_PATH = REPORTS_DIR / "registry/model_registry_decision.json"


# Class Prediction API
class ModelService:
    def __init__(self):
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_name: Optional[str] = None
        self.model_auc_roc: Optional[float] = None
        self.expected_columns: List[str] = None
        self.registry_decision: Optional[Dict[str, Any]] = None

    def load(self):
        self.model_path = self._resolve_model_path()

        if not self.model_path.exists():
            logger.error(f"Fichier modèle introuvable : {self.model_path}")
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        logger.info(f"Chargement du modèle depuis : {self.model_path}")
        self.model = joblib.load(self.model_path)

        self.expected_columns = self._extract_expected_columns()
        self.model_name = self.model_path.stem
        self.registry_decision = self._load_registry_decision()

        self.model_auc_roc = self._extract_auc_roc()

        logger.info(f"Modèle '{self.model_name}' chargé avec succès ({len(self.expected_columns)} features attendues).")
        return self

    def _resolve_model_path(self):
        if REGISTRY_DECISION_PATH.exists():
            with open(REGISTRY_DECISION_PATH, "r", encoding="utf-8") as file:
                registry_data = json.load(file)

            candidate = registry_data.get("candidate", {})
            model_path = candidate.get("model_path")

            if model_path:
                return Path(model_path)

        return DEFAULT_MODEL_PATH

    def _load_registry_decision(self):
        if not REGISTRY_DECISION_PATH.exists():
            return None

        try:
            with open(REGISTRY_DECISION_PATH, "r", encoding="utf-8") as file:
                return json.load(file)

        except Exception as e:
            logger.error(f"Impossible de charger le fichier registry decision : {e}")
            return None

    def _extract_auc_roc(self) -> Optional[float]:
        if self.registry_decision:
            return self.registry_decision.get("candidate", {}).get("metrics", {}).get("auc_roc")
        return None

    def _extract_expected_columns(self):
        if not hasattr(self.model, "named_steps"):
            raise ValueError("Le modèle chargé n'est pas un Pipeline sklearn valide.")

        preprocessor = self.model.named_steps.get("preprocessor")

        if preprocessor is None:
            raise ValueError("Le Pipeline ne contient pas d'étape 'preprocessor'.")

        if hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

        raise ValueError("Impossible de récupérer les colonnes attendues par le modèle.")

    def prepare_input(self, data: dict):
        """
        Transforme un dictionnaire utilisateur en DataFrame compatible modèle.
        Les colonnes manquantes sont ajoutées avec None.
        Les colonnes inconnues sont ignorées.
        """

        row = {}

        for col in self.expected_columns:
            row[col] = data.get(col, None)

        return pd.DataFrame([row], columns=self.expected_columns)

    def prepare_batch_input(self, records: list[dict]):
        """Prépare un batch de demandes."""

        rows = []

        for record in records:
            row = {}
            for col in self.expected_columns:
                row[col] = record.get(col, None)
            rows.append(row)

        return pd.DataFrame(rows, columns=self.expected_columns)

    def predict_one(self, data: dict):
        """Prédit le risque de défaut pour une demande."""

        X = self.prepare_input(data)

        probability_default = float(self.model.predict_proba(X)[:, 1][0])
        prediction = int(probability_default >= 0.5)

        return {
            "prediction": prediction,
            "probability_default": probability_default,
            "threshold": 0.5,
            "decision_label": ("risque_defaut" if prediction == 1 else "bon_payeur_probable"),
        }

    # Avoir plusieur demande pour des probabilités
    def predict_batch(self, records: list[dict]):
        X = self.prepare_batch_input(records)

        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []

        for idx, proba in enumerate(probabilities):
            pred = int(predictions[idx])

            results.append(
                {
                    "row_id": idx,
                    "prediction": pred,
                    "probability_default": float(proba),
                    "threshold": 0.5,
                    "decision_label": ("risque_defaut" if pred == 1 else "bon_payeur_probable"),
                }
            )

        return results


model_service = ModelService()
