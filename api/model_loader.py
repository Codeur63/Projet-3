"""
Chargement du modèle de scoring crédit
Le modèle est chargé une seule fois au démarrage de l'API.
"""

import json
import joblib
import pandas as pd
from pathlib import Path


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/optuna")

DEFAULT_MODEL_PATH = MODELS_DIR / "optuna/xgboost_optuna.pkl"
REGISTRY_DECISION_PATH = REPORTS_DIR / "model_registry_decision.json"


class ModelService:
    """Service de prédiction pour le modèle FinaScore."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_name = None
        self.expected_columns = None
        self.registry_decision = None

    def load(self):
        """Charge le modèle et ses métadonnées."""

        self.model_path = self._resolve_model_path()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.expected_columns = self._extract_expected_columns()

        self.model_name = self.model_path.stem

        self.registry_decision = self._load_registry_decision()

        return self

    def _resolve_model_path(self):
        """
        Sélectionne le meilleur modèle disponible.

        Priorité :
        1. modèle indiqué dans model_registry_decision.json
        2. models/xgboost_optuna.pkl
        """

        if REGISTRY_DECISION_PATH.exists():
            with open(REGISTRY_DECISION_PATH, "r", encoding="utf-8") as file:
                registry_data = json.load(file)

            candidate = registry_data.get("candidate", {})
            model_path = candidate.get("model_path")

            if model_path:
                return Path(model_path)

        return DEFAULT_MODEL_PATH

    def _load_registry_decision(self):
        """Charge la décision de registry si disponible."""

        if not REGISTRY_DECISION_PATH.exists():
            return None

        with open(REGISTRY_DECISION_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    def _extract_expected_columns(self):
        """
        Récupère les colonnes attendues par le pipeline sklearn.

        Le ColumnTransformer garde normalement les noms des colonnes utilisées
        pendant l'entraînement.
        """

        if not hasattr(self.model, "named_steps"):
            raise ValueError("Le modèle chargé n'est pas un Pipeline sklearn valide.")

        preprocessor = self.model.named_steps.get("preprocessor")

        if preprocessor is None:
            raise ValueError("Le Pipeline ne contient pas d'étape 'preprocessor'.")

        if hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

        raise ValueError(
            "Impossible de récupérer les colonnes attendues par le modèle."
        )

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
            "decision_label": (
                "risque_defaut" if prediction == 1 else "bon_payeur_probable"
            ),
        }

    def predict_batch(self, records: list[dict]):
        """Prédit le risque de défaut pour plusieurs demandes."""

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
                    "decision_label": (
                        "risque_defaut" if pred == 1 else "bon_payeur_probable"
                    ),
                }
            )

        return results


model_service = ModelService()