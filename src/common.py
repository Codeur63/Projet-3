"""
Utils partagées du pipeline ML FinaScore.

Centralise les constantes et fonctions dupliquées historiquement dans
05_train, 08_cross_validation, 09_tune_xgboost, 10_analysis et
12_train_anomalies :
    - liste des colonnes à exclure (identifiants, dates, nom partenaire)
    - typage des colonnes (numériques / catégorielles)
    - construction du préprocesseur sklearn
    - calcul des métriques de classification
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "defaut_paiement"
RANDOM_STATE = 42
THRESHOLD = 0.5
PERFORMANCE_THRESHOLD = 0.80

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

DROP_COLUMNS = [
    "applicant_id",
    "date_demande",
    "nom_partenaire",
]


def remove_columns(X):
    """Retire les colonnes identifiants / bruit du DataFrame."""
    cols_to_drop = [col for col in DROP_COLUMNS if col in X.columns]
    return X.drop(columns=cols_to_drop)


def select_column_types(X):
    """Sépare les colonnes numériques et catégorielles."""
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Construit le préprocesseur : imputation + scaling (num) / OHE (cat)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def compute_metrics(y_true, y_proba, threshold=THRESHOLD, include_accuracy=False):
    """Calcule les métriques de classification à partir des probabilités."""
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if include_accuracy:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    return metrics


def make_promotion_gate(auc_roc, required_threshold=PERFORMANCE_THRESHOLD):
    """Décision de promotion fondée sur le seuil AUC."""
    passed = auc_roc >= required_threshold
    return {
        "metric": "auc_roc",
        "required_threshold": required_threshold,
        "observed_value": float(auc_roc),
        "passed": bool(passed),
        "decision": "PROMOTE_TO_PRODUCTION" if passed else "NOT_PROMOTE",
    }
