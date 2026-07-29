"""
Cross-validation du meilleur modèle
    - Évaluer la stabilité du modèle avec Stratified K-Fold k=5 (Si les AUC sont presque toutes semblable)
    - Calculer AUC, F1, Precision, Recall avec moyenne, écart-type et intervalle de confiance
    - Produire des prédictions out-of-fold
    - Tester la stabilité par sous-populations : pays, secteur_activite, zone
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"
RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

COLS_TO_DROP = [
    "applicant_id",
    "date_demande",
    "nom_partenaire",
]


def load_train_data():
    X_train_path = SPLITS_DIR / "X_train.parquet"
    y_train_path = SPLITS_DIR / "y_train.parquet"

    if not X_train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {X_train_path}")

    if not y_train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {y_train_path}")

    X = pd.read_parquet(X_train_path)
    y = pd.read_parquet(y_train_path)[TARGET].astype(int)

    return X, y


def remove_excluded_columns(X):
    cols_to_drop = [col for col in COLS_TO_DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


def detect_column_types(X):
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def build_xgboost_pipeline(X, y):
    numeric_cols, categorical_cols = detect_column_types(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def compute_metrics(y_true, y_proba, threshold=THRESHOLD):
    "Calcule des métrics"

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


# Intervale de confiance
def confidence_interval_95(values):
    values = np.array(values, dtype=float)
    mean = values.mean()
    std = values.std(ddof=1)

    margin = 1.96 * std / np.sqrt(len(values))

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def main():
    print("=" * 60)
    print("CROSS-VALIDATION STRATIFIED K-FOLD")
    print("=" * 60)

    X_raw, y = load_train_data()

    print(f"X brut : {X_raw.shape}")
    print("Distribution target :")
    print(y.value_counts(normalize=True).round(4))

    X_model = remove_excluded_columns(X_raw)

    print(f"X modèle après suppression colonnes : {X_model.shape}")

    base_pipeline = build_xgboost_pipeline(X_model, y)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_model, y), start=1):
        print(f"\nFold {fold}/{N_SPLITS}")

        X_train_fold = X_model.iloc[train_idx]
        X_val_fold = X_model.iloc[val_idx]

        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        pipeline = clone(base_pipeline)

        pipeline.fit(X_train_fold, y_train_fold)

        y_val_proba = pipeline.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = y_val_proba

        metrics = compute_metrics(y_val_fold, y_val_proba)

        metrics["fold"] = fold
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)

        fold_metrics.append(metrics)

        print(f"AUC={metrics['auc_roc']:.4f} | " f"F1={metrics['f1']:.4f} | " f"Precision={metrics['precision']:.4f} | " f"Recall={metrics['recall']:.4f}")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(REPORTS_DIR / "validation/cv_fold_metrics.csv", index=False)

    summary = {}

    for metric in ["auc_roc", "f1", "precision", "recall"]:
        summary[metric] = confidence_interval_95(fold_metrics_df[metric].values)

    auc_mean = float(summary["auc_roc"]["mean"])
    auc_std = float(summary["auc_roc"]["std"])

    summary["n_splits"] = N_SPLITS
    summary["threshold"] = THRESHOLD
    passed_gate = bool(auc_mean >= 0.80)

    summary["performance_gate"] = {
        "metric": "auc_roc",
        "required_threshold": 0.80,
        "observed_mean_cv": auc_mean,
        "observed_std_cv": auc_std,
        "passed": passed_gate,
        "decision": ("PROMOTE_TO_PRODUCTION" if passed_gate else "DO_NOT_PROMOTE"),
    }

    with open(REPORTS_DIR / "validation/cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    oof_df = X_raw.copy()
    oof_df[TARGET] = y.values
    oof_df["y_proba"] = oof_predictions
    oof_df["y_pred"] = (oof_df["y_proba"] >= THRESHOLD).astype(int)

    oof_df.to_parquet(REPORTS_DIR / "validation/cv_oof_predictions.parquet", index=False)

    print("\nRésumé Cross Validation")
    print("---" * 10)
    for metric, values in summary.items():
        if isinstance(values, dict) and "mean" in values:
            print(f"{metric}: " f"mean={values['mean']:.4f} | " f"std={values['std']:.4f} | " f"CI95=[{values['ci95_low']:.4f}, {values['ci95_high']:.4f}]")

    print("\nPerformance Gate CV")
    print("---" * 10)
    print(summary["performance_gate"])

    print("\nRapports sauvegardés ")
    print("=" * 60)


if __name__ == "__main__":
    main()
