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
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (
    RANDOM_STATE,
    SPLITS_DIR,
    TARGET,
    THRESHOLD,
    build_preprocessor,
    compute_metrics,
    make_promotion_gate,
    remove_columns,
    select_column_types,
)

REPORTS_DIR = Path("reports")
VALIDATION_DIR = REPORTS_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5


def load_data():
    X_train_path = SPLITS_DIR / "X_train.parquet"
    y_train_path = SPLITS_DIR / "y_train.parquet"

    if not X_train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {X_train_path}")

    if not y_train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {y_train_path}")

    X = pd.read_parquet(X_train_path)
    y = pd.read_parquet(y_train_path)[TARGET].astype(int)

    return X, y


def build_xgboost_pipeline(X, y):
    numeric_cols, categorical_cols = select_column_types(X)
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    print("=" * 60)
    print("CROSS-VALIDATION STRATIFIED K-FOLD")
    print("=" * 60)

    X_raw, y = load_data()

    print(f"X brut : {X_raw.shape}")
    print("Distribution target :")
    print(y.value_counts(normalize=True).round(4))

    X_model = remove_columns(X_raw)

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

        print(f"AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(VALIDATION_DIR / "cv_fold_metrics.csv", index=False)

    metric_columns = ["auc_roc", "f1", "precision", "recall"]
    summary = {"n_splits": N_SPLITS, "threshold": THRESHOLD, "per_fold": fold_metrics_df[metric_columns + ["fold"]].to_dict(orient="records")}

    for metric in metric_columns:
        values = fold_metrics_df[metric].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if N_SPLITS > 1 else 0.0
        se = std / np.sqrt(N_SPLITS)
        ci_low = float(mean - stats.t.ppf(0.975, N_SPLITS - 1) * se)
        ci_high = float(mean + stats.t.ppf(0.975, N_SPLITS - 1) * se)
        summary[metric] = {"mean": mean, "std": std, "ci95_low": ci_low, "ci95_high": ci_high}

    auc_mean = summary["auc_roc"]["mean"]

    summary["performance_gate"] = make_promotion_gate(auc_mean)

    with open(VALIDATION_DIR / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    oof_df = X_raw.copy()
    oof_df[TARGET] = y.values
    oof_df["y_proba"] = oof_predictions
    oof_df["y_pred"] = (oof_df["y_proba"] >= THRESHOLD).astype(int)

    oof_df.to_parquet(VALIDATION_DIR / "cv_oof_predictions.parquet", index=False)

    print("\nRésumé Cross Validation")
    print("---" * 10)
    for metric, values in summary.items():
        if isinstance(values, dict) and "mean" in values:
            print(f"{metric}: mean={values['mean']:.4f} | std={values['std']:.4f} | CI95=[{values['ci95_low']:.4f}, {values['ci95_high']:.4f}]")

    print("\nPerformance Gate CV")
    print("---" * 10)
    print(summary["performance_gate"])

    print("\nRapports sauvegardés ")
    print("=" * 60)


if __name__ == "__main__":
    main()
