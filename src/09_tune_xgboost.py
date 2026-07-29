"""
Tuning XGBoost avec Optuna
    - Optimiser les hyperparamètres de XGBoost
    - Utiliser uniquement les données d'entraînement
    - Logger chaque trial dans MLflow
    - Sauvegarder les meilleurs paramètres et le meilleur modèle
    - Chercher une performance gate AUC >= 0.80
    
"""

import json
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from xgboost import XGBClassifier


SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models/optuna")
REPORTS_DIR = Path("reports/optuna")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"
RANDOM_STATE = 42
N_TRIALS = 100
PERFORMANCE_THRESHOLD = 0.80

COLS_TO_DROP = [
    "applicant_id",
    "date_demande",
    "nom_partenaire"
]


def load_train_data():
    X = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    y = pd.read_parquet(SPLITS_DIR / "y_train.parquet")[TARGET].astype(int)
    return X, y


def remove_excluded_columns(X):
    cols_to_drop = [col for col in COLS_TO_DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


def detect_column_types(X):
    numeric_cols = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def optuna_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def build_pipeline(params, preprocessor, scale_pos_weight):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    print("=" * 60)
    print("TUNING XGBOOST AVEC OPTUNA")
    print("=" * 60)

    mlflow.set_experiment("Finascore")

    X, y = load_train_data()
    X = remove_excluded_columns(X)

    print(f"X shape : {X.shape}")
    print("Distribution target :")
    print(y.value_counts(normalize=True).round(4))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    numeric_cols, categorical_cols = detect_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"Colonnes numériques : {len(numeric_cols)}")
    print(f"Colonnes catégorielles : {len(categorical_cols)}")
    print(f"scale_pos_weight : {scale_pos_weight:.2f}")

    trials_results = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
        }

        pipeline = build_pipeline(
            params=params,
            preprocessor=preprocessor,
            scale_pos_weight=scale_pos_weight,
        )

        with mlflow.start_run(run_name=f"optuna_xgboost_trial_{trial.number}", nested=True):
            pipeline.fit(X_train, y_train)

            y_val_proba = pipeline.predict_proba(X_val)[:, 1]
            metrics = optuna_metrics(y_val, y_val_proba)

            mlflow.log_params(params)
            mlflow.log_metric("auc_roc", metrics["auc_roc"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])

        trial_result = {
            "trial_number": trial.number,
            **params,
            **metrics,
        }

        trials_results.append(trial_result)

        return metrics["auc_roc"]

    with mlflow.start_run(run_name="optuna_xgboost_tuning"):
        study = optuna.create_study(
            direction="maximize",
            study_name="finascore_xgboost_optuna",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        best_params = study.best_params
        best_auc = study.best_value

        print("\nMeilleurs paramètres Optuna :")
        print(best_params)
        print(f"Meilleur AUC validation : {best_auc:.4f}")

        best_pipeline = build_pipeline(
            params=best_params,
            preprocessor=preprocessor,
            scale_pos_weight=scale_pos_weight,
        )

        best_pipeline.fit(X_train, y_train)

        y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
        final_metrics = optuna_metrics(y_val, y_val_proba)

        model_path = MODELS_DIR / "xgboost_optuna.pkl"
        joblib.dump(best_pipeline, model_path)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_auc_roc", final_metrics["auc_roc"])
        mlflow.log_metric("best_f1", final_metrics["f1"])
        mlflow.log_metric("best_precision", final_metrics["precision"])
        mlflow.log_metric("best_recall", final_metrics["recall"])

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

        trials_df = pd.DataFrame(trials_results)
        trials_path = REPORTS_DIR / "optuna_trials.csv"
        trials_df.to_csv(trials_path, index=False)

        best_params_path = REPORTS_DIR / "optuna_best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4, ensure_ascii=False)

        tuning_report = {
            "model_name": "xgboost_optuna",
            "n_trials": N_TRIALS,
            "best_params": best_params,
            "validation_metrics": final_metrics,
            "model_path": str(model_path),
            "performance_gate": {
                "metric": "auc_roc",
                "required_threshold": PERFORMANCE_THRESHOLD,
                "observed_value": final_metrics["auc_roc"],
                "passed": final_metrics["auc_roc"] >= PERFORMANCE_THRESHOLD,
                "decision":
                (
                    "PROMOTE_TO_PRODUCTION"
                    if final_metrics["auc_roc"] >= PERFORMANCE_THRESHOLD
                    else "NOT PROMOTION"
                ),
            },
        }

        tuning_report_path = REPORTS_DIR / "optuna_tuning_report.json"
        with open(tuning_report_path, "w", encoding="utf-8") as f:
            json.dump(tuning_report, f, indent=4, ensure_ascii=False)

        mlflow.log_artifact(str(trials_path))
        mlflow.log_artifact(str(best_params_path))
        mlflow.log_artifact(str(tuning_report_path))

    print("\nTuning terminé")
    print(f"Meilleur AUC validation : {final_metrics['auc_roc']:.4f}")
    print(f"F1 : {final_metrics['f1']:.4f}")
    print(f"Precision : {final_metrics['precision']:.4f}")
    print(f"Recall : {final_metrics['recall']:.4f}")
    print(f"Décision : {tuning_report['performance_gate']['decision']}")
    print(f"Modèle sauvegardé : {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()