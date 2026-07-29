"""
Réentraînement avec features d'anomalie
    - Réentraîner XGBoost avec anomaly_score et is_anomaly
    - Utiliser les meilleurs hyperparamètres Optuna si disponibles
    - Comparer les performances avec le modèle sans anomalies
    - Logger le run dans MLflow
    - Sauvegarder un rapport de comparaison

"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
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
from xgboost import XGBClassifier

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/anomaly/")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"
RANDOM_STATE = 42
THRESHOLD = 0.5
PERFORMANCE_THRESHOLD = 0.80

DROP = [
    "applicant_id",
    "date_demande",
    "nom_partenaire",
]


# Charger les datasets Anomalies
def load_data():
    x_train_path = SPLITS_DIR / "X_train_with_anomalies.parquet"
    x_test_path = SPLITS_DIR / "X_test_with_anomalies.parquet"
    y_train_path = SPLITS_DIR / "y_train.parquet"
    y_test_path = SPLITS_DIR / "y_test.parquet"

    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}. " "Exécute d'abord src/11_anomaly_detection.py.")

    X_train = pd.read_parquet(x_train_path)
    X_test = pd.read_parquet(x_test_path)
    y_train = pd.read_parquet(y_train_path)[TARGET].astype(int)
    y_test = pd.read_parquet(y_test_path)[TARGET].astype(int)

    return X_train, X_test, y_train, y_test


# Colonne inutile
def remove_excluded_columns(X):
    cols_to_drop = [col for col in DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


# Utiliser les meilleurs parametres de Optuna
def load_best_params():
    params_path = REPORTS_DIR / "optuna_best_params.json"

    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as file:
            params = json.load(file)

        print(f"Paramètres Optuna chargés : {params_path}")
        return params

    print("Aucun fichier Optuna trouvé. Utilisation de paramètres XGBoost par défaut.")

    return {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 1.0,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }


def select_column_types(X):
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_cols, categorical_cols


# Pipeline Scikit-learn
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


# Construction du mod1ele de ML
def build_model_pipeline(X_train, y_train, params):
    numeric_cols, categorical_cols = select_column_types(X_train)
    print(f"Colonnes numériques : {len(numeric_cols)}")
    print(f"Colonnes catégorielles : {len(categorical_cols)}")

    if "anomaly_score" not in X_train.columns or "is_anomaly" not in X_train.columns:
        raise ValueError("Les colonnes anomaly_score et is_anomaly sont absentes. " "Exécute d'abord src/10_anomaly_detection.py.")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, numeric_cols, categorical_cols, scale_pos_weight


# Métrique
def compute_metrics(y_true, y_proba, threshold=THRESHOLD):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# Résultats des précedents sans anomalies
def load_previous_results():
    candidates = [
        REPORTS_DIR / "optuna_tuning_report.json",
        REPORTS_DIR / "final_evaluation_report.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if "validation_metrics" in data:
            return {
                "source": str(path),
                "metrics": data["validation_metrics"],
            }

        if "final_test_metrics" in data:
            return {
                "source": str(path),
                "metrics": data["final_test_metrics"],
            }

    return None


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT AVEC FEATURES D'ANOMALIE")
    print("=" * 60)

    mlflow.set_experiment("Finascore")

    X_train, X_test, y_train, y_test = load_data()

    X_train = remove_excluded_columns(X_train)
    X_test = remove_excluded_columns(X_test)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print("Distribution y_train :")
    print(y_train.value_counts(normalize=True).round(4))

    params = load_best_params()

    pipeline, numeric_cols, categorical_cols, scale_pos_weight = build_model_pipeline(
        X_train=X_train,
        y_train=y_train,
        params=params,
    )

    with mlflow.start_run(run_name="xgboost_with_anomaly_features"):
        print("Entraînement XGBoost avec anomaly_score et is_anomaly...")
        pipeline.fit(X_train, y_train)

        print("Évaluation sur test final...")
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(
            y_true=y_test,
            y_proba=y_test_proba,
            threshold=THRESHOLD,
        )

        model_path = MODELS_DIR / "xgboost_with_anomalies.pkl"
        joblib.dump(pipeline, model_path)

        mlflow.log_params(params)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_numeric_cols", len(numeric_cols))
        mlflow.log_param("n_categorical_cols", len(categorical_cols))
        mlflow.log_param("uses_anomaly_score", True)
        mlflow.log_param("uses_is_anomaly", True)

        mlflow.log_metric("test_auc_roc", metrics["auc_roc"])
        mlflow.log_metric("test_f1", metrics["f1"])
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])
        mlflow.log_metric("test_accuracy", metrics["accuracy"])

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

    previous_results = load_previous_results()

    performance_gate = {
        "metric": "auc_roc",
        "required_threshold": PERFORMANCE_THRESHOLD,
        "observed_value": metrics["auc_roc"],
        "passed": metrics["auc_roc"] >= PERFORMANCE_THRESHOLD,
        "decision": ("PROMOTE_TO_PRODUCTION" if metrics["auc_roc"] >= PERFORMANCE_THRESHOLD else "DO_NOT_PROMOTE"),
    }

    comparison_report = {
        "model_name": "xgboost_with_anomaly_features",
        "model_path": str(model_path),
        "test_metrics_with_anomalies": metrics,
        "previous_results_without_anomalies": previous_results,
        "performance_gate": performance_gate,
        "interpretation": ("Les features d'anomalie améliorent le modèle si l'AUC augmente de façon notable " "par rapport au modèle sans anomalies. Si le gain est faible, elles peuvent être conservées " "comme diagnostic métier mais ne changent pas la décision de non-promotion."),
    }

    report_path = REPORTS_DIR / "anomaly_feature_model_comparison.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(comparison_report, file, indent=4, ensure_ascii=False)

    comparison_rows = [
        {
            "model": "xgboost_with_anomalies",
            **metrics,
            "source": "test_final",
        }
    ]

    if previous_results is not None:
        prev_metrics = previous_results["metrics"]
        comparison_rows.append(
            {
                "model": "previous_without_anomalies",
                "threshold": prev_metrics.get("threshold", None),
                "accuracy": prev_metrics.get("accuracy", None),
                "auc_roc": prev_metrics.get("auc_roc", None),
                "f1": prev_metrics.get("f1", None),
                "precision": prev_metrics.get("precision", None),
                "recall": prev_metrics.get("recall", None),
                "source": previous_results["source"],
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        REPORTS_DIR / "anomaly_feature_model_comparison.csv",
        index=False,
    )

    print("\nRésultats avec anomalies")
    print("-----------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if previous_results:
        print("\nRésultat précédent sans anomalies")
        print("--------------------------------")
        print(previous_results)

    print("\nPerformance Gate")
    print("----------------")
    print(f"Seuil requis : {PERFORMANCE_THRESHOLD}")
    print(f"AUC observée : {metrics['auc_roc']:.4f}")
    print(f"Décision : {performance_gate['decision']}")

    print("\nFichiers générés")
    print("----------------")
    print(f"- {model_path}")
    print(f"- {report_path}")
    print("- reports/anomaly_feature_model_comparison.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
