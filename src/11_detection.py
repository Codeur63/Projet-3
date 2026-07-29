"""
Détection d'anomalies avec Isolation Forest
    - Entraîner Isolation Forest uniquement sur X_train
    - Créer une feature anomaly_score et is_anomaly
    - Sauvegarder un dataset enrichi utilisable pour un futur entraînement
    - Faire un rapport d'analyses des anamalies

"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/detection/")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "defaut_paiement"
RANDOM_STATE = 42

# Proportion attendue d'anomalies.
# 0.03 signifie environ 3% des profils considérés comme atypiques.
CONTAMINATION = 0.03

DROP = [
    "applicant_id",
    "date_demande",
]

# Charger les données d'entrainement et de test
def load_data():
    X_train_path = SPLITS_DIR / "X_train.parquet"
    X_test_path = SPLITS_DIR / "X_test.parquet"
    y_train_path = SPLITS_DIR / "y_train.parquet"
    y_test_path = SPLITS_DIR / "y_test.parquet"

    for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)
    y_train = pd.read_parquet(y_train_path)[TARGET].astype(int)
    y_test = pd.read_parquet(y_test_path)[TARGET].astype(int)

    return X_train, X_test, y_train, y_test


# Recuperer les colones utilisables pour la technique
def select_numeric_features(X_train):
    cols_drop = [col for col in DROP if col in X_train.columns]

    X = X_train.drop(columns=cols_drop)

    numeric_cols = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    if not numeric_cols:
        raise ValueError("Aucune colonne numérique disponible pour Isolation Forest.")

    return numeric_cols


# Pipeline Isolation Forest
def build_anomaly_pipeline():
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=300,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def add_anomaly_features(pipeline, X, numeric_cols):
    """
    Ajoute anomaly_score et is_anomaly.

    decision_function :
    - score élevé = profil normal
    - score faible = profil atypique

    On inverse donc le score :
    - anomaly_score élevé = profil atypique
    """

    X_enriched = X.copy()
    X_numeric = X_enriched[numeric_cols]

    decision_scores = pipeline.decision_function(X_numeric)
    raw_predictions = pipeline.predict(X_numeric)

    X_enriched["anomaly_score"] = -decision_scores
    X_enriched["is_anomaly"] = (raw_predictions == -1).astype(int)

    return X_enriched 



def summarize_anomalies(X_enriched, y, dataset_name):
    """Produit un résumé global des anomalies."""

    df = X_enriched.copy()
    df[TARGET] = y.values
    
    anomaly_df = df[df["is_anomaly"] == 1]
    normal_df = df[df["is_anomaly"] == 0]

    summary = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_anomalies": int(len(anomaly_df)),
        "n_normal": int(len(normal_df)),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "default_rate_global": float(df[TARGET].mean()),
        "default_rate_anomalies": float(anomaly_df[TARGET].mean()) if len(anomaly_df) > 0 else None,
        "default_rate_normal": float(normal_df[TARGET].mean()) if len(normal_df) > 0 else None,
        "avg_anomaly_score": float(df['anomaly_score'].mean()),
        "contamination_parameter": CONTAMINATION,
    }

    return summary


# Trouver des anomalies par pays, secteur et Zone
def subgroup_anomaly_report(X_enriched, y, dataset_name):
    subgroup_cols = [
        "pays",
        "secteur_activite",
        "zone"
        ]

    rows = []

    for col in subgroup_cols:
        if col not in X_enriched.columns:
            continue

        for value, group in X_enriched.groupby(col, dropna=False):
            if len(group) < 50:
                continue

            rows.append(
                {
                    "dataset": dataset_name,
                    "subgroup_column": col,
                    "subgroup_value": str(value),
                    "n_samples": int(len(group)),
                    "anomaly_rate": float(group["is_anomaly"].mean()),
                    # "default_rate": float(group[TARGET].mean()),
                    "avg_anomaly_score": float(group["anomaly_score"].mean()),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("anomaly_rate", ascending=False)
    )


def numeric_profile_report(df, numeric_cols):
    """Compare le profil moyen des anomalies vs non-anomalies."""

    rows = []

    for col in numeric_cols:
        normal_values = pd.to_numeric(
            df.loc[df["is_anomaly"] == 0, col],
            errors="coerce",
        )

        anomaly_values = pd.to_numeric(
            df.loc[df["is_anomaly"] == 1, col],
            errors="coerce",
        )

        if normal_values.notna().sum() == 0 or anomaly_values.notna().sum() == 0:
            continue

        rows.append(
            {
                "feature": col,
                "normal_mean": float(normal_values.mean()),
                "anomaly_mean": float(anomaly_values.mean()),
                "normal_median": float(normal_values.median()),
                "anomaly_median": float(anomaly_values.median()),
                "absolute_mean_difference": float(
                    abs(anomaly_values.mean() - normal_values.mean())
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("absolute_mean_difference", ascending=False)
    )


def main():
    print("=" * 60)
    print("DÉTECTION D'ANOMALIES - ISOLATION FOREST")
    print("=" * 60)

    print(" Chargement des données splits ")
    X_train, X_test, y_train, y_test = load_data()

    print("Selection des colonnes")
    numeric_cols = select_numeric_features(X_train)

    print(f"Colonnes numériques utilisées : {len(numeric_cols)}")

    anomaly_pipeline = build_anomaly_pipeline()

    print("Entraînement Isolation Forest sur X_train ...")
    anomaly_pipeline.fit(X_train[numeric_cols])
    
    X_train_enriched = add_anomaly_features(
        pipeline=anomaly_pipeline,
        X=X_train,
        numeric_cols=numeric_cols,
    )

    X_test_enriched = add_anomaly_features(
        pipeline=anomaly_pipeline,
        X=X_test,
        numeric_cols=numeric_cols,
    )
    
    train_output_path = SPLITS_DIR / "X_train_with_anomalies.parquet"
    test_output_path = SPLITS_DIR / "X_test_with_anomalies.parquet"
    
    X_train_enriched.to_parquet(train_output_path, index=False)
    X_test_enriched.to_parquet(test_output_path, index=False)

    
    model_path = MODELS_DIR / "isolation_forest.pkl"
    joblib.dump(anomaly_pipeline, model_path)

    train_summary = summarize_anomalies(
        X_enriched=X_train_enriched,
        y=y_train,
        dataset_name="train",
    )

    test_summary = summarize_anomalies(
        X_enriched=X_test_enriched,
        y=y_test,
        dataset_name="test",
    )


    summary = {
        "method": "IsolationForest",
        "contamination": CONTAMINATION,
        "model_path": str(model_path),
        "numeric_columns_used": numeric_cols,
        "train_summary": train_summary,
        "test_summary": test_summary,
    }

    with open(REPORTS_DIR / "anomaly_detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    subgroup_train = subgroup_anomaly_report(
        X_enriched=X_train_enriched,
        y=y_train,
        dataset_name="train",
    )

    subgroup_test = subgroup_anomaly_report(
        X_enriched=X_test_enriched,
        y=y_test,
        dataset_name="test",
    )

    subgroup_report = pd.concat(
        [subgroup_train, subgroup_test],
        ignore_index=True,
    )

 
    subgroup_report.to_csv(
        REPORTS_DIR / "anomaly_detection_by_subgroup.csv",
        index=False,
    )

    top_100_train = X_train_enriched.copy()
    top_100_train[TARGET] = y_train.values
    top_100_train = top_100_train.sort_values(
        "anomaly_score",
        ascending=False,
    ).head(100)

    top_100_test = X_test_enriched.copy()
    top_100_test[TARGET] = y_test.values
    top_100_test = top_100_test.sort_values(
        "anomaly_score",
        ascending=False,
    ).head(100)

    top_100_train.to_csv(
        REPORTS_DIR / "top_100_anomalies_train.csv",
        index=False,
    )

    top_100_test.to_csv(
        REPORTS_DIR / "top_100_anomalies_test.csv",
        index=False,
    )

    print("\nFichiers générés")
    print("----------------")
    print(f"- {train_output_path}")
    print(f"- {test_output_path}")
    print(f"- {model_path}")
    print(f"- {REPORTS_DIR / 'anomaly_detection_summary.json'}")
    print(f"- {REPORTS_DIR / 'anomaly_detection_by_subgroup.csv'}")
    print(f"- {REPORTS_DIR / 'top_100_anomalies_train.csv'}")
    print(f"- {REPORTS_DIR / 'top_100_anomalies_test.csv'}")

    print("=" * 60)


if __name__ == "__main__":
    main()