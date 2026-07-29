"""
Entraiment des modèles
- recuperer les données split du train et du test
- Supprimer les colones qui pourront apporter du bruit
- Gerer les valeurs Nan avec les imputations pour le modele RandomForest, LogisticRegression
- Comparer les modeles et enregistrer le meilleur modele avec joblib
- L'evaluation dans le 06_evaluate ainsi que learning curve

"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/")
REPORTS_DIR_LEARN = Path("reports/learning/")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GENERATE_LEARNING_CURVES = True
TARGET = "defaut_paiement"
RANDOM_STATE = 42

COLS_TO_DROP = [
    "applicant_id",
    "date_demande",
    "nom_partenaire",
    # "historique_credit", "dernier_statut_credit",
    # "revenu_mensuel_xaf", "volume_entrant", "volume_sortant", "volume_mm_total",
    # "total_montant_xaf", "avg_credit_xaf", "total_retards", "nb_credit", "regularite_score", "anciennete_compte_mois",
    # "nom_partenaire", "pays_partenaire", "type_partenaire",
    # "seuil_score_partenaire", "volume_mensuel_partenaire",
    # "nb_credits_defaut_hist", "nb_credits_rembourses_hist",
    # "nb_credits_restructures_hist", "nb_credits_en_cours_hist",
    # "flag_surendette", "flag_no_credit_history",
]

# cols_to_drop_existing = [col for col in COLS_TO_DROP if col in df_finascore.columns]
# df_finascore = df_finascore.drop(columns=cols_to_drop_existing)


def load_data():
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")[TARGET]
    return X_train, y_train


def remove_excluded_columns(X):
    cols_to_drop = [col for col in COLS_TO_DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


def detect_column_types(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
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
        transformers=[("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)],
        remainder="drop",
    )


def get_base_models(y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE, solver="liblinear", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1),
    }
    return models, scale_pos_weight


def get_xgb_param_distributions():
    return {
        "model__n_estimators": randint(100, 800),
        "model__max_depth": randint(1, 8),
        "model__learning_rate": uniform(0.01, 0.14),
        "model__min_child_weight": randint(3, 30),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__gamma": uniform(0, 2),
    }


def plot_learning_curve_for_model(pipeline, X, y, model_name):
    print(f"  -> Génération learning curve pour {model_name}...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    train_sizes = np.linspace(0.2, 1.0, 5)

    train_sizes_abs, train_scores, val_scores = learning_curve(estimator=pipeline, X=X, y=y, train_sizes=train_sizes, cv=cv, scoring="roc_auc", n_jobs=-1)

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    curve_df = pd.DataFrame(
        {
            "train_size": train_sizes_abs,
            "train_auc_mean": train_mean,
            "train_auc_std": train_std,
            "val_auc_mean": val_mean,
            "val_auc_std": val_std,
        }
    )

    curve_csv_path = REPORTS_DIR_LEARN / f"learning_curve_{model_name}.csv"
    curve_df.to_csv(curve_csv_path, index=False)

    plt.figure(figsize=(8, 5))

    plt.plot(train_sizes_abs, train_mean, marker="o", label="Train AUC")

    plt.plot(train_sizes_abs, val_mean, marker="o", label="Validation AUC")

    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)

    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Nombre d'exemples d'entraînement")
    plt.ylabel("AUC-ROC")
    plt.legend()
    plt.tight_layout()

    curve_png_path = REPORTS_DIR_LEARN / f"learning_curve_{model_name}.png"
    plt.savefig(curve_png_path)
    plt.close()

    print(f"    Learning curve sauvegardée : {curve_png_path}")


def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    print(f"    -> AUC Train: {roc_auc_score(y_train, train_proba):.4f} | AUC Test: {roc_auc_score(y_test, y_proba):.4f}")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def train_model(model_name, model, preprocessor, X_train, X_test, y_train, y_test, param_distributions=None, n_iter=50):
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    best_params = "Default"

    with mlflow.start_run(run_name=model_name):
        if param_distributions is not None:
            print(f"  -> Lancement du RandomizedSearch ({n_iter} iterations)....")
            search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_distributions, n_iter=n_iter, cv=3, scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE)
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = search.best_params_
            print(f"    -> Meilleur AUC CV: {search.best_score_:.4f} and params : {best_params}")
        else:
            pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
        if GENERATE_LEARNING_CURVES:
            plot_learning_curve_for_model(pipeline=pipeline, X=X_train, y=y_train, model_name=model_name)
        curve_png_path = REPORTS_DIR / f"learning_curve_{model_name}.png"
        curve_csv_path = REPORTS_DIR / f"learning_curve_{model_name}.csv"

        if curve_png_path.exists():
            mlflow.log_artifact(str(curve_png_path))

        if curve_csv_path.exists():
            mlflow.log_artifact(str(curve_csv_path))

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(best_params if isinstance(best_params, dict) else {})
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(sk_model=pipeline, name="model", serialization_format="cloudpickle")

    return {"model_name": model_name, **metrics, "model_path": str(model_path)}


def main():
    mlflow.set_experiment("Finascore")

    X, y = load_data()
    X = remove_excluded_columns(X)
    # X_test = remove_excluded_columns(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print(f"X_train shape : {X_train.shape} | X_val shape : {X_val} ")

    numeric_cols, categorical_cols = detect_column_types(X_train)
    print(f"Numeriques: {len(numeric_cols)} | Categorielles: {len(categorical_cols)}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models, scale_pos_weight = get_base_models(y_train)
    xgb_dists = get_xgb_param_distributions()

    print(f"\nscale_pos_weight XGBoost: {round(scale_pos_weight, 2)}\n")

    results = []

    print("=" * 60)
    print("Debut Entrainement et evaluation des models")
    print("=" * 60)

    for model_name, model in models.items():
        print(f"Entrainement {model_name}...")

        if model_name == "xgboost":
            current_dists = xgb_dists
        else:
            current_dists = None

        result = train_model(model_name, model, preprocessor, X_train, X_val, y_train, y_val, param_distributions=current_dists)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="auc_roc", ascending=False)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model = results_df.iloc[0]
    with open(REPORTS_DIR / "best_model.json", "w") as f:
        json.dump(best_model.to_dict(), f, indent=4)

    print("Modèles sauvegardé .... ")

    print("\n--- COMPARAISON FINALE ---")
    print(results_df[["model_name", "auc_roc", "f1"]].to_string(index=False))
    print(f"\nMeilleur modele : {best_model['model_name']} (AUC: {best_model['auc_roc']:.4f})")


if __name__ == "__main__":
    main()
