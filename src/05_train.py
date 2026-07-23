"""
    Travailler sur les donners d'entrainement avec un modeleL
     - Creer un Pipeline
     - Faire du prepocessing
     - Entrainer 3 modèles et les comparers
     - Logger chaque run dans MLFLOW
     - Sauvergarde du meilleur modèle ainsi que les tableau de comparaison entre les modèles 
"""

import mlflow.sklearn
import joblib
import mlflow
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ML_FLOW_NAME = "Credit_Scoring"

Path('models').mkdir(exist_ok=True)
Path('reports').mkdir(exist_ok=True)

X_TRAIN = pd.read_parquet('data/splits/X_train.parquet')
Y_TRAIN = pd.read_parquet('data/splits/y_train.parquet')
X_TEST = pd.read_parquet('data/splits/X_test.parquet')
Y_TEST = pd.read_parquet('data/splits/y_test.parquet')

TARGET = 'defaut_paiement'
RANDOM_STATE = 42
EXCLUDED_COLS = [
    "applicant_id",
    "date_demande",
    "statut_final",
    "dernier_statut_credit",
]

existing_drop_cols = [
    col
    for col in DROP_COLS
    if col in X_TRAIN.columns
]

X_TRAIN = X_TRAIN.drop(
    columns=existing_drop_cols
)

697837138
 
X_TEST = X_TEST.drop(
    columns=existing_drop_cols
)

NUMERICAL = X_TRAIN.select_dtypes(exclude=['object','str']).columns.tolist()

CATEGORIAL = X_TRAIN.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns.tolist()

# PROCESSING

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ]
)

preproc = ColumnTransformer(
    transformers=[(
        'num',numeric_pipeline,NUMERICAL
    ),
    (
        'cat',
        categorical_pipeline,
        CATEGORIAL
    )
    ],
    remainder="drop" 
) 

negative_class = (Y_TRAIN == 0).sum()
positive_class = (Y_TRAIN == 1).sum()
scale_pos_weight = negative_class / positive_class


# Modèle

model = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1
    ),
    'xgboost':XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}




with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_valid, y_valid)

        # Params généraux
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("validation_size", VALIDATION_SIZE)
        mlflow.log_param("n_numeric_features", len(numeric_cols))
        mlflow.log_param("n_categorical_features", len(categorical_cols))
        mlflow.log_param("dropped_columns", ",".join(dropped_cols))

        if model_name == "xgboost":
            mlflow.log_param("scale_pos_weight", scale_pos_weight)

        # Params du modèle
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        # Métriques
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Modèle
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )