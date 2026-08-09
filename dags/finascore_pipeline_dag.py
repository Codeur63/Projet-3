"""
DAG Airflow pour orchestrer le pipeline ML FinaScore.
    - automatiser l'exécution des scripts de pipeline
    - tracer les logs de chaque étape
    - arrêter le pipeline si une étape échoue
    - Rendre le MLOps reproductible
    - décision de promotion du modèle (performance gate)
    - détection de drift après entraînement
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"
PYTHON = "python3"

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="finascore_ml_pipeline",
    description="Pipeline ML complet FinaScore SA",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "credit_scoring", "finascore"],
    default_args=default_args,
) as dag:
    collect_data = BashOperator(
        task_id="01_collect_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/01_collect.py",
    )

    clean_data = BashOperator(
        task_id="02_clean_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/02_clean.py",
    )

    feature_engineering = BashOperator(
        task_id="03_feature_engineering",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/03_features.py",
    )

    split_data = BashOperator(
        task_id="04_split_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/04_split.py",
    )

    train_models = BashOperator(
        task_id="05_train_models",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/05_train.py",
        execution_timeout=timedelta(hours=4),
    )

    evaluate_model = BashOperator(
        task_id="06_evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/06_evaluate.py",
    )

    register_model = BashOperator(
        task_id="13_register_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/13_register_model.py",
    )

    drift_monitoring = BashOperator(
        task_id="14_drift_monitoring",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/14_monitoring.py",
    )

    (collect_data >> clean_data >> feature_engineering >> split_data >> train_models >> evaluate_model >> register_model >> drift_monitoring)
