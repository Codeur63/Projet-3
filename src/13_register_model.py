"""
Enregistrement du modèle candidat dans MLflow Model Registry
    - Charger le meilleur modèle disponible
    - Vérifier le seuil de performance AUC >= 0.80
    - Logger le modèle dans MLflow
    - Tenter l'enregistrement dans le Model Registry
    - Sauvegarder une décision de promotion claire

"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "Finascore"
REGISTERED_MODEL_NAME = "Finascore_model"

PERFORMANCE_THRESHOLD = 0.80


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_best_available_candidate():
    """
    Cherche le meilleur candidat disponible dans l'ordre :
    1. Modèle avec features d'anomalie
    2. Modèle XGBoost Optuna
    3. Modèle champion de 05_train.py
    """

    anomaly_report = REPORTS_DIR / "anomaly/anomaly_feature_model_comparison.json"

    if anomaly_report.exists():
        data = load_json(anomaly_report)

        return {
            "source_report": str(anomaly_report),
            "model_name": data.get("model_name", "xgboost_with_anomaly_features"),
            "model_path": data["model_path"],
            "metrics": data["test_metrics_with_anomalies"],
            "selection_reason": "Modèle enrichi avec anomaly_score et is_anomaly.",
        }

    optuna_report = REPORTS_DIR / "optuna/optuna_tuning_report.json"

    if optuna_report.exists():
        data = load_json(optuna_report)

        return {
            "source_report": str(optuna_report),
            "model_name": data.get("model_name", "xgboost_optuna"),
            "model_path": data["model_path"],
            "metrics": data["validation_metrics"],
            "selection_reason": "Modèle XGBoost optimisé avec Optuna.",
        }

    best_model_report = REPORTS_DIR / "best_model.json"
    final_eval_report = REPORTS_DIR / "evaluation/final_evaluation_report.json"

    if best_model_report.exists() and final_eval_report.exists():
        best_model = load_json(best_model_report)
        final_eval = load_json(final_eval_report)

        return {
            "source_report": str(final_eval_report),
            "model_name": best_model["model_name"],
            "model_path": best_model["model_path"],
            "metrics": final_eval["final_test_metrics"],
            "selection_reason": "Modèle champion issu de 05_train.py.",
        }

    raise FileNotFoundError("Aucun candidat modèle trouvé. Exécute d'abord 05_train.py, 08_tune_xgboost_optuna.py ou 11_train_with_anomalies.py.")


def build_promotion_decision(metrics):
    """Construit la décision de promotion selon le seuil AUC."""

    auc = float(metrics["auc_roc"])
    passed = auc >= PERFORMANCE_THRESHOLD

    if passed:
        decision = "PROMOTE_TO_PRODUCTION"
        registry_status = "production_candidate"
        comment = "Le modèle respecte le seuil AUC >= 0.80. Il peut être considéré comme candidat à la production."
    else:
        decision = "NOT_PROMOTE"
        registry_status = "staging_only"
        comment = "Le modèle ne respecte pas le seuil AUC >= 0.80. Enregistrer mais pas promu."

    return {
        "metric": "auc_roc",
        "required_threshold": PERFORMANCE_THRESHOLD,
        "observed_value": auc,
        "passed": passed,
        "decision": decision,
        "registry_status": registry_status,
        "comment": comment,
    }


def log_related_artifacts():
    candidate_artifacts = [
        REPORTS_DIR / "registry/model_comparison.csv",
        REPORTS_DIR / "registry/final_evaluation_report.json",
        REPORTS_DIR / "registry/performance_gate.json",
        REPORTS_DIR / "registry/cv_metrics.json",
        REPORTS_DIR / "registry/cv_fold_metrics.csv",
        REPORTS_DIR / "registry/optuna_tuning_report.json",
        REPORTS_DIR / "registry/optuna_trials.csv",
        REPORTS_DIR / "registry/error_analysis_summary.json",
        REPORTS_DIR / "registry/error_analysis_business.md",
        REPORTS_DIR / "registry/anomaly_detection_summary.json",
        REPORTS_DIR / "registry/anomaly_feature_model_comparison.json",
    ]

    for artifact_path in candidate_artifacts:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))


def try_register_model(model, model_name, metrics, promotion_decision):
    """
    Loggue le modèle dans MLflow et tente de l'enregistrer dans le Model Registry.
    La fonction reste robuste si le registry local n'est pas disponible.
    """

    registry_info = {
        "registered": False,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": None,
        "alias": None,
        "error": None,
    }

    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )

        registry_info["registered"] = True

        model_version = getattr(model_info, "registered_model_version", None)

        if model_version is not None:
            registry_info["model_version"] = str(model_version)

            client = MlflowClient()

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version,
                key="model_name",
                value=model_name,
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version,
                key="auc_roc",
                value=str(metrics["auc_roc"]),
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version,
                key="promotion_decision",
                value=promotion_decision["decision"],
            )

            alias = promotion_decision["registry_status"]

            try:
                client.set_registered_model_alias(
                    name=REGISTERED_MODEL_NAME,
                    alias=alias,
                    version=model_version,
                )
                registry_info["alias"] = alias

            except Exception as alias_error:
                registry_info["alias"] = None
                registry_info["alias_error"] = str(alias_error)

    except Exception as error:
        registry_info["registered"] = False
        registry_info["error"] = str(error)

        # Fallback : logger seulement le modèle comme artefact MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

    return registry_info


def main():
    print("=" * 60)
    print("MODEL REGISTRY - DÉCISION DE PROMOTION")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    candidate = find_best_available_candidate()

    model_path = Path(candidate["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model_name = candidate["model_name"]
    metrics = candidate["metrics"]

    print(f"Modèle candidat : {model_name}")
    print(f"Chemin modèle : {model_path}")
    print(f"Source rapport : {candidate['source_report']}")

    promotion_decision = build_promotion_decision(metrics)

    print("\nPerformance Gate")
    print("----------------")
    print(f"AUC observée : {promotion_decision['observed_value']:.4f}")
    print(f"Seuil requis : {PERFORMANCE_THRESHOLD}")
    print(f"Décision : {promotion_decision['decision']}")

    model = joblib.load(model_path)

    with mlflow.start_run(run_name=f"registry_decision_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("source_report", candidate["source_report"])
        mlflow.log_param("selection_reason", candidate["selection_reason"])
        mlflow.log_param("promotion_decision", promotion_decision["decision"])
        mlflow.log_param("registry_status", promotion_decision["registry_status"])

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.set_tag("project", "FinaScore SA")
        mlflow.set_tag("stage", "week_14")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("promotion_decision", promotion_decision["decision"])

        registry_info = try_register_model(
            model=model,
            model_name=model_name,
            metrics=metrics,
            promotion_decision=promotion_decision,
        )

        log_related_artifacts()

    registry_decision = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "candidate": candidate,
        "promotion_decision": promotion_decision,
        "registry_info": registry_info,
    }

    output_path = REPORTS_DIR / "registry/model_registry_decision.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(registry_decision, file, indent=4, ensure_ascii=False)

    print("\nRegistry")
    print("--------")
    print(f"Modèle enregistré : {registry_info['registered']}")
    print(f"Nom registry : {REGISTERED_MODEL_NAME}")
    print(f"Version : {registry_info['model_version']}")
    print(f"Alias : {registry_info['alias']}")

    if registry_info["error"]:
        print(f"Erreur registry : {registry_info['error']}")

    print("\nFichier généré")
    print("--------------")
    print(f"- {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
