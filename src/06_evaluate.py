"""
Évaluation finale du modèle champion
    - Charger le meilleur modèle sauvegardé par 05_train.py
    - Évaluer une seule fois sur le jeu de test final
    - Calculer AUC, F1, précision, rappel
    - Générer matrice de confusion, courbe ROC, distribution des scores
    - Sauvegarder un rapport final
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("reports/evaluation")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"


COLS_TO_DROP = ["applicant_id", "date_demande", "nom_partenaire"]


# Load lest splits de test
def load_test_data():
    X_test_path = SPLITS_DIR / "X_test.parquet"
    y_test_path = SPLITS_DIR / "y_test.parquet"

    if not X_test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {X_test_path}")

    if not y_test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {y_test_path}")

    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)[TARGET].astype(int)

    return X_test, y_test


# Exclure les colonnes pour le bruit
def remove_excluded_columns(X):
    cols_to_drop = [col for col in COLS_TO_DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    return metrics, y_pred


def find_best_threshold_for_diagnostic(y_true, y_proba):
    thresholds = np.arange(0.10, 0.90, 0.01)

    scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        scores.append(score)

    best_idx = int(np.argmax(scores))

    return float(thresholds[best_idx]), float(scores[best_idx])


def plot_confusion_matrix(y_true, y_pred, model_name, threshold):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm)

    ax.set_title(f"Matrice de confusion - {model_name} - seuil {threshold:.2f}")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bon payeur", "Défaut"])
    ax.set_yticklabels(["Bon payeur", "Défaut"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eval_confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name):
    """Sauvegarde la courbe ROC."""

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Courbe ROC - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eval_roc_curve.png")
    plt.close()


def plot_score_distribution(y_true, y_proba, model_name):
    """Sauvegarde la distribution des probabilités prédites."""

    df_plot = pd.DataFrame(
        {
            "y_true": y_true,
            "y_proba": y_proba,
        }
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        df_plot.loc[df_plot["y_true"] == 0, "y_proba"],
        bins=30,
        alpha=0.6,
        label="Bon payeur réel",
        density=True,
    )

    plt.hist(
        df_plot.loc[df_plot["y_true"] == 1, "y_proba"],
        bins=30,
        alpha=0.6,
        label="Défaut réel",
        density=True,
    )

    plt.title(f"Distribution des scores prédits - {model_name}")
    plt.xlabel("Probabilité prédite de défaut")
    plt.ylabel("Densité")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eval_score_distribution.png")
    plt.close()


def plot_feature_importance(pipeline, model_name):
    """Sauvegarde les 20 variables les plus importantes si disponible."""

    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            print("Importance des variables non disponible pour ce modèle.")
            return

        feature_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .head(40)
        )

        feature_importance.to_csv(
            REPORTS_DIR / "eval_feature_importance.csv",
            index=False,
        )

        plt.figure(figsize=(9, 7))
        plt.barh(
            feature_importance["feature"][::-1],
            feature_importance["importance"][::-1],
        )
        plt.title(f"Top 30 variables importantes - {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "eval_feature_importance.png")
        plt.close()

    except Exception as error:
        print(f"Impossible de générer les importances : {error}")


def estimate_business_errors(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    business_report = {
        "true_negative_bons_clients_acceptes": tn,
        "false_positive_bons_clients_refuses": fp,
        "false_negative_defauts_non_detectes": fn,
        "true_positive_defauts_detectes": tp,
        "commentaire": ("Les faux négatifs sont les erreurs les plus coûteuses pour le risque crédit : " "le modèle accepte des clients qui feront défaut. " "Les faux positifs représentent des bons clients refusés, donc une perte d'opportunité."),
    }

    return business_report


def main():
    print("=" * 60)
    print("DÉBUT DE L'ÉVALUATION FINALE")
    print("=" * 60)

    best_model_path = REPORTS_DIR / "best_model.json"

    if not best_model_path.exists():
        raise FileNotFoundError("Exécute d'abord 05_train.py pour générer reports/best_model.json.")

    with open(best_model_path, "r", encoding="utf-8") as file:
        best_model_info = json.load(file)

    model_name = best_model_info["model_name"]
    model_path = Path(best_model_info["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print(f"Modèle champion chargé : {model_name}")
    print(f"Chemin modèle : {model_path}")

    pipeline = joblib.load(model_path)

    X_test, y_test = load_test_data()
    X_test = remove_excluded_columns(X_test)

    print(f"Shape X_test : {X_test.shape}")
    print("Distribution y_test :")
    print(y_test.value_counts(normalize=True).round(4))

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"Calcul des probabilités... de {y_proba}")

    # On défini un seuil que l'on pourrais manipuler
    threshold = float(best_model_info.get("best_threshold", 0.50))

    metrics, y_pred = compute_metrics(
        y_true=y_test,
        y_proba=y_proba,
        threshold=threshold,
    )

    diagnostic_threshold, diagnostic_f1 = find_best_threshold_for_diagnostic(
        y_test,
        y_proba,
    )

    business_report = estimate_business_errors(y_test, y_pred)

    classification_txt = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    final_report = {
        "model_name": model_name,
        "model_path": str(model_path),
        "final_test_metrics": metrics,
        "business_errors": business_report,
        "diagnostic_best_threshold_on_test": {
            "threshold": diagnostic_threshold,
            "f1": diagnostic_f1,
            "warning": ("Ce seuil est calculé sur le test final. " "Il sert uniquement au diagnostic et ne doit pas être présenté comme seuil de production."),
        },
    }

    with open(REPORTS_DIR / "final_evaluation_report.json", "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4, ensure_ascii=False)

    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(classification_txt)

    plot_confusion_matrix(y_test, y_pred, model_name, threshold)
    plot_roc_curve(y_test, y_proba, model_name)
    plot_score_distribution(y_test, y_proba, model_name)
    plot_feature_importance(pipeline, model_name)

    PERFORMANCE_THRESHOLD = 0.80

    performance_gate = {
        "metric": "auc_roc",
        "required_threshold": PERFORMANCE_THRESHOLD,
        "observed_value": metrics["auc_roc"],
        "passed": metrics["auc_roc"] >= PERFORMANCE_THRESHOLD,
        "decision": ("PROMOTE_TO_PRODUCTION" if metrics["auc_roc"] >= PERFORMANCE_THRESHOLD else "NOT_PRIMOTE"),
        "comment": ("Le modèle atteint le seuil de performance requis." if metrics["auc_roc"] >= PERFORMANCE_THRESHOLD else ("Le modèle ne respecte pas le seuil AUC >= 0.80. " "Il ne doit pas être promu en production. " "Le pipeline reste utilisable en environnement expérimental/staging.")),
    }

    with open(REPORTS_DIR / "evaluation/performance_gate.json", "w", encoding="utf-8") as f:
        json.dump(performance_gate, f, indent=4, ensure_ascii=False)

    print("\nMétriques finales sur test")
    print("-------------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nErreurs métier")
    print("--------------")
    for key, value in business_report.items():
        print(f"{key}: {value}")

    print("\nPerformance Gate")
    print("----------------")
    print("Seuil requis : 80")
    print(f"AUC observée : {metrics['auc_roc']:.4f}")
    print(f"Décision : {performance_gate['decision']}")

    print("\nSeuil diagnostic calculé sur test")
    print("---------------------------------")
    print(f"threshold: {diagnostic_threshold:.2f}")
    print(f"f1: {diagnostic_f1:.4f}")
    print("Attention : diagnostic seulement, pas seuil de production.")

    print("\nRapports sauvegardés dans reports/evaluation")
    print("=" * 60)


if __name__ == "__main__":
    main()
