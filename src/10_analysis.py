"""
Analyse métier des erreurs de prédiction
    - Identifier les faux positifs et faux négatifs
    - Analyser les erreurs par segment métier : pays, secteur, zone
    - Estimer un coût financier simplifié des erreurs
    - Produire un rapport métier exploitable pour la décision
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/analysis")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"

THRESHOLD = 0.5

LGD_RATE = 0.60
FP_OPPORTUNITY_RATE = 0.08

DEFAULT_AMOUNT_XAF = 1_000_000
DEFAULT_FN_COST_XAF = 600_000
DEFAULT_FP_COST_XAF = 100_000

PERFORMANCE_THRESHOLD = 0.80

COLS_TO_DROP = [
    "applicant_id",
    "date_demande",
    "nom_partenaire",
]


# Utilisation du modele avec Optuna ou meilleur modèle
def load_model():
    optuna_model_path = MODELS_DIR / "optuna/xgboost_optuna.pkl"

    if optuna_model_path.exists():
        print(f"Chargement du modèle Optuna : {optuna_model_path}")
        return joblib.load(optuna_model_path), "xgboost_optuna", optuna_model_path

    best_model_path = REPORTS_DIR / "best_model.json"

    if not best_model_path.exists():
        raise FileNotFoundError("Aucun modèle disponible. Exécuté les scripts antérieurs")

    with open(best_model_path, "r", encoding="utf-8") as file:
        best_model_info = json.load(file)

    model_path = Path(best_model_info["model_path"])
    model_name = best_model_info["model_name"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print(f"Chargement du meilleur modèle : {model_path}")

    return joblib.load(model_path), model_name, model_path


# Travaille sur les données de test
def load_data():
    X_test_path = SPLITS_DIR / "X_test.parquet"
    y_test_path = SPLITS_DIR / "y_test.parquet"

    if not X_test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {X_test_path}")

    if not y_test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {y_test_path}")

    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)[TARGET].astype(int)

    return X_test, y_test


# Colonne exclu lors de l'entrainement
def remove_excluded_columns(X):
    cols_to_drop = [col for col in COLS_TO_DROP if col in X.columns]
    return X.drop(columns=cols_to_drop)


# Metrique du modèle
def model_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def assign_error_type(y_true, y_pred):
    """Attribue un type d'erreur métier."""

    y_true = int(y_true)
    y_pred = int(y_pred)

    if y_true == 0 and y_pred == 0:
        return "TN_bon_client_accepte"

    if y_true == 0 and y_pred == 1:
        return "FP_bon_client_refuse"

    if y_true == 1 and y_pred == 0:
        return "FN_defaut_non_detecte"

    if y_true == 1 and y_pred == 1:
        return "TP_defaut_detecte"

    return "unknown"


def find_amount_column(df):
    candidates = [
        "avg_credit_xaf",
        "montant_moyen_credit_xaf",
        "montant_moyen_credit",
        "total_montant_xaf",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_amount_series(df):
    amount_col = find_amount_column(df)

    if amount_col:
        amount = pd.to_numeric(df[amount_col], errors="coerce")
        median_amount = amount.median()

        if pd.isna(median_amount) or median_amount <= 0:
            median_amount = DEFAULT_AMOUNT_XAF

        amount = amount.fillna(median_amount).clip(lower=0)
    else:
        amount = pd.Series(DEFAULT_AMOUNT_XAF, index=df.index)

    return amount, amount_col


def estimate_error_costs(df):
    """
    Estime le coût métier des erreurs.

    Faux négatif :
    - client réellement en défaut mais prédit bon payeur
    - coût = perte potentielle sur crédit accordé

    Faux positif :
    - bon client refusé
    - coût = opportunité commerciale perdue
    """

    amount, amount_col = get_amount_series(df)

    df["estimated_amount_xaf"] = amount
    df["estimated_cost_xaf"] = 0.0

    fn_mask = df["error_type"] == "FN_defaut_non_detecte"
    fp_mask = df["error_type"] == "FP_bon_client_refuse"

    df.loc[fn_mask, "estimated_cost_xaf"] = df.loc[fn_mask, "estimated_amount_xaf"] * LGD_RATE

    df.loc[fp_mask, "estimated_cost_xaf"] = df.loc[fp_mask, "estimated_amount_xaf"] * FP_OPPORTUNITY_RATE

    df.loc[fn_mask & df["estimated_cost_xaf"].isna(), "estimated_cost_xaf"] = DEFAULT_FN_COST_XAF

    df.loc[fp_mask & df["estimated_cost_xaf"].isna(), "estimated_cost_xaf"] = DEFAULT_FP_COST_XAF

    return df, amount_col


# Matrix de confusion
def summarize_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    total = tn + fp + fn + tp

    return {
        "true_negative_bons_clients_acceptes": tn,
        "false_positive_bons_clients_refuses": fp,
        "false_negative_defauts_non_detectes": fn,
        "true_positive_defauts_detectes": tp,
        "total_errors": fp + fn,
        "total_observations": total,
        "error_rate": float((fp + fn) / total) if total > 0 else None,
    }


def subgroup_error_analysis(df, subgroup_cols):
    """Analyse les erreurs par sous-groupe métier."""

    rows = []

    for col in subgroup_cols:
        if col not in df.columns:
            continue

        for value, group in df.groupby(col, dropna=False):
            if len(group) < 50:
                continue

            counts = group["error_type"].value_counts()

            fp = int(counts.get("FP_bon_client_refuse", 0))
            fn = int(counts.get("FN_defaut_non_detecte", 0))
            tp = int(counts.get("TP_defaut_detecte", 0))
            tn = int(counts.get("TN_bon_client_accepte", 0))

            total = len(group)

            rows.append(
                {
                    "subgroup_column": col,
                    "subgroup_value": str(value),
                    "n_samples": int(total),
                    "default_rate": float(group[TARGET].mean()),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "fp_rate": float(fp / total),
                    "fn_rate": float(fn / total),
                    "error_rate": float((fp + fn) / total),
                    "estimated_total_cost_xaf": float(group["estimated_cost_xaf"].sum()),
                    "estimated_avg_cost_xaf": float(group["estimated_cost_xaf"].mean()),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "subgroup_column",
                "subgroup_value",
                "n_samples",
                "default_rate",
                "tn",
                "fp",
                "fn",
                "tp",
                "fp_rate",
                "fn_rate",
                "error_rate",
                "estimated_total_cost_xaf",
                "estimated_avg_cost_xaf",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "estimated_total_cost_xaf",
        ascending=False,
    )


def threshold_cost_simulation(df):
    """
    Simule le coût métier pour plusieurs seuils.
    Diagnostic uniquement : ne pas sélectionner le seuil final sur le test.
    """

    rows = []
    thresholds = np.arange(0.10, 0.91, 0.05)

    amount, _ = get_amount_series(df)

    for threshold in thresholds:
        temp = df.copy()

        temp["y_pred_threshold"] = (temp["y_proba"] >= threshold).astype(int)

        temp["error_type_threshold"] = [assign_error_type(y_t, y_p) for y_t, y_p in zip(temp[TARGET], temp["y_pred_threshold"])]

        temp["estimated_cost_xaf"] = 0.0

        fn_mask = temp["error_type_threshold"] == "FN_defaut_non_detecte"
        fp_mask = temp["error_type_threshold"] == "FP_bon_client_refuse"

        temp.loc[fn_mask, "estimated_cost_xaf"] = amount.loc[fn_mask] * LGD_RATE
        temp.loc[fp_mask, "estimated_cost_xaf"] = amount.loc[fp_mask] * FP_OPPORTUNITY_RATE

        metrics = model_metrics(
            y_true=temp[TARGET],
            y_proba=temp["y_proba"],
            threshold=threshold,
        )

        counts = temp["error_type_threshold"].value_counts()

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "auc_roc": metrics["auc_roc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fp": int(counts.get("FP_bon_client_refuse", 0)),
                "fn": int(counts.get("FN_defaut_non_detecte", 0)),
                "tp": int(counts.get("TP_defaut_detecte", 0)),
                "tn": int(counts.get("TN_bon_client_accepte", 0)),
                "estimated_total_cost_xaf": float(temp["estimated_cost_xaf"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("estimated_total_cost_xaf")


def write_markdown_report(
    model_name,
    model_path,
    metrics,
    confusion_summary,
    cost_summary,
    amount_col,
    subgroup_report,
    cost_simulation,
):
    """Génère un rapport markdown métier."""

    if cost_simulation.empty:
        best_threshold = THRESHOLD
        best_cost = cost_summary["total_cost_xaf"]
    else:
        best_cost_row = cost_simulation.iloc[0]
        best_threshold = best_cost_row["threshold"]
        best_cost = best_cost_row["estimated_total_cost_xaf"]

    content = f"""# Analyse métier des erreurs — FinaScore SA

## 1. Objectif

Ce rapport analyse les erreurs du modèle de scoring crédit afin de comprendre leur impact métier.

Le modèle analysé est :

```text
{model_name}
```

Chemin du modèle :

```text
{model_path}
```

## 2. Métriques globales

Seuil utilisé pour l'analyse principale :

```text
{THRESHOLD}
```

Résultats :

| Métrique | Valeur |
|---|---:|
| AUC-ROC | {metrics["auc_roc"]:.4f} |
| F1-score | {metrics["f1"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| Accuracy | {metrics["accuracy"]:.4f} |

## 3. Matrice de confusion métier

| Type | Nombre | Interprétation |
|---|---:|---|
| True Negative | {confusion_summary["true_negative_bons_clients_acceptes"]} | Bons clients acceptés |
| False Positive | {confusion_summary["false_positive_bons_clients_refuses"]} | Bons clients refusés |
| False Negative | {confusion_summary["false_negative_defauts_non_detectes"]} | Défauts non détectés |
| True Positive | {confusion_summary["true_positive_defauts_detectes"]} | Défauts détectés |

Taux d'erreur global :

```text
{confusion_summary["error_rate"]:.4f}
```

## 4. Lecture métier

Les **faux négatifs** sont les erreurs les plus critiques pour le risque crédit.  
Ils correspondent aux demandeurs qui feront défaut, mais que le modèle classe comme bons payeurs.

Les **faux positifs** correspondent aux bons clients refusés.  
Ils représentent une perte d'opportunité commerciale et peuvent réduire le volume d'affaires.

## 5. Hypothèses de coût

Les coûts sont estimés avec les hypothèses suivantes :

| Élément | Hypothèse |
|---|---:|
| Loss Given Default | {LGD_RATE:.0%} |
| Coût d'opportunité faux positif | {FP_OPPORTUNITY_RATE:.0%} |
| Colonne de montant utilisée | {amount_col if amount_col else "Montant forfaitaire par défaut"} |

Ces montants sont des hypothèses de travail. Ils doivent être remplacés par les coûts réels de FinaScore si disponibles.

## 6. Coût estimé des erreurs

| Indicateur | Valeur |
|---|---:|
| Coût total estimé | {cost_summary["total_cost_xaf"]:,.0f} XAF |
| Coût moyen par observation | {cost_summary["avg_cost_xaf"]:,.0f} XAF |
| Coût des faux négatifs | {cost_summary["fn_total_cost_xaf"]:,.0f} XAF |
| Coût des faux positifs | {cost_summary["fp_total_cost_xaf"]:,.0f} XAF |

## 7. Simulation des seuils

Le seuil avec le coût estimé le plus faible dans la simulation est :

```text
{best_threshold:.2f}
```

Coût associé :

```text
{best_cost:,.0f} XAF
```

Attention : cette simulation est réalisée sur le test final. Elle sert au diagnostic métier, pas à définir directement un seuil de production.

## 8. Segments les plus coûteux

Les segments les plus coûteux sont sauvegardés dans :

```text
reports/error_analysis_by_subgroup.csv
```

Top 10 segments par coût estimé :

| Segment | Valeur | Nombre | Taux défaut | Coût estimé |
|---|---|---:|---:|---:|
"""

    if subgroup_report.empty:
        content += "| Aucun segment exploitable | - | 0 | - | - |\n"
    else:
        top_segments = subgroup_report.head(10)

        for _, row in top_segments.iterrows():
            content += f"| {row['subgroup_column']} | {row['subgroup_value']} | " f"{int(row['n_samples'])} | {row['default_rate']:.2%} | " f"{row['estimated_total_cost_xaf']:,.0f} XAF |\n"

    promotion_decision = "PROMOTE_TO_PRODUCTION" if metrics["auc_roc"] >= PERFORMANCE_THRESHOLD else "DO_NOT_PROMOTE"

    content += f"""

## 9. Décision de promotion

Seuil requis pour une promotion :

```text
AUC-ROC >= {PERFORMANCE_THRESHOLD}
```

AUC observée :

```text
{metrics["auc_roc"]:.4f}
```

Décision :

```text
{promotion_decision}
```

Le modèle peut rester en expérimentation ou en staging, mais ne doit pas être utilisé comme modèle de décision automatique en production tant que le seuil n'est pas atteint.

## 10. Recommandations

Avant toute mise en production, il est recommandé de :

- enrichir les données historiques de remboursement ;
- ajouter davantage de variables comportementales mobile money ;
- améliorer l'alignement temporel entre demande et historique crédit ;
- travailler avec les métiers pour obtenir de vrais coûts de faux positifs et faux négatifs ;
- définir un seuil métier sur validation, pas sur le test final ;
- conserver un humain dans la boucle tant que la performance reste insuffisante.
"""

    report_path = REPORTS_DIR / "error_analysis_business.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(content)

    return report_path


def main():
    print("=" * 60)
    print("ANALYSE MÉTIER DES ERREURS")
    print("=" * 60)

    model, model_name, model_path = load_model()

    X_test_raw, y_test = load_data()
    X_test = remove_excluded_columns(X_test_raw)

    print("Calcul des probabilités...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = model_metrics(y_test, y_proba, THRESHOLD)
    confusion_summary = summarize_confusion(y_test, y_pred)

    error_df = X_test_raw.copy()
    error_df[TARGET] = y_test.values
    error_df["y_proba"] = y_proba
    error_df["y_pred"] = y_pred

    error_df["error_type"] = [assign_error_type(y_t, y_p) for y_t, y_p in zip(error_df[TARGET], error_df["y_pred"])]

    error_df, amount_col = estimate_error_costs(error_df)

    error_cases_path = REPORTS_DIR / "error_cases.csv"
    error_df.to_csv(error_cases_path, index=False)

    subgroup_report = subgroup_error_analysis(
        df=error_df,
        subgroup_cols=["pays", "secteur_activite", "zone"],
    )

    subgroup_path = REPORTS_DIR / "error_analysis_by_subgroup.csv"
    subgroup_report.to_csv(subgroup_path, index=False)

    cost_simulation = threshold_cost_simulation(error_df)

    cost_simulation_path = REPORTS_DIR / "business_threshold_cost_simulation.csv"
    cost_simulation.to_csv(cost_simulation_path, index=False)

    fn_total_cost = error_df.loc[
        error_df["error_type"] == "FN_defaut_non_detecte",
        "estimated_cost_xaf",
    ].sum()

    fp_total_cost = error_df.loc[
        error_df["error_type"] == "FP_bon_client_refuse",
        "estimated_cost_xaf",
    ].sum()

    cost_summary = {
        "total_cost_xaf": float(error_df["estimated_cost_xaf"].sum()),
        "avg_cost_xaf": float(error_df["estimated_cost_xaf"].mean()),
        "fn_total_cost_xaf": float(fn_total_cost),
        "fp_total_cost_xaf": float(fp_total_cost),
    }

    promotion_decision = "PROMOTE_TO_PRODUCTION" if metrics["auc_roc"] >= PERFORMANCE_THRESHOLD else "DO_NOT_PROMOTE"

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "threshold": THRESHOLD,
        "metrics": metrics,
        "confusion_summary": confusion_summary,
        "cost_assumptions": {
            "lgd_rate": LGD_RATE,
            "fp_opportunity_rate": FP_OPPORTUNITY_RATE,
            "amount_column_used": amount_col,
            "default_amount_xaf": DEFAULT_AMOUNT_XAF,
            "default_fn_cost_xaf": DEFAULT_FN_COST_XAF,
            "default_fp_cost_xaf": DEFAULT_FP_COST_XAF,
        },
        "cost_summary": cost_summary,
        "performance_gate": {
            "metric": "auc_roc",
            "required_threshold": PERFORMANCE_THRESHOLD,
            "observed_value": metrics["auc_roc"],
            "passed": metrics["auc_roc"] >= PERFORMANCE_THRESHOLD,
            "decision": promotion_decision,
        },
    }

    summary_path = REPORTS_DIR / "error_analysis_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    report_path = write_markdown_report(
        model_name=model_name,
        model_path=model_path,
        metrics=metrics,
        confusion_summary=confusion_summary,
        cost_summary=cost_summary,
        amount_col=amount_col,
        subgroup_report=subgroup_report,
        cost_simulation=cost_simulation,
    )

    print("\nMétriques")
    print("---------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMatrice de confusion métier")
    print("--------------------------")
    for key, value in confusion_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nCoût estimé")
    print("-----------")
    for key, value in cost_summary.items():
        print(f"{key}: {value:,.0f} XAF")

    print("\nPerformance Gate")
    print("----------------")
    print(f"Seuil requis : {PERFORMANCE_THRESHOLD}")
    print(f"AUC observée : {metrics['auc_roc']:.4f}")
    print(f"Décision : {promotion_decision}")

    print("\nFichiers générés")
    print("----------------")
    print(f"- {error_cases_path}")
    print(f"- {subgroup_path}")
    print(f"- {cost_simulation_path}")
    print(f"- {summary_path}")
    print(f"- {report_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
