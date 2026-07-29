from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer


FEATURES_PATH = Path("data/features/features_dataset.csv")
REPORTS_DIR = Path("reports/diagnostics")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"


IMPORTANT_COLS = [
    "age",
    "pays",
    "secteur_activite",
    "zone",
    "revenu_mensuel_xaf",
    "anciennete_emploi",
    "ratio_endettement",
    "historique_credit",
    "nb_credits_actifs",
    "mobile_money_score",
    "flag_primo_demandeur",
    "flag_no_mobile_money",
    "flag_no_credit_history",
    "nb_credit",
    "total_retards",
    "max_retard",
    "taux_retard_credit",
    "taux_defaut_historique",
    "taux_remboursement_historique",
    "ratio_flux_mm",
    "score_regularite_pondere",
    "indice_surendettement",
    "anciennete_credit_normalisee",
]


def compute_numeric_univariate_auc(df, target):
    """Calcule l'AUC univariée pour chaque variable numérique."""

    y = df[target].astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]

    results = []

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_frame()

        if x[col].nunique(dropna=True) <= 1:
            continue

        x_imputed = SimpleImputer(strategy="median").fit_transform(x)
        score = x_imputed[:, 0]

        try:
            auc = roc_auc_score(y, score)
            auc_corrected = max(auc, 1 - auc)

            results.append({
                "feature": col,
                "auc_raw": auc,
                "auc_corrected": auc_corrected,
                "missing_rate": df[col].isna().mean(),
                "n_unique": df[col].nunique(dropna=True),
            })

        except Exception:
            continue

    return (
        pd.DataFrame(results)
        .sort_values("auc_corrected", ascending=False)
    )


def categorical_target_rate(df, target):
    """Analyse les taux de défaut par modalité catégorielle."""

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    rows = []

    for col in cat_cols:
        if col == target:
            continue

        temp = (
            df.groupby(col, dropna=False)[target]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        temp["feature"] = col
        temp = temp.rename(columns={"mean": "default_rate"})

        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main():
    print("=" * 60)
    print("DIAGNOSTIC DU SIGNAL")
    print("=" * 60)

    df = pd.read_csv(FEATURES_PATH, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(f"Target absente : {TARGET}")

    print("Shape :", df.shape)
    print("\nDistribution target :")
    print(df[TARGET].value_counts(normalize=True).round(4))

    print("\nPrésence des colonnes importantes :")
    for col in IMPORTANT_COLS:
        status = "OK" if col in df.columns else "ABSENT"
        print(f"{col}: {status}")

    existing_important_cols = [col for col in IMPORTANT_COLS if col in df.columns]

    print("\nTaux de valeurs manquantes sur colonnes importantes :")
    missing_report = (
        df[existing_important_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )
    missing_report.to_csv(REPORTS_DIR / "diagnostic_missing_important_cols.csv")

    print("\nAUC univariée des variables numériques, sauvergardé dans reports/diagnostic")
    auc_report = compute_numeric_univariate_auc(df, TARGET)
    auc_report.to_csv(
        REPORTS_DIR / "diagnostic_univariate_auc.csv",
        index=False
    )

    print("\nDiagnostic sauvegardé dans reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()