"""
Monitoring de drift des données FinaScore
    - Comparer les données de référence X_train avec les données courantes X_test
    - Générer un rapport Evidently HTML
    - Produire un résumé JSON exploitable pour le monitoring
    - Déclencher une alerte si plus de 15% des features dérivent

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("reports/monitoring")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = SPLITS_DIR / "X_train.parquet"
CURRENT_PATH = SPLITS_DIR / "X_test.parquet"

HTML_REPORT_PATH = REPORTS_DIR / "data_drift_report.html"
JSON_REPORT_PATH = REPORTS_DIR / "data_drift_report.json"
SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"

DRIFT_ALERT_THRESHOLD = 0.15


def load_data():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {REFERENCE_PATH}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CURRENT_PATH}")

    reference_data = pd.read_parquet(REFERENCE_PATH)
    current_data = pd.read_parquet(CURRENT_PATH)

    return reference_data, current_data


def clean_for_drift(reference_data, current_data):
    """
    Harmonise les colonnes entre reference et current.

    Evidently a besoin de colonnes comparables.
    On garde uniquement les colonnes communes.
    """

    common_cols = [col for col in reference_data.columns if col in current_data.columns]

    reference_data = reference_data[common_cols].copy()
    current_data = current_data[common_cols].copy()

    # Conversion simple des booléens pour éviter certains soucis de typage.
    for col in common_cols:
        if reference_data[col].dtype == bool:
            reference_data[col] = reference_data[col].astype(int)
        if current_data[col].dtype == bool:
            current_data[col] = current_data[col].astype(int)

    return reference_data, current_data, common_cols


def calculate_psi_numeric(reference_series, current_series, n_bins=10):
    """
    Calcule un PSI simple pour une variable numérique.

    PSI faible  : distribution stable
    PSI élevé   : distribution différente

    Seuils usuels :
    - PSI < 0.1  : pas de drift significatif
    - 0.1 à 0.2  : drift modéré
    - > 0.2      : drift significatif
    """

    reference = pd.to_numeric(reference_series, errors="coerce").dropna()
    current = pd.to_numeric(current_series, errors="coerce").dropna()

    if len(reference) == 0 or len(current) == 0:
        return None

    try:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(reference.quantile(quantiles).values)

        if len(bins) < 3:
            return None

        reference_counts = pd.cut(reference, bins=bins, include_lowest=True).value_counts(normalize=True)
        current_counts = pd.cut(current, bins=bins, include_lowest=True).value_counts(normalize=True)

        reference_counts, current_counts = reference_counts.align(current_counts, fill_value=0)

        eps = 1e-6
        reference_perc = reference_counts.values + eps
        current_perc = current_counts.values + eps

        psi = np.sum((current_perc - reference_perc) * np.log(current_perc / reference_perc))

        return float(psi)

    except Exception:
        return None


def calculate_categorical_drift(reference_series, current_series):
    """
    Mesure simple de drift catégoriel.

    On calcule la distance totale entre les distributions.
    Valeur proche de 0 : distributions proches.
    Valeur élevée : distributions différentes.
    """

    reference = reference_series.astype(str).fillna("missing")
    current = current_series.astype(str).fillna("missing")

    reference_dist = reference.value_counts(normalize=True)
    current_dist = current.value_counts(normalize=True)

    reference_dist, current_dist = reference_dist.align(current_dist, fill_value=0)

    drift_score = 0.5 * np.abs(reference_dist - current_dist).sum()

    return float(drift_score)


def build_custom_drift_summary(reference_data, current_data):
    """
    Produit un résumé maison pour avoir une alerte simple et contrôlable.

    Règles :
    - numérique : drift si PSI > 0.2
    - catégoriel : drift si distance de distribution > 0.2
    - alerte globale si plus de 15% des colonnes dérivent
    """

    rows = []

    for col in reference_data.columns:
        ref_col = reference_data[col]
        cur_col = current_data[col]

        if pd.api.types.is_numeric_dtype(ref_col):
            score = calculate_psi_numeric(ref_col, cur_col)
            method = "psi"
            drift_detected = bool(score is not None and score > 0.2)

        else:
            score = calculate_categorical_drift(ref_col, cur_col)
            method = "categorical_distribution_distance"
            drift_detected = bool(score > 0.2)

        rows.append(
            {
                "feature": col,
                "method": method,
                "drift_score": score,
                "drift_detected": drift_detected,
            }
        )

    drift_df = pd.DataFrame(rows)

    n_features = int(len(drift_df))
    n_drifted_features = int(drift_df["drift_detected"].sum())
    drift_share = float(n_drifted_features / n_features) if n_features > 0 else 0.0

    alert_triggered = bool(drift_share > DRIFT_ALERT_THRESHOLD)

    summary = {
        "reference_dataset": str(REFERENCE_PATH),
        "current_dataset": str(CURRENT_PATH),
        "n_reference_rows": int(len(reference_data)),
        "n_current_rows": int(len(current_data)),
        "n_features": n_features,
        "n_drifted_features": n_drifted_features,
        "drift_share": drift_share,
        "alert_threshold": DRIFT_ALERT_THRESHOLD,
        "alert_triggered": alert_triggered,
        "decision": "DRIFT_ALERT" if alert_triggered else "NO_DRIFT_ALERT",
        "top_drifted_features": (drift_df.sort_values("drift_score", ascending=False).head(10).to_dict(orient="records")),
    }

    return summary, drift_df


def generate_evidently_report(reference_data, current_data):
    """
    Génère le rapport Evidently.

    Avec l'API récente, Evidently utilise :
    report = Report([DataDriftPreset()])
    report.run(current_data=current, reference_data=reference)
    """

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )

    snapshot.save_html(str(HTML_REPORT_PATH))
    snapshot.save_json(str(JSON_REPORT_PATH))

    return snapshot


def main():
    print("=" * 60)
    print("MONITORING DATA DRIFT - FINASCORE")
    print("=" * 60)

    reference_data, current_data = load_data()

    print(f"Reference data : {reference_data.shape}")
    print(f"Current data   : {current_data.shape}")

    reference_data, current_data, common_cols = clean_for_drift(
        reference_data=reference_data,
        current_data=current_data,
    )

    print(f"Colonnes communes analysées : {len(common_cols)}")

    print("Génération du rapport Evidently...")
    generate_evidently_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    print("Calcul du résumé de drift custom...")
    summary, drift_df = build_custom_drift_summary(
        reference_data=reference_data,
        current_data=current_data,
    )

    drift_df.to_csv(
        REPORTS_DIR / "drift_by_feature.csv",
        index=False,
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\nRésumé drift")
    print("------------")
    for key, value in summary.items():
        if key != "top_drifted_features":
            print(f"{key}: {value}")

    print("\nFichiers générés")
    print("----------------")
    print(f"- {HTML_REPORT_PATH}")
    print(f"- {JSON_REPORT_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {REPORTS_DIR / 'drift_by_feature.csv'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
