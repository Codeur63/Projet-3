"""
Application Streamlit FinaScore SA
    - Communique avec l'API FastAPI (endpoints /predict, /predict/batch, /model/info)
    - Jeu de prédiction individuelle avec formulaire dynamique
    - Prédiction par lot depuis un fichier CSV
"""

import sys
from pathlib import Path

# Correction d'ombre : le dossier local `streamlit/` masque le paquet Streamlit.
# On retire du sys.path le dossier du script et la racine du projet (résolue via CWD)
# afin que `import streamlit` retrouve le paquet installé.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _keep_path(p):
    try:
        if not p or Path(p).name == "streamlit":
            return False
        return Path(p).resolve() != _PROJECT_ROOT
    except OSError:
        return True


sys.path = [p for p in sys.path if _keep_path(p)]

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="FinaScore Playground", page_icon=":credit_card:", layout="wide")

TRAIN_PARQUET = _PROJECT_ROOT / "data/splits/X_train.parquet"

# Colonnes dérivées internes du modèle : non saisissables par l'utilisateur
DERIVED_COLS = {
    "anomaly_score",
    "is_anomaly",
    "taux_defaut_historique",
    "taux_remboursement_historique",
}

# Options par défaut pour les colonnes catégorielles (si données d'entraînement indisponibles)
CATEGORICAL_FALLBACK = {
    "pays": ["CMR", "GAB", "COG", "CAF", "TCD", "GNQ"],
    "secteur_activite": ["commerce", "agriculture", "artisanat", "services", "elevage"],
    "zone": ["urbain", "periurbain", "rural"],
    "operateur": ["MTN", "ORANGE", "CAMTEL", "mixte"],
    "type_partenaire": ["microfinance", "banque", "mobile_money", "cooperative"],
    "pays_partenaire": ["CMR", "GAB", "COG", "CAF", "TCD", "GNQ"],
    "dernier_statut_credit": ["rembourse", "defaut", "en_cours", "restructure"],
}


def api_url():
    return st.sidebar.text_input("URL de l'API", value="http://localhost:8000").rstrip("/")


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info(base_url):
    try:
        resp = requests.get(f"{base_url}/model/info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"API injoignable ({base_url}) : {e}"}


@st.cache_data(show_spinner=False)
def load_train_stats():
    """Statistiques (médiane/min/max, modalités) issues du jeu d'entraînement."""
    if not TRAIN_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(TRAIN_PARQUET)
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    "default": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "type": "numeric",
                }
            else:
                stats[col] = {
                    "options": [str(v) for v in df[col].dropna().unique()],
                    "default": str(df[col].dropna().mode().iloc[0]) if df[col].notna().any() else "",
                    "type": "categorical",
                }
        return stats
    except Exception:
        return None


def render_feature_inputs(expected_columns, stats):
    features = {}
    usable = [col for col in expected_columns if col not in DERIVED_COLS]

    for idx in range(0, len(usable), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            if idx + col_idx >= len(usable):
                break
            col = usable[idx + col_idx]
            with cols[col_idx]:
                col_stats = (stats or {}).get(col)

                if col_stats and col_stats["type"] == "categorical":
                    options = col_stats["options"] or CATEGORICAL_FALLBACK.get(col, ["Oui", "Non"])
                    features[col] = st.selectbox(col, options, index=0)
                elif col in CATEGORICAL_FALLBACK:
                    features[col] = st.selectbox(col, CATEGORICAL_FALLBACK[col], index=0)
                else:
                    default = col_stats["default"] if col_stats else 0.0
                    min_val = col_stats["min"] if col_stats else None
                    max_val = col_stats["max"] if col_stats else None
                    step = 1.0 if (col_stats and default == int(default)) else 0.01
                    features[col] = st.number_input(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default,
                        step=step,
                    )

    return features


def call_predict(base_url, features):
    try:
        resp = requests.post(f"{base_url}/predict", json={"features": features}, timeout=15)
        if resp.status_code == 400:
            st.error(f"Requête invalide : {resp.json().get('detail', resp.text)}")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de communication avec l'API : {e}")
        return None


def render_prediction(result):
    proba = result["probability_default"]
    prediction = result["prediction"]
    label = result["decision_label"]

    color = "red" if prediction == 1 else "green"
    st.markdown(f"## Décision : <span style='color:{color}'>{label.upper()}</span>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilité de défaut", f"{proba:.2%}")
    m2.metric("Prédiction", "Défaut (1)" if prediction == 1 else "Bon payeur (0)")
    m3.metric("Seuil", f"{result.get('threshold', 0.5):.2f}")

    st.progress(min(proba, 1.0), text="Risque de défaut")

    if result.get("cached"):
        st.caption("Résultat servi depuis le cache Redis.")
    if result.get("warning"):
        st.warning(result["warning"])


def main():
    st.title("FinaScore SA — Scoring crédit")
    st.subheader("Playground de prédiction (démo, modèle staging)")

    base_url = api_url()
    info = get_model_info(base_url)

    with st.sidebar:
        st.header("Modèle servi")
        if "error" in info:
            st.error(info["error"])
        else:
            st.write(f"**Modèle :** {info.get('model_name')}")
            st.write(f"**AUC-ROC :** {info.get('model_metric', 'N/A')}")
            st.write(f"**Décision :** {info.get('promotion_decision') or 'N/A'}")
            st.write(f"**Alias registre :** {info.get('registry_alias') or 'N/A'}")
            if info.get("production_warning") == "NOT_PROMOTE":
                st.warning("Modèle non promu en production")

    if "error" in info:
        st.stop()

    expected_columns = info["expected_columns"]
    stats = load_train_stats()

    tab_individual, tab_batch = st.tabs(["Prédiction individuelle", "Prédiction par lot"])

    with tab_individual:
        st.markdown(f"Formulaire basé sur les **{len(expected_columns)}** variables attendues par le modèle.")
        with st.form("prediction_form"):
            features = render_feature_inputs(expected_columns, stats)
            submitted = st.form_submit_button("Prédire le risque", type="primary")

        if submitted:
            with st.spinner("Appel de l'API en cours..."):
                result = call_predict(base_url, features)
            if result:
                st.divider()
                render_prediction(result)

    with tab_batch:
        st.markdown("Chargez un CSV dont les colonnes correspondent aux variables du modèle.")
        uploaded = st.file_uploader("Fichier CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV illisible : {e}")
                st.stop()

            missing = [col for col in expected_columns if col not in DERIVED_COLS and col not in batch_df.columns]
            if missing:
                st.warning(f"Colonnes absentes (remplies comme manquantes) : {', '.join(missing[:8])}")

            if st.button("Prédire le lot", type="primary"):
                records = batch_df.to_dict(orient="records")
                try:
                    resp = requests.post(f"{base_url}/predict/batch", json={"records": records}, timeout=30)
                    resp.raise_for_status()
                    results = resp.json()["predictions"]
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur batch : {e}")
                    st.stop()

                result_df = pd.DataFrame(results)
                st.success(f"{len(result_df)} prédictions reçues.")
                st.dataframe(result_df)
                st.download_button(
                    "Télécharger les prédictions (CSV)",
                    result_df.to_csv(index=False),
                    file_name="predicted_batch.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
