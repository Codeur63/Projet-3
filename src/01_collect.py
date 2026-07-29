"""
    - Collecter les données de différentes sources. 
    - Agréger les données transactionnelles et historiques
    - Fusionner les données  
    """

import pandas as pd
import numpy as np
from pathlib import Path

Path("data/merge").mkdir(parents=True, exist_ok=True)

# Nettoyage des dates pour eviter des erreurs
def parse_date(val):
    """Gère ISO, FR (dd/mm/yyyy), tiret, et timestamp Unix."""
    val = str(val).strip()
    if val.isdigit() and len(val) >= 9:
        return pd.to_datetime(int(val), unit='s')
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y/%d/%m" ):
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    return pd.NaT


def statut(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"encours", "actif", "en cours"}:
        return "en_cours"
    if value in {"cloture"}:
        return "rembourse"
    if value in {'impaye'}:
        return "defaut"
    
    return value.lower()
   

def pays(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.lower()
    if value in {"cameroun"}:
        return "CMR"
    if value in {"congo"}:
        return "COG"
    if value in {"guinee eq."}:
        return "GNQ"
    if value in {"tchad"}:
        return "TCD"
    if value in {"gabon"}:
        return "GAB"
    return value.upper()  


def zone(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"periurbain"}:
        return "Peri-urbain"
    if value in {"urban"}:
        return "Urbain"
    return value.capitalize()


def secteur(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"agri"}:
        return "Agriculture"
    if value in {"artisan"}:
        return "Artisanat"
    if value in {"services"}:
        return "Service"
    if value in {"negoce"}:
        return "Commerce"
    return value.capitalize()


def operateur(value:str) -> str:
    if pd.isna(value):
        return np.nan
    
    value = value.lower()
    if value in {'mtn momo'}:
        return 'MTN'
    if value in {'multi-operateur'}:
        return "MIXTE"
    if value in {'orange money'}:
        return "ORANGE"
    return value.upper()



def main():
    # Charger les données
    applicants = pd.read_csv("data/raw/applicants.csv", low_memory=False)
    mobile = pd.read_csv("data/raw/mobile_money_transactions.csv", low_memory=False)
    partners = pd.read_csv("data/raw/partners_metadata.csv", low_memory=False)
    credit = pd.read_csv("data/raw/credit_history.csv", low_memory=False)
    
    credit['date_credit'] = credit['date_credit'].apply(parse_date)
    applicants['date_demande'] = applicants['date_demande'].apply(parse_date)
    
    # Petit Nettoyage
    credit['statut_final'] = credit['statut_final'].apply(statut)  
    applicants['pays'] = applicants['pays'].apply(pays)
    applicants['zone'] = applicants['zone'].apply(zone)    
    applicants['secteur_activite'] = applicants['secteur_activite'].apply(secteur)
    
    mobile['operateur'] = mobile['operateur'].apply(operateur)
        
    applicants = (
        applicants
        .sort_values("date_demande")
        .drop_duplicates(subset=["applicant_id"], keep="last")
    )

    # Enlever ou supprimer les clients qui sont duplicated
    applicants_ref = (
        applicants
        [["applicant_id", "date_demande"]]
    )

    # # Fusionner credit avec date_demande pour calculer l'ancienneté du dernier crédit
    credit = credit.merge(
        applicants_ref[['applicant_id', 'date_demande']],
        on="applicant_id",
        how='left'
    )

    # Trier et utiliser les dernieres date de credit  
    credit = credit.sort_values(['applicant_id', 'date_credit'])

    credit = credit[
        credit["date_credit"].notna()
        & credit["date_demande"].notna()
        & (credit["date_credit"] < credit["date_demande"])
    ]


    credit = credit.merge(
        partners, on='partenaire_id', how='left'
    )

    mobile_agg = (
        mobile
        .groupby("applicant_id")
        .agg(
            nb_transaction_mois = ('nb_transactions_mois', 'mean'),
            volume_entrant = ('volume_entrant_xaf', 'mean'),
            volume_sortant = ('volume_sortant_xaf', 'mean'),
            regularite_score = ('regularite_score', 'mean'),
            operateur = ('operateur', 'last'),
            anciennete_compte_mois = ('anciennete_compte_mois', 'max')        
        )
    ).reset_index()


    # # Aggrégation des crédits 
    credit_agg = (
        credit
        .groupby("applicant_id")
        .agg(
            total_montant_xaf=('montant_xaf', 'sum'),
            avg_credit_xaf=('montant_xaf', 'mean'),
            total_retards=('nb_retards', 'sum'),
            max_retard=('jours_retard_max', 'max'),
            nb_credit=('credit_id', 'count'),
            derniere_date_credit=('date_credit', 'max'),
            date_demande=('date_demande', 'last'),
            nom_partenaire=('nom', 'last'),
            type_partenaire=('type', 'last'),
            pays_partenaire=('pays', 'last'),
            dernier_statut_credit=('statut_final', 'last'),
            seuil_score_partenaire=('seuil_score', 'last'),
            volume_mensuel_partenaire=('volume_mensuel', 'last'),
            nb_credits_defaut_hist=('statut_final', lambda s: (s == "defaut").sum()),
            nb_credits_rembourses_hist=('statut_final', lambda s: (s == "rembourse").sum()),
            nb_credits_restructures_hist=('statut_final', lambda s: (s == "restructure").sum()),
            nb_credits_en_cours_hist=('statut_final', lambda s: (s == "en_cours").sum())
        )
    ) 

    # Protege contre la division 0
    credit_agg['nb_credit_safe'] = credit_agg['nb_credit'].replace(0, np.nan)


    credit_agg["taux_defaut_historique"] = (
        credit_agg["nb_credits_defaut_hist"] / credit_agg["nb_credit_safe"]
    )

    credit_agg["taux_remboursement_historique"] = (
        credit_agg["nb_credits_rembourses_hist"] / credit_agg["nb_credit_safe"]
    )

    credit_agg = credit_agg.drop(columns=['nb_credit_safe'])


    # Calculer les jours depuis le dernier credit et indexer sur applicant_id
    credit_agg['jours_depuis_dernier_credit'] = (credit_agg["date_demande"] - credit_agg['derniere_date_credit']).dt.days

    credit_agg = credit_agg.drop(columns=['derniere_date_credit', 'date_demande']).reset_index()


    # Fusion avec applicant et credit
    df_finascore = pd.merge(
        applicants, credit_agg, on='applicant_id', how='left'
    )


    # # Fusion avec mobile_money a la suite
    df_finascore = pd.merge(
        df_finascore, mobile_agg, on='applicant_id', how='left'
    )
    
    # df_finascore = applicants.copy()

    # Verification des dimensions
    print(f"Collecte terminée - dimension : {df_finascore.shape}")
    print("="*60)

    # Sauvegarde
    df_finascore.to_csv(
        "data/merge/finascore.csv",
        index=False
    )




if __name__=='__main__':    
    print("="*60)
    print("Nettoyage et collecte des données")
    print("="*60)
    main()
    print('='*10 + 'Collect et Nettoyage Sauvergade' + "="*10)
