"""
    Collecter les données de différentes sources. 
    - Collecter les données les concat et nettoyers
    -  Collecter les données les nettoyers et concat 
    """

import pandas as pd

# Charger les données
applicants = pd.read_csv("data/raw/applicants.csv", low_memory=False)
mobile = pd.read_csv("data/raw/mobile_money_transactions.csv", low_memory=False)
partners = pd.read_csv("data/raw/partners_metadata.csv", low_memory=False)
credit = pd.read_csv("data/raw/credit_history.csv", low_memory=False)


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

# Nettoyer les dates
credit['date_credit'] = credit['date_credit'].apply(parse_date)
applicants['date_demande'] = applicants['date_demande'].apply(parse_date)

# Enlever ou supprimer les clients qui sont duplicated
applicants_ref = (
    applicants
    .sort_values("date_demande")
    .drop_duplicates(subset=["applicant_id"], keep="last")
    [["applicant_id", "date_demande"]]
)

# Merde credit et date_demande pour calculer les jours de demande depuis la dernier demande 
credit = credit.merge(
    applicants_ref[['applicant_id', 'date_demande']],
    on="applicant_id",
    how='left'
)

# Ne garder que les credits connus avant le scoring
credit = credit[
    credit["date_credit"].notna()
    & credit["date_demande"].notna()
    & (credit["date_credit"] < credit["date_demande"])
]

# Trier et utiliser les dernieres date de credit  
credit = credit.sort_values(['applicant_id', 'date_credit'])

credit = credit.merge(
    partners, on='partenaire_id', how='left'
)
# Affichage
print(credit.head(5))


# Aggrégation des crédits 
credit_agg = (
    credit
    .groupby("applicant_id")
    .agg({
        "montant_xaf":["sum", "mean"],
        "nb_retards":"sum",
        "jours_retard_max":"max",
        "credit_id":"count",
        "date_credit": "max",
        "statut_final": "last",
        "date_demande": "last", 
        "nom":"last",
        "type":"last",
        "pays":"last",
        "seuil_score":"last",
        "volume_mensuel":"last" 
    })
) 

# Renommer les columns
credit_agg.columns = ["total_montant_xaf","avg_credit_xaf","total_retards", "max_retard","nb_credit", 'derniere_date_credit', "statut_final", "date_demande", "nom", "type", "pays", "seuil_score", "volume_mensuel"]

# Calculer les jours depuis le dernier credit et indexer sur applicant_id
credit_agg['jours_depuis_dernier_credit'] = (credit_agg["date_demande"] - credit_agg['derniere_date_credit']).dt.days
credit_agg = credit_agg.drop(columns=['derniere_date_credit', 'date_demande'])
credit_agg = credit_agg.reset_index()

print(credit_agg.head(5))

# Fusion avec applicant et credit
df_finascore = pd.merge(
    applicants, credit_agg, on='applicant_id', how='left'
)

# Fusion avec mobile_money a la suite
df_finascore = pd.merge(
    df_finascore, mobile, on='applicant_id', how='left'
)

# Fusion avec les partenaires

print(df_finascore.shape)

# Sauvegarde
df_finascore.to_csv(
    "data/merge/finascore.csv",
    index=False
)

# df = (
#     applicants
#     .merge(
#         mobile,
#         on="applicant_id",
#         how="left"
#     )
#     .merge(
#         credit_agg,
#         on="applicant_id",
#         how="left"
#     )
# )

# df.to_csv(
#     "data/merge/merged_dataset.csv",
#     index=False
# )
