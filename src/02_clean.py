import pandas as pd
import numpy as np


df_finascore = pd.read_csv('data/merge/finascore.csv', low_memory=True)


df_finascore = df_finascore.sort_values("date_demande")

df_finascore.drop_duplicates("applicant_id", keep="last")
df_finascore["age"] = df_finascore["age"].clip(18, 75)
df_finascore['nb_transactions_mois'].isna().astype(int)
df_finascore['jours_depuis_dernier_credit'].isna().astype(int)
df_finascore['nb_credit'].isna().astype(int)
df_finascore['max_retard'].isna().astype(int)
df_finascore['anciennete_compte_mois'].isna().astype(int)
df_finascore['anciennete_emploi'].isna().astype(int)
df_finascore["ratio_endettement"] = df_finascore["ratio_endettement"].clip(lower=0)

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

df_finascore['date_demande'] = df_finascore['date_demande'].apply(parse_date)

 
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
    
df_finascore['secteur_activite'] = df_finascore['secteur_activite'].apply(secteur)


def operateur(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"orange money", "orangemoney"}:
        return "ORANGE"
    if value in {"multi-operateur"}:
        return "Mixte"
    if value in {"mtn momo", "mtmmomo"}:
        return "MTN"       
    return value.upper()

df_finascore['operateur'] = df_finascore['operateur'].apply(operateur)


def zone(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"periurbain"}:
        return "Peri-urbain"
    if value in {"urban"}:
        return "Urbain"
    return value.capitalize()

df_finascore['zone'] = df_finascore['zone'].apply(zone) 


def pays(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
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

df_finascore['pays_x'] = df_finascore['pays_x'].apply(pays)


def statut(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"encours", "actif", "en cours"}:
        return "En_cours"
    if value in {"cloture"}:
        return "Rembourse"
    if value in {'impaye'}:
        return "Defaut"
    
    return value.capitalize()
   
df_finascore['statut_final'] = df_finascore['statut_final'].apply(statut)  

df_finascore.to_csv("data/processed/finascore_clean.csv", index=False)

print("Nettoyage effectué de finascore")