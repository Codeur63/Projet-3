import pandas as pd
import numpy as np

df_a = pd.read_csv("data/raw/applicants.csv", low_memory=False, sep=',')
df_c = pd.read_csv("data/raw/credit_history.csv", low_memory=False, sep=',')
df_m = pd.read_csv("data/raw/mobile_money_transactions.csv", low_memory=False, sep=',')
df_p = pd.read_csv("data/raw/partners_metadata.csv", low_memory=False, sep=",")
 

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
df_c['date_credit'].apply(parse_date)
df_1['date_demande'].apply(parse_date)

 
def secteur(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"agri"}:
        return "Agriculture"
    if value in {"artisan"}:
        return "Artisanat"
    if value in {"negoce"}:
        return "Commerce"
    if value in {"services"}:
        return "Service"
    return value.strip().lower().capitalize()
    
df_a['secteur_activite'] = df_a['secteur_activite'].apply(secteur)

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

df_m['operateur'] = df_m['operateur'].apply(operateur)


def zone(value:str) -> str:
    if pd.isna(value):
        return np.nan
    value = value.strip().lower()
    if value in {"periurbain"}:
        return "Peri-urbain"
    if value in {"urban"}:
        return "Urbain"
    return value.capitalize()

df_a['zone'] = df_a['zone'].apply(zone) 

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

df_a['pays'] = df_a['pays'].apply(pays)


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
    
    return value.capitalize()e()
   
df_a['statut_final'] = df_c['statut_final'].apply(statut)  