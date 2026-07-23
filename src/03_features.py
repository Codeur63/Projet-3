"""
    Travaille de features engineering a éffectuer : 
        - Normalisation des valeurs numeriques et categorielle
        - Encodage
        - Standardisation si possible
        - Creer des variables (flag_primo) pour savoir sur c'est un primo demandeur ou non  
        - Suppression ou imputation a la medain des valeurs manqantes\
        - feature selection (selection des variables)    

"""
import pandas as pd
import numpy as np


df_finascore = pd.read_csv("data/processed/finascore_clean.csv", low_memory=True)

print(df_finascore.duplicated().sum())

# Présence ou absence d'historique 
df_finascore['flag_primo_demandeur'] = df_finascore['historique_credit'].isna()

# Voir si le client ne depense plus qu'il ne gagne (pour la capacité réel à rembourser et normaliser la comparaison des montants)
df_finascore['solde_flux_mm'] = df_finascore['volume_entrant_xaf'] - df_finascore['volume_sortant_xaf']
df_finascore['ratio_flux_mm'] =  df_finascore["volume_entrant_xaf"]/df_finascore["volume_sortant_xaf"]

# Intensité de ses depenses et revenue 
df_finascore["volume_mm_total"] = df_finascore["volume_entrant_xaf"] + df_finascore["volume_sortant_xaf"] 
df_finascore["ratio_volume_mm_revenu"] =  df_finascore["volume_mm_total"]/ df_finascore["revenu_mensuel_xaf"]

# Fiabilité du score sur le temps, la regularité de son argent sur le temps
df_finascore["score_regularite_pondere"] = df_finascore["regularite_score"] * np.log1p(df_finascore["anciennete_compte_mois"])

# Savoir son taux de retard pour les crédits
df_finascore["taux_retard_credit"] = df_finascore["total_retards"]/df_finascore["nb_credit"]

# Permet de savoir si le client est risqué 
df_finascore["montant_moyen_credit_sur_revenu"] = df_finascore["avg_credit_xaf"]/df_finascore["revenu_mensuel_xaf"]
df_finascore["montant_total_credit_sur_revenu"] = df_finascore["total_montant_xaf"]/df_finascore["revenu_mensuel_xaf"]

# réduit l'effet des très hauts revenus
df["log_revenu_mensuel_xaf"] = np.log1p(df["revenu_mensuel_xaf"])

# Plus le dernier crédit est ancien, moins l'information sur le client est récente.
df_finascore["anciennete_credit_normalisee"] = 1 / (
        1 + df_finascore["jours_depuis_dernier_credit"]
    )

# Si aucun crédit précédent, cette variable doit rester NaN.
df_finascore.loc[
        df_finascore["jours_depuis_dernier_credit"].isna(),
        "anciennete_credit_normalisee"
    ] = np.nan

# Feature qui permet de mettre directement un default
df_finascore["flag_surendette"] = (
        df_finascore["ratio_endettement"] > 1
    ).astype(int)

df_finascore.to_csv("data/features/features_dataset.csv",index=False)