"""
Travaille de features engineering a éffectuer :
- Création des variables métier utiles (Ratios, Flags, Interactions)
- Sécurisation des divisions par zéro
- Faire la Feature Selection crée des variables utiles
- la standardisation, l'imputation et l'encodage doivent pouvoir se faire par le pipeline

"""

from pathlib import Path

import numpy as np
import pandas as pd

Path("data/features").mkdir(parents=True, exist_ok=True)


def divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = numerator / denominator.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def main():
    print("=" * 60)
    print("Feature engineering")
    print("=" * 60)

    df_finascore = pd.read_csv("data/processed/finascore_clean.csv", low_memory=True)
    print(f"Shape initiale : {df_finascore.shape}")
    print(f"Duplications exactes : {df_finascore.duplicated().sum()}")

    # Présence ou absence d'historique
    df_finascore["flag_primo_demandeur"] = df_finascore["historique_credit"].isna().astype(int)
    # Pas d'historique credit

    df_finascore["flag_no_credit_history"] = (df_finascore["nb_credits_actifs"].isna() | (df_finascore["nb_credits_actifs"] == 0)).astype(int)

    # Manque d'information
    df_finascore["flag_anciennete_emploi_missing"] = df_finascore["anciennete_emploi"].isna().astype(int)
    # Flag de surendettement
    # df_finascore["flag_surendette"] = (
    #     df_finascore["ratio_endettement"] > 1
    # ).astype(int)

    # # Voir si le client depense plus qu'il ne gagne (pour la capacité réel à rembourser et normaliser la comparaison des montants)
    # df_finascore['solde_flux_mm'] = df_finascore['volume_entrant'] - df_finascore['volume_sortant']
    # df_finascore['ratio_flux_mm'] = divide(df_finascore["volume_entrant"] , df_finascore["volume_sortant"])

    # # Intensité de ses depenses et revenue
    # df_finascore["volume_mm_total"] = df_finascore["volume_entrant"] + df_finascore["volume_sortant"]
    # df_finascore["ratio_volume_mm_revenu"] =divide(df_finascore["volume_mm_total"] , df_finascore["revenu_mensuel_xaf"])

    # # Fiabilité du score sur le temps, la regularité de son argent sur le temps
    df_finascore["score_regularite_pondere"] = df_finascore["regularite_score"] * np.log1p(df_finascore["anciennete_compte_mois"])

    # # Ratio flux sortant / revenu : capacité de dépense relative
    df_finascore["flux_sortant_sur_revenu"] = divide(df_finascore["volume_sortant"], df_finascore["revenu_mensuel_xaf"])

    # # Savoir son taux de retard pour les crédits (sécuriser division par zéro)
    # df_finascore["taux_retard_credit"] = divide(df_finascore["total_retards"] , df_finascore["nb_credit"])

    # # Permet de savoir si le client est risqué
    # df_finascore["montant_moyen_credit_sur_revenu"] = divide(df_finascore["avg_credit_xaf"],df_finascore["revenu_mensuel_xaf"])
    # df_finascore["montant_total_credit_sur_revenu"] = divide(df_finascore["total_montant_xaf"] , df_finascore["revenu_mensuel_xaf"])

    # # Plus le dernier crédit est ancien, moins l'information sur le client est récente.
    df_finascore["anciennete_credit_normalisee"] = 1 / (1 + df_finascore["jours_depuis_dernier_credit"])

    # # Si aucun crédit précédent, cette variable doit rester NaN.
    # df_finascore.loc[
    #         df_finascore["jours_depuis_dernier_credit"].isna(),
    #         "anciennete_credit_normalisee"
    #     ] = np.nan

    # # Indice de surendement
    df_finascore["indice_surendettement"] = np.where(df_finascore["ratio_endettement"] > 1, df_finascore["ratio_endettement"] - 1, 0)

    # # réduit l'effet des très hauts revenus
    # df_finascore["log_revenu_mensuel_xaf"] = np.where(
    #     df_finascore["revenu_mensuel_xaf"] > 0,
    #     np.log1p(df_finascore["revenu_mensuel_xaf"]),
    #     np.nan
    # )

    df_finascore["revenu_par_credit_actif"] = divide(df_finascore["revenu_mensuel_xaf"], df_finascore["nb_credits_actifs"] + 1)

    # # Score mobile / revenu : intensité du mobile money relative au revenu
    # # df_finascore["mobile_score_sur_revenu"] = divide(df_finascore["mobile_money_score"] , df_finascore["revenu_mensuel_xaf"])

    # # # Ratio endettement × ancienneté emploi : client endetté depuis longtemps = plus risqué
    # # df_finascore["ratio_endettement_x_anciennete"] = df_finascore["ratio_endettement"] * df_finascore["anciennete_emploi"]

    LEAKAGE_COLS = [
        "taux_defaut_historique",
        "taux_remboursement_historique",
    ]
    existing_leakage_cols = [col for col in LEAKAGE_COLS if col in df_finascore.columns]
    if existing_leakage_cols:
        df_finascore = df_finascore.drop(columns=existing_leakage_cols)
        print(f"Colonnes leakage supprimées : {existing_leakage_cols}")

    numeric_cols = df_finascore.select_dtypes(include=["number"]).columns
    inf_count = np.isinf(df_finascore[numeric_cols]).sum().sum()
    print(f"Infinity count : {inf_count}")
    if inf_count > 0:
        raise ValueError(f"Des valeurs infinies existent encore : {inf_count}")

    print(f"Shape finale (Après purge) : {df_finascore.shape}")
    print(f"Colonnes conservées : {len(df_finascore.columns)}")
    df_finascore.to_csv("data/features/features_dataset.csv", index=False)

    print("=" * 60)
    print("Sauvegarde effectuée ")
    print("=" * 60)


if __name__ == "__main__":
    main()
