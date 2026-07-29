import pandas as pd
import numpy as np
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)


def main():

    df_finascore = pd.read_csv('data/merge/finascore.csv')
    df_finascore["date_demande"] = pd.to_datetime(
            df_finascore["date_demande"],
            errors="coerce"
    )
    
    df_finascore = (
        df_finascore.sort_values('date_demande').drop_duplicates("applicant_id", keep="last")
    )

    # Retenir les ages > 18
    df_finascore = df_finascore[df_finascore["age"] >= 18]
    df_finascore["age"] = df_finascore["age"].clip(upper=75)
    
    # Mettre les erreurs de montant à NAN
    df_finascore.loc[
            df_finascore["revenu_mensuel_xaf"] < 0,
            "revenu_mensuel_xaf"
        ] = np.nan
    
    # Clip du ration endemente
    df_finascore["ratio_endettement"] = df_finascore["ratio_endettement"].clip(lower=0)
    
    
    
    # df_finascore['nb_transactions_mois'] = df_finascore['nb_transactions_mois'].fillna(0).astype(int)
    
    # df_finascore['max_retard'] = df_finascore['max_retard'].fillna(0).astype(int)
    
    # df_finascore['anciennete_compte_mois'] = df_finascore['anciennete_compte_mois'].fillna(0).astype(int)


    df_finascore.to_csv("data/processed/finascore_clean.csv", index=False)

    print("Nettoyage effectué de finascore")

    
    
if __name__ == '__main__':   
    print('='*60)
    print('Debut second Nettoyage')
    print('='*60)
    main()
    print('='*10 + 'Sauvergade effectué' + "="*10)
    
    