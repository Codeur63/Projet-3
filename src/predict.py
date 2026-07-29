import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/xgboost.pkl")


client = pd.DataFrame(
    [
        {
            "age": 34,
            "log_revenu_mensuel_xaf": np.log1p(200000),
            "ratio_endettement": 0.3,
            "operateur": "MTN",
            "zone": "urbain",
            "secteur_activite": "Commerce",
            "flag_primo_demandeur": 0,
            "nb_credits_actifs": 1,
            "mobile_money_score": 80,
            "mobile_score_sur_revenu": 80 / 200000,
        }
    ]
)

prediction = model.predict(client)
proba = model.predict_proba(client)

print("Prédiction (0=Bon, 1=Défaut) :", prediction[0])
print("Probabilité de défaut :", proba[0][1])
