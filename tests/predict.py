import joblib
import pandas as pd 

model  = joblib.load(
    "models/finascore_model.pkl"
)

client = pd.DataFrame([
    {
        "age":34,
        "revenu_mensuel_xaf":200000,
        "ratio_endettement":0.3,
        "historique_credit":500,
        "mobile_money_score":80
    }
])

prediction = model.predict(client)

proba= model.predict_proba(client)

print(f"Prediction du client : {prediction}")
print(f"Proba : {proba}") 