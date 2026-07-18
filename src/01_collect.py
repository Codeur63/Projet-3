import pandas as pd

applicants = pd.read_csv("data/raw/applicants.csv", low_memory=False)
mobile = pd.read_csv("data/raw/mobile_money_transactions.csv", low_memory=False)
partners = pd.read_csv("data/raw/partners_metadata.csv", low_memory=False)
credit = pd.read_csv("data/raw/credit_history.csv", low_memory=False)

credit_agg = (
    credit
    .groupby("applicant_id")
    .agg({
        "montant_xaf":["sum", "mean"],
        "nb_retards":"sum",
        "jours_retard_max":"max",
        "credit_id":"count"
    })
) 

credit_agg.columns = ["total_montant_xaf","avg_credit_xaf","total_retards", "max_retard","nb_credit"]


df = (
    applicants
    .merge(
        mobile,
        on="applicant_id",
        how="left"
    )
    .merge(
        credit_agg,
        on="applicant_id",
        how="left"
    )
)

df.to_csv(
    "data/merge/merged_dataset.csv",
    index=False
)

print(df.shape)