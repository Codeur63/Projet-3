import pandas as pd
from sklearn.model_selection import train_test_split

df_finascore = pd.read_csv("data/features/finascore_clean.csv", low_memory=True)

TARGET = "defaut_paiement"

X = df_finascore.drop(columns=TARGET)
y = df_finascore[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42) 

X_train.to_parquet('data/splits/X_train.parquet', index=False)

X_test.to_parquet('data/splits/X_test.parquet', index=False)

y_train.to_frame().to_parquet('data/splits/y_train.parquet', index=False)

y_test.to_frame().to_parquet('data/splits/y_test.parquet', index=False)

print("Train/Test Sauvegardés")