from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path("data/features/features_dataset.csv")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "defaut_paiement"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("Debut du Train Test Split")
    print("=" * 60)

    df_finascore = pd.read_csv(FEATURES_PATH, low_memory=False)

    X = df_finascore.drop(columns=[TARGET])
    y = df_finascore[TARGET].astype(int)

    print("Shape dataset complet :", df_finascore.shape)
    print("Distribution target complète :")
    print(y.value_counts(normalize=True).round(2))

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Sauvegarde des fichiers
    X_train.to_parquet(SPLITS_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(SPLITS_DIR / "X_test.parquet", index=False)
    y_train.to_frame(name=TARGET).to_parquet(SPLITS_DIR / "y_train.parquet", index=False)
    y_test.to_frame(name=TARGET).to_parquet(SPLITS_DIR / "y_test.parquet", index=False)

    #  Contrôles post-split
    print("\nTrain/Test sauvegardés")
    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    print("\nDistribution target train :")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nDistribution target test :")
    print(y_test.value_counts(normalize=True).round(4))

    print("=" * 10 + "Sauvergade effectué des trains et des TESTS" + "=" * 10)


if __name__ == "__main__":
    main()
