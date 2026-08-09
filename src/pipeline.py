"""
- Collecte de données.
- Nettoyage de données.
- Création de Features.
- Splits des données.
- Entrainement des données.
- Evaluation du modèle.
- Diagnostic des variables d'entrainement.
"""

import subprocess
import sys
from pathlib import Path

STEPS = ["01_collect.py", "02_clean.py", "03_features.py", "04_split.py", "05_train.py", "06_evaluate.py", "07_diagnostics.py", "13_register_model.py", "14_monitoring.py"]


SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def run_step(script: str):
    script_path = SRC_DIR / script

    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable : {script_path}")

    print("=" * 80)
    print(f"Script : {script_path}")
    print("=" * 80)

    result = subprocess.run([sys.executable, str(script_path)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Echec de {script}")

    print(f"{script} terminé\n")
    print("=" * 60)


def main():
    print("\nLANCEMENT DU PIPELINE COMPLET FINASCORE\n")

    for script_name in STEPS:
        run_step(script_name)

    print("=" * 80)
    print("PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


if __name__ == "__main__":
    main()
