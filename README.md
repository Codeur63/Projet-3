# FinaScore SA — Système de scoring crédit ML/MLOps

## 1. Contexte du projet

FinaScore SA est une fintech camerounaise spécialisée dans le scoring crédit pour les PME et micro-entrepreneurs de la zone CEMAC.

L’objectif du projet est de remplacer un scoring manuel par un pipeline ML reproductible permettant de prédire le risque de défaut de paiement d’un demandeur de crédit.

La variable cible est :

- `defaut_paiement = 1` : défaut de paiement
- `defaut_paiement = 0` : client solvable

Le projet s’inscrit dans une logique ML/MLOps avec :

- pipeline de données reproductible ;
- feature engineering métier ;
- entraînement de plusieurs modèles ;
- tracking MLflow ;
- évaluation finale ;
- contrôle de performance avant promotion.

---

## 2. Données utilisées

Les fichiers sources sont placés dans `data/raw/`.

| Fichier | Rôle |
|---|---|
| `applicants.csv` | Profils des demandeurs et target |
| `credit_history.csv` | Historique de remboursement |
| `mobile_money_transactions.csv` | Données comportementales mobile money |
| `partners_metadata.csv` | Informations sur les partenaires |

Le dataset est volontairement imparfait : doublons, valeurs manquantes, formats de dates mixtes, catégories non standardisées et valeurs aberrantes.

---

## 3. Installation 

```text
git clone <url depot>
cd <depot>
pip install -r requirements.txt
```

voir le tracking des modeles ``` mlfow ui ``

## 4. Run pipeline
- For unix 
    ```python
    python3 src/pipeline.py 
    ```
- For other SE
    ``` python 
        python src/pipeline.py 
    ```
