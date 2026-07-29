# FinaScore SA — Système de scoring crédit ML/MLOps

## 1. Contexte du projet

FinaScore SA est une fintech camerounaise spécialisée dans le scoring crédit pour les PME et micro-entrepreneurs de la zone CEMAC qui vend leur scores a des parteniares effectuant des crédits.

L’objectif du projet est de remplacer un scoring manuel par un pipeline ML reproductible permettant de prédire le risque de défaut de paiement d’un demandeur de crédit, en fonction de son passé.

La variable cible est :
- `defaut_paiement = 1` : défaut de paiement
- `defaut_paiement = 0` : client solvable

Le projet s’inscrit dans une logique :
    - Obtenir un ML Performant versionné et déployé en API REST conteneurisée
    - Avoir un monitoring actif du drift
    - Un pipeline CI/CD automatisé  

Nous comprenons ici donc facilement que notre projet intègre ML/MLOPS, FastAPI, Docker, CI/CD. Pour ça nous procédons aux étapes suivante : 
   - Créer un pipeline de données reproductible ;
   - Faire de la feature engineering métier pour les données ;
   - Pourvoir entraîné de plusieurs modèles (Logistics Regresion , Random Forest, XGBoost) ;
   - Faire du tracking avec MLflow ;
   - Faire de l'évaluation finale du meilleur modèle ;
   - Pourvoir contrôle la performance du modèle avant promotion.
   - Pourvoir déployé la version du modèle qui est performant (AUC>= 80) ;
   - Contenerisation notre API et pouvant etre deployer avec Docker Compose; 
   - Test de charge avec Locust
   - Entraiment automatique du modèle, evaluation et promotion avec CI/CD ;
   - Dasboard Grafana

---

## 2. Données utilisées

Les fichiers sources sont placés dans `data/raw/`.

| Fichier | Rôle |
|---|---|
| `applicants.csv` | Profils des demandeurs et target |
| `credit_history.csv` | Historique de remboursement |
| `mobile_money_transactions.csv` | Données comportementales mobile money |
| `partners_metadata.csv` | Informations sur les partenaires |

Le dataset est volontairement imparfait : doublons, valeurs manquantes, formats de dates mixtes, catégories non standardisées, valeurs aberrantes et abscente.
 
---

## 3. Installation 

Pour avoir le projet, ensuite cd/projet-3 et installer les dépendances avec pip 
```text
git clone <url depot>
cd <depot>
pip install -r requirements.txt
```

voir le tracking des modeles ``` mlfow ui ```, lancer l'API ```docker-compose up -d ou docker compose up -d ``` selon les versions docker

## 4. Run pipeline
- For unix 
    ```python
    python3 src/pipeline.py 
    ```
- For other SE
    ``` python 
        python src/pipeline.py 
    ```
