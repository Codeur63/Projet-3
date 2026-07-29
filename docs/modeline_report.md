# Rapport de modélisation — FinaScore SA

## 1. Objectif

L’objectif de cette phase est d’entraîner plusieurs modèles supervisés afin de prédire le défaut de paiement des demandeurs de crédit.

La variable cible est :

- `defaut_paiement = 1` : défaut
- `defaut_paiement = 0` : solvable

Les modèles testés sont :

- Logistic Regression
- Random Forest
- XGBoost

Les métriques suivies sont :

- AUC-ROC
- F1-score
- Precision
- Recall

L’accuracy est suivie à titre informatif uniquement, car la target est déséquilibrée.

---

## 2. Methode Appliqué
- Analayse des données EDA et répérer les features importantes
- Collecter les données et les nettoyers
- Créer des features utilisable
- Spliter les données de Train et les données de Test
- Entrainer le modèle avec le tracking MLFLOW ainsi que les expériences
- Evaluer les modèles qui sont en compétition
- Faire la Cross-validation sur les données de Test
- Optimisation du XGBoost avec Optuna (100 trials) et documenter
- Analyser les métiers des Erreurs (Calcul des couts estimé des erreurs pour un paternaire )
- Detecter les anomalies (avec Isolation Forest)
- Creer API, dockerisé, tester , avec l'API utilisant le model enregister dans MLFlow
- Monitoring du Drift + Dashboard Grafana 
- Pipeline CI/CD pour entrainer le modele et le mettre a jour automatiquement
   

## 3. Pipeline utilisé

Le pipeline est structuré en plusieurs étapes :

```text
01_collect.py   -> collecte, agrégation et fusion des données
02_clean.py     -> nettoyage minimal des anomalies
03_features.py  -> création des variables métier
04_split.py     -> séparation train/test stratifiée
05_train.py     -> entraînement et tracking MLflow
06_evaluate.py  -> évaluation finale sur le test set
```

## 4. Resultats

Modéle Champion: XGBoost
AUC-test : environ 0.64
AUC-train : environ 0.63

## 5. Learning
Avec le diagnostics nous pouvons ainsi voir is on  a des variables importantes qu'on pourrais priveligíe à d'autre. Le learning Curve nous permet de savoir si on a besoin de plus de données pour que notre modèle soit plus performant ou bien meme avec nos données le modèle ne seras pas toujours performant. Ce qui indique un probleme de signal au niveau des données qu'un simple overfitting remarque a travers nos auc-roc

## 6.Decision
La décision pour la promotion d'un modèle doit etre supérieur à 80 concernant le AUC-ROC 


## 7. Recommandation

- Utilisation d'un model selection ensemble (Voting/Bagging), pour associer les 3 modeles et entrainer un nouveau modèle sur cette base. Car nos trois modele optienne des AUC-ROC >= 0.60 
- Enrichir les données avec plus d'information

## 8.Cross Validation
La cross-validation Stratified K-Fold 5 confirme que le modèle champion ne généralise pas suffisamment. L’AUC moyenne est de 0.59, avec une faible variance entre folds.
Le problème semble donc lié à un signal prédictif limité dans les variables disponibles, plutôt qu’à un simple problème de surapprentissage. Conformément au seuil de performance défini dans le cahier des charges, le modèle ne doit pas être promu en production. Il permet de determinée entre les FOLD si notre modèle est stable ou pas.

## 9.Optuna
IL vas permettre à ameliorer notre modèle champion, il a été exécuté sur XGBoost avec 100 trials. Malgré l’optimisation des hyperparamètres, le modèle reste sous le seuil AUC <= 0.65. Le gain obtenu est insuffisant, ce qui confirme que la limitation principale vient du signal disponible dans les données plutôt que du choix d’hyperparamètres.

## 10. Analyse
A travers l'analyse nous pouvons voir que les partenaires perdent enormement d'argent car les données que nous avons reçu ne sont pas bonne ou une mauvaise modélisation. Nous trouvons évidement des signaux faible que le modèle n'arrive pas à pouvoir gerer. Ici nous cherchons à savoir : TN : bons client acceptés, FP: bons client refusés, FN: défauts non détectés, TP: défauts détectés.

## Isolation
Permet de voir si les anomalies de notre dataset ont un petit signal. Et nous remarquons que le anomalies ont un petit signal, mais pas assez fort pour transformer le modèle.

## Enregistrer le Modèle
On enregistre le modèle meme si il n'est pas bon pour la traçalabilité, mais il ne seras pas promu en production car l'AUC est inférieur au seuil attendu

## FAST API
