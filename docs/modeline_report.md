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

## 2. Pipeline utilisé

Le pipeline est structuré en plusieurs étapes :

```text
01_collect.py   -> collecte, agrégation et fusion des données
02_clean.py     -> nettoyage minimal des anomalies
03_features.py  -> création des variables métier
04_split.py     -> séparation train/test stratifiée
05_train.py     -> entraînement et tracking MLflow
06_evaluate.py  -> évaluation finale sur le test set
```

## 3. Resultats

Modéle Champion: XGBoost
AUC-test : environ 0.64
AUC-train : environ 0.63

## 4. Learning
Avec le diagnostics nous pouvons ainsi voir is on  a des variables importantes qu'on pourrais priveligíe à d'autre. Le learning Curve nous permet de savoir si on a besoin de plus de données pour que notre modèle soit plus performant ou bien meme avec nos données le modèle ne seras pas toujours performant. Ce qui indique un probleme de signal au niveau des données qu'un simple overfitting remarque a travers nos auc-roc

## 5.Decision
La décision pour la promotion d'un modèle doit etre supérieur à 80 concernant le AUC-ROC 


## 6. Recommandation

- Utilisation d'un model selection ensemble (Voting/Bagging), pour associer les 3 modeles et entrainer un nouveau modèle sur cette base. Car nos trois modele optienne des AUC-ROC >= 0.60 
- Enrichir les données avec plus d'information
