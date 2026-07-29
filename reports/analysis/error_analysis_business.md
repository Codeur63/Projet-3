# Analyse métier des erreurs — FinaScore SA

## 1. Objectif

Ce rapport analyse les erreurs du modèle de scoring crédit afin de comprendre leur impact métier.

Le modèle analysé est :

```text
xgboost_optuna
```

Chemin du modèle :

```text
models/optuna/xgboost_optuna.pkl
```

## 2. Métriques globales

Seuil utilisé pour l'analyse principale :

```text
0.5
```

Résultats :

| Métrique | Valeur |
|---|---:|
| AUC-ROC | 0.6380 |
| F1-score | 0.2663 |
| Precision | 0.1747 |
| Recall | 0.5599 |
| Accuracy | 0.6245 |

## 3. Matrice de confusion métier

| Type | Nombre | Interprétation |
|---|---:|---|
| True Negative | 5536 | Bons clients acceptés |
| False Positive | 3203 | Bons clients refusés |
| False Negative | 533 | Défauts non détectés |
| True Positive | 678 | Défauts détectés |

Taux d'erreur global :

```text
0.3755
```

## 4. Lecture métier

Les **faux négatifs** sont les erreurs les plus critiques pour le risque crédit.  
Ils correspondent aux demandeurs qui feront défaut, mais que le modèle classe comme bons payeurs.

Les **faux positifs** correspondent aux bons clients refusés.  
Ils représentent une perte d'opportunité commerciale et peuvent réduire le volume d'affaires.

## 5. Hypothèses de coût

Les coûts sont estimés avec les hypothèses suivantes :

| Élément | Hypothèse |
|---|---:|
| Loss Given Default | 60% |
| Coût d'opportunité faux positif | 8% |
| Colonne de montant utilisée | avg_credit_xaf |

Ces montants sont des hypothèses de travail. Ils doivent être remplacés par les coûts réels de FinaScore si disponibles.

## 6. Coût estimé des erreurs

| Indicateur | Valeur |
|---|---:|
| Coût total estimé | 215,962,380 XAF |
| Coût moyen par observation | 21,705 XAF |
| Coût des faux négatifs | 121,061,800 XAF |
| Coût des faux positifs | 94,900,580 XAF |

## 7. Simulation des seuils

Le seuil avec le coût estimé le plus faible dans la simulation est :

```text
0.45
```

Coût associé :

```text
211,005,653 XAF
```

Attention : cette simulation est réalisée sur le test final. Elle sert au diagnostic métier, pas à définir directement un seuil de production.

## 8. Segments les plus coûteux

Les segments les plus coûteux sont sauvegardés dans :

```text
reports/error_analysis_by_subgroup.csv
```

Top 10 segments par coût estimé :

| Segment | Valeur | Nombre | Taux défaut | Coût estimé |
|---|---|---:|---:|---:|
| pays | CMR | 4928 | 12.05% | 105,389,177 XAF |
| zone | Urbain | 4459 | 12.20% | 99,469,097 XAF |
| secteur_activite | Commerce | 3483 | 12.58% | 78,525,237 XAF |
| zone | Peri-urbain | 2757 | 11.68% | 59,648,667 XAF |
| secteur_activite | Agriculture | 2439 | 11.77% | 50,094,917 XAF |
| zone | Rural | 2351 | 12.34% | 48,711,833 XAF |
| secteur_activite | Artisanat | 2025 | 12.10% | 40,039,983 XAF |
| secteur_activite | Service | 1516 | 11.94% | 35,312,777 XAF |
| pays | COG | 1207 | 13.67% | 32,341,233 XAF |
| pays | GAB | 1363 | 10.64% | 25,233,543 XAF |


## 9. Décision de promotion

Seuil requis pour une promotion :

```text
AUC-ROC >= 0.8
```

AUC observée :

```text
0.6380
```

Décision :

```text
DO_NOT_PROMOTE
```

Le modèle peut rester en expérimentation ou en staging, mais ne doit pas être utilisé comme modèle de décision automatique en production tant que le seuil n'est pas atteint.

## 10. Recommandations

Avant toute mise en production, il est recommandé de :

- enrichir les données historiques de remboursement ;
- ajouter davantage de variables comportementales mobile money ;
- améliorer l'alignement temporel entre demande et historique crédit ;
- travailler avec les métiers pour obtenir de vrais coûts de faux positifs et faux négatifs ;
- définir un seuil métier sur validation, pas sur le test final ;
- conserver un humain dans la boucle tant que la performance reste insuffisante.
