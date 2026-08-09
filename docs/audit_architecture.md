# Audit Architecture & Code — FinaScore SA (Scoring crédit)

> **Type** : Revue d'architecture (Lead Data)
> **Périmètre** : Pipeline ML (src/), API (api/), MLOps (MLflow, Docker, Monitoring)
> **Statut** : Documentation — aucune modification de code

---

## 1. Vue d'ensemble de l'architecture

```
data/raw/ ──► 01_collect ──► 02_clean ──► 03_features ──► 04_split ──► 05_train
   (4 CSV)     (agrégation)   (nettoyage)  (feature eng.)  (split stratifié) (RandomSearch+MLflow)
                                                                   │
                                              ┌────────────────────┘
                                              ▼
                        09_tune (Optuna) ◄── 08_cross_validation ◄── 06_evaluate
                              │                    │
                              ▼                    ▼
                        13_register_model ──► registry decision JSON (seuil AUC ≥ 0.80)
                              │
                              ▼
                    api/ (FastAPI + Redis + Prometheus) ──► Docker Compose (API, Redis, Prometheus, Grafana)
                              │
                    dags/ + airflow/ + 14_monitoring (drift Evidently)
```

Chaîne de valeur sur le papier : ingestion → nettoyage → features → split → entraînement
→ validation → tuning → registry → serving → monitoring. Démarche MLOps complète
(FastAPI, Redis, Prometheus, Grafana, Airflow, Evidently).

---

## 2. Forces (bon patterns en place)

| # | Force | Localisation |
|---|-------|--------------|
| F1 | Pipeline numéroté `01→14` : ordre d'exécution clair et reproductible | `src/` |
| F2 | Split stratifié sur cible déséquilibrée (≈12% défaut) | `04_split.py` |
| F3 | Pipeline sklearn + `ColumnTransformer` : préprocessing fit sur train uniquement → pas de data leakage dans la CV | `05_train.py` |
| F4 | Prévention leakage temporel : filtrage `date_credit < date_demande` | `01_collect.py:67` |
| F5 | Gestion du déséquilibre : `scale_pos_weight`, `class_weight="balanced"` | `05_train.py` |
| F6 | Tracking MLflow (params + metrics + modèles) | `05 / 09 / 13` |
| F7 | Performance gate : logique explicite `AUC ≥ 0.80 → PROMOTE / NOT_PROMOTE` | `13_register_model.py` |
| F8 | API bien découpée : `model_loader`, `cache`, `schemas`, `metrics` séparés | `api/` |
| F9 | Cache Redis fail-open (l'API ne casse pas si Redis est down) | `api/cache.py` |
| F10 | Monitoring drift Evidently + PSI maison + alerte à 15% | `14_monitoring.py` |
| F11 | Monitoring Prometheus/Grafana via middleware propre | `api/metrics.py` |
| F12 | Honnêteté produit : l'API avertit que le modèle n'est pas promu | `api/main.py` |

Le composant API est le plus solide du projet : propre, découplé, robuste.

---

## 3. Faiblesses

### 3.1 Bugs bloquants (le code ne tourne pas)

| # | Fichier | Bug |
|---|---------|-----|
| B1 | `src/pipeline.py` | `BASE_DIR` utilisé mais jamais défini → `NameError` à l'exécution. `STEPS` ne contient que `01→06` (07-14 exclus). |
| B2 | `src/09_tune_xgboost.py:121` | Appelle `load_data()` mais seule `load_ata()` est définie → `NameError`. |
| B3 | `src/08_cross_validation.py:99` | Appelle `detect_column_types()` (inexistante, c'est `select_column_types`) → `NameError`. |
| B4 | `src/08_cross_validation.py:198` | `summary["auc_roc"]` → `KeyError` : le résumé statistique des folds n'est jamais calculé (`summary = {}`). |
| B5 | `tests/predict.py` | Charge `models/finascore_model.pkl` qui n'existe pas (le vrai nom est `xgboost.pkl`). |
| B6 | `docker-compose.yml` | Définit `MODEL_PATH=...` mais `api/model_loader.py` n'utilise jamais cette variable d'env → conflit de configuration. |

### 3.2 Risque de data leakage (crucial pour l'AUC)

- `05_train.py` **exclut** `nb_credits_defaut_hist` (considéré comme fuite) mais **garde**
  `taux_defaut_historique` et `taux_remboursement_historique`, qui en sont des transformations
  **directes** → incohérence, le ratio peut "téléphoner" le statut du client.
- `statut_final` (dans la donnée clean) est dérivé du statut du dernier crédit =
  quasi-définition de la cible. Il est exclu en entraînement, mais toutes ses dérivées
  doivent l'être aussi.

### 3.3 Reproducibilité et cohérence MLflow

- 3 expériences différentes : `finascore_credit_scoring` (05), `Finascore` (09/13)
  → tracking fragmenté.
- Le modèle servi par l'API (`models/optuna/xgboost_optuna.pkl`) n'est produit par aucun
  script de `src/` en l'état (09 plante). La boucle entraînement → registry → API est cassée.
- **Modèles `.pkl` commités dans git** (`models/*.pkl`) — anti-pattern MLOps
  (devrait être dans un artifact store / registry).

### 3.4 DRY — duplication massive

- `build_preprocessor` dupliqué 3× quasi à l'identique (`05`, `08`, `09`).
- `parse_date` / `statut` dupliqués 3× (`01`, `02`, notebooks EDA).
- `RANDOM_STATE=42`, `TARGET`, `EXCLUDED_COLS` / `DROP` (2 listes **divergentes**)
  redéfinis partout → toute évolution des colonnes crée des incohérences.

### 3.5 Quality gates manquants

- `06_evaluate.py` est un stub (print de stats, pas d'évaluation réelle du modèle test).
- Pas de **seuil de décision optimisé** (hardcodé à 0.5 partout) alors que la target est à 12% de positifs.
- Pas de **split temporel** alors que `date_demande` existe (un vrai scoring doit simuler le futur).
- Pas de CI/CD finalisé : `.github/workflows/ci.yml` présent mais incomplet
  (git status montre des fichiers modifiés/supprimés incohérents).
- `requirements.txt` **incomplet** : `lightgbm`, `pyarrow` (indispensables au pipeline)
  et `optuna` **absents**.

---

## 4. Notes

| Domaine | Note /10 | Justification |
|---|---|---|
| Architecture & MLOps | **7.0** | Vision MLOps complète et cohérente, séparation des responsabilités, monitoring, gate de promotion. Manque CI/CD finalisé + découplage entraînement/serving. |
| Qualité du code | **4.0** | Bugs bloquants (B1→B6), DRY violé, noms de fonctions erronés, chemins hardcodés. |
| Rigueur ML | **5.0** | Bonnes pratiques de base (stratify, pipeline, imbalance). Mais leakage partiel, pas de threshold tuning, AUC resté ~0.61-0.64. |
| Feature engineering | **5.0** | Bugs corrigés (solde/ratio), mais signal limité et plusieurs fichiers de features en circulation. |
| Serving & monitoring | **7.5** | Le point fort du projet. |
| Reproductibilité | **5.5** | MLflow ok, mais expériences fragmentées, `.pkl` dans git, pas de versioning données. |
| **Note globale** | **5.8** | Bonne fondation MLOps, mais code d'entraînement non exécutable en l'état et signal ML insuffisant. |

---

## 5. Leviers pour atteindre AUC > 0.80

D'après `docs/modeline_report.md`, même Optuna (100 trials) plafonne à ~0.64 et la
cross-validation confirme un signal limité dans les données. Un AUC > 0.80 nécessitera,
par ordre de priorité :

1. **Corriger la chaîne d'entraînement cassée** (B1→B4) — on ne peut pas optimiser ce qui ne tourne pas.
2. **Éliminer toute fuite** (`taux_defaut_historique`) pour avoir un AUC *honnête* avant de chercher à le gonfler.
3. **Split temporel** (train avant date X, test après) — les modèles actuels sont évalués sur un split aléatoire optimiste.
4. **Features vraiment informatives** : le signal mobile-money est sous-exploité. Des agrégats
   transactionnels plus riches (montant moyen/transaction, variance, tendance sur 6 mois)
   ont plus de potentiel que des ratios.
5. **Enrichissement externe** si possible (score bureautique, secteur économique du pays).
6. **Optimiser le seuil de décision** + calibration — même avec AUC ~0.65, un seuil business
   + pondération des coûts FN/FP peut dépasser 0.80 en **gain économique**, même si pas en AUC brut.

---

## 6. Actions recommandées (récapitulatif)

| Priorité | Action | Effort |
|----------|--------|--------|
| P0 | Corriger B1, B2, B3, B4 (pipeline, 09, 08) | Faible |
| P0 | Harmoniser les listes d'exclusion de colonnes (`EXCLUDED_COLS` / `DROP`) | Faible |
| P1 | Retirer `taux_defaut_historique` / `taux_remboursement_historique` (leakage) | Faible |
| P1 | Split temporel + seuil optimisé | Moyen |
| P1 | Factoriser le préprocesseur et les constantes dans un module commun | Moyen |
| P2 | Enrichissement des features mobile-money | Moyen |
| P2 | Finaliser CI/CD + compléter `requirements.txt` | Moyen |
| P2 | Sortir les `.pkl` de git → artifact store | Faible |
