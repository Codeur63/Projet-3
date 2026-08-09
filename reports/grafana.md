# Rapport Prometheus / Grafana - FinaScore SA

## 1. Objectif

Ce rapport décrit la mise en place du monitoring technique de l'API FinaScore avec Prometheus et Grafana.

## 2. Architecture

L'environnement Docker Compose contient :

- API FastAPI : exposition des endpoints de scoring et de `/metrics`
- Redis : cache applicatif
- Prometheus : collecte des métriques API
- Grafana : visualisation des métriques

## 3. Métriques exposées

Les métriques principales sont :

- `finascore_api_requests_total`
- `finascore_api_request_latency_seconds`
- `finascore_predictions_total`
- `finascore_model_loaded`
- `finascore_last_default_probability`

## 4. Dashboard Grafana

Le dashboard contient 6 panels :

1. Requêtes par seconde
2. Latence p95 par endpoint
3. Erreurs HTTP
4. Nombre de prédictions par classe
5. Dernière probabilité de défaut
6. Statut de chargement du modèle

## 5. Interprétation

Le monitoring permet de suivre la disponibilité de l'API, sa latence, ses erreurs et son activité de prédiction.

La métrique p95 permet de vérifier la contrainte de performance API. Dans les tests Locust, le p95 observé était inférieur à 40 ms, donc largement inférieur au seuil attendu de 200 ms.

## 6. Conclusion

La brique Prometheus / Grafana complète le dispositif MLOps en ajoutant une supervision technique de l'API de scoring.

http://localhost:9090
http://localhost:3000