# FINASCORE SA - IA DE SCORING

Dans la suite de ce projet notre objectif était de construire un système complet de scoring crédit et de le deployer pour FINASCORE enfin qu'il puisse utilisé les modèles pour les partenaires. Enfin de pouvoir le faire aisement de la construction du système au déployement et enfin au monitoring, nous procédons par les étapes suivant :
 - EDA (Explorer les données à disposition)  
 - Collecte et Netoyage des données
 - Feature engineering (Retenir les variables importantes et en créer des nouvelles qui seront utiles)
 - Entrainer les modèles (XGBoost, Logistic Regression, RandomForest)
 - Validation des données (AUC >= 80) 
 - Enregistrer les meilleurs modeles  et tracker avec MLFlor enfin d'avoir les historiques 
 - Tunning du modèle avec Optuna
 - Enregistrer le meilleur modéles dans MLFlow (AUC>=80)
 - Faire une API (Fast API) pour utiliser le modèle depuis n'importe où.
 - Tester l'API
 - Dockeriser API
 - Voir les performances et faire la simulation de l'API avec locust
 - CI/CD github Actions (Automatisation de l'intégration et du déployement)
 - Apache Airflow pour automatiser le processus


## Pipeline ML
La premiere partie etait d'initialiser le projet avec un gestionaire pip ou poetry ou uv selon les préférences, ainsi que l'installation d'un environnement virtuelle (selon le gestionaire) pour installer toutes les dépendances que nous aurons besoin dans projet. 

Récupérer les données et le mettre dans un dossier ```data/raw/```, faire un script qui vas nous permettre de collecter nos différent données depuis le dossier, les nettoyers et les charger dans un autre dossier.  

Dans cette partie nous avons nettoyer le dataset, collecter les données et nettoyer les residus pour une dernier fois. 
Puis est venu les travailles de features engineering à l'aide de EDA, nous avons pu localiser les varaibles qui serais importante dans notre systeme de scoring, et en ajouter certaines au passage, et l'enregister dans ```data/features/```. Le prochain point était de diviser le dataset en deux partie. ```DataSet d'entrainement et de test``` pour le modele.  On utiliser les données d'entrainement pour entrainer et évaluer les 03 modèles en compétitions et enregistrer les meilleurs tout en faisant le tracking dans MLFlow, pour la traçabilité.  Pour le tracking et gestion du cycle de vie des modèles.

## Validation du Modèle
Ici il s'agiras de valider nos modeles avec la technique startified KFOLd pour la cross validation, car nous avons un jeux de données très désequilibre, l'optimisation des hyperparmetres de notre meilleure modèle avec OPTUNA. Avec cette évaluation du modéle nous sommmes rendu compte que le modèle tournais toujours autour de 0.59 de AUC ce qui prouvais un peu que notre jeu de données n'avais pas assez de signal pour bien implémenter notre jeu de données. L'optimisation de notre modèle avec OPTUNA, n'as pas été d'une grande utilité pour notre modèle. Nous sommes passés au couts estimer des mauvaises predictions de du modèles
et les valuers manquantes comme les revenus (passer à 1.000.000 XAF), faire la dectection des anomolies avec L'Isolation Forest (détecter les progils atypiques) qui peuvent servir à evaluer les demandeurs dont le profil financier n'est pas correcte. Faire l'entrainement sur ses anomalies et voir le resultat, et enfin d'enregistrer le modèle dans MLFlow registry

## Docker
Ici nous devons concevoir une API qui utiliseras le modèle pour les prédiction, locus pour pouvoir tester la charger des API, redis pour guader les predictions deja effectuer en cache, et docker pour enfin conteneuriser toutes l'application API, et nous avons vue que pour 95% de ŕeponse nous avons les temps de réponse inférieurs à 25 ms. L'API est performante. 

## GitHub Action
Ici nous proceder à l'utilisation du CI/CD pour automatiser les taches. Le CI pour verifier la structure du projet, les test et éviter les régressions de code à chaque push du projet. Le CD nous serviras à déployer notre image docker. On a utiliser Evidently, qui permet de savoir si les donnés changent après déploiement. Prometheus nous permettras de collecter les métriqus techniques de notre API, et Grafana visualiser les métriques. 


