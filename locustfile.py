"""
Tests de charge de l'API FinaScore
    - Simuler plusieurs utilisateur de l'APO
    - Tester les endpoints principaux de l'API
    - Mesurer les temps de réponse pour verifier le seuil p95 < 200ms
    - Vérifier le comportement sous charge
"""

from locust import HttpUser, between, task

SAMPLE_FEATURES_1 = {
    "age": 35,
    "pays": "CMR",
    "secteur_activite": "commerce",
    "zone": "urbaine",
    "revenu_mensuel_xaf": 250000,
    "anciennete_emploi": 4,
    "ratio_endettement": 0.35,
    "historique_credit": 650,
    "nb_credits_actifs": 1,
    "mobile_money_score": 72,
}


SAMPLE_FEATURES_2 = {
    "age": 48,
    "pays": "GAB",
    "secteur_activite": "commerce",
    "zone": "semi_urbaine",
    "revenu_mensuel_xaf": 180000,
    "anciennete_emploi": 2,
    "ratio_endettement": 0.65,
    "historique_credit": 580,
    "nb_credits_actifs": 2,
    "mobile_money_score": 55,
}


SAMPLE_FEATURES_3 = {
    "age": 29,
    "pays": "COG",
    "secteur_activite": "services",
    "zone": "urbaine",
    "revenu_mensuel_xaf": 320000,
    "anciennete_emploi": 6,
    "ratio_endettement": 0.25,
    "historique_credit": 710,
    "nb_credits_actifs": 0,
    "mobile_money_score": 80,
}


class FinaScoreUser(HttpUser):
    """Utilisateur simulé pour l'API FinaScore."""

    wait_time = between(1, 3)

    @task(2)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def model_info(self):
        self.client.get("/model/info", name="/model/info")

    @task(5)
    def predict_one(self):
        payload = {"features": SAMPLE_FEATURES_1}

        self.client.post(
            "/predict",
            json=payload,
            name="/predict",
        )

    @task(2)
    def predict_batch(self):
        payload = {
            "records": [
                SAMPLE_FEATURES_1,
                SAMPLE_FEATURES_2,
                SAMPLE_FEATURES_3,
            ]
        }

        self.client.post(
            "/predict/batch",
            json=payload,
            name="/predict/batch",
        )
