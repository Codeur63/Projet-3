"""
Service Redis pour le cache des prédictions
"""

import hashlib
import json
import os
from typing import Any

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/")
CACHE_TTL_SECONDS = 3600


class CacheService:
    """Service de cache Redis."""

    def __init__(self):
        self.client = None
        self.connected = False
        self.error = None

    def connect(self):
        """Connexion à Redis."""

        try:
            self.client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )

            self.client.ping()
            self.connected = True
            self.error = None

        except Exception as error:
            self.client = None
            self.connected = False
            self.error = str(error)

        return self.connected

    def status(self):
        """Retourne le statut du cache."""

        return {
            "redis_url": REDIS_URL,
            "connected": self.connected,
            "error": self.error,
            "ttl_seconds": CACHE_TTL_SECONDS,
        }

    def make_prediction_key(self, features: dict[str, Any]) -> str:
        """
        Crée une clé stable à partir des features.
        Deux requêtes identiques auront la même clé.
        """

        payload = json.dumps(
            features,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        return f"prediction:{digest}"

    def get(self, key: str):
        """Lit une valeur dans Redis."""

        if not self.connected or self.client is None:
            return None

        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except Exception as error:
            self.error = str(error)
            return None

    def set(self, key: str, value: dict[str, Any], ttl: int = CACHE_TTL_SECONDS):
        """Écrit une valeur dans Redis."""

        if not self.connected or self.client is None:
            return False

        try:
            self.client.setex(
                key,
                ttl,
                json.dumps(value, ensure_ascii=False, default=str),
            )
            return True

        except Exception as error:
            self.error = str(error)
            return False


cache_service = CacheService()
