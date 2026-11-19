"""Scénarios Locust pour l'API de prédiction.

Quatre scénarios, sélectionnables par `--class-picker` ou par tag :

* `PredictionUnitaire` — un disque par requête (cas « alerte en ligne ») ;
* `PredictionLot100`   — lot de 100 disques (cas « rack ») ;
* `PredictionLot1000`  — lot de 1 000 disques (cas « scoring de parc ») ;
* `TraficRealiste`     — mélange 80 % unitaire / 18 % lot 100 / 2 % lot 1 000,
  plus quelques appels `/health` et `/model`, qui est ce qu'on attend en
  production.

Par défaut, les prédictions de test ne sont pas persistées (`persist: false`) : on mesure la
capacité de calcul de l'API, pas le débit d'insertion de PostgreSQL, et la
campagne ne pollue pas les traces exploitées par Grafana.

`LOADTEST_PERSIST=1` les fait écrire en base, pour mesurer le coût des traces.

Protocole et résultats : loadtest/PROTOCOLE.md.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, constant, tag, task

MODELS = [
    ("ST4000DM000", 4_000_787_030_016),
    ("ST12000NM0008", 12_000_138_625_024),
    ("TOSHIBA MG07ACA14TA", 14_000_519_643_136),
    ("HGST HUH721212ALN604", 12_000_138_625_024),
    ("WDC WUH721414ALE6L4", 14_000_519_643_136),
]


PERSIST = os.environ.get("LOADTEST_PERSIST") == "1"


def make_observation(index: int = 0) -> dict[str, object]:
    """Observation plausible : disque sain, douteux ou franchement dégradé."""
    model, capacity = random.choice(MODELS)
    severity = random.random()
    age_hours = random.randint(2_000, 60_000)
    return {
        "serial_number": f"LT{index:08d}{random.randint(10, 99)}",
        "model": model,
        "capacity_bytes": capacity,
        "smart_5_raw": int(severity**3 * 400),
        "smart_9_raw": age_hours,
        "smart_12_raw": age_hours // 900,
        "smart_187_raw": int(severity**3 * 150),
        "smart_188_raw": int(severity**3 * 30),
        "smart_193_raw": int(age_hours * 1.6),
        "smart_194_raw": random.randint(24, 46),
        "smart_197_raw": int(severity**3 * 90),
        "smart_198_raw": int(severity**3 * 55),
        "smart_199_raw": int(severity**3 * 12),
    }


def batch_payload(size: int) -> dict[str, object]:
    return {
        "observations": [make_observation(i) for i in range(size)],
        "persist": PERSIST,
    }


class PredictionUnitaire(HttpUser):
    """Prédiction en ligne, un disque par requête."""

    wait_time = constant(0)

    @tag("unitaire")
    @task
    def predict(self) -> None:
        self.client.post(
            "/predict",
            json={"observation": make_observation(random.randint(0, 10**6)), "persist": PERSIST},
            name="POST /predict",
        )


class PredictionLot100(HttpUser):
    """Lot de 100 disques : taille typique d'un rack."""

    wait_time = constant(0)

    @tag("lot100")
    @task
    def predict_batch(self) -> None:
        self.client.post(
            "/predict/batch",
            json=batch_payload(100),
            headers={"X-Batch-Size": "100"},
            name="POST /predict/batch [100]",
        )


class PredictionLot1000(HttpUser):
    """Lot de 1 000 disques : scoring d'un fragment de parc."""

    wait_time = constant(0)

    @tag("lot1000")
    @task
    def predict_batch(self) -> None:
        self.client.post(
            "/predict/batch",
            json=batch_payload(1_000),
            headers={"X-Batch-Size": "1000"},
            name="POST /predict/batch [1000]",
        )


class TraficRealiste(HttpUser):
    """Mélange représentatif, avec des appels de supervision."""

    wait_time = between(0.1, 0.5)

    @tag("realiste")
    @task(80)
    def predict(self) -> None:
        self.client.post(
            "/predict",
            json={"observation": make_observation(random.randint(0, 10**6)), "persist": PERSIST},
            name="POST /predict",
        )

    @tag("realiste")
    @task(18)
    def predict_batch_100(self) -> None:
        self.client.post(
            "/predict/batch",
            json=batch_payload(100),
            headers={"X-Batch-Size": "100"},
            name="POST /predict/batch [100]",
        )

    @tag("realiste")
    @task(2)
    def predict_batch_1000(self) -> None:
        self.client.post(
            "/predict/batch",
            json=batch_payload(1_000),
            headers={"X-Batch-Size": "1000"},
            name="POST /predict/batch [1000]",
        )

    @tag("realiste")
    @task(3)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @tag("realiste")
    @task(1)
    def model(self) -> None:
        self.client.get("/model", name="GET /model")
