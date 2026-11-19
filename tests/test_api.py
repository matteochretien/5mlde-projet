"""API de prédiction : validation des entrées, erreurs, version du modèle.

Le modèle est réellement entraîné (jeu synthétique réduit, MLflow en SQLite
temporaire) : ces tests couvrent donc aussi le chargement depuis le registre.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.ingestion.bronze import build_bronze
from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats
from hdd_rul.training.train import train_and_publish

OBSERVATION = {
    "serial_number": "ZA00000001",
    "model": "ST4000DM000",
    "capacity_bytes": 4_000_787_030_016,
    "date": "2026-05-20",
    "smart_5_raw": 120,
    "smart_9_raw": 41_714,
    "smart_12_raw": 48,
    "smart_187_raw": 80,
    "smart_188_raw": 5,
    "smart_193_raw": 66_742,
    "smart_194_raw": 34,
    "smart_197_raw": 40,
    "smart_198_raw": 12,
    "smart_199_raw": 2,
}


@pytest.fixture(scope="module")
def served_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    root = tmp_path_factory.mktemp("api")
    config = load_config(
        overrides={
            "paths": {"root": str(root / "data"), "artifacts": str(root / "artifacts")},
            "sampling": {"healthy_rate_pct": 70, "healthy_stride_days": 2},
            "mlflow": {
                "tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
                "experiment_name": "api",
                "registered_model_name": "hdd-rul-api",
            },
            "synthetic": {
                "n_drives": 160,
                "start_date": "2025-12-01",
                "end_date": "2026-06-30",
                "daily_failure_rate": 0.003,
                "seed": 23,
            },
            "training": {"hgb": {"max_iter": 30, "early_stopping": False}},
        }
    )
    generate_synthetic_drive_stats(config)
    build_bronze(config)
    train_and_publish(config, trigger="test")
    return config


@pytest.fixture
def client(served_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client de test. Les traces PostgreSQL sont désactivées."""
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    monkeypatch.setattr("hdd_rul.common.config.load_config", lambda *a, **k: served_config)
    import api.main as api_main

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: served_config)
    with TestClient(api_main.app) as test_client:
        yield test_client


@pytest.fixture
def client_sans_modele(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    """API démarrée sans champion disponible : doit répondre, en mode dégradé."""
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    vide = load_config(
        overrides={
            "paths": {"root": str(tmp_path / "data")},
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_path / 'vide.db'}",
                "registered_model_name": "modele-inexistant",
            },
        }
    )
    import api.main as api_main

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: vide)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_sante_nominale(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["model_loaded"] is True
    assert body["model"]["model_alias"] == "champion"


def test_registre_injoignable_rend_la_sante_degradee(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow arrêté : l'API sert toujours (200), mais le signale (degraded)."""
    import api.main as api_main

    monkeypatch.setattr(
        api_main,
        "registry_healthcheck",
        lambda *a, **k: {"enabled": True, "reachable": False, "error": "ConnectionError"},
    )
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is True
    assert body["registry"]["reachable"] is False


def test_verification_du_registre_rapide_et_sans_exception() -> None:
    """Serveur injoignable : réponse en moins de 2 s, sans lever."""
    import time

    from hdd_rul.common.config import load_config
    from hdd_rul.training.registry import registry_healthcheck

    injoignable = load_config(overrides={"mlflow": {"tracking_uri": "http://127.0.0.1:9"}})
    debut = time.monotonic()
    etat = registry_healthcheck(injoignable, timeout=1.0)
    assert time.monotonic() - debut < 2.0
    assert etat == {"enabled": True, "reachable": False, "error": "ConnectionError"}

    local = load_config(overrides={"mlflow": {"tracking_uri": "sqlite:///x.db"}})
    assert registry_healthcheck(local)["enabled"] is False


def test_rechargement_rate_garde_le_modele_servi(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow arrêté pendant un /admin/reload : 503 explicite, et la version déjà
    chargée continue de prédire (avant : 500, ou modèle perdu)."""
    from mlflow.exceptions import MlflowException

    import hdd_rul.serving.predictor as predictor_module

    version_avant = client.get("/model").json()["model_version"]

    def registre_injoignable(*args, **kwargs):
        raise MlflowException("API request to http://mlflow:5000 failed: Max retries exceeded")

    monkeypatch.setattr(predictor_module, "configure_mlflow", registre_injoignable)
    reponse = client.post("/admin/reload")
    assert reponse.status_code == 503
    assert "registre MLflow injoignable" in reponse.json()["detail"]

    assert client.get("/model").json()["model_version"] == version_avant
    prediction = client.post("/predict", json={"observation": OBSERVATION, "persist": False})
    assert prediction.status_code == 200


def test_description_du_modele(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["available"] is True
    assert body["model_version"] == "1"
    assert body["horizon_days"] == 90
    assert body["alert_threshold_days"] == 30
    assert "smart_5_raw" in body["features"]
    assert body["metrics"]["mae"] > 0


def test_prediction_unitaire_porte_la_version_du_modele(client: TestClient) -> None:
    response = client.post("/predict", json={"observation": OBSERVATION, "persist": False})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["prediction"]["predicted_rul_days"] <= 90
    assert body["prediction"]["alert"] is (body["prediction"]["predicted_rul_days"] < 30)
    # Exigence du sujet : identifier la version utilisée dans chaque réponse.
    assert body["model"]["model_name"] == "hdd-rul-api"
    assert body["model"]["model_version"] == "1"
    assert body["model"]["run_id"]
    assert "X-Process-Time-Ms" in response.headers


def test_prediction_par_lot(client: TestClient) -> None:
    observations = [dict(OBSERVATION, serial_number=f"ZA{i:08d}") for i in range(50)]
    body = client.post(
        "/predict/batch", json={"observations": observations, "persist": False}
    ).json()
    assert body["n_items"] == 50
    assert len(body["predictions"]) == 50
    assert body["n_alerts"] <= 50
    assert body["persisted"] == 0  # persist: false


def test_attributs_smart_absents_toleres(client: TestClient) -> None:
    """Un disque qui ne remonte pas tous ses attributs reste prédictible."""
    minimal = {"serial_number": "ZA99999999", "model": "ST4000DM000"}
    response = client.post("/predict", json={"observation": minimal, "persist": False})
    assert response.status_code == 200
    assert response.json()["prediction"]["warnings"]  # capacité inconnue signalée


def test_modele_de_disque_inconnu_avertit_sans_echouer(client: TestClient) -> None:
    inconnu = dict(OBSERVATION, model="CONSTRUCTEUR INCONNU XYZ")
    body = client.post("/predict", json={"observation": inconnu, "persist": False}).json()
    assert body["prediction"]["predicted_rul_days"] >= 0
    assert any("inconnu" in warning for warning in body["prediction"]["warnings"])


@pytest.mark.parametrize(
    "observation",
    [
        {"model": "ST4000DM000"},  # serial absent
        {"serial_number": "   ", "model": "ST4000DM000"},  # serial vide
        dict(OBSERVATION, smart_5_raw=-5),  # compteur négatif
        dict(OBSERVATION, smart_194_raw=500),  # température absurde
        dict(OBSERVATION, date="2099-01-01"),  # date future
    ],
)
def test_entrees_invalides_renvoient_422(client: TestClient, observation: dict) -> None:
    response = client.post("/predict", json={"observation": observation, "persist": False})
    assert response.status_code == 422


def test_lot_vide_refuse(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"observations": []}).status_code == 422


def test_champ_inattendu_refuse(client: TestClient) -> None:
    """`extra = forbid` sur la requête : une faute de frappe ne passe pas en silence."""
    response = client.post("/predict", json={"observation": OBSERVATION, "persiste": False})
    assert response.status_code == 422


def test_feedback_calcule_la_rul_reelle(client: TestClient) -> None:
    body = client.post(
        "/feedback",
        json={
            "feedbacks": [
                {
                    "serial_number": "ZA00000001",
                    "observation_date": "2026-05-01",
                    "failure_date": "2026-05-15",
                },
                {
                    "serial_number": "ZA00000002",
                    "observation_date": "2026-01-01",
                    "still_alive_after_horizon": True,
                },
                {  # incohérent : panne avant l'observation
                    "serial_number": "ZA00000003",
                    "observation_date": "2026-05-01",
                    "failure_date": "2026-04-01",
                },
            ]
        },
    ).json()
    assert body["received"] == 3
    assert len(body["rejected"]) == 1
    assert "antérieure" in body["rejected"][0]["raison"]


def test_rechargement_du_modele(client: TestClient) -> None:
    """Permet de prendre en compte une promotion sans redémarrer l'API."""
    assert client.post("/admin/reload").status_code == 200


def test_sans_modele_l_api_repond_503_et_ne_plante_pas(client_sans_modele: TestClient) -> None:
    sante = client_sans_modele.get("/health")
    assert sante.status_code == 503
    body = sante.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert body["model_error"]

    prediction = client_sans_modele.post(
        "/predict", json={"observation": OBSERVATION, "persist": False}
    )
    assert prediction.status_code == 503
    assert "champion" in prediction.json()["detail"]

    # `/model` doit rester interrogeable pour diagnostiquer.
    modele = client_sans_modele.get("/model").json()
    assert modele["available"] is False


def test_racine_documente_le_service(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["documentation"] == "/docs"
    assert os.environ.get("HDD_RUL_TRACING_ENABLED") == "0"
