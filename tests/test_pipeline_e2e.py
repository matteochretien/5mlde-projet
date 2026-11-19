"""Intégration de bout en bout : données -> bronze -> entraînement -> registre -> API.

Ce test est la preuve automatisée que la chaîne circule : il n'utilise aucun
service externe (MLflow en SQLite temporaire, traces désactivées), et vérifie
que les artefacts et les alias attendus existent réellement.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.ingestion.bronze import build_bronze
from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats
from hdd_rul.serving.predictor import load_champion, predict_frame
from hdd_rul.training.models import BASELINE_NAME, CANDIDATE_NAME
from hdd_rul.training.registry import (
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS,
    get_client,
    get_version_by_alias,
    rollback_champion,
)
from hdd_rul.training.train import train_and_publish


@pytest.fixture(scope="module")
def pipeline_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    """Jeu synthétique + couche bronze, construits une fois pour le module."""
    root = tmp_path_factory.mktemp("e2e")
    config = load_config(
        overrides={
            "paths": {"root": str(root / "data"), "artifacts": str(root / "artifacts")},
            "sampling": {"healthy_rate_pct": 70, "healthy_stride_days": 2},
            "mlflow": {
                "tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
                "experiment_name": "e2e",
                "registered_model_name": "hdd-rul-e2e",
            },
            "synthetic": {
                "n_drives": 220,
                "start_date": "2025-12-01",
                "end_date": "2026-06-30",
                "daily_failure_rate": 0.0025,
                "seed": 11,
            },
            "training": {"hgb": {"max_iter": 40, "early_stopping": False}},
        }
    )
    generate_synthetic_drive_stats(config)
    build_bronze(config)
    return config


def test_le_premier_entrainement_publie_un_champion(pipeline_config: AppConfig) -> None:
    outcome = train_and_publish(pipeline_config, trigger="test")

    assert outcome.decision == "promoted"
    assert outcome.reason == "no_champion_yet"
    assert outcome.registered_version == "1"
    assert outcome.aliases[ALIAS_CHAMPION] == "1"
    assert set(outcome.models) == {BASELINE_NAME, CANDIDATE_NAME}

    # Le modèle doit faire nettement mieux que la médiane : sinon, quelque chose
    # ne circule pas correctement en amont (features vides, labels décalés...).
    baseline = outcome.models[BASELINE_NAME]["metrics"]["mae"]
    candidate = outcome.models[CANDIDATE_NAME]["metrics"]["mae"]
    assert candidate < baseline

    assert outcome.data_window["window_start"] < outcome.data_window["window_end"]


def test_le_champion_est_servable_et_predit_dans_l_horizon(pipeline_config: AppConfig) -> None:
    import pandas as pd

    model = load_champion(pipeline_config)
    assert model.alias == ALIAS_CHAMPION
    assert model.version == "1"

    observations = pd.DataFrame(
        [
            {  # disque très dégradé : RUL courte attendue
                "serial_number": "ZA00000001",
                "model": "ST4000DM000",
                "capacity_bytes": 4_000_787_030_016,
                "smart_5_raw": 400,
                "smart_9_raw": 50_000,
                "smart_12_raw": 50,
                "smart_187_raw": 140,
                "smart_188_raw": 28,
                "smart_193_raw": 80_000,
                "smart_194_raw": 41,
                "smart_197_raw": 85,
                "smart_198_raw": 50,
                "smart_199_raw": 11,
            },
            {  # disque sain
                "serial_number": "ZA00000002",
                "model": "ST4000DM000",
                "capacity_bytes": 4_000_787_030_016,
                "smart_5_raw": 0,
                "smart_9_raw": 9_000,
                "smart_12_raw": 8,
                "smart_187_raw": 0,
                "smart_188_raw": 0,
                "smart_193_raw": 14_000,
                "smart_194_raw": 29,
                "smart_197_raw": 0,
                "smart_198_raw": 0,
                "smart_199_raw": 0,
            },
        ]
    )
    scored = predict_frame(model, observations, pipeline_config)
    assert scored["predicted_rul"].between(0, pipeline_config.horizon_days).all()
    # Le modèle doit au minimum ordonner correctement les deux cas.
    assert scored.loc[0, "predicted_rul"] < scored.loc[1, "predicted_rul"]


def test_un_second_entrainement_applique_la_regle_de_promotion(
    pipeline_config: AppConfig,
) -> None:
    """Le challenger n'est promu que s'il améliore la zone critique."""
    outcome = train_and_publish(pipeline_config, trigger="test", as_of=date(2026, 5, 1))

    assert outcome.registered_version == "2"
    assert outcome.decision in {"promoted", "rejected"}
    if outcome.decision == "rejected":
        assert outcome.reason.startswith(("mae_lt30_gain_insuffisant", "mae_globale_degradee"))
    client = get_client(pipeline_config)
    name = pipeline_config.mlflow.registered_model_name
    version = client.get_model_version(name, "2")
    assert version.tags["promotion.decision"] == outcome.decision
    assert version.tags["promotion.reason"] == outcome.reason
    # Le champion a été note sur la validation du challenger, pas sur la sienne.
    assert version.tags["promotion.champion_eval"] == "same_validation"
    assert version.tags["promotion.champion_version"] == "1"
    assert float(version.tags["promotion.champion_mae_lt30"]) >= 0


def test_le_champion_est_reevalue_sur_la_validation_du_challenger(
    pipeline_config: AppConfig,
) -> None:
    """Régression : la règle comparait le challenger aux métriques *stockées* du
    champion, calculées sur un autre jeu de validation."""
    import mlflow

    from hdd_rul.features.build import select_feature_frame
    from hdd_rul.features.labels import LABEL_COLUMN
    from hdd_rul.training.evaluate import evaluate_predictions
    from hdd_rul.training.registry import configure_mlflow, version_metrics
    from hdd_rul.training.train import champion_metrics_on, prepare_training_data

    configure_mlflow(pipeline_config)
    client = get_client(pipeline_config)
    name = pipeline_config.mlflow.registered_model_name
    champion = get_version_by_alias(client, name, ALIAS_CHAMPION)
    split, _ = prepare_training_data(pipeline_config, as_of=date(2026, 4, 1))

    metrics, source = champion_metrics_on(pipeline_config, client, champion, split.valid)
    assert source == "same_validation"
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS_CHAMPION}")
    attendu = evaluate_predictions(
        split.valid[LABEL_COLUMN],
        pipeline.predict(select_feature_frame(split.valid, pipeline_config)),
        alert_threshold_days=pipeline_config.split.alert_threshold_days,
    )
    assert metrics["mae"] == pytest.approx(attendu.mae)
    assert metrics["mae_lt30"] == pytest.approx(attendu.mae_lt30, nan_ok=True)

    # Sans jeu de validation : repli explicite sur les métriques stockées.
    stockees, source = champion_metrics_on(pipeline_config, client, champion, None)
    assert source == "stored_metrics"
    assert stockees == version_metrics(client, champion)
    assert champion_metrics_on(pipeline_config, client, None, split.valid) == ({}, "none")


def test_les_alias_du_registre_sont_recopies_dans_les_traces(
    pipeline_config: AppConfig,
) -> None:
    """La vue croisée de Grafana lit `tracing.model_versions` : elle doit refléter
    l'état réel des alias, quel que soit le chemin d'entraînement (ici la CLI)."""
    from hdd_rul.training.registry import sync_model_versions

    class StoreEnMemoire:
        def __init__(self) -> None:
            self.lignes: dict[tuple[str, str], dict] = {
                ("hdd-rul-e2e", "challenger"): {"version": "99"}
            }

        def record_model_version(self, *, model_name, version, alias, run_id, metrics):
            self.lignes[(model_name, alias)] = {"version": version, "metrics": metrics}

        def prune_alias(self, *, model_name, alias, keep_version):
            ligne = self.lignes.get((model_name, alias))
            if ligne and (keep_version is None or ligne["version"] != keep_version):
                del self.lignes[(model_name, alias)]

    store = StoreEnMemoire()
    client = get_client(pipeline_config)
    name = pipeline_config.mlflow.registered_model_name
    attendu = {
        alias: str(get_version_by_alias(client, name, alias).version)
        for alias in ("champion", "challenger", "previous")
        if get_version_by_alias(client, name, alias) is not None
    }

    assert sync_model_versions(pipeline_config, store=store) == attendu
    assert {alias: ligne["version"] for (_, alias), ligne in store.lignes.items()} == attendu
    # Les métriques de validation de chaque version accompagnent la ligne.
    assert "mae_lt30" in store.lignes[(name, "champion")]["metrics"]


def test_les_artefacts_de_tracabilite_sont_produits(pipeline_config: AppConfig) -> None:
    """Chaque run doit porter de quoi reconstituer l'entraînement."""
    client = get_client(pipeline_config)
    champion = get_version_by_alias(
        client, pipeline_config.mlflow.registered_model_name, ALIAS_CHAMPION
    )
    run = client.get_run(champion.run_id)

    for tag in ("data.window_start", "data.sha256", "git.commit", "trigger", "training.group_id"):
        assert tag in run.data.tags, f"tag manquant: {tag}"
    for metrique in ("mae", "mae_lt30", "alert_recall"):
        assert metrique in run.data.metrics

    artefacts = {item.path for item in client.list_artifacts(run.info.run_id, "rapports")}
    assert "rapports/features.json" in artefacts
    assert "rapports/data_manifest.json" in artefacts
    assert "rapports/residus_rul.png" in artefacts

    summary = Path(pipeline_config.paths.artifacts) / "training" / "derniere_execution.json"
    assert summary.exists()


def test_retour_arriere(pipeline_config: AppConfig) -> None:
    """`previous` doit permettre de revenir en arrière sans chercher un numéro."""
    client = get_client(pipeline_config)
    name = pipeline_config.mlflow.registered_model_name
    avant = get_version_by_alias(client, name, ALIAS_CHAMPION)

    # Si le challenger a été rejeté, il n'y a pas de `previous` : on force une
    # promotion pour pouvoir éprouver le retour arrière.
    if get_version_by_alias(client, name, ALIAS_PREVIOUS) is None:
        client.set_registered_model_alias(name, ALIAS_CHAMPION, "2")
        client.set_registered_model_alias(name, ALIAS_PREVIOUS, avant.version)

    result = rollback_champion(client, name)
    apres = get_version_by_alias(client, name, ALIAS_CHAMPION)
    assert str(apres.version) == result[ALIAS_CHAMPION]
    assert str(apres.version) != str(get_version_by_alias(client, name, ALIAS_PREVIOUS).version)
