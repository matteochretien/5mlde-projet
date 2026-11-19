"""Flows Prefect : enchaînement nominal et arrêt sur lot invalide.

`prefect_test_harness` fournit une base Prefect temporaire : les flows tournent
réellement (tâches, retries, états) sans serveur partagé.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.ingestion.bronze import build_bronze
from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats
from hdd_rul.orchestration.deployments import build_deployments
from hdd_rul.training.train import train_and_publish
from hdd_rul.validation.contract import DataContractError


@pytest.fixture(scope="module", autouse=True)
def prefect_harness():
    with prefect_test_harness():
        yield


@pytest.fixture(scope="module")
def flow_config(tmp_path_factory: pytest.TempPathFactory) -> tuple[AppConfig, Path]:
    """Jeu synthétique, bronze et champion publié ; retourne aussi le chemin config."""
    root = tmp_path_factory.mktemp("flows")
    config_path = root / "flows.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"paths:\n  root: {root / 'data'}\n  artifacts: {root / 'artifacts'}",
                "sampling:\n  healthy_rate_pct: 70\n  healthy_stride_days: 2",
                "mlflow:"
                f"\n  tracking_uri: sqlite:///{root / 'mlflow.db'}"
                "\n  experiment_name: flows"
                "\n  registered_model_name: hdd-rul-flows",
                "synthetic:"
                "\n  n_drives: 150"
                '\n  start_date: "2025-12-01"'
                '\n  end_date: "2026-06-30"'
                "\n  daily_failure_rate: 0.003"
                "\n  seed: 31",
                "training:\n  hgb:\n    max_iter: 30\n    early_stopping: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    generate_synthetic_drive_stats(config)
    build_bronze(config)
    train_and_publish(config, trigger="test")
    return config, config_path


def test_le_lot_journalier_valide_score_et_reconcilie(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.orchestration.flows import daily_batch

    config, config_path = flow_config
    source = sorted(config.paths.raw.glob("*.csv"))[100]
    target = config.paths.landing / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())

    result = daily_batch(config_path=str(config_path), batch_path=str(target))

    assert result["status"] == "completed"
    assert result["batch"] == source.name
    assert result["rows"] > 0
    assert result["scoring"]["scored"] == result["rows"]
    assert result["scoring"]["model_version"] == "1"
    # Le lot traité quitte le dépôt d'arrivée : la reprise n'est pas ambiguë.
    assert not target.exists()
    assert (config.paths.processed / source.name).exists()


def test_un_lot_invalide_arrete_le_flow(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test bloquant au niveau de l'orchestration : rien ne tourne en aval."""
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.orchestration.flows import daily_batch

    config, config_path = flow_config
    source = sorted(config.paths.raw.glob("*.csv"))[101]
    frame = pd.read_csv(source)
    frame.loc[frame.index[:3], "failure"] = 2  # violation bloquante
    target = config.paths.landing / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)

    import hdd_rul.orchestration.flows as flows

    envoyees = []
    monkeypatch.setattr(flows, "notify", lambda alert, **_: envoyees.append(alert) or False)
    with pytest.raises(DataContractError):
        daily_batch(config_path=str(config_path), batch_path=str(target))

    # Deux alertes : le refus du lot (tâche validation), puis l'échec du flow (hook).
    refus, echec = envoyees
    assert refus.title == f"Lot {source.name} refusé par le contrat de données"
    assert "expect_column_values_to_be_in_set (failure)" in refus.sections["Diagnostic"]
    assert echec.status["Étape"] == "validation"  # lue dans l'API Prefect
    assert echec.title.startswith("Échec du flow daily_batch : Failed (étape validation)")
    assert echec.sections["Diagnostic"]["Nature"].startswith("donnée invalide")
    assert echec.sections["Diagnostic"]["Cause"].startswith("DataContractError")
    # Lien direct vers le flow run concerné, le même dans les deux alertes.
    lien = dict(echec.links)["Ouvrir le flow run dans Prefect"]
    assert lien.startswith("http://localhost:4200/runs/flow-run/")
    assert dict(refus.links)["Ouvrir le flow run dans Prefect"] == lien
    assert "Rejouer le lot" in echec.commands

    # Le lot refusé reste en place : on peut le corriger et le rejouer.
    assert target.exists()
    assert not (config.paths.processed / source.name).exists()
    # Aucun fichier propre n'a été produit : le prétraitement n'a pas tourne.
    assert not (config.paths.bronze / "landing_clean" / f"{source.stem}.parquet").exists()


def test_un_disque_deja_en_panne_qui_reapparait_est_ecarte(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Règle de non-résurrection active en production : le flow calcule lui-même
    la liste des disques déjà en panne (avant, il la laissait vide)."""
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.ingestion.landing import batch_date_from_name, known_failed_serials_before
    from hdd_rul.orchestration.flows import daily_batch

    config, config_path = flow_config
    source = sorted(config.paths.raw.glob("*.csv"))[110]
    deja_en_panne = known_failed_serials_before(config, batch_date_from_name(source))
    assert deja_en_panne, "le jeu synthétique doit contenir des pannes antérieures"

    frame = pd.read_csv(source)
    revenant = frame.iloc[[0]].copy()
    revenant["serial_number"] = deja_en_panne[0]
    target = config.paths.landing / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame, revenant], ignore_index=True).to_csv(target, index=False)

    result = daily_batch(config_path=str(config_path), batch_path=str(target))

    assert result["status"] == "completed"
    propre = pd.read_parquet(config.paths.bronze / "landing_clean" / f"{source.stem}.parquet")
    assert deja_en_panne[0] not in set(propre["serial_number"])
    assert (config.paths.quarantine / f"{source.stem}.parquet").exists()


def test_flow_sans_lot_en_attente(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.orchestration.flows import daily_batch

    _, config_path = flow_config
    config = load_config(config_path)
    for leftover in config.paths.landing.glob("*.csv"):
        leftover.unlink()

    result = daily_batch(config_path=str(config_path))
    assert result["status"] == "idle"


def test_entrainement_orchestre_en_parallele(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.orchestration.flows import train_pipeline

    _, config_path = flow_config
    outcome = train_pipeline(config_path=str(config_path), trigger="test", as_of="2026-05-15")

    assert set(outcome["models"]) == {"baseline_median", "hist_gradient_boosting"}
    assert outcome["decision"] in {"promoted", "rejected"}
    assert outcome["registered_version"] == "2"
    assert outcome["data_window"]["as_of"] == "2026-05-15"


def test_les_deployments_sont_declares(flow_config: tuple[AppConfig, Path]) -> None:
    """Déclenchement automatique : scrutation du dépôt + planification du training."""
    _, config_path = flow_config
    deployments = build_deployments(str(config_path), batch_interval=30, train_cron="17 2 * * *")

    noms = {deployment.name for deployment in deployments}
    assert noms == {
        "daily-batch-landing",
        "train-pipeline-nightly",
        "ingest-quarter-manual",
    }

    # Jamais deux traitements du même lot en parallèle (OOM constaté sur lot réel).
    par_nom = {deployment.name: deployment for deployment in deployments}
    for nom in ("daily-batch-landing", "train-pipeline-nightly"):
        limite = par_nom[nom].concurrency_limit
        assert limite is not None and getattr(limite, "limit", limite) == 1


def test_l_entrainement_planifie_se_place_au_dernier_lot_recu(
    flow_config: tuple[AppConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sans `as_of`, le flow ne doit pas s'entraîner sur l'avenir de la simulation."""
    monkeypatch.setenv("HDD_RUL_TRACING_ENABLED", "0")
    from hdd_rul.ingestion.landing import latest_processed_day
    from hdd_rul.orchestration.flows import train_pipeline

    config, config_path = flow_config
    dernier = latest_processed_day(config)
    assert dernier is not None, "les tests de lot journalier doivent avoir traité des lots"

    outcome = train_pipeline(config_path=str(config_path), trigger="schedule")
    assert outcome["data_window"]["as_of"] == dernier.isoformat()
