"""Contenu des alertes : diagnostic, références et liens directs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from hdd_rul.common.config import AlertsConfig, AppConfig
from hdd_rul.common.notifications import build_payload, render_text
from hdd_rul.orchestration.alerts import (
    batch_refused_alert,
    classify_failure,
    flow_failure_alert,
    sample_alert,
    training_outcome_alert,
)
from hdd_rul.validation.contract import ExpectationOutcome, ValidationReport


def _etat_en_echec(message: str, nom: str = "Failed") -> SimpleNamespace:
    def result(raise_on_failure: bool = True):
        raise RuntimeError("résultat non persisté")

    return SimpleNamespace(
        name=nom, type=SimpleNamespace(value=nom.upper()), message=message, result=result
    )


def _flow_run(**parameters) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name="brave-otter",
        parameters=parameters,
        run_count=1,
        start_time=datetime.now(UTC) - timedelta(seconds=75),
        deployment_id=None,
    )


def test_echec_de_flow_dependance_indisponible() -> None:
    run = _flow_run(trigger="schedule")
    state = _etat_en_echec(
        "Flow run encountered an exception: MlflowException: API request failed "
        "(Connection refused)"
    )
    contexte = {"failed_tasks": ["training", "training"], "task_errors": [], "deployment": "train"}
    alert = flow_failure_alert("train_pipeline", run, state, AlertsConfig(), contexte)

    assert alert.title == "Échec du flow train_pipeline : Failed (étape training)"
    assert alert.level == "critical"
    diagnostic = alert.sections["Diagnostic"]
    assert diagnostic["Nature"].startswith("dépendance indisponible")
    assert diagnostic["Cause"].startswith("MlflowException: API request failed")
    assert "champion en service est inchangé" in diagnostic["Conséquence"]
    references = alert.sections["Références"]
    assert references["Flow run"] == f"brave-otter ({run.id})"
    assert references["Lancement"] == "déploiement train, déclencheur schedule"
    assert references["Durée"].startswith("1 min")
    assert dict(alert.links)["Ouvrir le flow run dans Prefect"] == (
        f"http://localhost:4200/runs/flow-run/{run.id}"
    )
    assert alert.commands["Logs du flow run"] == f"uv run prefect flow-run logs {run.id}"
    assert alert.note  # liens locaux : le destinataire est prévenu


def test_echec_sans_api_prefect_reste_exploitable() -> None:
    """Sans contexte Prefect (API injoignable), l'alerte part quand même."""
    alert = flow_failure_alert(
        "daily_batch", _flow_run(), _etat_en_echec("boom", nom="Crashed"), AlertsConfig()
    )
    assert alert.title == "Échec du flow daily_batch : Crashed"
    assert alert.status["Étape"] == "inconnue"
    assert alert.sections["Diagnostic"]["Nature"].startswith("arrêt brutal")


def test_classification_des_causes() -> None:
    assert classify_failure("DataContractError", "")[0].startswith("donnée invalide")
    assert classify_failure("OperationalError", "")[0].startswith("dépendance")
    assert classify_failure("RuntimeError", "Max retries exceeded")[0].startswith("dépendance")
    assert classify_failure("KeyError", "'mae'")[0].startswith("erreur technique")


def test_lot_refuse_detaille_les_regles_violees(tmp_path) -> None:
    report = ValidationReport(
        batch_name="2026-04-07.csv",
        batch_date="2026-04-07",
        total_rows=1200,
        quarantined_rows=0,
        outcomes=[
            ExpectationOutcome(
                "expect_column_values_to_be_in_set",
                "failure",
                "blocking",
                False,
                unexpected_count=3,
                unexpected_sample=[2, 2, 2],
            ),
            ExpectationOutcome("expect_column_to_exist", "smart_197_raw", "blocking", False),
        ],
        validated_at="2026-04-07T00:00:00Z",
        report_path=tmp_path / "rapport.json",
    )
    alert = batch_refused_alert(report, AlertsConfig(), flow_run_id="abc", flow_run_name="run")
    diagnostic = alert.sections["Diagnostic"]
    assert diagnostic["Nature"] == "donnée invalide : 2 règle(s) bloquante(s) violée(s)"
    assert diagnostic["expect_column_values_to_be_in_set (failure)"] == (
        "3 ligne(s) en défaut, ex. 2, 2, 2"
    )
    assert alert.status["Lot"] == "2026-04-07.csv"
    assert alert.sections["Références"]["Rapport de validation"].endswith("rapport.json")
    assert alert.commands["Rejouer le lot"].endswith("--batch data/landing/2026-04-07.csv")


def test_fin_d_entrainement_renvoie_vers_mlflow() -> None:
    outcome = SimpleNamespace(
        group_id="0123456789abcdef",
        decision="rejected",
        reason="gain MAE<30j insuffisant (0.01 < 0.05)",
        registered_version="7",
        aliases={"challenger": "7"},
        data_window={"window_start": "2026-01-01", "window_end": "2026-03-31"},
        models={
            "hgb": {
                "run_id": "r-hgb",
                "metrics": {"mae": 12.3, "mae_lt30": 8.1, "alert_recall": 0.06},
            },
            "baseline": {"run_id": "r-base", "metrics": {"mae": 20.0}},
        },
    )
    alert = training_outcome_alert(
        outcome, AppConfig(), candidate_name="hgb", baseline_name="baseline", trigger="replay"
    )
    assert alert.title == "Entraînement terminé : version 7 non promue"
    assert alert.level == "warning"
    assert alert.sections["Résultat"]["MAE sur RUL < 30 j"] == "8.10 j"
    liens = dict(alert.links)
    assert liens["Version 7 dans le registre MLflow"] == (
        "http://localhost:5050/#/models/hdd-rul-hgb/versions/7"
    )
    assert liens["Run MLflow du candidat"] == "http://localhost:5050/#/runs/r-hgb"


def test_liens_surcharges_par_l_environnement(monkeypatch) -> None:
    """Dans les conteneurs, MLflow est `mlflow:5000` : le lien doit rester celui du navigateur."""
    monkeypatch.setenv("HDD_RUL_MLFLOW_UI_URL", "https://mlflow.exemple.test/")
    monkeypatch.setenv("HDD_RUL_ALERT_ENV", "pile Compose")
    alerts = AlertsConfig()
    assert alerts.links.mlflow_run("x") == "https://mlflow.exemple.test/#/runs/x"
    assert alerts.resolved_environment() == "pile Compose"
    assert not alerts.links.all_local()


def test_carte_teams_complete_et_serialisable() -> None:
    alerts = AlertsConfig()
    payload = build_payload(sample_alert(alerts), alerts)
    card = payload["attachments"][0]["content"]
    assert card["msteams"] == {"width": "Full"}
    textes = json.dumps(card, ensure_ascii=False)
    for attendu in ("Diagnostic", "Références", "Accès", "[Tableau de bord Grafana](http"):
        assert attendu in textes
    assert payload["potentialAction"][0]["@type"] == "OpenUri"
    assert "Statut : Failed | Étape : validation" in render_text(sample_alert(alerts), alerts)


def test_l_enveloppe_failedrun_de_prefect_est_ignoree() -> None:
    """Une tâche en échec renvoie `FailedRun` : la cause utile est dans le message."""
    from prefect.exceptions import FailedRun

    from hdd_rul.orchestration.alerts import failure_exception

    state = SimpleNamespace(
        message="Task run encountered an exception MlflowException: API request failed",
        result=lambda raise_on_failure=True: FailedRun("enveloppe"),
    )
    assert failure_exception(state) == ("MlflowException", "API request failed")
