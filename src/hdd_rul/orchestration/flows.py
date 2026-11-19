"""Flows Prefect 3 : ingestion, lot journalier, entraînement.

Trois flows, trois déclencheurs :

* `ingest_quarter`  — téléchargement et construction de la couche bronze. Rejoué
  sans effet de bord grâce au cache (clé construite sur l'empreinte de l'entrée).
* `daily_batch`     — déclenché par l'arrivée d'un fichier dans `data/landing/`.
  Valide, met en quarantaine, score et réconcilié les feedbacks.
* `train_pipeline`  — planifié, ou déclenché à la main / par le rejeu. Entraîne
  les deux candidats **en parallèle**, évalue et publie.

Nommage des tâches reprenant celui du TP : `ingestion`, `validation`,
`preprocessing`, `training`, `evaluation-publication`, avec les sous-tâches
`steam-http-get` (rattachée à l'ingestion) et `feature-table` (rattachée au
prétraitement).

Pas de sous-flow : un lot repris doit apparaître dans **un seul** flow run, avec
tout son historique, plutôt que dans deux runs sans lien dans l'interface.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import flow_run

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.common.notifications import notify
from hdd_rul.common.tracing import get_store
from hdd_rul.ingestion.bronze import build_bronze
from hdd_rul.ingestion.download import download_quarter, extract_quarter
from hdd_rul.ingestion.landing import (
    batch_date_from_name,
    claim_batch,
    known_failed_serials_before,
    latest_processed_day,
    mark_processed,
)
from hdd_rul.orchestration.alerts import (
    batch_refused_alert,
    flow_failure_alert,
    prefect_failure_context,
    training_outcome_alert,
    training_window_refused_alert,
)
from hdd_rul.serving.feedback import collect_batch_feedbacks
from hdd_rul.serving.predictor import ModelUnavailableError, PredictorCache, predict_frame
from hdd_rul.training.models import BASELINE_NAME, CANDIDATE_NAME
from hdd_rul.training.train import (
    TrainedModel,
    TrainingContext,
    finalize_training,
    prepare_training_data,
    train_single_model,
)
from hdd_rul.validation.contract import (
    DataContractError,
    ValidationReport,
    quarantine_rows,
    validate_batch,
)

CACHE = INPUTS + TASK_SOURCE  # le code de la tâche fait partie de la clé de cache
RETRY_DELAYS = [2, 4, 8, 16]  # MLflow ou PostgreSQL momentanément indisponible


# ---------------------------------------------------------------------------
# Hooks d'alerte
# ---------------------------------------------------------------------------
def alert_on_failure(flow_object, run, state) -> None:
    """Hook `on_failure` : une exécution échouée ne passe jamais inaperçue."""
    config = load_config((getattr(run, "parameters", None) or {}).get("config_path"))
    alert = flow_failure_alert(
        flow_object.name, run, state, config.alerts, prefect_failure_context(run)
    )
    notify(alert, alerts=config.alerts)
    get_store().record_pipeline_run(
        flow_name=flow_object.name,
        flow_run_id=str(run.id),
        status="failed",
        trigger="hook",
        details={"message": state.message or ""},
    )


def _current_flow_run_id() -> str | None:
    try:
        return str(flow_run.id)
    except Exception:
        return None


def _current_flow_run_name() -> str | None:
    try:
        return flow_run.name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------
@task(name="steam-http-get", retries=3, retry_delay_seconds=RETRY_DELAYS, cache_policy=CACHE)
def fetch_source(config_dict: dict[str, Any], quarter: str) -> str:
    """Sous-tâche d'ingestion : téléchargement cache d'une archive trimestrielle."""
    config = AppConfig.model_validate(config_dict)
    archive = download_quarter(config, quarter)
    return str(archive)


@task(name="ingestion", retries=2, retry_delay_seconds=RETRY_DELAYS)
def load_data(config_dict: dict[str, Any], archive: str) -> int:
    """Décompresse les CSV journaliers du trimestre dans `data/raw/`."""
    config = AppConfig.model_validate(config_dict)
    files = extract_quarter(config, Path(archive))
    return len(files)


@task(name="validation", retries=0)
def validate_data(
    config_dict: dict[str, Any],
    batch_path: str,
    known_failed_serials: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Contrat de données. Lève `DataContractError` si une règle bloquante échoue.

    C'est **le** point d'arrêt du pipeline : aucune étape suivante ne tourne si
    le lot est refusé, et l'alerte part avant l'arrêt.
    """
    logger = get_run_logger()
    config = AppConfig.model_validate(config_dict)
    path = Path(batch_path)
    frame = pd.read_csv(path, low_memory=False)
    report = validate_batch(
        config,
        frame,
        batch_name=path.name,
        expected_date=batch_date_from_name(path),
        known_failed_serials=known_failed_serials,
    )
    if not report.passed:
        for outcome in report.blocking_failures:
            logger.error("règle violée sur %s : %s", path.name, outcome.describe(report.total_rows))
        alert = batch_refused_alert(
            report,
            config.alerts,
            flow_run_id=_current_flow_run_id(),
            flow_run_name=_current_flow_run_name(),
        )
        notify(alert, alerts=config.alerts)
        get_store().record_pipeline_run(
            flow_name="daily_batch",
            flow_run_id=_current_flow_run_id(),
            status="blocked_by_validation",
            trigger="landing",
            batch_name=path.name,
            details=report.to_dict(),
        )
        raise DataContractError(report)
    for outcome in report.warnings:
        logger.warning(
            "avertissement de qualité sur %s : %s", path.name, outcome.describe(report.total_rows)
        )
    if report.warnings:
        logger.info("lot accepté ; rapport complet : %s", report.report_path)
    return report.to_dict() | {"report_path": str(report.report_path or "")}


@task(name="preprocessing", retries=1, retry_delay_seconds=RETRY_DELAYS)
def preprocess_data(
    config_dict: dict[str, Any],
    batch_path: str,
    known_failed_serials: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Quarantaine des lignes non conformes puis mise à disposition du lot propre."""
    config = AppConfig.model_validate(config_dict)
    path = Path(batch_path)
    frame = pd.read_csv(path, low_memory=False)
    clean, quarantined = quarantine_rows(config, frame, path.name, known_failed_serials)
    if quarantined:
        resurrected = (
            int(frame["serial_number"].isin(set(known_failed_serials)).sum())
            if known_failed_serials and "serial_number" in frame.columns
            else 0
        )
        reasons = []
        if quarantined - resurrected > 0:
            reasons.append(f"{quarantined - resurrected} capacity_bytes <= 0 ou non numérique")
        if resurrected:
            reasons.append(f"{resurrected} disque(s) déjà en panne qui réapparaissent")
        get_run_logger().info(
            "%d ligne(s) de %s en quarantaine : %s", quarantined, path.name, " ; ".join(reasons)
        )
        get_store().record_quarantine(
            batch_name=path.name,
            batch_date=str(batch_date_from_name(path) or ""),
            rows=quarantined,
            reason=" ; ".join(reasons),
        )
    staged = config.paths.bronze / "landing_clean" / f"{path.stem}.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(staged, index=False)
    return {"rows": len(clean), "quarantined": quarantined, "path": str(staged)}


@task(name="feature-table", retries=1, retry_delay_seconds=RETRY_DELAYS)
def build_feature_table(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Sous-tâche de prétraitement : reconstruit la couche bronze d'entraînement."""
    config = AppConfig.model_validate(config_dict)
    manifest = build_bronze(config)
    return {"rows": manifest.rows_total, "fingerprint": manifest.source_fingerprint}


@task(name="score-batch", retries=2, retry_delay_seconds=RETRY_DELAYS)
def score_batch(config_dict: dict[str, Any], staged_path: str) -> dict[str, Any]:
    """Prédiction par lot avec le champion, puis persistance des traces."""
    import uuid

    logger = get_run_logger()
    config = AppConfig.model_validate(config_dict)
    frame = pd.read_parquet(staged_path)
    cache = PredictorCache(config)
    try:
        model = cache.get()
    except ModelUnavailableError as exc:
        # Pas de champion : le lot est quand même valide et archive, mais rien
        # n'est score. On ne fait pas échouer le lot pour autant.
        logger.warning("scoring ignoré: %s", exc)
        return {"scored": 0, "reason": "modèle indisponible"}

    scored = predict_frame(model, frame, config)
    rows = [
        {
            "prediction_id": str(uuid.uuid4()),
            "observation_date": row.get("date"),
            "serial_number": row.get("serial_number"),
            "model": row.get("model"),
            "capacity_bytes": row.get("capacity_bytes"),
            "features": {
                column: row.get(column)
                for column in config.features.smart_columns
                if column in scored.columns
            },
            "predicted_rul": float(row["predicted_rul"]),
            "alert": bool(row["alert"]),
            "model_name": model.model_name,
            "model_version": model.version,
            "run_id": model.run_id,
            "source": "batch",
        }
        for row in scored.to_dict(orient="records")
    ]
    written = get_store().record_predictions(rows)
    at_risk = int(scored["alert"].sum())
    logger.info("%d lignes scorées, %d disques à risque (<30 j)", len(scored), at_risk)
    return {
        "scored": len(scored),
        "persisted": written,
        "at_risk": at_risk,
        "model_version": model.version,
    }


@task(name="feedbacks", retries=2, retry_delay_seconds=RETRY_DELAYS)
def collect_feedbacks(config_dict: dict[str, Any], staged_path: str, batch_day: str) -> int:
    """Réconcilié les prédictions passées avec ce que le lot du jour révèle."""
    config = AppConfig.model_validate(config_dict)
    frame = pd.read_parquet(staged_path)
    return collect_batch_feedbacks(
        get_store(), frame, date.fromisoformat(batch_day), config.horizon_days
    )


@task(name="training", retries=1, retry_delay_seconds=RETRY_DELAYS)
def train_models(
    config_dict: dict[str, Any],
    model_name: str,
    split_payload: dict[str, Any],
    context_payload: dict[str, Any],
) -> dict[str, Any]:
    """Entraîne un candidat. Deux instances de cette tâche tournent en parallèle."""
    from hdd_rul.training.split import SplitResult

    config = AppConfig.model_validate(config_dict)
    split = SplitResult(
        train=pd.read_parquet(split_payload["train"]),
        valid=pd.read_parquet(split_payload["valid"]),
        cutoff_date=split_payload["cutoff_date"],
        valid_drives=split_payload["valid_drives"],
        train_drives=split_payload["train_drives"],
        rows_dropped_leakage=split_payload["rows_dropped_leakage"],
    )
    context = TrainingContext(**context_payload)
    trained = train_single_model(model_name, config, split, context)
    return trained.as_dict() | {"model_uri": trained.model_uri}


@task(name="evaluation-publication", retries=2, retry_delay_seconds=RETRY_DELAYS)
def evaluate_models(
    config_dict: dict[str, Any],
    trained_payloads: list[dict[str, Any]],
    context_payload: dict[str, Any],
    valid_path: str | None = None,
) -> dict[str, Any]:
    """Compare les candidats, applique la règle de mise en service et publie.

    Le champion est réévalué sur le jeu de validation des candidats (`valid_path`).
    """
    from hdd_rul.training.evaluate import EvaluationMetrics

    config = AppConfig.model_validate(config_dict)
    context = TrainingContext(**context_payload)
    trained = {
        payload["name"]: TrainedModel(
            name=payload["name"],
            run_id=payload["run_id"],
            model_uri=payload["model_uri"],
            metrics=EvaluationMetrics(**payload["metrics"]),
            pipeline=None,  # le pipeline reste dans MLflow : inutile de le rapatrier
        )
        for payload in trained_payloads
    }
    valid = pd.read_parquet(valid_path) if valid_path else None
    # finalize_training synchronise aussi les alias vers les traces (vue croisée).
    outcome = finalize_training(config, trained, context, valid=valid)  # sauvegarde le résumé
    alert = training_outcome_alert(
        outcome,
        config,
        candidate_name=CANDIDATE_NAME,
        baseline_name=BASELINE_NAME,
        trigger=context.trigger,
        flow_run_id=context.flow_run_id or _current_flow_run_id(),
        flow_run_name=_current_flow_run_name(),
    )
    notify(alert, alerts=config.alerts)
    return outcome.as_dict()


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
@flow(name="ingest_quarter", on_failure=[alert_on_failure], log_prints=True)
def ingest_quarter(quarter: str, config_path: str | None = None) -> dict[str, Any]:
    """Télécharge un trimestre, le décompresse et reconstruit la couche bronze."""
    config = load_config(config_path)
    config.paths.ensure()
    payload = config.model_dump(mode="json")

    archive = fetch_source(payload, quarter)
    extracted = load_data(payload, archive)
    table = build_feature_table(payload)

    result = {"quarter": quarter, "extracted_files": extracted, **table}
    get_store().record_pipeline_run(
        flow_name="ingest_quarter",
        flow_run_id=_current_flow_run_id(),
        status="completed",
        trigger="manual",
        details=result,
    )
    return result


@flow(name="daily_batch", on_failure=[alert_on_failure], log_prints=True)
def daily_batch(
    config_path: str | None = None,
    batch_path: str | None = None,
    *,
    known_failed_serials: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Traite un lot journalier déposé dans `data/landing/`.

    Enchaînement : validation (bloquante) -> quarantaine -> scoring -> feedbacks.
    Le lot n'est déplacé dans `processed/` qu'en cas de succès, ce qui rend la
    reprise possible après correction.
    """
    logger = get_run_logger()
    config = load_config(config_path)
    config.paths.ensure()
    payload = config.model_dump(mode="json")

    target = Path(batch_path) if batch_path else claim_batch(config)
    if target is None:
        logger.info("aucun lot en attente dans %s", config.paths.landing)
        return {"status": "idle", "batch": None}

    batch_day = batch_date_from_name(target) or date.today()
    if not known_failed_serials:
        # Règle de non-résurrection : disques déjà en panne avant ce lot.
        known_failed_serials = known_failed_serials_before(config, batch_day)
        logger.info("%d disque(s) déjà en panne avant le %s", len(known_failed_serials), batch_day)
    report = validate_data(payload, str(target), known_failed_serials)
    staged = preprocess_data(payload, str(target), known_failed_serials)
    scoring = score_batch(payload, staged["path"])
    feedbacks = collect_feedbacks(payload, staged["path"], batch_day.isoformat())

    if target.parent == config.paths.landing:
        mark_processed(config, target)

    result = {
        "status": "completed",
        "batch": target.name,
        "batch_date": batch_day.isoformat(),
        "rows": staged["rows"],
        "quarantined": staged["quarantined"],
        "scoring": scoring,
        "feedbacks": feedbacks,
        "validation_warnings": len(report.get("warnings", [])),
    }
    get_store().record_pipeline_run(
        flow_name="daily_batch",
        flow_run_id=_current_flow_run_id(),
        status="completed",
        trigger="landing",
        batch_name=target.name,
        details=result,
    )
    return result


@flow(name="train_pipeline", on_failure=[alert_on_failure], log_prints=True)
def train_pipeline(
    config_path: str | None = None,
    trigger: str = "schedule",
    as_of: str | None = None,
    window_days: int | None = None,
    validate_window: bool = True,
) -> dict[str, Any]:
    """Entraîne, évalue et publie. Les deux candidats tournent en parallèle.

    `as_of` simule la date du jour : seules les lignes dont le label était déjà
    résolu à cette date entrent dans l'entraînement. C'est ce qui fait varier la
    période d'entraînement d'un run à l'autre sur un jeu historique figé.
    """
    logger = get_run_logger()
    config = load_config(config_path)
    config.paths.ensure()
    payload = config.model_dump(mode="json")

    as_of_date = date.fromisoformat(as_of) if as_of else None
    if as_of_date is None:
        # Entraînement planifié : sans date explicite, on se place au dernier
        # lot reçu. Sinon, sur un jeu historique figé, le modèle apprendrait sur
        # l'avenir de la simulation (tout le bronze, jusqu'au 30 juin).
        as_of_date = latest_processed_day(config)
        if as_of_date is not None:
            logger.info("as_of non fourni : dernier lot traité, %s", as_of_date)
    split, context = prepare_training_data(config, as_of=as_of_date, window_days=window_days)
    context.trigger = trigger
    context.flow_run_id = _current_flow_run_id()

    # Contrôle qualité du jeu d'entraînement lui-même, avant tout apprentissage.
    if validate_window:
        report = validate_training_window(config, split)
        context.validation_passed = report.passed
        context.validation_report_path = report.report_path
        if not report.passed:
            alert = training_window_refused_alert(
                report,
                config.alerts,
                flow_run_id=context.flow_run_id,
                flow_run_name=_current_flow_run_name(),
                trigger=trigger,
            )
            notify(alert, alerts=config.alerts)
            raise DataContractError(report)

    staging = config.paths.bronze / "splits"
    staging.mkdir(parents=True, exist_ok=True)
    train_path = staging / f"train_{context.group_id[:8]}.parquet"
    valid_path = staging / f"valid_{context.group_id[:8]}.parquet"
    split.train.to_parquet(train_path, index=False)
    split.valid.to_parquet(valid_path, index=False)
    split_payload = {
        "train": str(train_path),
        "valid": str(valid_path),
        "cutoff_date": split.cutoff_date,
        "valid_drives": split.valid_drives,
        "train_drives": split.train_drives,
        "rows_dropped_leakage": split.rows_dropped_leakage,
    }
    context_payload = {
        "group_id": context.group_id,
        "trigger": context.trigger,
        "flow_run_id": context.flow_run_id,
        "validation_passed": context.validation_passed,
        "validation_report_path": str(context.validation_report_path or "") or None,
        "data_window": context.data_window,
        "data_manifest": context.data_manifest,
        "label_stats": context.label_stats,
    }

    # Parallélisation : les deux entraînements sont indépendants.
    futures = [
        train_models.submit(payload, name, split_payload, context_payload)
        for name in (BASELINE_NAME, CANDIDATE_NAME)
    ]
    trained_payloads = [future.result() for future in futures]
    logger.info("%d candidats entraînés en parallèle", len(trained_payloads))

    outcome = evaluate_models(
        payload, trained_payloads, context_payload, valid_path=split_payload["valid"]
    )
    get_store().record_pipeline_run(
        flow_name="train_pipeline",
        flow_run_id=context.flow_run_id,
        status="completed",
        trigger=trigger,
        details=outcome,
    )
    return outcome


def validate_training_window(config: AppConfig, split: Any) -> ValidationReport:
    """Contrat appliqué au jeu d'entraînement (et non à un lot journalier).

    On ne peut pas réutiliser le contrat journalier tel quel : un jeu
    d'entraînement couvre plusieurs dates et contient plusieurs lignes par
    disque. On vérifie donc ce qui doit rester vrai à ce niveau : volume
    suffisant, label dans [0, H], aucune valeur manquante dans le label.
    """
    from datetime import UTC
    from datetime import datetime as _dt

    from hdd_rul.validation.contract import ExpectationOutcome

    frame = pd.concat([split.train, split.valid], ignore_index=True)
    horizon = config.horizon_days
    label = pd.to_numeric(frame.get("rul"), errors="coerce")
    outcomes = [
        ExpectationOutcome(
            expectation="expect_training_rows_to_be_sufficient",
            column=None,
            severity="blocking",
            success=len(frame) >= 100,
            observed=len(frame),
        ),
        ExpectationOutcome(
            expectation="expect_label_to_be_within_horizon",
            column="rul",
            severity="blocking",
            success=bool(label.between(0, horizon).all()),
            observed=[float(label.min()), float(label.max())] if len(label) else None,
            unexpected_count=int((~label.between(0, horizon)).sum()),
        ),
        ExpectationOutcome(
            expectation="expect_label_not_null",
            column="rul",
            severity="blocking",
            success=bool(label.notna().all()),
            unexpected_count=int(label.isna().sum()),
        ),
        ExpectationOutcome(
            expectation="expect_both_populations_present",
            column="drive_kind",
            severity="warning",
            success=frame["drive_kind"].nunique() >= 2,
            observed=sorted(frame["drive_kind"].unique().tolist()),
        ),
    ]
    report = ValidationReport(
        batch_name="jeu_entrainement",
        batch_date=split.cutoff_date,
        total_rows=len(frame),
        quarantined_rows=0,
        outcomes=outcomes,
        validated_at=_dt.now(UTC).isoformat(timespec="seconds"),
    )
    from hdd_rul.validation.contract import save_report

    report.report_path = save_report(config, report)
    return report


def daily_batch_until_empty(
    config_path: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """Traite les lots en attente, un flow run par lot (utilisé par le rejeu)."""
    results: list[dict[str, Any]] = []
    for _ in range(limit):
        outcome = daily_batch(config_path=config_path)
        results.append(outcome)
        if outcome.get("status") == "idle":
            break
    return results
