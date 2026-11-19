"""Contenu des alertes du pipeline : échec de flow, lot refusé, entraînement.

Chaque fonction construit une `Alert` (cf. `hdd_rul.common.notifications`) à
partir de ce que le pipeline sait au moment de l'incident. Rien ici n'envoie :
c'est `notify` qui s'en charge, et ces fonctions restent testables sans réseau.

Les informations demandées à l'API Prefect (tâche en échec, déploiement) sont
récupérées « au mieux » : si l'API ne répond pas, l'alerte part quand même,
simplement moins détaillée.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from hdd_rul.common.config import AlertsConfig, AppConfig
from hdd_rul.common.logging import get_logger
from hdd_rul.common.notifications import Alert, context_references

logger = get_logger(__name__)

LOCAL_LINKS_NOTE = "Liens accessibles uniquement depuis le poste hébergeant la pile Compose."

# Ce que l'échec laisse derrière lui, par flow : la première question d'astreinte.
CONSEQUENCES = {
    "daily_batch": "lot ni scoré ni archivé (il reste dans data/landing/) ; "
    "l'API continue de servir le champion",
    "train_pipeline": "aucune version publiée ; le champion en service est inchangé",
    "ingest_quarter": "couche bronze inchangée ; les flows suivants lisent la précédente",
}

_DEPENDENCY_ERRORS = {
    "MlflowException",
    "RestException",
    "ModelUnavailableError",
    "ConnectionError",
    "ConnectionRefusedError",
    "URLError",
    "OperationalError",
    "TimeoutError",
    "MaxRetryError",
    "HTTPError",
}
_DEPENDENCY_HINTS = (
    "connection refused",
    "max retries exceeded",
    "connection to server",
    "name or service not known",
    "timed out",
    "failed to resolve",
)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
def classify_failure(
    error_type: str | None, message: str, *, crashed: bool = False
) -> tuple[str, str]:
    """Nature de l'échec et action attendue, à partir du type d'exception."""
    lowered = message.lower()
    if crashed:
        return (
            "arrêt brutal du processus (infrastructure, mémoire ou conteneur)",
            "vérifier l'état du conteneur prefect-runner, puis relancer le flow run",
        )
    if error_type == "DataContractError":
        return (
            "donnée invalide : une règle bloquante du contrat de données est violée",
            "corriger le fichier puis le rejouer : il n'a pas été déplacé",
        )
    if error_type in _DEPENDENCY_ERRORS or any(hint in lowered for hint in _DEPENDENCY_HINTS):
        return (
            "dépendance indisponible (MLflow, PostgreSQL ou réseau), après retries",
            "vérifier `docker compose ps`, redémarrer le service, puis relancer le flow run",
        )
    return (
        "erreur technique (code, configuration ou environnement)",
        "lire les logs du flow run pour localiser l'erreur",
    )


_EXC_IN_MESSAGE = re.compile(r"exception:?\s*([A-Za-z_][\w.]*)\s*:\s*(.*)", re.S)


def failure_exception(state: Any) -> tuple[str | None, str]:
    """(type, message) de l'exception à l'origine de l'état en échec."""
    try:
        result = state.result(raise_on_failure=False)
        # `FailedRun` n'est qu'une enveloppe Prefect : la vraie exception est
        # dans le message de l'état (« … exception MlflowException: … »).
        if isinstance(result, BaseException) and type(result).__name__ != "FailedRun":
            return type(result).__name__, str(result)
    except Exception:  # résultat non persisté ou API injoignable
        pass
    message = getattr(state, "message", None) or ""
    if match := _EXC_IN_MESSAGE.search(message):
        return match.group(1).rsplit(".", 1)[-1], match.group(2).strip()
    return None, message


def _shorten(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def prefect_failure_context(run: Any) -> dict[str, Any]:
    """Tâches en échec et nom du déploiement, lus dans l'API Prefect (au mieux)."""
    context: dict[str, Any] = {"failed_tasks": [], "task_errors": [], "deployment": None}
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import (
            TaskRunFilter,
            TaskRunFilterFlowRunId,
            TaskRunFilterState,
            TaskRunFilterStateType,
        )
        from prefect.client.schemas.objects import StateType
        from prefect.client.schemas.sorting import TaskRunSort

        with get_client(sync_client=True) as client:
            task_runs = client.read_task_runs(
                task_run_filter=TaskRunFilter(
                    flow_run_id=TaskRunFilterFlowRunId(any_=[UUID(str(run.id))]),
                    state=TaskRunFilterState(
                        type=TaskRunFilterStateType(any_=[StateType.FAILED, StateType.CRASHED])
                    ),
                ),
                # Explicites : certaines versions du serveur refusent `null` (422).
                sort=TaskRunSort.EXPECTED_START_TIME_ASC,
                limit=20,
            )
            for task_run in task_runs:
                # « validation-3f2 » -> « validation » : le suffixe est propre au run.
                context["failed_tasks"].append(task_run.name.rsplit("-", 1)[0])
                if task_run.state is not None:
                    context["task_errors"].append(failure_exception(task_run.state))
            if getattr(run, "deployment_id", None):
                context["deployment"] = client.read_deployment(run.deployment_id).name
    except Exception as exc:  # l'alerte doit partir même sans API Prefect
        logger.debug("contexte Prefect indisponible pour l'alerte: %s", exc)
    return context


# ---------------------------------------------------------------------------
# Constructeurs d'alertes
# ---------------------------------------------------------------------------
def _flow_run_label(flow_run_id: str | None, flow_run_name: str | None) -> str | None:
    if not flow_run_id:
        return None
    return f"{flow_run_name} ({flow_run_id})" if flow_run_name else flow_run_id


def _common_links(alerts: AlertsConfig, flow_run_id: str | None) -> list[tuple[str, str]]:
    links = []
    if flow_run_id:
        links.append(
            ("Ouvrir le flow run dans Prefect", alerts.links.prefect_flow_run(flow_run_id))
        )
    links.append(("Tableau de bord Grafana", alerts.links.grafana_dashboard()))
    return links


def _note(alerts: AlertsConfig) -> str | None:
    return LOCAL_LINKS_NOTE if alerts.links.all_local() else None


def _batch_from_parameters(parameters: dict[str, Any]) -> str | None:
    if batch := parameters.get("batch_path"):
        return Path(str(batch)).name
    return None


def _duration(start: datetime | None) -> str | None:
    if start is None:
        return None
    seconds = int((datetime.now(UTC) - start.astimezone(UTC)).total_seconds())
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


def flow_failure_alert(
    flow_name: str,
    run: Any,
    state: Any,
    alerts: AlertsConfig,
    prefect_context: dict[str, Any] | None = None,
) -> Alert:
    """Hook `on_failure` : qui a échoué, où, pourquoi, et où regarder."""
    ctx = prefect_context or {"failed_tasks": [], "task_errors": [], "deployment": None}
    state_name = getattr(state, "name", None) or "Failed"
    state_type = getattr(getattr(state, "type", None), "value", "") or ""
    failed_tasks: list[str] = ctx.get("failed_tasks") or []
    step = ", ".join(dict.fromkeys(failed_tasks)) or None

    error_type, error_message = failure_exception(state)
    if error_type is None and ctx.get("task_errors"):
        error_type, error_message = ctx["task_errors"][0]
    nature, action = classify_failure(
        error_type, error_message, crashed=state_type.upper() == "CRASHED"
    )
    cause = f"{error_type}: {error_message}" if error_type else error_message
    parameters = dict(getattr(run, "parameters", None) or {})
    batch = _batch_from_parameters(parameters)
    run_id = str(run.id)
    run_count = getattr(run, "run_count", 1) or 1

    deployment = ctx.get("deployment")
    launch = f"déploiement {deployment}" if deployment else "appel direct (CLI ou script)"
    if trigger := parameters.get("trigger"):
        launch += f", déclencheur {trigger}"

    title = f"Échec du flow {flow_name} : {state_name}"
    if step:
        title += f" (étape {step})"

    commands = {"Logs du flow run": f"uv run prefect flow-run logs {run_id}"}
    if flow_name == "daily_batch" and batch and error_type == "DataContractError":
        commands["Rejouer le lot"] = (
            f"uv run python -m hdd_rul.cli batch --batch data/landing/{batch}"
        )

    return Alert(
        title=title,
        summary=f"Le flow {flow_name} s'est arrêté"
        + (f" à l'étape {step}." if step else ".")
        + " Les liens ci-dessous mènent au run concerné.",
        level="critical",
        status={
            "Statut": state_name,
            "Étape": step or "inconnue",
            "Flow": flow_name,
            "Lot": batch,
            "Trimestre": parameters.get("quarter"),
            "Date simulée": parameters.get("as_of"),
            "Tentative": run_count if run_count > 1 else None,
        },
        sections={
            "Diagnostic": {
                "Nature": nature,
                "Cause": _shorten(cause) or "message d'erreur absent",
                "Conséquence": CONSEQUENCES.get(flow_name),
                "Action": action,
            },
            "Références": {
                "Flow run": _flow_run_label(run_id, getattr(run, "name", None)),
                "Lancement": launch,
                "Durée": _duration(getattr(run, "start_time", None)),
                **context_references(alerts),
            },
        },
        links=_common_links(alerts, run_id),
        commands=commands,
        note=_note(alerts),
    )


def _rule_detail(outcome: Any) -> str:
    if outcome.unexpected_count:
        detail = f"{outcome.unexpected_count} ligne(s) en défaut"
        if outcome.unexpected_sample:
            sample = ", ".join(str(v) for v in outcome.unexpected_sample[:3])
            detail += f", ex. {sample}"
        return _shorten(detail, 160)
    if outcome.observed not in (None, "", [], {}):
        return _shorten(f"observé : {outcome.observed}", 160)
    return "violée"


def _validation_sections(report: Any) -> dict[str, Any]:
    rules = {
        f"{o.expectation} ({o.column})" if o.column else o.expectation: _rule_detail(o)
        for o in report.blocking_failures
    }
    return {
        "Nature": f"donnée invalide : {len(report.blocking_failures)} "
        "règle(s) bloquante(s) violée(s)",
        **rules,
        "Lignes": report.total_rows,
        "Quarantaine": report.quarantined_rows or None,
        "Avertissements": len(report.warnings) or None,
    }


def batch_refused_alert(
    report: Any,
    alerts: AlertsConfig,
    *,
    flow_run_id: str | None = None,
    flow_run_name: str | None = None,
) -> Alert:
    """Lot journalier refusé par le contrat de données (tâche `validation`)."""
    diagnostic = _validation_sections(report)
    diagnostic["Conséquence"] = CONSEQUENCES["daily_batch"]
    diagnostic["Action"] = "corriger le fichier dans data/landing/, puis rejouer le lot"
    return Alert(
        title=f"Lot {report.batch_name} refusé par le contrat de données",
        summary="Le lot est bloqué avant tout traitement : aucune prédiction n'a été produite.",
        level="critical",
        status={
            "Statut": "Bloqué",
            "Étape": "validation",
            "Flow": "daily_batch",
            "Lot": report.batch_name,
            "Date du lot": report.batch_date,
        },
        sections={
            "Diagnostic": diagnostic,
            "Références": {
                "Flow run": _flow_run_label(flow_run_id, flow_run_name),
                "Rapport de validation": str(report.report_path) if report.report_path else None,
                **context_references(alerts),
            },
        },
        links=_common_links(alerts, flow_run_id),
        commands={
            "Rejouer le lot": "uv run python -m hdd_rul.cli batch "
            f"--batch data/landing/{report.batch_name}",
        },
        note=_note(alerts),
    )


def training_window_refused_alert(
    report: Any,
    alerts: AlertsConfig,
    *,
    flow_run_id: str | None = None,
    flow_run_name: str | None = None,
    trigger: str | None = None,
) -> Alert:
    """Jeu d'entraînement non conforme : l'entraînement n'est pas lancé."""
    diagnostic = _validation_sections(report)
    diagnostic["Conséquence"] = CONSEQUENCES["train_pipeline"]
    diagnostic["Action"] = "examiner le rapport de validation, corriger la fenêtre ou les lots"
    return Alert(
        title="Entraînement interrompu : jeu d'entraînement non conforme",
        summary="Le contrôle qualité de la fenêtre d'entraînement a échoué "
        "avant tout apprentissage.",
        level="critical",
        status={
            "Statut": "Bloqué",
            "Étape": "validation",
            "Flow": "train_pipeline",
            "Fenêtre": report.batch_name,
            "Déclencheur": trigger,
        },
        sections={
            "Diagnostic": diagnostic,
            "Références": {
                "Flow run": _flow_run_label(flow_run_id, flow_run_name),
                "Rapport de validation": str(report.report_path) if report.report_path else None,
                **context_references(alerts),
            },
        },
        links=_common_links(alerts, flow_run_id),
        note=_note(alerts),
    )


def _fmt_days(value: Any) -> str | None:
    try:
        return f"{float(value):.2f} j"
    except (TypeError, ValueError):
        return None


def training_outcome_alert(
    outcome: Any,
    config: AppConfig,
    *,
    candidate_name: str,
    baseline_name: str,
    trigger: str | None = None,
    flow_run_id: str | None = None,
    flow_run_name: str | None = None,
) -> Alert:
    """Fin d'entraînement : décision de mise en service et liens MLflow."""
    alerts = config.alerts
    promoted = outcome.decision == "promoted"
    version = outcome.registered_version
    candidate = outcome.models.get(candidate_name, {})
    baseline = outcome.models.get(baseline_name, {})
    metrics = candidate.get("metrics", {})
    base_metrics = baseline.get("metrics", {})
    window = outcome.data_window or {}

    if promoted:
        title = f"Entraînement terminé : version {version} promue champion"
    else:
        title = f"Entraînement terminé : version {version or '-'} non promue"
    model_name = config.mlflow.registered_model_name
    links: list[tuple[str, str]] = []
    if version:
        links.append(
            (
                f"Version {version} dans le registre MLflow",
                alerts.links.mlflow_model_version(model_name, str(version)),
            )
        )
    if run_id := candidate.get("run_id"):
        links.append(("Run MLflow du candidat", alerts.links.mlflow_run(run_id)))
    links += _common_links(alerts, flow_run_id)

    champion = outcome.aliases.get("champion")
    previous = outcome.aliases.get("previous")
    return Alert(
        title=title,
        summary=f"Règle de mise en service : {outcome.reason}.",
        level="info" if promoted else "warning",
        status={
            "Décision": "promue" if promoted else "rejetée",
            "Version": version or "-",
            "Étape": "evaluation-publication",
            "Déclencheur": trigger,
        },
        sections={
            "Résultat": {
                "MAE globale": _fmt_days(metrics.get("mae")),
                "MAE sur RUL < 30 j": _fmt_days(metrics.get("mae_lt30")),
                "Rappel de l'alerte < 30 j": (
                    f"{float(metrics['alert_recall']):.2f}" if "alert_recall" in metrics else None
                ),
                "MAE de la baseline": _fmt_days(base_metrics.get("mae")),
                "Champion": (
                    f"version {champion}" + (f" (précédent : {previous})" if previous else "")
                    if champion
                    else "inchangé"
                ),
                "Fenêtre de données": (
                    f"{window.get('window_start')} → {window.get('window_end')}" if window else None
                ),
            },
            "Références": {
                "Run MLflow": candidate.get("run_id"),
                "Modèle": f"{model_name} v{version}" if version else None,
                "Flow run": _flow_run_label(flow_run_id, flow_run_name),
                "Groupe d'entraînement": outcome.group_id[:8] if outcome.group_id else None,
                **context_references(alerts),
            },
        },
        links=links,
        note=_note(alerts),
    )


def sample_alert(alerts: AlertsConfig) -> Alert:
    """Alerte de test : vérifie le rendu de la carte et le webhook, sans incident."""
    return Alert(
        title="[TEST] Échec du flow daily_batch : Failed (étape validation)",
        summary="Message de test : rendu du modèle d'alerte sur un incident fictif. "
        "Aucune action requise.",
        level="warning",
        status={
            "Statut": "Failed",
            "Étape": "validation",
            "Flow": "daily_batch",
            "Lot": "2026-04-07.csv",
        },
        sections={
            "Diagnostic": {
                "Nature": "donnée invalide : une règle bloquante du contrat de données est violée",
                "Cause": "DataContractError: contrat violé sur 2026-04-07.csv: "
                "expect_column_values_to_be_in_set(failure)",
                "Conséquence": CONSEQUENCES["daily_batch"],
                "Action": "aucune (message de test)",
            },
            "Références": {
                "Flow run": "exemple (aucun run réel)",
                "Lancement": "commande hdd_rul.cli alert-test",
                **context_references(alerts),
            },
        },
        links=[
            ("Ouvrir les runs dans Prefect", alerts.links.prefect_runs()),
            ("Tableau de bord Grafana", alerts.links.grafana_dashboard()),
        ],
        commands={"Envoyer une alerte de test": "uv run python -m hdd_rul.cli alert-test"},
        note=_note(alerts),
    )
