"""Registre de modèles MLflow : alias, règle de mise en service, retour arrière.

Trois alias, et un seul sert en production :

* `champion`  — la version servie par l'API. C'est le seul alias que l'API lit.
* `challenger`— le dernier candidat entraîné, promu ou non. Il garde une trace du
  candidat rejeté et de la raison du rejet (tag `promotion.reason`).
* `previous`  — l'ancien champion, conservé pour le retour arrière. Sans lui, un
  rollback demanderait de retrouver à la main le bon numéro de version.

La décision de promotion est une fonction pure (`decide_promotion`) : elle est
testable unitairement, sans serveur MLflow, et la raison du rejet est toujours
explicite plutôt que déduite des logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"
ALIAS_PREVIOUS = "previous"


@dataclass(frozen=True)
class PromotionDecision:
    """Résultat de la règle de mise en service, journalisé en tags de version."""

    promote: bool
    reason: str
    details: dict[str, float] | None = None

    @property
    def decision(self) -> str:
        return "promoted" if self.promote else "rejected"


def configure_mlflow(config: AppConfig) -> str:
    """Pointe MLflow sur le tracking server et l'expérience de la config."""
    uri = config.mlflow.resolved_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    return uri


def get_client(config: AppConfig) -> MlflowClient:
    return MlflowClient(tracking_uri=config.mlflow.resolved_tracking_uri())


def get_version_by_alias(client: MlflowClient, name: str, alias: str) -> Any | None:
    """Version portant l'alias, ou None si l'alias n'existe pas encore."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def version_metrics(client: MlflowClient, version: Any | None) -> dict[str, float]:
    """Métriques du run qui a produit une version du modèle."""
    if version is None or not getattr(version, "run_id", None):
        return {}
    try:
        run = client.get_run(version.run_id)
    except MlflowException:
        return {}
    return {key: float(value) for key, value in run.data.metrics.items()}


def decide_promotion(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    config: AppConfig,
    *,
    validation_passed: bool = True,
) -> PromotionDecision:
    """Applique la règle de mise en service. Fonction pure, sans effet de bord.

    Règle : on promeut si les contrôles qualité sont verts et si la MAE de la
    zone critique (RUL < 30 j) s'améliore d'au moins `min_mae_lt30_improvement`
    sans dégrader la MAE globale de plus de `max_mae_global_degradation`.
    """
    rules = config.promotion
    if rules.require_validation_passed and not validation_passed:
        return PromotionDecision(False, "validation_failed")

    required = ("mae", "mae_lt30")
    missing = [key for key in required if key not in challenger]
    if missing:
        return PromotionDecision(False, f"metrics_missing:{','.join(missing)}")

    if not champion or any(key not in champion for key in required):
        return PromotionDecision(True, "no_champion_yet")

    improvement = (champion["mae_lt30"] - challenger["mae_lt30"]) / champion["mae_lt30"]
    degradation = (challenger["mae"] - champion["mae"]) / champion["mae"]
    details = {
        "mae_lt30_improvement": round(improvement, 6),
        "mae_global_degradation": round(degradation, 6),
        "champion_mae_lt30": champion["mae_lt30"],
        "challenger_mae_lt30": challenger["mae_lt30"],
        "champion_mae": champion["mae"],
        "challenger_mae": challenger["mae"],
    }

    if improvement < rules.min_mae_lt30_improvement:
        return PromotionDecision(
            False,
            f"mae_lt30_gain_insuffisant:{improvement:.4f}<{rules.min_mae_lt30_improvement}",
            details,
        )
    if degradation > rules.max_mae_global_degradation:
        return PromotionDecision(
            False,
            f"mae_globale_degradee:{degradation:.4f}>{rules.max_mae_global_degradation}",
            details,
        )
    if improvement < 0:
        # Seuil négatif (overlay de démonstration) : la promotion est tracée comme
        # une régression tolérée, jamais comme une amélioration.
        return PromotionDecision(
            True,
            f"regression_toleree:{improvement:.4f}>={rules.min_mae_lt30_improvement}",
            details,
        )
    return PromotionDecision(True, "ameliore_zone_critique", details)


def apply_decision(
    client: MlflowClient,
    name: str,
    version: str | int,
    decision: PromotionDecision,
) -> dict[str, str]:
    """Positionne les alias et les tags de version selon la décision.

    Le candidat reçoit toujours `challenger` : même rejeté, il doit rester
    identifiable, avec la raison du rejet lisible dans l'interface MLflow.
    """
    version = str(version)
    client.set_model_version_tag(name, version, "promotion.decision", decision.decision)
    client.set_model_version_tag(name, version, "promotion.reason", decision.reason)
    client.set_registered_model_alias(name, ALIAS_CHALLENGER, version)
    aliases = {ALIAS_CHALLENGER: version}
    if not decision.promote:
        logger.warning("version %s non promue (%s)", version, decision.reason)
        return aliases

    current = get_version_by_alias(client, name, ALIAS_CHAMPION)
    if current is not None and current.version != version:
        client.set_registered_model_alias(name, ALIAS_PREVIOUS, current.version)
        aliases[ALIAS_PREVIOUS] = str(current.version)
    client.set_registered_model_alias(name, ALIAS_CHAMPION, version)
    aliases[ALIAS_CHAMPION] = version
    logger.info("version %s promue champion (%s)", version, decision.reason)
    return aliases


def rollback_champion(client: MlflowClient, name: str) -> dict[str, str]:
    """Retour arrière : `previous` redevient `champion` (et inversement).

    L'échange est volontaire : après un rollback, `previous` pointe sur la
    version retirée, donc un second appel annule le premier.
    """
    previous = get_version_by_alias(client, name, ALIAS_PREVIOUS)
    if previous is None:
        raise RuntimeError(
            f"aucune version '{ALIAS_PREVIOUS}' pour {name}: retour arrière impossible"
        )
    current = get_version_by_alias(client, name, ALIAS_CHAMPION)

    client.set_registered_model_alias(name, ALIAS_CHAMPION, previous.version)
    result = {ALIAS_CHAMPION: str(previous.version)}
    if current is not None:
        client.set_registered_model_alias(name, ALIAS_PREVIOUS, current.version)
        result[ALIAS_PREVIOUS] = str(current.version)
        client.set_model_version_tag(name, current.version, "promotion.decision", "rolled_back")
    logger.warning(
        "retour arrière: champion %s -> %s",
        getattr(current, "version", "aucun"),
        previous.version,
    )
    return result


def sync_model_versions(config: AppConfig, store: Any | None = None) -> dict[str, str]:
    """Recopie l'état des alias du registre dans `tracing.model_versions`.

    C'est cette table que lit la vue croisée de Grafana (MAE de validation
    contre MAE de production, par version). Appelé après chaque publication et
    chaque retour arrière, quel que soit le chemin (CLI ou flow Prefect) : avant,
    seul le flow l'alimentait, et le champion entraîné par la CLI n'y figurait pas.
    Ne lève jamais : une trace manquante ne doit pas faire échouer un entraînement.
    """
    from hdd_rul.common.tracing import get_store

    store = store or get_store()
    name = config.mlflow.registered_model_name
    synced: dict[str, str] = {}
    try:
        client = get_client(config)
        for alias in (ALIAS_CHAMPION, ALIAS_CHALLENGER, ALIAS_PREVIOUS):
            version = get_version_by_alias(client, name, alias)
            if version is None:
                store.prune_alias(model_name=name, alias=alias, keep_version=None)
                continue
            store.record_model_version(
                model_name=name,
                version=str(version.version),
                alias=alias,
                run_id=version.run_id,
                metrics=version_metrics(client, version),
            )
            store.prune_alias(model_name=name, alias=alias, keep_version=str(version.version))
            synced[alias] = str(version.version)
    except Exception as exc:
        logger.warning("synchronisation des versions vers les traces impossible (%s)", exc)
    return synced


def registry_healthcheck(config: AppConfig, *, timeout: float = 2.0) -> dict[str, Any]:
    """Joignabilite du serveur MLflow, exposée par `/health`.

    Appel HTTP direct et court (pas le client MLflow, qui relance plusieurs
    fois) : `/health` doit répondre vite même quand MLflow est arrêté. Un store
    local (SQLite) n'a pas de serveur à vérifier.
    """
    import requests

    uri = config.mlflow.resolved_tracking_uri()
    if not uri.startswith(("http://", "https://")):
        return {"enabled": False, "reachable": None, "error": None}
    try:
        response = requests.get(f"{uri.rstrip('/')}/health", timeout=timeout)
        reachable = response.status_code == 200
        error = None if reachable else f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        reachable, error = False, type(exc).__name__
    return {"enabled": True, "reachable": reachable, "error": error}


def describe_champion(config: AppConfig) -> dict[str, Any]:
    """Résumé de la version servie, exposé par l'endpoint `/model` de l'API."""
    client = get_client(config)
    name = config.mlflow.registered_model_name
    version = get_version_by_alias(client, name, ALIAS_CHAMPION)
    if version is None:
        return {"name": name, "alias": ALIAS_CHAMPION, "available": False}
    return {
        "name": name,
        "alias": ALIAS_CHAMPION,
        "available": True,
        "version": str(version.version),
        "run_id": version.run_id,
        "tags": dict(version.tags or {}),
        "metrics": version_metrics(client, version),
    }
