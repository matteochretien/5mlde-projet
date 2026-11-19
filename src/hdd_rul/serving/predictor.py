"""Chargement du modèle servi et scoring d'un lot.

Le modèle est chargé **une fois par processus** depuis l'alias `champion` du
registre MLflow, puis gardé en mémoire : un worker uvicorn ne retélécharge pas
l'artefact à chaque requête. Le rechargement explicite (après une promotion ou
un retour arrière) passe par `PredictorCache.reload()`, exposé par l'endpoint
d'administration de l'API.

Le même code sert l'API et le flow `daily_batch` : la prédiction en ligne et la
prédiction par lot ne peuvent pas divergees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger
from hdd_rul.features.build import select_feature_frame
from hdd_rul.training.registry import ALIAS_CHAMPION, configure_mlflow, get_client

logger = get_logger(__name__)

PREDICTION_COLUMN = "predicted_rul"
ALERT_COLUMN = "alert"


class ModelUnavailableError(RuntimeError):
    """Aucune version `champion` disponible : l'API doit répondre 503, pas 500."""


@dataclass
class LoadedModel:
    """Modèle servi et son identité, renvoyée dans chaque réponse de l'API."""

    pipeline: Any
    model_name: str
    version: str
    run_id: str | None
    alias: str = ALIAS_CHAMPION
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    metrics: dict[str, float] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.version,
            "model_alias": self.alias,
            "run_id": self.run_id,
            "loaded_at": self.loaded_at,
        }


def load_champion(config: AppConfig, *, alias: str = ALIAS_CHAMPION) -> LoadedModel:
    """Charge la version portant l'alias demandé (par défaut `champion`).

    Toute erreur (registre injoignable, alias absent, artefact illisible) devient
    une `ModelUnavailableError` : l'API la traduit en `503` explicite, jamais en
    `500`.
    """
    name = config.mlflow.registered_model_name
    try:
        configure_mlflow(config)
        client = get_client(config)
    except Exception as exc:
        raise ModelUnavailableError(
            f"registre MLflow injoignable ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise ModelUnavailableError(
            f"aucune version '{alias}' pour le modèle '{name}' ({type(exc).__name__}: {exc}). "
            "Lancer un entraînement avant de servir."
        ) from exc

    uri = f"models:/{name}@{alias}"
    logger.info("chargement du modèle %s (version %s)", uri, version.version)
    try:
        pipeline = mlflow.sklearn.load_model(uri)
    except Exception as exc:
        raise ModelUnavailableError(
            f"artefact de {uri} illisible ({type(exc).__name__}: {exc})"
        ) from exc
    metrics: dict[str, float] = {}
    if version.run_id:
        try:
            metrics = {k: float(v) for k, v in client.get_run(version.run_id).data.metrics.items()}
        except Exception:
            metrics = {}
    return LoadedModel(
        pipeline=pipeline,
        model_name=name,
        version=str(version.version),
        run_id=version.run_id,
        alias=alias,
        metrics=metrics,
    )


def predict_frame(model: LoadedModel, frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Score un lot et ajoute `predicted_rul` et `alert`.

    Les colonnes attendues sont celles de `features.required_input_columns` ; une
    colonne absente est traitée comme manquante (NaN) plutôt que comme une
    erreur : c'est ce qui permet de servir un trimestre au schéma différent.
    """
    if frame.empty:
        return frame.assign(**{PREDICTION_COLUMN: [], ALERT_COLUMN: []})
    features = select_feature_frame(frame, config)
    predictions = model.pipeline.predict(features)
    out = frame.copy()
    # La RUL est bornée par construction : [0, H]. Un modèle peut sortir de
    # l'intervalle par extrapolation, ce qui n'a pas de sens métier.
    out[PREDICTION_COLUMN] = pd.Series(predictions, index=out.index).clip(0, config.horizon_days)
    out[ALERT_COLUMN] = out[PREDICTION_COLUMN] < config.split.alert_threshold_days
    return out


class PredictorCache:
    """Conteneur du modèle servi, avec rechargement explicite.

    Le chargement est paresseux : l'API démarre même si MLflow n'est pas encore
    joignable, et `/health` signale alors l'indisponibilité au lieu de planter
    au démarrage (comportement attendu par le scénario d'incident).
    """

    def __init__(self, config: AppConfig, *, alias: str = ALIAS_CHAMPION) -> None:
        self.config = config
        self.alias = alias
        self._model: LoadedModel | None = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def get(self) -> LoadedModel:
        if self._model is None:
            self.reload()
        if self._model is None:  # pragma: no cover — reload lève déjà
            raise ModelUnavailableError(self._error or "modèle non chargé")
        return self._model

    def try_get(self) -> LoadedModel | None:
        try:
            return self.get()
        except ModelUnavailableError:
            return None

    def reload(self, *, alias: str | None = None) -> LoadedModel:
        """Rechargé depuis le registre. Appelé après une promotion ou un rollback."""
        self.alias = alias or self.alias
        try:
            self._model = load_champion(self.config, alias=self.alias)
            self._error = None
        except ModelUnavailableError as exc:
            # On garde le modèle déjà chargé : un rechargement raté (registre
            # injoignable pendant une panne) ne doit pas couper les prédictions.
            self._error = str(exc)
            if self._model is not None:
                logger.warning(
                    "rechargement impossible, la version %s reste servie (%s)",
                    self._model.version,
                    exc,
                )
            raise
        return self._model

    def describe(self) -> dict[str, Any]:
        model = self.try_get()
        if model is None:
            return {"available": False, "alias": self.alias, "error": self._error}
        return {"available": True, **model.identity(), "metrics": model.metrics}
