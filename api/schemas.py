"""Schémas Pydantic de l'API : validation des entrées et forme des réponses.

La validation est stricte sur ce qui a un sens physique (bornes des compteurs
SMART, capacité positive, date non future) et tolérante sur le reste : un
modèle de disque inconnu n'est pas une erreur, il tombe dans la modalité
« autre » du prétraitement, avec un avertissement dans la réponse. Refuser la
requête serait pire : le parc évolue en permanence.
"""

from __future__ import annotations

# `date` est aliase : un champ du schéma s'appelle `date`, ce qui masquerait
# le type dans l'espace de noms de la classe et casserait l'annotation.
from datetime import date as date_type
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_BATCH_SIZE = 5_000


class DriveObservation(BaseModel):
    """État d'un disque un jour donné, tel que publié par Drive Stats."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    serial_number: str = Field(min_length=1, max_length=64, description="Numéro de série")
    model: str = Field(min_length=1, max_length=120, description="Modèle constructeur")
    capacity_bytes: int | None = Field(
        default=None, description="Capacité en octets (-1 ou 0 traité comme inconnu)"
    )
    date: date_type | None = Field(default=None, description="Jour de l'observation")

    # Attributs SMART *raw* exploités par la v1. Tous optionnels : un disque qui
    # ne remonte pas un attribut est un cas normal, géré nativement par le modèle.
    smart_5_raw: float | None = Field(default=None, ge=0, description="Secteurs réallouées")
    smart_9_raw: float | None = Field(default=None, ge=0, description="Heures de fonctionnement")
    smart_12_raw: float | None = Field(default=None, ge=0, description="Cycles d'alimentation")
    smart_187_raw: float | None = Field(default=None, ge=0, description="Erreurs non corrigées")
    smart_188_raw: float | None = Field(default=None, ge=0, description="Commandes expirées")
    smart_193_raw: float | None = Field(default=None, ge=0, description="Cycles de parcage")
    smart_194_raw: float | None = Field(default=None, ge=-50, le=150, description="Température °C")
    smart_197_raw: float | None = Field(default=None, ge=0, description="Secteurs en attente")
    smart_198_raw: float | None = Field(default=None, ge=0, description="Secteurs incorrigibles")
    smart_199_raw: float | None = Field(default=None, ge=0, description="Erreurs CRC UDMA")

    @field_validator("date")
    @classmethod
    def _not_in_future(cls, value: date_type | None) -> date_type | None:
        if value is not None and value > date_type.today():
            raise ValueError("la date d'observation ne peut pas être dans le futur")
        return value

    @field_validator("serial_number", "model")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("valeur vide")
        return cleaned


class PredictRequest(BaseModel):
    """Prédiction en ligne, un disque."""

    model_config = ConfigDict(extra="forbid")

    observation: DriveObservation
    persist: bool = Field(default=True, description="Tracer la prédiction en base")


class BatchPredictRequest(BaseModel):
    """Prédiction par lot : une liste de disques (ex. une journée de parc)."""

    model_config = ConfigDict(extra="forbid")

    observations: list[DriveObservation] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    persist: bool = True


class ModelIdentity(BaseModel):
    """Identité de la version servie, présente dans **chaque** réponse."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_alias: str
    run_id: str | None = None
    loaded_at: str | None = None


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_id: str
    serial_number: str
    predicted_rul_days: float = Field(description="RUL prédite, bornée à [0, H]")
    alert: bool = Field(description="True si la RUL prédite est sous le seuil d'alerte")
    horizon_days: int
    alert_threshold_days: int
    warnings: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    prediction: Prediction
    model: ModelIdentity
    duration_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelIdentity
    n_items: int
    n_alerts: int
    duration_ms: float
    persisted: int


class FeedbackItem(BaseModel):
    """RUL réellement constatée pour une observation déjà prédite."""

    model_config = ConfigDict(extra="forbid")

    serial_number: str = Field(min_length=1, max_length=64)
    observation_date: date_type
    failure_date: date_type | None = Field(
        default=None, description="Date de panne constatée ; absente si le disque a survécu"
    )
    still_alive_after_horizon: bool = Field(
        default=False, description="True si le disque était encore vivant à J+H"
    )
    actual_rul_days: float | None = Field(
        default=None, ge=0, description="RUL réelle, si déjà calculée par l'appelant"
    )

    @field_validator("observation_date")
    @classmethod
    def _not_future(cls, value: date_type) -> date_type:
        if value > date_type.today():
            raise ValueError("observation_date ne peut pas être dans le futur")
        return value


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedbacks: list[FeedbackItem] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class FeedbackResponse(BaseModel):
    received: int
    persisted: int
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = Field(description="ok, degraded ou unavailable")
    api_version: str
    model_loaded: bool
    model: ModelIdentity | None = None
    model_error: str | None = None
    database: dict[str, Any]
    registry: dict[str, Any] | None = Field(
        default=None,
        description="Serveur MLflow : injoignable = degraded (le modèle en mémoire reste servi)",
    )
    checked_at: datetime


class ModelResponse(BaseModel):
    """Détail de la version servie et de ses métriques d'évaluation."""

    model_config = ConfigDict(protected_namespaces=())

    available: bool
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    run_id: str | None = None
    loaded_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    horizon_days: int
    alert_threshold_days: int
    features: list[str] = Field(default_factory=list)
    error: str | None = None
