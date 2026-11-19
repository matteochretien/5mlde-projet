"""Service de prédiction : chargement du champion, scoring, feedbacks."""

from hdd_rul.serving.feedback import build_feedbacks, collect_batch_feedbacks
from hdd_rul.serving.predictor import (
    ALERT_COLUMN,
    PREDICTION_COLUMN,
    LoadedModel,
    ModelUnavailableError,
    PredictorCache,
    load_champion,
    predict_frame,
)

__all__ = [
    "ALERT_COLUMN",
    "PREDICTION_COLUMN",
    "LoadedModel",
    "ModelUnavailableError",
    "PredictorCache",
    "build_feedbacks",
    "collect_batch_feedbacks",
    "load_champion",
    "predict_frame",
]
