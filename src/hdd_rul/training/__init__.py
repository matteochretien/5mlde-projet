"""Entraînement, évaluation, registre de modèles et règle de mise en service."""

from hdd_rul.training.evaluate import EvaluationMetrics, evaluate_predictions
from hdd_rul.training.models import BASELINE_NAME, CANDIDATE_NAME, build_pipeline
from hdd_rul.training.registry import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS,
    PromotionDecision,
    configure_mlflow,
    decide_promotion,
    describe_champion,
    get_client,
    rollback_champion,
)
from hdd_rul.training.split import SplitResult, temporal_group_split
from hdd_rul.training.train import (
    TrainingContext,
    TrainingOutcome,
    finalize_training,
    prepare_training_data,
    save_outcome,
    train_and_publish,
    train_single_model,
)

__all__ = [
    "ALIAS_CHALLENGER",
    "ALIAS_CHAMPION",
    "ALIAS_PREVIOUS",
    "BASELINE_NAME",
    "CANDIDATE_NAME",
    "EvaluationMetrics",
    "PromotionDecision",
    "SplitResult",
    "TrainingContext",
    "TrainingOutcome",
    "build_pipeline",
    "configure_mlflow",
    "decide_promotion",
    "describe_champion",
    "evaluate_predictions",
    "finalize_training",
    "get_client",
    "prepare_training_data",
    "rollback_champion",
    "save_outcome",
    "temporal_group_split",
    "train_and_publish",
    "train_single_model",
]
