"""Fabriqués de modèles : baseline et modèle retenu.

Deux candidats, entraînés en parallèle par le flow Prefect :

* **baseline** `DummyRegressor(median)` — prédit toujours la RUL médiane. Sans
  elle, une MAE n'a pas de sens : c'est le point de référence qui permet de dire
  qu'un modèle apporte quelque chose, et c'est aussi un garde-fou (si le modèle
  ne fait pas mieux que la médiane, il y a un problème en amont).
* **modèle** `HistGradientBoostingRegressor` — gère nativement les valeurs
  manquantes, très présentés dans les attributs SMART (colonnes absentes d'un
  trimestre, disques qui ne remontent pas un attribut), insensible à l'échelle,
  et s'entraîne en quelques secondes sur quelques centaines de milliers de
  lignes. La performance n'est pas notée : la rapidité et la robustesse comptent
  plus ici que le dernier point de MAE.

Les deux partagent exactement le même prétraitement, embarqué dans le pipeline.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from hdd_rul.common.config import AppConfig
from hdd_rul.features.build import build_feature_steps

BASELINE_NAME = "baseline_median"
CANDIDATE_NAME = "hist_gradient_boosting"


def build_baseline_pipeline(config: AppConfig) -> Pipeline:
    return Pipeline(
        [
            *build_feature_steps(config),
            ("modele", DummyRegressor(strategy=config.training.baseline.strategy)),
        ]
    )


def build_candidate_pipeline(config: AppConfig) -> Pipeline:
    hgb = config.training.hgb
    return Pipeline(
        [
            *build_feature_steps(config),
            (
                "modele",
                HistGradientBoostingRegressor(
                    loss="absolute_error",  # cohérent avec la MAE suivie en production
                    max_iter=hgb.max_iter,
                    learning_rate=hgb.learning_rate,
                    max_depth=hgb.max_depth,
                    min_samples_leaf=hgb.min_samples_leaf,
                    l2_regularization=hgb.l2_regularization,
                    early_stopping=hgb.early_stopping,
                    validation_fraction=hgb.validation_fraction,
                    random_state=hgb.random_state,
                ),
            ),
        ]
    )


def build_pipeline(name: str, config: AppConfig) -> Pipeline:
    if name == BASELINE_NAME:
        return build_baseline_pipeline(config)
    if name == CANDIDATE_NAME:
        return build_candidate_pipeline(config)
    raise ValueError(f"modèle inconnu: {name!r} (attendu: {BASELINE_NAME} ou {CANDIDATE_NAME})")


def model_params(name: str, config: AppConfig) -> dict[str, object]:
    """Hyperparamètres journalisés dans MLflow (lisibles sans désérialiser)."""
    if name == BASELINE_NAME:
        return {"strategy": config.training.baseline.strategy}
    return config.training.hgb.model_dump()
