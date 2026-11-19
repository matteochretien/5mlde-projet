"""Features v1 et prétraitement sklearn.

Choix v1 : **l'état du jour seulement**. Le vecteur de features se déduit d'une
seule ligne Drive Stats, donc l'API peut répondre sans historique du disque. Les
deltas sur 7 jours (attributs 5, 197, 198) seraient plus informatifs mais
imposeraient de connaître le passe du disque : ce serait une v2 limitée à la
prédiction par lot (cf. limites du README).

Le prétraitement est embarqué dans le `Pipeline` sklearn, donc tracé et versionné
avec le modèle : l'API applique exactement les mêmes transformations que
l'entraînement, sans code dupliqué.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from hdd_rul.common.config import AppConfig

# Préfixe de `model` -> constructeur. Les attributs SMART n'ont pas la même
# sémantique selon le fabricant : la feature constructeur donne au modèle de
# quoi en tenir compte.
VENDOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("ST", "Seagate"),
    ("SEAGATE", "Seagate"),
    ("TOSHIBA", "Toshiba"),
    ("HGST", "HGST"),
    ("HITACHI", "Hitachi"),
    ("WDC", "WDC"),
    ("WD", "WDC"),
    ("SAMSUNG", "Samsung"),
    ("MTFD", "Micron"),
    ("CT", "Crucial"),
    ("DELLBOSS", "Dell"),
)
UNKNOWN_VENDOR = "unknown"
RAW_INPUT_COLUMNS = ("model", "capacity_bytes")


def vendor_from_model(model: str | None) -> str:
    """Constructeur déduit du préfixe du champ `model`."""
    if not model or not isinstance(model, str):
        return UNKNOWN_VENDOR
    normalized = model.strip().upper()
    for prefix, vendor in VENDOR_PREFIXES:
        if normalized.startswith(prefix):
            return vendor
    return UNKNOWN_VENDOR


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `capacity_tb` et `vendor`. Fonction de module : picklable par MLflow."""
    out = frame.copy()
    if "capacity_bytes" in out.columns:
        capacity = pd.to_numeric(out["capacity_bytes"], errors="coerce")
        # Backblaze publie parfois -1 : traité comme une valeur manquante, que
        # HistGradientBoosting sait gérer nativement.
        out["capacity_tb"] = np.where(capacity > 0, capacity / 1e12, np.nan)
    else:
        out["capacity_tb"] = np.nan
    out["vendor"] = out.get("model", pd.Series(dtype="object")).map(vendor_from_model)
    return out


def required_input_columns(config: AppConfig) -> list[str]:
    """Colonnes brutes attendues en entrée du pipeline (et donc de l'API)."""
    return [*RAW_INPUT_COLUMNS, *config.features.smart_columns]


def select_feature_frame(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Réduit un lot aux colonnes brutes du pipeline, en complétant les absentes.

    Une colonne SMART absente du trimestre devient NaN plutôt qu'une erreur :
    c'est le comportement voulu face aux variations de schéma de Backblaze.
    """
    columns = required_input_columns(config)
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        out[column] = frame[column] if column in frame.columns else np.nan
    for column in config.features.smart_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    # `capacity_bytes` est converti en flottant : un entier ne peut pas porter
    # de valeur manquante, et la signature MLflow refuserait alors un lot ou la
    # capacité est inconnue (cas réel : Backblaze publie des -1).
    out["capacity_bytes"] = pd.to_numeric(out["capacity_bytes"], errors="coerce").astype("float64")
    return out


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    """Prétraitement : numériques inchangées, catégorielles en one-hot top-K.

    * Les numériques ne sont ni imputées ni standardisées : `HistGradientBoosting`
      gère nativement les valeurs manquantes — fréquentes dans les SMART — et est
      insensible à l'échelle. Imputer introduirait une valeur inventée.
    * `OneHotEncoder(max_categories=K, handle_unknown="infrequent_if_exist")`
      implémente exactement la règle « top-K modèles + modalité autre » : les
      modèles rares sont regroupés à l'apprentissage, et un modèle inconnu à la
      prédiction tombe dans cette même modalité au lieu de faire échouer l'appel.
    """
    features = config.features
    numeric = features.numeric_columns
    categorical = features.categorical_columns

    encoder = OneHotEncoder(
        max_categories=features.top_k_models,
        handle_unknown="infrequent_if_exist",
        min_frequency=2,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("numeriques", "passthrough", numeric),
            ("categorielles", encoder, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_steps(config: AppConfig) -> list[tuple[str, object]]:
    """Étapes de prétraitement, communes à la baseline et au modèle."""
    return [
        ("derivation", FunctionTransformer(derive_features, validate=False)),
        ("encodage", build_preprocessor(config)),
    ]


def feature_names(config: AppConfig) -> list[str]:
    """Liste lisible des features, journalisée comme artefact MLflow."""
    return [*config.features.numeric_columns, *config.features.categorical_columns]
