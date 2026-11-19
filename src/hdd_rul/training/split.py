"""Split groupe par disque ET temporel.

Deux fuites de données sont possibles sur ce jeu, et il faut traiter les deux :

* **fuite par groupe** — deux jours consécutifs d'un même disque sont presque
  identiques (même age, mêmes compteurs SMART à quelques unités près). Un disque
  présent des deux côtés donnerait une MAE de validation artificiellement basse.
  Les disques sont donc répartis par un hachage déterministe de leur numéro de
  série : un disque est *entièrement* en entraînement ou *entièrement* en
  validation.
* **fuite temporelle** — valider sur des dates mélangées à l'entraînement revient
  à prédire le passe. On fixe donc une date de coupure : l'entraînement ne voit
  que les lignes antérieures, la validation que les lignes postérieures.

Les deux contraintes sont appliquées ensemble, ce qui écarte des lignes (celles
des disques de validation avant la coupure, et celles des disques
d'entraînement après). C'est le prix à payer pour une validation honnête ; le
compte est journalisé et tracé en tag MLflow.

Une coupure purement temporelle sans la contrainte de groupe ne marche pas ici :
un disque sain vit des années et chevauche n'importe quelle coupure, si bien que
l'anti-fuite viderait la validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.common.hashing import stable_bucket
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

SPLIT_SALT = "hdd-rul-split"


@dataclass
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    cutoff_date: str
    valid_drives: int
    train_drives: int
    rows_dropped_leakage: int

    def as_tags(self) -> dict[str, str]:
        return {
            "split.cutoff_date": self.cutoff_date,
            "split.rows_train": str(len(self.train)),
            "split.rows_valid": str(len(self.valid)),
            "split.drives_train": str(self.train_drives),
            "split.drives_valid": str(self.valid_drives),
            "split.rows_dropped_leakage": str(self.rows_dropped_leakage),
        }


def assign_validation_drives(serials: pd.Series, valid_fraction: float) -> pd.Series:
    """Masqué booléen : True pour les disques affectés à la validation.

    Déterministe (hachage du numéro de série) : la composition des deux
    ensembles ne change pas d'une exécution à l'autre, ce qui rend les MAE
    comparables entre runs — condition nécessaire à la règle de promotion.
    """
    buckets = serials.map(lambda serial: stable_bucket(serial, salt=SPLIT_SALT, buckets=1000))
    return buckets < round(valid_fraction * 1000)


def temporal_group_split(frame: pd.DataFrame, config: AppConfig) -> SplitResult:
    """Coupe le jeu étiqueté en entraînement / validation."""
    if frame.empty:
        raise ValueError("jeu étiqueté vide : rien à entraîner")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Date de coupure : la validation porte sur les dates les plus récentes,
    # en gardant les journées entières.
    daily_counts = data.groupby("date").size().sort_index()
    cumulative = daily_counts.cumsum() / daily_counts.sum()
    target = 1.0 - config.split.valid_fraction
    candidates = cumulative.loc[cumulative >= target]
    cutoff = candidates.index[0] if len(candidates) else daily_counts.index[-1]

    is_valid_drive = assign_validation_drives(data["serial_number"], config.split.valid_fraction)
    before_cutoff = data["date"] <= cutoff

    train = data.loc[~is_valid_drive & before_cutoff]
    valid = data.loc[is_valid_drive & ~before_cutoff]
    dropped = int(len(data) - len(train) - len(valid))

    logger.info(
        "split: coupure %s | train=%d lignes / %d disques | valid=%d lignes / %d disques "
        "| %d lignes écartées pour éviter les fuites",
        cutoff.date(),
        len(train),
        train["serial_number"].nunique(),
        len(valid),
        valid["serial_number"].nunique(),
        dropped,
    )
    if train.empty or valid.empty:
        raise ValueError(
            "entraînement ou validation vide après anti-fuite : élargir la fenêtre de "
            "données, augmenter split.valid_fraction ou le taux d'échantillonnage des sains"
        )
    return SplitResult(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        cutoff_date=str(cutoff.date()),
        valid_drives=int(valid["serial_number"].nunique()),
        train_drives=int(train["serial_number"].nunique()),
        rows_dropped_leakage=dropped,
    )
