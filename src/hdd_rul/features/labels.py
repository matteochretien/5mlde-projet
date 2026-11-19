"""Calcul du label RUL : le cœur du prétraitement.

Définition retenue : `rul = date_panne - date`, **plafonnée à H jours**
(`horizon_days`). La valeur H signifie « pas de panne en vue dans l'horizon ».

Deux populations, deux règles :

* **disques en panne** — la date de panne est connue (`failure = 1` ce jour-là,
  dernière ligne du disque). Le label de chaque ligne de la fenêtre
  [date_panne - H, date_panne] vaut `min(date_panne - date, H)`.
* **disques sains** — leur vraie RUL est inconnue (censure à droite). Une ligne
  datée `d` n'est étiquetée que si le disque est **encore observé vivant à
  `d + H`**. Alors RUL >= H, donc le label plafonné H est *exact* : pas de biais
  de censure. Les lignes trop récentes pour cela sont écartées, avec le motif.

Cette séparation est la raison pour laquelle le label est « en retard » : une
ligne du jour n'est étiquetable que lorsque le disque tombe en panne, ou H jours
plus tard. Chaque réentraînement travaille donc sur une fenêtre glissante de
lignes dont le label est résolu (cf. `resolve_training_window`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

LABEL_COLUMN = "rul"


@dataclass
class LabelStats:
    """Comptes utiles en tags MLflow et dans les logs du flow."""

    rows_in: int
    rows_labelled: int
    rows_dropped_censored: int
    rows_dropped_invalid: int
    rows_failed: int
    rows_healthy: int
    label_min: float | None
    label_max: float | None
    label_mean: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "rows_in": self.rows_in,
            "rows_labelled": self.rows_labelled,
            "rows_dropped_censored": self.rows_dropped_censored,
            "rows_dropped_invalid": self.rows_dropped_invalid,
            "rows_failed": self.rows_failed,
            "rows_healthy": self.rows_healthy,
            "label_min": self.label_min,
            "label_max": self.label_max,
            "label_mean": self.label_mean,
        }


def compute_labels(samples: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, LabelStats]:
    """Ajoute la colonne `rul` et écarte les lignes non étiquetables.

    Attend les colonnes produites par la couche bronze : `date`, `drive_kind`,
    `failure_date`, `last_seen`, `rul_days_observed`.
    """
    horizon = config.horizon_days
    frame = samples.copy()
    rows_in = len(frame)

    for column in ("date", "failure_date", "last_seen"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])

    is_failed = frame["drive_kind"].eq("failed")
    observed = pd.to_numeric(frame.get("rul_days_observed"), errors="coerce")

    # Disques en panne : RUL observée, plafonnée à H. Une ligne antérieure à la
    # fenêtre (RUL observée > H) reste correctement étiquetée H : sa vraie RUL
    # dépasse l'horizon, donc le label plafonné est exact. Une RUL négative, en
    # revanche, est une incohérence (ligne postérieure à la panne) : on l'écarte.
    label = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    valid_failed = is_failed & observed.notna() & (observed >= 0)
    label[valid_failed] = observed[valid_failed].clip(upper=horizon)

    # Disques sains : label exact H si et seulement si le disque est encore
    # observé à date + H ; sinon la vraie RUL est censurée.
    healthy = ~is_failed
    horizon_delta = pd.to_timedelta(horizon, unit="D")
    survived_horizon = healthy & (frame["last_seen"] >= frame["date"] + horizon_delta)
    label[survived_horizon] = float(horizon)

    frame[LABEL_COLUMN] = label
    censored = int((healthy & ~survived_horizon).sum())
    invalid = int((is_failed & ~valid_failed).sum())

    labelled = frame.loc[frame[LABEL_COLUMN].notna()].copy()
    labelled[LABEL_COLUMN] = labelled[LABEL_COLUMN].astype(float)
    labelled = labelled.loc[labelled[LABEL_COLUMN].between(0, horizon)]

    stats = LabelStats(
        rows_in=rows_in,
        rows_labelled=len(labelled),
        rows_dropped_censored=censored,
        rows_dropped_invalid=invalid,
        rows_failed=int(labelled["drive_kind"].eq("failed").sum()),
        rows_healthy=int(labelled["drive_kind"].eq("healthy").sum()),
        label_min=float(labelled[LABEL_COLUMN].min()) if len(labelled) else None,
        label_max=float(labelled[LABEL_COLUMN].max()) if len(labelled) else None,
        label_mean=float(labelled[LABEL_COLUMN].mean()) if len(labelled) else None,
    )
    logger.info(
        "labels: %d/%d lignes étiquetées (%d pannes, %d sains), %d censurées, %d invalides",
        stats.rows_labelled,
        stats.rows_in,
        stats.rows_failed,
        stats.rows_healthy,
        stats.rows_dropped_censored,
        stats.rows_dropped_invalid,
    )
    return labelled.reset_index(drop=True), stats


def resolve_training_window(
    labelled: pd.DataFrame,
    *,
    as_of: date | None = None,
    window_days: int | None = None,
    horizon_days: int = 90,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Restreint le jeu d'entraînement aux lignes dont le label est résolu à `as_of`.

    C'est ce qui fait bouger les périodes d'entraînement d'un run à l'autre : à
    la date `as_of`, une ligne n'est utilisable que si son label était déjà
    connu, c'est-à-dire si le disque est tombe en panne avant `as_of` ou si
    `date + H <= as_of`. `window_days` limite en plus l'ancienneté des lignes.
    """
    if labelled.empty:
        return labelled, {"window_start": "", "window_end": "", "as_of": str(as_of or "")}

    frame = labelled.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        failure_known = pd.to_datetime(frame["failure_date"]).le(as_of_ts)
        horizon_elapsed = frame["date"] + pd.to_timedelta(horizon_days, unit="D") <= as_of_ts
        frame = frame.loc[failure_known.fillna(False) | horizon_elapsed]
    if window_days is not None and not frame.empty:
        newest = frame["date"].max()
        frame = frame.loc[frame["date"] >= newest - pd.to_timedelta(window_days, unit="D")]

    bounds = {
        "window_start": str(frame["date"].min().date()) if not frame.empty else "",
        "window_end": str(frame["date"].max().date()) if not frame.empty else "",
        "as_of": str(as_of or (frame["date"].max().date() if not frame.empty else "")),
    }
    logger.info(
        "fenêtre d'entraînement résolue: %s -> %s (%d lignes, as_of=%s)",
        bounds["window_start"],
        bounds["window_end"],
        len(frame),
        bounds["as_of"],
    )
    return frame.reset_index(drop=True), bounds
