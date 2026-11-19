"""Feedbacks : reconstitution de la RUL réellement observée.

Le label d'une prédiction n'est pas connu au moment où on la sert : il arrive
plus tard, quand le disque tombe en panne, ou quand on constate qu'il a survécu
à l'horizon. Pendant le rejeu, ces deux événements se lisent dans le lot du
jour, donc **les feedbacks arrivent automatiquement**, sans intervention :

* un disque passe à `failure = 1` le jour D : pour chacune de ses prédictions
  passées datée `d`, la vraie RUL vaut `min(D - d, H)` ;
* un disque est toujours présent le jour D : les prédictions datées `D - H` ont
  une vraie RUL de `H` exactement (il a survécu à tout l'horizon).

C'est ce qui permet de mesurer la MAE *en production* et de la comparer, dans
Grafana, à la MAE de validation annoncée par le modèle.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from hdd_rul.common.logging import get_logger
from hdd_rul.common.tracing import TracingStore

logger = get_logger(__name__)


def build_feedbacks(
    predictions: pd.DataFrame,
    batch: pd.DataFrame,
    batch_date: date,
    horizon_days: int,
) -> list[dict[str, Any]]:
    """Dérive les feedbacks d'un lot journalier à partir des prédictions passées.

    `predictions` : colonnes `serial_number`, `observation_date`.
    `batch`       : lot du jour, colonnes `serial_number`, `failure`.
    Fonction pure : testable sans base de données.
    """
    if predictions.empty or batch.empty:
        return []

    observed = predictions.copy()
    observed["observation_date"] = pd.to_datetime(observed["observation_date"]).dt.date

    failures = batch.loc[pd.to_numeric(batch["failure"], errors="coerce") >= 1, "serial_number"]
    failed = set(failures.astype(str))
    present = set(batch["serial_number"].astype(str))

    feedbacks: list[dict[str, Any]] = []
    horizon_cutoff = batch_date - timedelta(days=horizon_days)

    for row in observed.itertuples(index=False):
        serial = str(row.serial_number)
        observation_date = row.observation_date
        if observation_date is None or observation_date > batch_date:
            continue
        if serial in failed:
            # Panne constatée ce jour : RUL réelle = D - d, plafonnée à H.
            actual = min((batch_date - observation_date).days, horizon_days)
            feedbacks.append(
                {
                    "serial_number": serial,
                    "observation_date": observation_date,
                    "actual_rul": float(actual),
                    "failure_date": batch_date,
                    "source": "replay_failure",
                }
            )
        elif serial in present and observation_date == horizon_cutoff:
            # Toujours vivant H jours plus tard : RUL réelle >= H, label exact H.
            feedbacks.append(
                {
                    "serial_number": serial,
                    "observation_date": observation_date,
                    "actual_rul": float(horizon_days),
                    "failure_date": None,
                    "source": "replay_survived",
                }
            )
    return feedbacks


def collect_batch_feedbacks(
    store: TracingStore,
    batch: pd.DataFrame,
    batch_date: date,
    horizon_days: int,
) -> int:
    """Lit les prédictions concernées en base, dérive puis enregistre les feedbacks."""
    serials = batch["serial_number"].astype(str).tolist()
    horizon_cutoff = batch_date - timedelta(days=horizon_days)
    predictions = store.fetch_predictions(serials=serials, until=batch_date)
    if predictions.empty:
        logger.info("aucune prédiction antérieure à réconcilier pour %s", batch_date)
        return 0
    feedbacks = build_feedbacks(predictions, batch, batch_date, horizon_days)
    written = store.record_feedbacks(feedbacks)
    logger.info(
        "%d feedback(s) enregistré(s) pour le lot %s (seuil de survie: %s)",
        written,
        batch_date,
        horizon_cutoff,
    )
    return written
