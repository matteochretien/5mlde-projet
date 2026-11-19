"""Le calcul du label RUL est le cœur du prétraitement : il est testé en détail."""

from __future__ import annotations

from datetime import date

import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.features.labels import compute_labels, resolve_training_window


def test_rul_des_disques_en_panne_est_plafonnee(
    config: AppConfig, bronze_samples: pd.DataFrame
) -> None:
    labelled, stats = compute_labels(bronze_samples, config)
    failed = labelled.loc[labelled["serial_number"] == "FAILED1"].set_index("rul_days_observed")

    assert failed.loc[0, "rul"] == 0  # jour de la panne
    assert failed.loc[30, "rul"] == 30
    assert failed.loc[90, "rul"] == 90
    # Une ligne antérieure à la fenêtre garde un label *exact* : sa vraie RUL
    # (120 j) dépasse l'horizon, donc le label plafonné vaut bien 90.
    assert failed.loc[120, "rul"] == 90
    assert stats.rows_failed == 7
    assert stats.rows_dropped_invalid == 0


def test_les_sains_ne_sont_etiquetes_que_si_l_horizon_est_observe(
    config: AppConfig, bronze_samples: pd.DataFrame
) -> None:
    labelled, stats = compute_labels(bronze_samples, config)
    healthy = labelled.loc[labelled["serial_number"] == "HEALTHY1"]

    # last_seen = 2026-06-30 ; seules les lignes datées avant le 2026-04-01
    # inclus ont 90 jours d'observation derrière elles.
    assert set(healthy["date"].dt.date) == {date(2026, 1, 15), date(2026, 4, 1)}
    assert set(healthy["rul"]) == {90.0}
    assert stats.rows_dropped_censored == 2  # les deux lignes trop récentes


def test_horizon_configurable(config: AppConfig, bronze_samples: pd.DataFrame) -> None:
    court = config.model_copy(update={"horizon_days": 30})
    labelled, _ = compute_labels(bronze_samples, court)
    assert labelled["rul"].max() == 30.0


def test_fenetre_d_entrainement_ne_garde_que_les_labels_resolus(
    config: AppConfig, bronze_samples: pd.DataFrame
) -> None:
    labelled, _ = compute_labels(bronze_samples, config)

    # Au 2026-02-01, la panne du 2026-04-10 n'est pas encore connue et aucune
    # ligne n'a 90 jours d'ancienneté : la fenêtre doit être vide.
    window, bounds = resolve_training_window(
        labelled, as_of=date(2026, 2, 1), horizon_days=config.horizon_days
    )
    assert window.empty
    assert bounds["as_of"] == "2026-02-01"

    # Au 2026-05-01, la panne est connue : toutes ses lignes deviennent utilisables.
    window, _ = resolve_training_window(
        labelled, as_of=date(2026, 5, 1), horizon_days=config.horizon_days
    )
    assert (window["serial_number"] == "FAILED1").sum() == 7


def test_fenetre_glissante_limite_l_anciennete(
    config: AppConfig, bronze_samples: pd.DataFrame
) -> None:
    labelled, _ = compute_labels(bronze_samples, config)
    window, bounds = resolve_training_window(
        labelled, as_of=date(2026, 5, 1), window_days=30, horizon_days=config.horizon_days
    )
    assert bounds["window_start"] >= "2026-03-11"
    assert not window.empty


def test_rul_negative_est_ecartee(config: AppConfig, bronze_samples: pd.DataFrame) -> None:
    """Une ligne datée après la panne du disque est une incohérence, pas un label."""
    corrupted = bronze_samples.copy()
    corrupted.loc[0, "rul_days_observed"] = -5
    labelled, stats = compute_labels(corrupted, config)
    assert stats.rows_dropped_invalid == 1
    assert (labelled["rul"] >= 0).all()
