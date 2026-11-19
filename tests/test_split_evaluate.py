"""Anti-fuite du split et métriques d'évaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hdd_rul.common.config import AppConfig
from hdd_rul.training.evaluate import evaluate_predictions
from hdd_rul.training.split import assign_validation_drives, temporal_group_split


def _jeu(n_drives: int = 60, n_days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    for drive in range(n_drives):
        for day in dates:
            rows.append(
                {
                    "date": day,
                    "serial_number": f"ZA{drive:08d}",
                    "drive_kind": "failed" if drive % 2 else "healthy",
                    "rul": float((drive * 7 + day.dayofyear) % 91),
                }
            )
    return pd.DataFrame(rows)


def test_aucun_disque_des_deux_cotes(config: AppConfig) -> None:
    """La fuite par groupe est la plus dangereuse : deux jours d'un disque se ressemblent."""
    split = temporal_group_split(_jeu(), config)
    communs = set(split.train["serial_number"]) & set(split.valid["serial_number"])
    assert communs == set()


def test_la_validation_est_posterieure_a_l_entrainement(config: AppConfig) -> None:
    split = temporal_group_split(_jeu(), config)
    assert split.train["date"].max() < split.valid["date"].min()
    assert split.cutoff_date == str(split.train["date"].max().date())


def test_le_split_est_reproductible(config: AppConfig) -> None:
    premier = temporal_group_split(_jeu(), config)
    second = temporal_group_split(_jeu(), config)
    assert set(premier.valid["serial_number"]) == set(second.valid["serial_number"])


def test_affectation_des_disques_respecte_la_proportion() -> None:
    serials = pd.Series([f"ZA{i:08d}" for i in range(4_000)])
    part = assign_validation_drives(serials, 0.2).mean()
    assert 0.17 <= part <= 0.23


def test_jeu_vide_refuse(config: AppConfig) -> None:
    with pytest.raises(ValueError, match="vide"):
        temporal_group_split(pd.DataFrame(), config)


def test_metriques_sur_prediction_parfaite() -> None:
    truth = np.array([0.0, 10.0, 45.0, 90.0])
    metrics = evaluate_predictions(truth, truth, alert_threshold_days=30)
    assert metrics.mae == 0.0
    assert metrics.mae_lt30 == 0.0
    assert metrics.alert_precision == 1.0
    assert metrics.alert_recall == 1.0


def test_mae_zone_critique_ne_porte_que_sur_les_rul_courtes() -> None:
    truth = np.array([5.0, 10.0, 80.0])
    prediction = np.array([7.0, 14.0, 20.0])
    metrics = evaluate_predictions(truth, prediction, alert_threshold_days=30)
    assert metrics.mae == pytest.approx((2 + 4 + 60) / 3)
    assert metrics.mae_lt30 == pytest.approx((2 + 4) / 2)  # la ligne à 80 j est exclue


def test_alerte_precision_et_rappel() -> None:
    truth = np.array([5.0, 40.0, 10.0, 70.0])
    prediction = np.array([20.0, 25.0, 50.0, 80.0])
    metrics = evaluate_predictions(truth, prediction, alert_threshold_days=30)
    # Alertes prédites : lignes 0 et 1 ; vraies alertes : lignes 0 et 2.
    assert metrics.alert_precision == pytest.approx(0.5)
    assert metrics.alert_recall == pytest.approx(0.5)


def test_tailles_incompatibles_refusees() -> None:
    with pytest.raises(ValueError, match="incompatibles"):
        evaluate_predictions(np.array([1.0, 2.0]), np.array([1.0]))
