"""Dérivation automatique des feedbacks pendant le rejeu."""

from __future__ import annotations

from datetime import date

import pandas as pd

from hdd_rul.serving.feedback import build_feedbacks


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"serial_number": "DEAD1", "observation_date": date(2026, 3, 1)},
            {"serial_number": "DEAD1", "observation_date": date(2026, 3, 20)},
            {"serial_number": "ALIVE1", "observation_date": date(2026, 1, 1)},
            {"serial_number": "ALIVE1", "observation_date": date(2026, 3, 20)},
        ]
    )


def _lot(failure: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"serial_number": "DEAD1", "failure": failure},
            {"serial_number": "ALIVE1", "failure": 0},
        ]
    )


def test_une_panne_resout_toutes_les_predictions_du_disque() -> None:
    feedbacks = build_feedbacks(_predictions(), _lot(), date(2026, 4, 1), horizon_days=90)
    dead = {f["observation_date"]: f for f in feedbacks if f["serial_number"] == "DEAD1"}

    assert dead[date(2026, 3, 1)]["actual_rul"] == 31.0  # 1er mars -> 1er avril
    assert dead[date(2026, 3, 20)]["actual_rul"] == 12.0
    assert dead[date(2026, 3, 1)]["failure_date"] == date(2026, 4, 1)
    assert dead[date(2026, 3, 1)]["source"] == "replay_failure"


def test_la_rul_reelle_est_plafonnee_a_l_horizon() -> None:
    predictions = pd.DataFrame([{"serial_number": "DEAD1", "observation_date": date(2025, 1, 1)}])
    feedbacks = build_feedbacks(predictions, _lot(), date(2026, 4, 1), horizon_days=90)
    assert feedbacks[0]["actual_rul"] == 90.0


def test_survie_a_l_horizon_donne_un_label_exact() -> None:
    """Le disque vu H jours plus tard a une vraie RUL >= H : le label H est exact."""
    feedbacks = build_feedbacks(_predictions(), _lot(), date(2026, 4, 1), horizon_days=90)
    alive = [f for f in feedbacks if f["serial_number"] == "ALIVE1"]
    assert len(alive) == 1
    assert alive[0]["observation_date"] == date(2026, 1, 1)  # 2026-04-01 moins 90 jours
    assert alive[0]["actual_rul"] == 90.0
    assert alive[0]["failure_date"] is None
    assert alive[0]["source"] == "replay_survived"


def test_aucun_feedback_sans_evenement() -> None:
    """Sans panne et sans horizon atteint, il n'y a rien à réconcilier."""
    predictions = pd.DataFrame([{"serial_number": "ALIVE1", "observation_date": date(2026, 3, 25)}])
    assert build_feedbacks(predictions, _lot(failure=0), date(2026, 4, 1), horizon_days=90) == []


def test_lot_ou_predictions_vides() -> None:
    assert build_feedbacks(pd.DataFrame(), _lot(), date(2026, 4, 1), 90) == []
    assert build_feedbacks(_predictions(), pd.DataFrame(), date(2026, 4, 1), 90) == []
