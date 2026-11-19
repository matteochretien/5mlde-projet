"""Fixtures partagées : configuration de test et jeu de données synthétique."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from hdd_rul.common.config import AppConfig, load_config

# Les tests ne doivent jamais écrire dans la base de traces ni dans un MLflow
# partagé : ces variables sont posées avant tout import de l'API.
os.environ.setdefault("HDD_RUL_TRACING_ENABLED", "0")
# `mise` exporte `.env.example` (HDD_RUL_DATA_DIR=data, MLFLOW_TRACKING_URI, ...).
# Les flows rechargent leur configuration et ces variables l'emportent sur
# l'overlay : sans ce nettoyage, `mise run test` reconstruit la VRAIE couche
# bronze et déposé de vrais lots dans data/landing.
for _variable in (
    "HDD_RUL_CONFIG",
    "HDD_RUL_DATA_DIR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "HDD_RUL_WEBHOOK_URL",
):
    os.environ.pop(_variable, None)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """Configuration de test : petites données, tout dans `tmp_path`."""
    return load_config(
        overrides={
            "paths": {"root": str(tmp_path / "data"), "artifacts": str(tmp_path / "artifacts")},
            "horizon_days": 90,
            "sampling": {"healthy_rate_pct": 60, "healthy_stride_days": 2},
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                "experiment_name": "tests",
                "registered_model_name": "hdd-rul-tests",
            },
            "synthetic": {
                "n_drives": 120,
                "start_date": "2026-01-01",
                "end_date": "2026-06-30",
                "daily_failure_rate": 0.002,
                "seed": 7,
            },
            "training": {"hgb": {"max_iter": 30, "early_stopping": False}},
        }
    )


@pytest.fixture
def synthetic_raw(config: AppConfig) -> AppConfig:
    """Génère le jeu synthétique dans `data/raw` et retourne la configuration."""
    from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats

    generate_synthetic_drive_stats(config)
    return config


@pytest.fixture
def daily_batch_frame() -> pd.DataFrame:
    """Lot journalier minimal mais conforme au contrat de données."""
    day = date(2026, 3, 15)
    rows = []
    for index in range(200):
        rows.append(
            {
                "date": day.isoformat(),
                "serial_number": f"ZA{index:08d}",
                "model": "ST4000DM000" if index % 2 else "TOSHIBA MG07ACA14TA",
                "capacity_bytes": 4_000_787_030_016,
                "failure": 0,
                "smart_5_raw": index % 7,
                "smart_9_raw": 10_000 + index * 3,
                "smart_12_raw": 12,
                "smart_187_raw": 0,
                "smart_188_raw": 0,
                "smart_193_raw": 20_000,
                "smart_194_raw": 30 + (index % 8),
                "smart_197_raw": 0,
                "smart_198_raw": 0,
                "smart_199_raw": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def bronze_samples() -> pd.DataFrame:
    """Échantillon bronze construit à la main, labels connus à l'avance."""
    failure_date = pd.Timestamp("2026-04-10")
    rows = []
    # Disque en panne : RUL observée de 0 à 120 jours (donc plafonnée à 90).
    for delta in (0, 1, 10, 30, 89, 90, 120):
        rows.append(
            {
                "date": failure_date - timedelta(days=delta),
                "serial_number": "FAILED1",
                "model": "ST4000DM000",
                "capacity_bytes": 4_000_787_030_016,
                "failure": 1 if delta == 0 else 0,
                "drive_kind": "failed",
                "failure_date": failure_date,
                "last_seen": failure_date,
                "rul_days_observed": delta,
                "smart_5_raw": 10.0,
                "smart_9_raw": 30_000.0,
                "smart_194_raw": 31.0,
            }
        )
    # Disque sain observé jusqu'au 2026-06-30 : étiquetable jusqu'au 2026-04-01.
    last_seen = pd.Timestamp("2026-06-30")
    for day in ("2026-01-15", "2026-04-01", "2026-05-01", "2026-06-20"):
        rows.append(
            {
                "date": pd.Timestamp(day),
                "serial_number": "HEALTHY1",
                "model": "WDC WUH721414ALE6L4",
                "capacity_bytes": 14_000_519_643_136,
                "failure": 0,
                "drive_kind": "healthy",
                "failure_date": pd.NaT,
                "last_seen": last_seen,
                "rul_days_observed": None,
                "smart_5_raw": 0.0,
                "smart_9_raw": 12_000.0,
                "smart_194_raw": 29.0,
            }
        )
    return pd.DataFrame(rows)
