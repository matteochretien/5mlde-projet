"""Dépôt `data/landing/` : point d'entrée des lots journaliers rejoués.

Le simulateur y dépose `2026-04-07.csv`; le flow `daily_batch` prend le plus
ancien fichier en attente, le traite, puis le déplace dans `data/processed/`
(ou le laisse en place en cas d'échec, pour que la reprise soit possible).
"""

from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


def batch_date_from_name(path: str | Path) -> date | None:
    """Date portée par le nom du fichier, pour la confronter à la colonne `date`."""
    match = _DATE_IN_NAME.search(Path(path).name)
    return date.fromisoformat(match.group(1)) if match else None


def list_pending_batches(config: AppConfig) -> list[Path]:
    config.paths.landing.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in config.paths.landing.glob("*.csv") if not p.name.startswith("_"))


def claim_batch(config: AppConfig) -> Path | None:
    """Renvoie le plus ancien lot en attente, sans le déplacer."""
    pending = list_pending_batches(config)
    if not pending:
        return None
    return pending[0]


def latest_processed_day(config: AppConfig) -> date | None:
    """Date du dernier lot journalier traité (`processed/`), ou None.

    C'est « aujourd'hui » du point de vue du pipeline : en rejeu comme en
    production, rien de plus récent n'a été reçu.
    """
    days = [
        day
        for day in (batch_date_from_name(p) for p in config.paths.processed.glob("*.csv"))
        if day is not None
    ]
    return max(days) if days else None


def known_failed_serials_before(config: AppConfig, batch_day: date) -> tuple[str, ...]:
    """Disques déjà déclarés en panne avant `batch_day`.

    Deux sources, causales toutes les deux (rien après `batch_day`) :
      * l'historique ingéré (`bronze/drive_events.parquet`, `failure_date`) ;
      * les lots journaliers déjà traités (`bronze/landing_clean/*.parquet`).
    Alimente la règle de non-résurrection du contrat de données. Une source
    absente est simplement ignorée : la règle se contente de ce qui est connu.
    """
    import pandas as pd

    cutoff = pd.Timestamp(batch_day)
    serials: set[str] = set()

    events = config.paths.bronze / "drive_events.parquet"
    if events.exists():
        frame = pd.read_parquet(events, columns=["serial_number", "failure_date"])
        failed = frame["failure_date"].notna() & (pd.to_datetime(frame["failure_date"]) < cutoff)
        serials.update(frame.loc[failed, "serial_number"].astype(str))

    clean_dir = config.paths.bronze / "landing_clean"
    for lot in sorted(clean_dir.glob("*.parquet")) if clean_dir.exists() else []:
        lot_day = batch_date_from_name(lot)
        if lot_day is None or lot_day >= batch_day:
            continue
        frame = pd.read_parquet(lot, columns=["serial_number", "failure"])
        failed = pd.to_numeric(frame["failure"], errors="coerce").fillna(0) >= 1
        serials.update(frame.loc[failed, "serial_number"].astype(str))

    return tuple(sorted(serials))


def mark_processed(config: AppConfig, batch: Path) -> Path:
    config.paths.processed.mkdir(parents=True, exist_ok=True)
    destination = config.paths.processed / batch.name
    shutil.move(str(batch), destination)
    logger.info("lot %s traité -> %s", batch.name, destination)
    return destination


def drop_batch(config: AppConfig, source: Path, *, batch_date: date | None = None) -> Path:
    """Copie un CSV journalier dans `landing/` (utilisé par le simulateur de rejeu)."""
    config.paths.landing.mkdir(parents=True, exist_ok=True)
    name = f"{batch_date.isoformat()}.csv" if batch_date else source.name
    destination = config.paths.landing / name
    shutil.copyfile(source, destination)
    logger.info("lot déposé dans landing: %s", destination.name)
    return destination
