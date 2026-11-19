"""Simulateur de rejeu : fait avancer le temps sur un jeu historique figé.

Le trimestre de simulation (`data.quarters.replay`) est rejoué **jour par
jour** : le script copie `2026-04-07.csv` dans `data/landing/`, ce qui est le
déclencheur du flow `daily_batch`. Chaque jour rejoué produit donc des
prédictions, des feedbacks et des traces, exactement comme une arrivée réelle.

C'est la réponse explicite à la contrainte du sujet : « un dataset historique
peut être rejoué par lots pour simuler l'arrivée de nouvelles données, à
condition de le préciser ». C'est précisé ici, dans le README et dans la
synthèse.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.common.logging import get_logger, setup_logging
from hdd_rul.ingestion.landing import drop_batch

logger = get_logger(__name__)


def available_days(config: AppConfig) -> list[Path]:
    """CSV journaliers disponibles dans `data/raw/`, triés par date."""
    return sorted(p for p in config.paths.raw.glob("*.csv") if not p.name.startswith("_"))


def replay_start(config: AppConfig) -> date | None:
    """Premier jour du trimestre de simulation (`data.quarters.replay`).

    `Q2_2026` donne le 2026-04-01. Sans cette borne, le rejeu repartait du
    premier fichier de `data/raw/` (le 2025-10-01, trimestre d'historique) et
    retraitait des journées qui ont servi à l'entraînement.
    """
    debuts = []
    for quarter in config.data.quarters.replay:
        match = re.fullmatch(r"Q([1-4])_(\d{4})", quarter)
        if match:
            debuts.append(date(int(match.group(2)), 3 * int(match.group(1)) - 2, 1))
    return min(debuts) if debuts else None


def select_days(
    config: AppConfig,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Jours à rejouer, filtres par fenêtre puis tronqués à `limit`.

    Par défaut on rejoue à partir du jour qui suit le dernier lot déjà traité :
    le rejeu est donc reprenable, et relancer la commande ne retraite pas deux
    fois la même journée.
    """
    days = available_days(config)
    processed = {p.name for p in config.paths.processed.glob("*.csv")}
    pending = {p.name for p in config.paths.landing.glob("*.csv")}
    days = [p for p in days if p.name not in processed and p.name not in pending]

    start = start or replay_start(config)
    if start is not None:
        days = [p for p in days if p.stem >= start.isoformat()]
    if end is not None:
        days = [p for p in days if p.stem <= end.isoformat()]
    if limit is not None:
        days = days[:limit]
    return days


def replay_days(
    config: AppConfig,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = 5,
    delay_seconds: float = 0.0,
    run_flow: bool = True,
) -> list[Path]:
    """Dépose les lots un par un et déclenche le flow pour chacun."""
    days = select_days(config, start=start, end=end, limit=limit)
    if not days:
        logger.warning("aucun jour à rejouer (déjà traité, ou fenêtre vide)")
        return []

    deposited: list[Path] = []
    for source in days:
        target = drop_batch(config, source)
        deposited.append(target)
        if run_flow:
            from hdd_rul.orchestration.flows import daily_batch

            outcome = daily_batch(
                config_path=str(config.source_path) if config.source_path else None,
                batch_path=str(target),
            )
            logger.info("lot %s: %s", target.name, outcome.get("status"))
        if delay_seconds:
            # Pause volontaire : simule l'écoulement du temps entre deux jours
            # pour que la démonstration soit lisible dans l'interface Prefect.
            time.sleep(delay_seconds)
    return deposited


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rejoue les lots journaliers")
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default=None, help="AAAA-MM-JJ")
    parser.add_argument("--end", default=None, help="AAAA-MM-JJ")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--deposit-only", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    config = load_config(args.config)
    deposited = replay_days(
        config,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        limit=args.limit,
        delay_seconds=args.delay,
        run_flow=not args.deposit_only,
    )
    print(f"{len(deposited)} lot(s) rejoué(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
