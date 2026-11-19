"""Enregistrement et exécution des deployments Prefect.

`python -m hdd_rul.orchestration.deployments` enregistre les trois deployments
et lance un *runner* qui les exécute. Les déclencheurs :

* `daily-batch-landing`   — scrute `data/landing/` toutes les 30 s. C'est le
  déclenchement « automatique » attendu : déposer un fichier suffit.
* `train-pipeline-nightly`— planification quotidienne (cron).
* `ingest-quarter-manual` — à la demande, pour (re)construire la couche bronze.

Les intervalles sont volontairement courts en démonstration : l'objectif est de
voir le cycle complet pendant une soutenance, pas de tenir un vrai rythme
quotidien.
"""

from __future__ import annotations

import argparse
import os

from prefect import serve
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy

from hdd_rul.common.logging import get_logger, setup_logging
from hdd_rul.orchestration.flows import daily_batch, ingest_quarter, train_pipeline

logger = get_logger(__name__)

# Un seul run à la fois par deployment. La scrutation part toutes les 30 s, et un
# lot réel (~345 000 lignes) prend plus longtemps : sans limite, plusieurs runs
# chargeaient le même lot en parallèle jusqu'à l'OOM du conteneur (SIGKILL).
# CANCEL_NEW plutôt qu'ENQUEUE : un tick de scrutation manque ne perd rien, le
# lot reste dans data/landing/ et sera pris au tick suivant.
ONE_AT_A_TIME = ConcurrencyLimitConfig(
    limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
)

DEFAULT_CONFIG = os.environ.get("HDD_RUL_CONFIG", "config/base.yaml")


def build_deployments(config_path: str, *, batch_interval: int, train_cron: str) -> list:
    """Construit les trois deployments sans les exécuter (testable)."""
    return [
        daily_batch.to_deployment(
            name="daily-batch-landing",
            description="Traite le plus ancien lot en attente dans data/landing/",
            parameters={"config_path": config_path},
            interval=batch_interval,
            concurrency_limit=ONE_AT_A_TIME,
            tags=["hdd-rul", "batch"],
        ),
        train_pipeline.to_deployment(
            name="train-pipeline-nightly",
            description="Réentraînement planifié, avec règle de mise en service",
            parameters={"config_path": config_path, "trigger": "schedule"},
            cron=train_cron,
            concurrency_limit=ONE_AT_A_TIME,
            tags=["hdd-rul", "training"],
        ),
        ingest_quarter.to_deployment(
            name="ingest-quarter-manual",
            description="Téléchargement d'un trimestre et reconstruction de la couche bronze",
            parameters={"config_path": config_path},
            tags=["hdd-rul", "ingestion"],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sert les deployments Prefect du projet")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--batch-interval",
        type=int,
        default=int(os.environ.get("HDD_RUL_BATCH_INTERVAL", "30")),
        help="intervalle en secondes de la scrutation de data/landing/",
    )
    parser.add_argument(
        "--train-cron",
        default=os.environ.get("HDD_RUL_TRAIN_CRON", "17 2 * * *"),
        help="planification du réentraînement (cron)",
    )
    args = parser.parse_args(argv)
    setup_logging()

    deployments = build_deployments(
        args.config, batch_interval=args.batch_interval, train_cron=args.train_cron
    )
    logger.info(
        "mise à disposition de %d deployments (config=%s, scrutation=%ss, cron=%r)",
        len(deployments),
        args.config,
        args.batch_interval,
        args.train_cron,
    )
    serve(*deployments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
