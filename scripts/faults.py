"""Injection d'incidents contrôles, pour la démonstration.

Trois incidents, qui correspondent aux scénarios de la soutenance :

1. `lot-invalide`  — fabrique un lot journalier corrompu (colonne SMART
   supprimée, `failure = 2`, date incohérente avec le nom du fichier, serial
   dupliqué). Attendu : le contrat de données bloqué, une alerte part, aucun
   modèle n'est publié, l'API continue de servir le champion.
2. `lot-quarantaine` — injecte des `capacity_bytes = -1` sous le seuil de
   tolérance. Attendu : le lot passe, les lignes fautives sont mises en
   quarantaine et comptées dans Grafana.
3. `dependance`    — arrête un service Compose (MLflow ou PostgreSQL).
   Attendu : retries, échec propre avec alerte, `/health` en dégradé ou 503,
   et l'API qui ne plante pas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from hdd_rul.common.config import load_config
from hdd_rul.common.logging import get_logger, setup_logging
from hdd_rul.ingestion.landing import batch_date_from_name

logger = get_logger(__name__)


def corrupt_batch(source: Path, destination: Path) -> Path:
    """Écrit une copie corrompue d'un lot journalier (4 violations bloquantes)."""
    frame = pd.read_csv(source, low_memory=False)
    if frame.empty:
        raise ValueError(f"lot source vide: {source}")

    # 1. valeur hors domaine : failure doit valoir 0 ou 1
    frame.loc[frame.index[:3], "failure"] = 2
    # 2. date incohérente avec le nom du fichier
    batch_day = batch_date_from_name(source)
    if batch_day is not None:
        frame.loc[frame.index[4], "date"] = (batch_day + timedelta(days=1)).isoformat()
    # 3. colonne SMART obligatoire absente
    if "smart_197_raw" in frame.columns:
        frame = frame.drop(columns=["smart_197_raw"])
    # 4. numéro de série dupliqué dans la même journée
    frame.loc[frame.index[5], "serial_number"] = frame.loc[frame.index[0], "serial_number"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    logger.warning("lot corrompu écrit: %s (source: %s)", destination, source.name)
    return destination


def quarantine_batch(source: Path, destination: Path, fraction: float = 0.02) -> Path:
    """Copie un lot avec quelques `capacity_bytes = -1` (règle non bloquante)."""
    frame = pd.read_csv(source, low_memory=False)
    count = max(1, int(len(frame) * fraction))
    frame.loc[frame.index[:count], "capacity_bytes"] = -1
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    logger.warning("%d ligne(s) à capacité invalide dans %s", count, destination.name)
    return destination


def stop_service(service: str) -> int:
    """Arrête un service Compose pour simuler une dépendance indisponible."""
    logger.warning("arrêt du service %s", service)
    return subprocess.run(["docker", "compose", "stop", service], check=False).returncode


def start_service(service: str) -> int:
    logger.info("redémarrage du service %s", service)
    return subprocess.run(["docker", "compose", "start", service], check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Injecte un incident contrôlé")
    parser.add_argument(
        "incident",
        choices=["lot-invalide", "lot-quarantaine", "dependance", "retablir"],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="CSV journalier à corrompre")
    parser.add_argument("--service", default="mlflow", help="service Compose à arrêter")
    parser.add_argument(
        "--clean-landing",
        action="store_true",
        help="vide d'abord data/landing/ pour que seul le lot injecté soit en attente",
    )
    args = parser.parse_args(argv)

    setup_logging()
    config = load_config(args.config)
    config.paths.ensure()

    if args.incident in {"dependance", "retablir"}:
        action = stop_service if args.incident == "dependance" else start_service
        return action(args.service)

    if args.source:
        source = Path(args.source)
    else:
        candidates = sorted(config.paths.raw.glob("*.csv"))
        if not candidates:
            logger.error("aucun CSV dans %s : générer d'abord les données", config.paths.raw)
            return 2
        source = candidates[len(candidates) // 2]

    if args.clean_landing:
        for leftover in config.paths.landing.glob("*.csv"):
            leftover.unlink()
            logger.info("lot en attente retiré: %s", leftover.name)

    target = config.paths.landing / source.name
    if args.incident == "lot-invalide":
        corrupt_batch(source, target)
    else:
        quarantine_batch(source, target)
    logger.info("lot déposé dans %s", target)
    # Seule la sortie standard porte le chemin : les scripts de démonstration
    # l'utilisent directement (`--batch "$LOT"`).
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
