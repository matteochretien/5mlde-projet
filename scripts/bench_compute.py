"""Coût de calcul du champion, hors HTTP (section 7 de loadtest/PROTOCOLE.md).

Charge le pipeline servi (alias `champion`), puis chronomètre `predict` sur des
lots de taille croissante tirés de la couche bronze, avec 1 thread OpenMP puis
avec tous les cœurs. Sépare le coût du modèle du coût de l'API (HTTP, JSON,
validation Pydantic) : c'est ce qui permet de dire si un plafond de débit vient
du serveur ou du modèle.

    uv run python scripts/bench_compute.py [--repetitions 30]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import mlflow
import pandas as pd
from threadpoolctl import threadpool_limits

from hdd_rul.common.config import load_config
from hdd_rul.features.build import select_feature_frame
from hdd_rul.ingestion.bronze import load_bronze_samples
from hdd_rul.training.registry import ALIAS_CHAMPION, configure_mlflow

TAILLES = (1, 10, 100, 1_000, 10_000)


def chrono(pipeline, frame: pd.DataFrame, repetitions: int) -> float:
    """Médiane du temps d'un appel `predict`, en millisecondes."""
    pipeline.predict(frame)  # chauffe
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        pipeline.predict(frame)
        durees.append((time.perf_counter() - debut) * 1000)
    return statistics.median(durees)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()

    config = load_config()
    configure_mlflow(config)
    uri = f"models:/{config.mlflow.registered_model_name}@{ALIAS_CHAMPION}"
    pipeline = mlflow.sklearn.load_model(uri)
    echantillon = load_bronze_samples(config).sample(max(TAILLES), random_state=0)
    features = select_feature_frame(echantillon, config)

    print(f"modèle : {uri} | cœurs : {os.cpu_count()} | répétitions : {args.repetitions}\n")
    print(
        "| Taille du lot | 1 thread : ms/appel | 1 thread : lignes/s "
        f"| {os.cpu_count()} threads : ms/appel | {os.cpu_count()} threads : lignes/s |"
    )
    print("|---|---|---|---|---|")
    for taille in TAILLES:
        lot = features.head(taille)
        with threadpool_limits(limits=1):
            un = chrono(pipeline, lot, args.repetitions)
        with threadpool_limits(limits=os.cpu_count()):
            tous = chrono(pipeline, lot, args.repetitions)
        print(
            f"| {taille:,} | {un:.1f} | {taille / un * 1000:,.0f} "
            f"| {tous:.1f} | {taille / tous * 1000:,.0f} |".replace(",", " ")
        )


if __name__ == "__main__":
    main()
