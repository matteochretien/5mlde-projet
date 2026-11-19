"""Agrège les CSV produits par Locust en un tableau Markdown.

Objectif : que les chiffres du protocole de charge soient **relus depuis les
mesures**, et non recopiés à la main.

    uv run python scripts/collect_loadtest_results.py loadtest/results
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

# sw_<classe>_w<workers>_u<utilisateurs>_stats.csv
PATTERN = re.compile(r"^sw_(?P<scenario>\w+?)_w(?P<workers>\d+)_u(?P<users>\d+)_stats\.csv$")
SEUIL_P95_MS = 200.0
SEUIL_ERREURS = 0.01


@dataclass
class Palier:
    scenario: str
    workers: int
    users: int
    requests: int
    failures: int
    rps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @property
    def lignes_par_seconde(self) -> float:
        """Débit en *lignes scorées*, qui est la grandeur utile pour dimensionner."""
        # « PredictionLot100Persist » est aussi un lot de 100 : on lit la taille
        # dans le nom plutôt que dans une table de noms exacts.
        lot = re.search(r"Lot(\d+)", self.scenario)
        return self.rps * (int(lot.group(1)) if lot else 1)

    @property
    def taux_erreur(self) -> float:
        return self.failures / self.requests if self.requests else 0.0

    @property
    def soutenable(self) -> bool:
        return self.p95_ms < SEUIL_P95_MS and self.taux_erreur < SEUIL_ERREURS


def lire(path: Path) -> Palier | None:
    match = PATTERN.match(path.name)
    if not match:
        return None
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Name") != "Aggregated":
                continue
            return Palier(
                scenario=match["scenario"],
                workers=int(match["workers"]),
                users=int(match["users"]),
                requests=int(row["Request Count"]),
                failures=int(row["Failure Count"]),
                rps=float(row["Requests/s"]),
                avg_ms=float(row["Average Response Time"]),
                p50_ms=float(row["50%"]),
                p95_ms=float(row["95%"]),
                p99_ms=float(row["99%"]),
            )
    return None


def tableau(paliers: list[Palier]) -> str:
    entetes = (
        "| Scénario | Workers | Concurrence | Requêtes | Erreurs | RPS | "
        "Lignes/s | moy. (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Soutenable |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lignes = [
        f"| {p.scenario} | {p.workers} | {p.users} | {p.requests} | {p.failures} | "
        f"{p.rps:.1f} | {p.lignes_par_seconde:.0f} | {p.avg_ms:.0f} | {p.p50_ms:.0f} | "
        f"{p.p95_ms:.0f} | {p.p99_ms:.0f} | {'oui' if p.soutenable else 'non'} |"
        for p in sorted(paliers, key=lambda x: (x.scenario, x.workers, x.users))
    ]
    return entetes + "\n".join(lignes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agrège les résultats Locust")
    parser.add_argument("directory", nargs="?", default="loadtest/results")
    args = parser.parse_args(argv)

    paliers = [p for p in (lire(f) for f in sorted(Path(args.directory).glob("*_stats.csv"))) if p]
    if not paliers:
        print(f"aucun palier trouvé dans {args.directory}")
        return 1
    print(tableau(paliers))
    print()
    soutenables = [p for p in paliers if p.soutenable]
    if soutenables:
        meilleur = max(soutenables, key=lambda p: p.lignes_par_seconde)
        print(
            f"Meilleur débit soutenable (p95 < {SEUIL_P95_MS:.0f} ms) : "
            f"{meilleur.lignes_par_seconde:.0f} lignes/s "
            f"({meilleur.scenario}, {meilleur.workers} worker(s), "
            f"{meilleur.users} utilisateur(s) simultané(s))"
        )
        par_jour = meilleur.lignes_par_seconde * 3600
        print(f"Soit environ {par_jour:,.0f} disques scorés par heure de calcul.".replace(",", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
