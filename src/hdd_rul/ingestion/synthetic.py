"""Générateur de données Drive Stats synthétiques.

Pourquoi : le jeu réel représenté ~3 x 2 Go de zip (~35 Go décompressés), ce qui
rend impossible une exécution de bout en bout en quelques minutes, en CI ou dans
les tests. Le générateur produit des fichiers journaliers au *même format* que
Backblaze (mêmes colonnes, mêmes coquilles `smart_211_normailized`), ce qui
permet de faire tourner et de tester toute la chaîne sans réseau. Le pipeline ne
sait pas d'où viennent les CSV : les deux sources passent par `data/raw/`.

Le mécanisme de dégradation est volontairement simple et monotone : l'objectif
du projet n'est pas la performance prédictive mais la chaîne opérationnelle.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

# Modèles inspirés du parc Backblaze : (modèle, capacité en octets).
FLEET_MODELS: list[tuple[str, int]] = [
    ("ST4000DM000", 4_000_787_030_016),
    ("ST8000NM0055", 8_001_563_222_016),
    ("ST12000NM0008", 12_000_138_625_024),
    ("ST16000NM001G", 16_000_900_661_248),
    ("TOSHIBA MG07ACA14TA", 14_000_519_643_136),
    ("TOSHIBA MG08ACA16TE", 16_000_900_661_248),
    ("HGST HMS5C4040ALE640", 4_000_787_030_016),
    ("HGST HUH721212ALN604", 12_000_138_625_024),
    ("WDC WUH721414ALE6L4", 14_000_519_643_136),
    ("WDC WUH722222ALE6L4", 22_000_969_973_760),
]

# Attributs émis dans les CSV. Le pipeline n'en consomme qu'un sous-ensemble
# (config `features.smart_raw`) : le reste sert à éprouver `union_by_name`.
EMITTED_SMART = [1, 4, 5, 7, 9, 12, 187, 188, 190, 192, 193, 194, 197, 198, 199, 240, 241, 242]
# Coquilles bien présentés dans le schéma Backblaze courant : le contrat de
# données doit les tolérer (cf. Drive_Stats_Schema_Current.csv).
TYPO_SMART = [211, 212]

BASE_COLUMNS = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
    "pod_slot_num",
    "is_legacy_format",
]

DATACENTERS = ["sac0", "sac2", "ashburn", "phoenix", "amsterdam"]


def csv_header() -> list[str]:
    header = list(BASE_COLUMNS)
    for attr in EMITTED_SMART:
        header += [f"smart_{attr}_normalized", f"smart_{attr}_raw"]
    for attr in TYPO_SMART:
        header += [f"smart_{attr}_normailized", f"smart_{attr}_raw"]
    return header


@dataclass(frozen=True)
class _Drive:
    serial: str
    model: str
    capacity_bytes: int
    datacenter: str
    entry_day: int
    exit_day: int  # dernier jour observé (sortie de parc ou fin de fenêtre)
    failure_day: int | None  # jour de la panne, s'il y en a une
    base_hours: int
    severity: float  # intensite de la dégradation SMART avant panne


def _build_fleet(config: AppConfig, start: date, end: date) -> list[_Drive]:
    synthetic = config.synthetic
    rng = np.random.default_rng(synthetic.seed)
    n_days = (end - start).days + 1
    drives: list[_Drive] = []

    for index in range(synthetic.n_drives):
        model, capacity = FLEET_MODELS[int(rng.integers(len(FLEET_MODELS)))]
        vendor_prefix = {"S": "ZA", "T": "X9", "H": "PL", "W": "QG"}.get(model[0], "ZZ")
        serial = f"{vendor_prefix}{index:06d}{int(rng.integers(10, 99))}"

        # 85 % du parc est déjà en place au premier jour ; le reste arrive en cours
        # de période (installations), ce qui crée de vraies dates d'entrée.
        entry_day = 0 if rng.random() < 0.85 else int(rng.integers(1, max(2, n_days - 30)))

        # Probabilité de panne sur la fenêtre observée.
        hazard = synthetic.daily_failure_rate * float(rng.gamma(2.0, 0.5))
        p_failure = 1.0 - (1.0 - min(hazard, 0.02)) ** (n_days - entry_day)
        failure_day: int | None = None
        if rng.random() < p_failure:
            failure_day = int(rng.integers(entry_day + 5, n_days))

        # Sorties de parc sans panne (migrations) : source de censure à droite,
        # exactement ce que la règle d'étiquetage des sains doit gérer.
        exit_day = n_days - 1
        if failure_day is None and rng.random() < 0.08:
            exit_day = int(rng.integers(entry_day + 10, n_days))
        if failure_day is not None:
            exit_day = failure_day

        drives.append(
            _Drive(
                serial=serial,
                model=model,
                capacity_bytes=capacity,
                datacenter=DATACENTERS[int(rng.integers(len(DATACENTERS)))],
                entry_day=entry_day,
                exit_day=exit_day,
                failure_day=failure_day,
                base_hours=int(rng.integers(2_000, 55_000)),
                severity=float(rng.uniform(0.3, 1.0)),
            )
        )
    return drives


def _smart_values(
    drive: _Drive, day_index: int, rng: np.random.Generator
) -> dict[str, float | int]:
    """Valeurs SMART du jour, avec une dégradation monotone avant la panne."""
    age_hours = drive.base_hours + (day_index - drive.entry_day) * 24
    if drive.failure_day is None:
        ramp = 0.0
    else:
        days_to_failure = drive.failure_day - day_index
        ramp = max(0.0, 1.0 - days_to_failure / 60.0) ** 2

    noise = rng.normal(0.0, 1.0)
    reallocated = int(max(0, ramp * drive.severity * 420 + max(0.0, noise * 2)))
    pending = int(max(0, ramp * drive.severity * 90 + max(0.0, noise)))
    uncorrectable = int(max(0, ramp * drive.severity * 55))
    reported_uncorrect = int(max(0, ramp * drive.severity * 140))
    command_timeout = int(max(0, ramp * drive.severity * 30))
    crc_errors = int(max(0, ramp * drive.severity * 12))
    temperature = int(np.clip(30 + ramp * 9 + rng.normal(0, 1.5), 18, 62))
    seek_errors = int(age_hours * 11 + ramp * drive.severity * 3_000)

    return {
        "smart_1_raw": seek_errors,
        "smart_4_raw": int(age_hours / 180) + 4,
        "smart_5_raw": reallocated,
        "smart_7_raw": int(age_hours * 7),
        "smart_9_raw": age_hours,
        "smart_12_raw": int(age_hours / 900) + 2,
        "smart_187_raw": reported_uncorrect,
        "smart_188_raw": command_timeout,
        "smart_190_raw": temperature,
        "smart_192_raw": int(age_hours / 1_200),
        "smart_193_raw": int(age_hours * 1.6),
        "smart_194_raw": temperature,
        "smart_197_raw": pending,
        "smart_198_raw": uncorrectable,
        "smart_199_raw": crc_errors,
        "smart_240_raw": int(age_hours * 0.93),
        "smart_241_raw": int(age_hours * 21_000),
        "smart_242_raw": int(age_hours * 18_500),
        "smart_211_raw": 0,
        "smart_212_raw": 0,
    }


def _normalized(attr: int, raw: float, ramp_hint: int) -> int | str:
    """Valeur `normalized` : échelle 100 -> 1 propre au constructeur.

    Volontairement grossière : les valeurs normalisées ne sont pas utilisées en
    features (sémantique différente selon le fabricant, cf. limites du README).
    """
    if attr in {194, 190}:
        return int(max(1, 120 - raw))
    if attr in {5, 187, 188, 197, 198, 199}:
        return int(max(1, 100 - min(99, ramp_hint)))
    if attr in {211, 212}:
        return ""  # colonne présenté mais vide, comme chez Backblaze
    return 100


def generate_synthetic_drive_stats(
    config: AppConfig,
    *,
    destination: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Écrit un CSV par jour dans `data/raw/` (ou `destination`).

    Retourne la liste des fichiers écrits, triée par date.
    """
    synthetic = config.synthetic
    start = date.fromisoformat(synthetic.start_date)
    end = date.fromisoformat(synthetic.end_date)
    if end < start:
        raise ValueError("synthetic.end_date doit être postérieure à synthetic.start_date")

    target_dir = destination or config.paths.raw
    target_dir.mkdir(parents=True, exist_ok=True)
    header = csv_header()
    fleet = _build_fleet(config, start, end)
    rng = np.random.default_rng(synthetic.seed + 1)

    n_days = (end - start).days + 1
    written: list[Path] = []
    total_rows = 0
    total_failures = 0

    for day_index in range(n_days):
        current = start + timedelta(days=day_index)
        path = target_dir / f"{current.isoformat()}.csv"
        if path.exists() and not force:
            written.append(path)
            continue

        rows: list[list[object]] = []
        for drive in fleet:
            if day_index < drive.entry_day or day_index > drive.exit_day:
                continue
            failed_today = int(drive.failure_day == day_index)
            smart = _smart_values(drive, day_index, rng)
            row: list[object] = [
                current.isoformat(),
                drive.serial,
                drive.model,
                drive.capacity_bytes,
                failed_today,
                drive.datacenter,
                f"{drive.datacenter}-c{drive.entry_day % 7:02d}",
                1_000 + drive.entry_day % 97,
                f"pod-{drive.entry_day % 13:02d}",
                drive.entry_day % 60,
                0,
            ]
            ramp_hint = int(smart["smart_5_raw"] / 5)
            for attr in EMITTED_SMART:
                raw = smart[f"smart_{attr}_raw"]
                row += [_normalized(attr, raw, ramp_hint), raw]
            for attr in TYPO_SMART:
                raw = smart[f"smart_{attr}_raw"]
                row += [_normalized(attr, raw, ramp_hint), raw]
            rows.append(row)
            total_failures += failed_today

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        total_rows += len(rows)
        written.append(path)

    logger.info(
        "jeu synthétique: %d fichiers (%s -> %s), %d lignes, %d pannes, %d disques",
        len(written),
        start,
        end,
        total_rows,
        total_failures,
        len(fleet),
    )
    return sorted(written)
