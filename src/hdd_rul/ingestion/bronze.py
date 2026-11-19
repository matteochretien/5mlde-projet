"""Construction de la couche bronze avec DuckDB, en deux passes.

DuckDB lit les CSV en streaming : à aucun moment les ~35 Go décompressés ne
passent en mémoire pandas. `union_by_name=true` absorbe les écarts de schéma
entre trimestres (colonnes SMART ajoutees ou retirées d'un trimestre à l'autre).

Passe 1 — `drive_events` : un enregistrement par `serial_number` (première et
dernière observation, date de panne éventuelle). C'est ce qui permet, en passe 2,
de ne garder que les lignes utiles.

Passe 2 — `samples` : pour les disques en panne, les lignes de la fenêtre
[date_panne - H, date_panne] ; pour les disques sains, un échantillon
déterministe (hachage du numéro de série) sous-échantillonné dans le temps.

L'échantillonnage des sains est fait en Python (`stable_bucket`) sur la table
`drive_events`, pas en SQL : une seule implémentation, testable unitairement, et
aucun risque de divergence entre la logique SQL et la logique Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.common.hashing import sha256_text, stable_bucket
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

DRIVE_EVENTS_FILE = "drive_events.parquet"
SAMPLES_FILE = "samples.parquet"
BRONZE_MANIFEST_FILE = "bronze_manifest.json"


@dataclass
class BronzeManifest:
    """Trace de ce qui a été construit : repris en artefact et en tags MLflow."""

    built_at: str
    source_glob: str
    source_files: int
    source_fingerprint: str
    horizon_days: int
    date_min: str
    date_max: str
    drives_total: int
    drives_failed: int
    drives_healthy_sampled: int
    rows_failed: int
    rows_healthy: int
    rows_total: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _source_fingerprint(files: list[Path]) -> str:
    """Empreinte du périmètre source : (nom, taille) de chaque fichier.

    Sert de clé de cache Prefect et de tag `data.sha256` dans MLflow. On évite
    le sha256 du contenu : illisoire sur 35 Go, alors que nom + taille suffit à
    détecter un changement de périmètre ou un fichier remplacé.
    """
    payload = "\n".join(f"{p.name}:{p.stat().st_size}" for p in sorted(files))
    return sha256_text(payload)


def _csv_types(config: AppConfig) -> dict[str, str]:
    """Types imposés aux colonnes exploitées : l'inférence ne les devine pas bien.

    Les SMART *raw* sont lus en DOUBLE (certains compteurs dépassent l'int32 et
    les valeurs manquantes sont fréquentes), `failure` en BIGINT pour pouvoir
    détecter une valeur hors {0, 1} au lieu de la faire planter au parsing.
    """
    types = {
        "date": "DATE",
        "serial_number": "VARCHAR",
        "model": "VARCHAR",
        "capacity_bytes": "BIGINT",
        "failure": "BIGINT",
    }
    for column in config.features.smart_columns:
        types[column] = "DOUBLE"
    return types


def _read_csv_expression(config: AppConfig, glob: str) -> str:
    types = ", ".join(f"'{name}': '{sql_type}'" for name, sql_type in _csv_types(config).items())
    return (
        f"read_csv('{glob}', union_by_name = true, header = true, "
        f"nullstr = '', ignore_errors = false, types = {{{types}}})"
    )


def build_drive_events(
    connection: duckdb.DuckDBPyConnection, config: AppConfig, glob: str
) -> pd.DataFrame:
    """Passe 1 : agrège l'historique de chaque disque.

    `failure = 1` signifie que le disque est tombe en panne *ce jour-là* (c'est
    sa dernière ligne dans les données), d'où `max(date) FILTER (failure = 1)`.
    """
    query = f"""
        SELECT
            serial_number,
            min(date)                                   AS first_seen,
            max(date)                                   AS last_seen,
            max(date) FILTER (WHERE failure >= 1)       AS failure_date,
            max(CASE WHEN failure >= 1 THEN 1 ELSE 0 END) AS ever_failed,
            arg_max(model, date)                        AS model,
            arg_max(capacity_bytes, date)               AS capacity_bytes,
            count(*)                                    AS observations
        FROM {_read_csv_expression(config, glob)}
        GROUP BY serial_number
    """
    return connection.sql(query).df()


def select_sampled_serials(events: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Décidé, disque par disque, s'il entre dans le jeu bronze et à quel titre.

    * `failed`  : tous les disques dont on connaît la date de panne ;
    * `healthy` : échantillon déterministe `stable_bucket(serial) < taux`, et
      seulement ceux observés au moins H jours (sinon aucune de leurs lignes ne
      peut porter un label non censuré).
    """
    sampling = config.sampling
    horizon = config.horizon_days
    frame = events.copy()
    frame["kind"] = "excluded"

    failed_mask = frame["ever_failed"].astype(int) == 1
    frame.loc[failed_mask, "kind"] = "failed"

    buckets = frame["serial_number"].map(
        lambda serial: stable_bucket(serial, salt=sampling.hash_salt, buckets=100)
    )
    observed_days = (frame["last_seen"] - frame["first_seen"]).dt.days
    healthy_mask = (
        (~failed_mask) & (buckets < sampling.healthy_rate_pct) & (observed_days >= horizon)
    )
    frame.loc[healthy_mask, "kind"] = "healthy"

    selected = frame.loc[
        frame["kind"] != "excluded", ["serial_number", "kind", "failure_date", "last_seen"]
    ]
    return selected.reset_index(drop=True)


def build_samples(
    connection: duckdb.DuckDBPyConnection,
    config: AppConfig,
    glob: str,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Passe 2 : extrait les lignes utiles et calcule la RUL non plafonnée.

    Conditions retenues :
      * disque en panne  : `date` dans [date_panne - H, date_panne] ;
      * disque sain      : encore observé à `date + H` (sinon la vraie RUL est
        censurée et le label H serait faux), et sous-échantillonné dans le temps
        (un jour sur `healthy_stride_days`, sur un calendrier global).
    """
    smart_columns = ", ".join(f"raw.{column}" for column in config.features.smart_columns)
    horizon = config.horizon_days
    stride = config.sampling.healthy_stride_days
    connection.register("selected_drives", selected)

    query = f"""
        SELECT
            raw.date,
            raw.serial_number,
            raw.model,
            raw.capacity_bytes,
            CAST(raw.failure AS INTEGER)         AS failure,
            sel.kind                             AS drive_kind,
            sel.failure_date,
            sel.last_seen,
            CASE WHEN sel.kind = 'failed'
                 THEN date_diff('day', raw.date, sel.failure_date)
                 ELSE NULL END                   AS rul_days_observed,
            {smart_columns}
        FROM {_read_csv_expression(config, glob)} AS raw
        JOIN selected_drives AS sel USING (serial_number)
        WHERE
            (
                sel.kind = 'failed'
                AND raw.date BETWEEN sel.failure_date - INTERVAL {horizon} DAY AND sel.failure_date
            )
            OR
            (
                sel.kind = 'healthy'
                AND sel.last_seen >= raw.date + INTERVAL {horizon} DAY
                AND CAST(epoch(raw.date) / 86400 AS BIGINT) % {stride} = 0
            )
        ORDER BY raw.date, raw.serial_number
    """
    return connection.sql(query).df()


def build_bronze(
    config: AppConfig,
    *,
    source_dir: Path | None = None,
    pattern: str = "*.csv",
) -> BronzeManifest:
    """Construit `drive_events.parquet`, `samples.parquet` et leur manifeste."""
    config.paths.ensure()
    source = source_dir or config.paths.raw
    files = sorted(p for p in source.glob(pattern) if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(
            f"aucun CSV dans {source}. Lancer `mise run download` (données réelles) "
            "ou `mise run data:synthetic` (jeu synthétique)."
        )
    glob = str(source / pattern)

    with duckdb.connect() as connection:
        connection.execute("PRAGMA enable_progress_bar = false")
        logger.info("passe 1/2 — agrégation de l'historique des disques (%d fichiers)", len(files))
        events = build_drive_events(connection, config, glob)
        selected = select_sampled_serials(events, config)
        logger.info(
            "passe 1/2 — %d disques, %d en panne, %d sains échantillonnés",
            len(events),
            int((selected["kind"] == "failed").sum()),
            int((selected["kind"] == "healthy").sum()),
        )
        logger.info("passe 2/2 — extraction des lignes utiles")
        samples = build_samples(connection, config, glob, selected)

    config.paths.bronze.mkdir(parents=True, exist_ok=True)
    events.to_parquet(config.paths.bronze / DRIVE_EVENTS_FILE, index=False)
    samples.to_parquet(config.paths.bronze / SAMPLES_FILE, index=False)

    manifest = BronzeManifest(
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_glob=glob,
        source_files=len(files),
        source_fingerprint=_source_fingerprint(files),
        horizon_days=config.horizon_days,
        date_min=str(samples["date"].min()) if len(samples) else "",
        date_max=str(samples["date"].max()) if len(samples) else "",
        drives_total=len(events),
        drives_failed=int((selected["kind"] == "failed").sum()),
        drives_healthy_sampled=int((selected["kind"] == "healthy").sum()),
        rows_failed=int((samples["drive_kind"] == "failed").sum()) if len(samples) else 0,
        rows_healthy=int((samples["drive_kind"] == "healthy").sum()) if len(samples) else 0,
        rows_total=len(samples),
    )
    (config.paths.bronze / BRONZE_MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
    logger.info(
        "bronze construit: %d lignes (%d pannes / %d sains) sur %s -> %s",
        manifest.rows_total,
        manifest.rows_failed,
        manifest.rows_healthy,
        manifest.date_min,
        manifest.date_max,
    )
    return manifest


def load_bronze_samples(config: AppConfig) -> pd.DataFrame:
    path = config.paths.bronze / SAMPLES_FILE
    if not path.exists():
        raise FileNotFoundError(f"couche bronze absente: {path}. Lancer `mise run ingest`.")
    return pd.read_parquet(path)


def load_drive_events(config: AppConfig) -> pd.DataFrame:
    path = config.paths.bronze / DRIVE_EVENTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"couche bronze absente: {path}. Lancer `mise run ingest`.")
    return pd.read_parquet(path)


def read_bronze_manifest(config: AppConfig) -> dict[str, object]:
    path = config.paths.bronze / BRONZE_MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
