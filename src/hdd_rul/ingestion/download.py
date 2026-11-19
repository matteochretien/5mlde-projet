"""Téléchargement et décompression des archives trimestrielles Backblaze.

Deux propriétés attendues par le sujet (« ingestion reproductible ») :
  * cache : une archive déjà présente et conforme n'est jamais retéléchargée,
    ce qui rend la tâche Prefect idempotente et cachable sur son sha256 ;
  * manifeste : url, sha256, taille et nombre de fichiers sont écrits dans
    `data/raw/_manifest.json`, repris comme artefact MLflow pour savoir
    exactement sur quelles données un modèle a été entraîné.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from hdd_rul.common.config import AppConfig
from hdd_rul.common.hashing import sha256_file
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "_manifest.json"
_DOWNLOAD_CHUNK = 8 * 1024 * 1024


class ChecksumMismatchError(RuntimeError):
    """Le sha256 téléchargé ne correspond pas à celui figé dans la configuration."""


def manifest_path(config: AppConfig) -> Path:
    return config.paths.raw / MANIFEST_NAME


def read_manifest(config: AppConfig) -> dict[str, Any]:
    path = manifest_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(config: AppConfig, entries: dict[str, Any]) -> Path:
    path = manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    return path


def download_quarter(config: AppConfig, quarter: str, *, force: bool = False) -> Path:
    """Télécharge `data_<quarter>.zip` dans le cache et vérifie son empreinte."""
    config.paths.ensure()
    url = config.data.archive_url(quarter)
    target = config.paths.cache / config.data.archive_name(quarter)
    expected = config.data.checksums.get(quarter)

    if target.exists() and not force:
        digest = sha256_file(target)
        if expected and digest != expected:
            raise ChecksumMismatchError(
                f"{target.name}: sha256 attendu {expected}, obtenu {digest}. "
                "Archive corrompue ou trimestre republie par Backblaze."
            )
        logger.info("cache: %s déjà présent (sha256=%s)", target.name, digest[:12])
        return target

    logger.info("téléchargement de %s", url)
    temporary = target.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                handle.write(chunk)
    digest = sha256_file(temporary)
    if expected and digest != expected:
        temporary.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{target.name}: sha256 attendu {expected}, obtenu {digest}; téléchargement rejeté."
        )
    temporary.replace(target)

    entries = read_manifest(config)
    entries[quarter] = {
        "url": url,
        "sha256": digest,
        "bytes": target.stat().st_size,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_manifest(config, entries)
    logger.info(
        "téléchargé %s (%.1f Mo, sha256=%s)",
        target.name,
        target.stat().st_size / 1e6,
        digest[:12],
    )
    return target


def extract_quarter(config: AppConfig, archive: Path, *, force: bool = False) -> list[Path]:
    """Décompresse les CSV journaliers à plat dans `data/raw/`.

    Les archives Backblaze imbriquent parfois les CSV dans un dossier ; on
    aplatit pour que le glob `data/raw/*.csv` reste la seule convention.
    """
    config.paths.ensure()
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        members = [
            m for m in zf.infolist() if m.filename.lower().endswith(".csv") and not m.is_dir()
        ]
        for member in members:
            name = Path(member.filename).name
            if name.startswith("."):  # entrées __MACOSX
                continue
            destination = config.paths.raw / name
            if destination.exists() and not force:
                continue
            with zf.open(member) as source, destination.open("wb") as handle:
                while chunk := source.read(_DOWNLOAD_CHUNK):
                    handle.write(chunk)
            written.append(destination)
    logger.info(
        "%s: %d fichiers journaliers écrits dans %s", archive.name, len(written), config.paths.raw
    )
    return written


def ingest_quarters(config: AppConfig, quarters: list[str] | None = None) -> dict[str, Any]:
    """Téléchargé puis décompresse les trimestres demandes (ou tous ceux déclarés)."""
    selected = quarters or config.data.quarters.all()
    summary: dict[str, Any] = {}
    for quarter in selected:
        archive = download_quarter(config, quarter)
        files = extract_quarter(config, archive)
        summary[quarter] = {"archive": str(archive), "extracted_files": len(files)}
    return summary
