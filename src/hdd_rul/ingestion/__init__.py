"""Ingestion reproductible : téléchargement cache, génération synthétique, bronze."""

from hdd_rul.ingestion.bronze import build_bronze
from hdd_rul.ingestion.download import download_quarter, extract_quarter, read_manifest
from hdd_rul.ingestion.landing import claim_batch, list_pending_batches, mark_processed
from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats

__all__ = [
    "build_bronze",
    "claim_batch",
    "download_quarter",
    "extract_quarter",
    "generate_synthetic_drive_stats",
    "list_pending_batches",
    "mark_processed",
    "read_manifest",
]
