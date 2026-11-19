"""Ingestion : générateur synthétique, téléchargement cache, couche bronze DuckDB."""

from __future__ import annotations

import hashlib
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hdd_rul.common.config import AppConfig
from hdd_rul.ingestion.bronze import build_bronze, load_bronze_samples, select_sampled_serials
from hdd_rul.ingestion.download import ChecksumMismatchError, download_quarter, extract_quarter
from hdd_rul.ingestion.landing import batch_date_from_name, claim_batch, drop_batch, mark_processed
from hdd_rul.ingestion.synthetic import csv_header, generate_synthetic_drive_stats


def test_le_jeu_synthetique_respecte_le_format_backblaze(config: AppConfig) -> None:
    files = generate_synthetic_drive_stats(config)
    assert len(files) == 181  # du 1er janvier au 30 juin 2026

    frame = pd.read_csv(files[50])
    attendu = csv_header()
    assert list(frame.columns) == attendu
    # Les coquilles du schéma courant doivent être reproduites telles quelles.
    assert "smart_211_normailized" in frame.columns
    assert frame["date"].nunique() == 1
    assert frame["serial_number"].is_unique
    assert set(frame["failure"].unique()).issubset({0, 1})


def test_generation_idempotente(config: AppConfig) -> None:
    premier = generate_synthetic_drive_stats(config)
    tailles = {p.name: p.stat().st_size for p in premier}
    second = generate_synthetic_drive_stats(config)  # sans force : ne réécrit pas
    assert {p.name: p.stat().st_size for p in second} == tailles


def test_un_disque_disparait_apres_sa_panne(config: AppConfig) -> None:
    """Invariant du jeu : `failure = 1` est la dernière ligne du disque."""
    generate_synthetic_drive_stats(config)
    frames = [
        pd.read_csv(p, usecols=["date", "serial_number", "failure"])
        for p in sorted(config.paths.raw.glob("*.csv"))
    ]
    everything = pd.concat(frames, ignore_index=True)
    panne = everything.loc[everything["failure"] == 1]
    assert not panne.empty

    derniere_vue = everything.groupby("serial_number")["date"].max()
    for row in panne.itertuples(index=False):
        assert derniere_vue[row.serial_number] == row.date


def test_bronze_deux_passes(synthetic_raw: AppConfig) -> None:
    config = synthetic_raw
    manifest = build_bronze(config)

    assert manifest.rows_total > 0
    assert manifest.drives_failed > 0
    assert manifest.rows_failed > 0
    assert manifest.source_files == 181
    assert len(manifest.source_fingerprint) == 64

    samples = load_bronze_samples(config)
    assert set(samples["drive_kind"].unique()).issubset({"failed", "healthy"})
    # Les lignes des disques en panne tiennent dans la fenêtre [panne - H, panne].
    failed = samples.loc[samples["drive_kind"] == "failed"]
    assert failed["rul_days_observed"].between(0, config.horizon_days).all()
    for column in config.features.smart_columns:
        assert column in samples.columns


def test_bronze_sans_source_leve_une_erreur_explicite(config: AppConfig) -> None:
    with pytest.raises(FileNotFoundError, match="mise run"):
        build_bronze(config)


def test_selection_des_disques(config: AppConfig) -> None:
    events = pd.DataFrame(
        [
            {
                "serial_number": "FAILED1",
                "first_seen": pd.Timestamp("2026-01-01"),
                "last_seen": pd.Timestamp("2026-03-01"),
                "failure_date": pd.Timestamp("2026-03-01"),
                "ever_failed": 1,
            },
            {  # sain observé longtemps : éligible si le hachage le retient
                "serial_number": "LONG1",
                "first_seen": pd.Timestamp("2026-01-01"),
                "last_seen": pd.Timestamp("2026-06-01"),
                "failure_date": pd.NaT,
                "ever_failed": 0,
            },
            {  # sain observé 10 jours : aucune ligne étiquetable, à exclure
                "serial_number": "SHORT1",
                "first_seen": pd.Timestamp("2026-01-01"),
                "last_seen": pd.Timestamp("2026-01-11"),
                "failure_date": pd.NaT,
                "ever_failed": 0,
            },
        ]
    )
    selected = select_sampled_serials(events, config)
    kinds = dict(zip(selected["serial_number"], selected["kind"], strict=True))
    assert kinds["FAILED1"] == "failed"  # tous les disques en panne sont pris
    assert "SHORT1" not in kinds  # horizon non observable


def test_telechargement_depuis_le_cache(config: AppConfig, monkeypatch) -> None:
    """Une archive déjà présente et conforme n'est pas retéléchargée."""
    config.paths.ensure()
    archive = config.paths.cache / config.data.archive_name("Q1_2026")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("2026-01-01.csv", "date,serial_number\n2026-01-01,ZA1\n")

    def _interdit(*args, **kwargs):  # pragma: no cover
        raise AssertionError("le cache n'a pas été utilisé")

    monkeypatch.setattr("hdd_rul.ingestion.download.requests.get", _interdit)
    # « Conforme » = empreinte attendue égale à celle de l'archive. On ne dépend
    # pas des sha256 réels figés dans config/base.yaml.
    empreinte = hashlib.sha256(archive.read_bytes()).hexdigest()
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"checksums": {"Q1_2026": empreinte}})}
    )
    assert download_quarter(config, "Q1_2026") == archive

    files = extract_quarter(config, archive)
    assert [p.name for p in files] == ["2026-01-01.csv"]


def test_empreinte_non_conforme_refusee(config: AppConfig) -> None:
    config.paths.ensure()
    archive = config.paths.cache / config.data.archive_name("Q1_2026")
    archive.write_bytes(b"contenu quelconque")
    avec_checksum = config.model_copy(
        update={"data": config.data.model_copy(update={"checksums": {"Q1_2026": "0" * 64}})}
    )
    with pytest.raises(ChecksumMismatchError):
        download_quarter(avec_checksum, "Q1_2026")


def test_cycle_de_vie_d_un_lot(config: AppConfig, tmp_path: Path) -> None:
    config.paths.ensure()
    source = tmp_path / "2026-04-07.csv"
    source.write_text("date,serial_number\n2026-04-07,ZA1\n", encoding="utf-8")

    deposited = drop_batch(config, source)
    assert batch_date_from_name(deposited).isoformat() == "2026-04-07"
    assert claim_batch(config) == deposited

    processed = mark_processed(config, deposited)
    assert processed.exists()
    assert claim_batch(config) is None


def test_disques_deja_en_panne_avant_un_lot(config: AppConfig) -> None:
    """Liste causale : rien de ce qui tombe en panne le jour du lot ou après."""
    import pandas as pd

    from hdd_rul.ingestion.landing import known_failed_serials_before

    config.paths.ensure()
    assert known_failed_serials_before(config, date(2026, 4, 10)) == ()  # aucune source

    pd.DataFrame(
        {
            "serial_number": ["ZA1", "ZA2", "ZA3"],
            "failure_date": pd.to_datetime(["2026-04-01", "2026-04-10", None]),
        }
    ).to_parquet(config.paths.bronze / "drive_events.parquet")
    clean = config.paths.bronze / "landing_clean"
    clean.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"serial_number": ["ZB1", "ZB2"], "failure": [1, 0]}).to_parquet(
        clean / "2026-04-05.parquet"
    )
    pd.DataFrame({"serial_number": ["ZB3"], "failure": [1]}).to_parquet(
        clean / "2026-04-12.parquet"
    )

    assert known_failed_serials_before(config, date(2026, 4, 10)) == ("ZA1", "ZB1")


def test_dernier_lot_traite(config: AppConfig) -> None:
    from hdd_rul.ingestion.landing import latest_processed_day

    config.paths.ensure()
    assert latest_processed_day(config) is None
    for name in ("2026-04-02.csv", "2026-04-09.csv", "notes.csv"):
        (config.paths.processed / name).write_text("x", encoding="utf-8")
    assert latest_processed_day(config) == date(2026, 4, 9)


def test_le_rejeu_part_du_trimestre_de_simulation(config: AppConfig) -> None:
    """Sans `--start`, le rejeu ne doit pas repartir du trimestre d'historique."""
    from scripts.replay import replay_start, select_days

    config.paths.ensure()
    for day in ("2025-10-01", "2026-03-31", "2026-04-01", "2026-04-02"):
        (config.paths.raw / f"{day}.csv").write_text("x", encoding="utf-8")
    rejeu = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"quarters": config.data.quarters.model_copy(update={"replay": ["Q2_2026"]})}
            )
        }
    )
    assert replay_start(rejeu) == date(2026, 4, 1)
    assert [p.stem for p in select_days(rejeu, limit=5)] == ["2026-04-01", "2026-04-02"]
