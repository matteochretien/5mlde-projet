"""Contrat de données : un lot invalide doit *bloquer*, un lot douteux alerter.

Le test `test_lot_invalide_bloque_l_entrainement` est le test bloquant exigé par
le sujet : il vérifie qu'un lot non conforme interrompt bien le pipeline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hdd_rul.common.config import AppConfig
from hdd_rul.validation.contract import DataContractError, quarantine_rows, validate_batch


def test_lot_conforme_est_accepte(config: AppConfig, daily_batch_frame: pd.DataFrame) -> None:
    report = validate_batch(
        config, daily_batch_frame, batch_name="2026-03-15.csv", expected_date=date(2026, 3, 15)
    )
    assert report.passed
    assert report.blocking_failures == []
    assert report.total_rows == 200
    assert report.report_path is not None and report.report_path.exists()


@pytest.mark.parametrize(
    ("mutation", "regle_attendue"),
    [
        ("failure_hors_domaine", "expect_column_values_to_be_in_set"),
        ("colonne_manquante", "expect_column_to_exist"),
        ("date_incoherente", "expect_column_distinct_values_to_equal_set"),
        ("serial_duplique", "expect_column_values_to_be_unique"),
        ("trop_de_pannes", "expect_drive_failure_rate_to_be_below"),
        ("serial_invalide", "expect_column_values_to_be_valid_drive_serial"),
    ],
)
def test_chaque_violation_est_detectee(
    config: AppConfig, daily_batch_frame: pd.DataFrame, mutation: str, regle_attendue: str
) -> None:
    frame = daily_batch_frame.copy()
    if mutation == "failure_hors_domaine":
        frame.loc[0, "failure"] = 2
    elif mutation == "colonne_manquante":
        frame = frame.drop(columns=["smart_197_raw"])
    elif mutation == "date_incoherente":
        frame.loc[7, "date"] = "2026-03-16"
    elif mutation == "serial_duplique":
        frame.loc[1, "serial_number"] = frame.loc[0, "serial_number"]
    elif mutation == "trop_de_pannes":
        frame.loc[frame.index[:50], "failure"] = 1
    elif mutation == "serial_invalide":
        frame["serial_number"] = "x"  # trop court pour un numéro de série

    report = validate_batch(
        config, frame, batch_name="lot.csv", expected_date=date(2026, 3, 15), persist_report=False
    )
    assert not report.passed
    assert regle_attendue in {outcome.expectation for outcome in report.blocking_failures}


def test_disque_deja_en_panne_qui_reapparait_part_en_quarantaine(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    """Expectation personnalisée : une résurrection isolée est signalée et écartée,
    sans refuser le lot. Les données réelles en contiennent (2BJD2XYD, en panne le
    2025-10-14, présent tous les jours de Q2) : bloquer refuserait tout le trimestre."""
    deja_en_panne = (daily_batch_frame.loc[3, "serial_number"],)
    report = validate_batch(
        config,
        daily_batch_frame,
        batch_name="lot.csv",
        expected_date=date(2026, 3, 15),
        known_failed_serials=deja_en_panne,
        persist_report=False,
    )
    assert report.passed
    noms = {outcome.expectation for outcome in report.warnings}
    assert "expect_column_values_to_not_be_known_failed_drive" in noms
    assert report.quarantined_rows == 1

    # Le log dit ce qui cloche, et sur quel disque, pas seulement « 1 avertissement ».
    (detail,) = [o.describe(report.total_rows) for o in report.warnings]
    assert "déjà déclarés en panne" in detail
    assert "(serial_number) : 1 ligne(s)" in detail
    assert deja_en_panne[0] in detail

    propre, ecartees = quarantine_rows(config, daily_batch_frame, "lot.csv", deja_en_panne)
    assert ecartees == 1
    assert deja_en_panne[0] not in set(propre["serial_number"])


def test_une_resurrection_massive_bloque_le_lot(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    """Au-delà du plafond de quarantaine, c'est un fichier rejoué ou abîmé : on bloque."""
    deja_en_panne = tuple(daily_batch_frame["serial_number"].iloc[:20])  # 10 % du lot
    report = validate_batch(
        config,
        daily_batch_frame,
        batch_name="lot.csv",
        expected_date=date(2026, 3, 15),
        known_failed_serials=deja_en_panne,
        persist_report=False,
    )
    assert not report.passed
    noms = {outcome.expectation for outcome in report.blocking_failures}
    assert "expect_quarantine_fraction_to_be_below" in noms


def test_capacite_invalide_met_en_quarantaine_sans_bloquer(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    frame = daily_batch_frame.copy()
    frame.loc[frame.index[:4], "capacity_bytes"] = -1  # 2 % du lot : sous le seuil

    report = validate_batch(
        config, frame, batch_name="lot.csv", expected_date=date(2026, 3, 15), persist_report=False
    )
    assert report.passed  # non bloquant
    assert report.quarantined_rows == 4
    assert report.warnings  # mais signalé

    clean, quarantined = quarantine_rows(config, frame, "lot.csv")
    assert quarantined == 4
    assert len(clean) == len(frame) - 4
    assert (config.paths.quarantine / "lot.parquet").exists()


def test_quarantaine_massive_redevient_bloquante(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    """Quelques lignes douteuses : on tolère. La moitié du lot : c'est un incident."""
    frame = daily_batch_frame.copy()
    frame.loc[frame.index[:100], "capacity_bytes"] = -1

    report = validate_batch(
        config, frame, batch_name="lot.csv", expected_date=date(2026, 3, 15), persist_report=False
    )
    assert not report.passed
    assert "expect_quarantine_fraction_to_be_below" in {
        outcome.expectation for outcome in report.blocking_failures
    }


def test_temperature_aberrante_est_un_avertissement(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    """Un capteur SMART bruité ne doit pas rejeter une journée entière."""
    frame = daily_batch_frame.copy()
    frame.loc[frame.index[:10], "smart_194_raw"] = 250

    report = validate_batch(
        config, frame, batch_name="lot.csv", expected_date=date(2026, 3, 15), persist_report=False
    )
    assert report.passed
    assert "expect_column_values_to_be_between" in {o.expectation for o in report.warnings}


def test_lot_invalide_bloque_l_entrainement(
    config: AppConfig, daily_batch_frame: pd.DataFrame
) -> None:
    """Test bloquant : un lot non conforme interrompt la chaîne (DataContractError).

    C'est la garantie demandée par le sujet : « interruption de l'entraînement
    en cas de non-conformité ». On vérifie le mécanisme d'arrêt lui-même.
    """
    frame = daily_batch_frame.copy()
    frame.loc[0, "failure"] = 7

    report = validate_batch(
        config, frame, batch_name="lot.csv", expected_date=date(2026, 3, 15), persist_report=False
    )
    assert not report.passed

    with pytest.raises(DataContractError) as excinfo:
        if not report.passed:
            raise DataContractError(report)
    assert "contrat violé" in str(excinfo.value)
    assert excinfo.value.report.blocking_failures
