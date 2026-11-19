"""Contrat de qualité appliqué AVANT tout prétraitement.

Deux niveaux de sévérité, conformément à l'attendu « interruption de
l'entraînement en cas de non-conformité » :

* **bloquant** — le lot est refusé : le flow s'arrête, une alerte part, aucun
  modèle n'est publié et l'API continue de servir le champion en place.
  Exemples : colonne obligatoire absente, `failure` hors {0, 1}, date du
  fichier incohérente, disque déjà en panne qui réapparaît.
* **non bloquant** — les lignes fautives partent en quarantaine et le lot
  continue. Backblaze publié par exemple des `capacity_bytes = -1` : rejeter la
  journée entière pour quelques lignes serait disproportionné. Si la part de
  quarantaine dépasse `validation.max_quarantine_fraction`, la règle redevient
  bloquante : ce n'est plus un bruit isolé mais un problème de fichier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger
from hdd_rul.validation.custom import (
    ExpectColumnValuesToBeValidDriveSerial,
    ExpectColumnValuesToNotBeKnownFailedDrive,
    ExpectDriveFailureRateToBeBelow,
)

logger = get_logger(__name__)

SUITE_NAME = "hdd_rul_daily_batch"
BLOCKING = {"severity": "blocking"}
WARNING = {"severity": "warning"}

# Libellés lisibles des règles, pour les logs : le nom GX seul ne dit pas ce qui
# cloche dans le fichier.
RULE_LABELS = {
    "expect_column_to_exist": "colonne obligatoire absente",
    "expect_table_row_count_to_be_between": "fichier vide",
    "expect_column_values_to_not_be_null": "valeurs manquantes",
    "expect_column_values_to_be_unique": "valeurs en double",
    "expect_column_values_to_be_valid_drive_serial": "numéros de série invalides",
    "expect_column_distinct_values_to_equal_set": "date différente de celle du fichier",
    "expect_column_values_to_be_in_set": "valeurs hors de l'ensemble autorisé",
    "expect_drive_failure_rate_to_be_below": "taux de pannes anormalement élevé",
    "expect_column_values_to_not_be_known_failed_drive": (
        "disques déjà déclarés en panne qui réapparaissent"
    ),
    "expect_column_values_to_be_between": "valeurs hors plage",
    "expect_quarantine_fraction_to_be_below": "trop de lignes en quarantaine",
}


class DataContractError(RuntimeError):
    """Levée quand une règle bloquante est violée : interrompt le flow."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass
class ExpectationOutcome:
    expectation: str
    column: str | None
    severity: str
    success: bool
    observed: Any = None
    unexpected_count: int | None = None
    unexpected_sample: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation": self.expectation,
            "column": self.column,
            "severity": self.severity,
            "success": self.success,
            "observed": self.observed,
            "unexpected_count": self.unexpected_count,
            "unexpected_sample": self.unexpected_sample[:5],
        }

    def describe(self, total_rows: int | None = None) -> str:
        """Une ligne lisible : règle en clair, colonne, ampleur et exemples."""
        label = RULE_LABELS.get(self.expectation, self.expectation)
        text = f"{label} ({self.column or 'table'})"
        if self.unexpected_count is not None:
            text += f" : {self.unexpected_count} ligne(s)"
            if total_rows:
                share = 100 * self.unexpected_count / total_rows
                text += (
                    f" sur {total_rows} ({share:.2f} %)" if share >= 0.01 else f" sur {total_rows}"
                )
        elif self.observed is not None:
            text += f" : observé {self.observed}"
        if self.unexpected_sample:
            text += ", ex. " + ", ".join(str(value) for value in self.unexpected_sample[:5])
        return f"{text} [{self.expectation}]"


@dataclass
class ValidationReport:
    """Résultat exploitable par le flow, l'alerte et Grafana."""

    batch_name: str
    batch_date: str | None
    total_rows: int
    quarantined_rows: int
    outcomes: list[ExpectationOutcome]
    validated_at: str
    report_path: Path | None = None

    @property
    def blocking_failures(self) -> list[ExpectationOutcome]:
        return [o for o in self.outcomes if not o.success and o.severity == "blocking"]

    @property
    def warnings(self) -> list[ExpectationOutcome]:
        return [o for o in self.outcomes if not o.success and o.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    @property
    def quarantine_fraction(self) -> float:
        return self.quarantined_rows / self.total_rows if self.total_rows else 0.0

    def summary(self) -> str:
        if self.passed:
            return (
                f"contrat respecté sur {self.batch_name}: {self.total_rows} lignes, "
                f"{len(self.warnings)} avertissement(s), {self.quarantined_rows} en quarantaine"
            )
        names = ", ".join(f"{o.expectation}({o.column or '-'})" for o in self.blocking_failures)
        return f"contrat violé sur {self.batch_name}: {names}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "batch_date": self.batch_date,
            "validated_at": self.validated_at,
            "passed": self.passed,
            "total_rows": self.total_rows,
            "quarantined_rows": self.quarantined_rows,
            "quarantine_fraction": round(self.quarantine_fraction, 6),
            "blocking_failures": [o.to_dict() for o in self.blocking_failures],
            "warnings": [o.to_dict() for o in self.warnings],
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def build_suite(
    context: gx.data_context.AbstractDataContext,
    config: AppConfig,
    *,
    expected_date: date | None,
    known_failed_serials: tuple[str, ...] = (),
) -> gx.ExpectationSuite:
    """Construit la suite d'expectations du contrat journalier."""
    validation = config.validation
    suite = context.suites.add_or_update(gx.ExpectationSuite(name=SUITE_NAME))
    temperature_min, temperature_max = validation.temperature_range

    # --- structure du fichier ------------------------------------------------
    for column in [*validation.required_columns, *config.features.smart_columns]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=column, meta=dict(BLOCKING)))
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1, meta=dict(BLOCKING)))

    # --- clés et identités ---------------------------------------------------
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(column="serial_number", meta=dict(BLOCKING))
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(column="serial_number", meta=dict(BLOCKING))
    )
    suite.add_expectation(
        ExpectColumnValuesToBeValidDriveSerial(
            column="serial_number", mostly=0.999, meta=dict(BLOCKING)
        )
    )
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="model", meta=dict(BLOCKING)))

    # --- cohérence temporelle ------------------------------------------------
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="date", meta=dict(BLOCKING)))
    if expected_date is not None:
        # Un fichier journalier porte exactement la date de son nom.
        suite.add_expectation(
            gxe.ExpectColumnDistinctValuesToEqualSet(
                column="date", value_set=[expected_date.isoformat()], meta=dict(BLOCKING)
            )
        )

    # --- sémantique métier ---------------------------------------------------
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="failure", value_set=[0, 1], meta=dict(BLOCKING))
    )
    suite.add_expectation(
        ExpectDriveFailureRateToBeBelow(
            max_rate=validation.max_daily_failure_rate, meta=dict(BLOCKING)
        )
    )
    if known_failed_serials:
        # Avertissement + quarantaine, et non blocage : les données réelles
        # contiennent des disques déclarés en panne qui continuent d'émettre
        # (2BJD2XYD, en panne le 2025-10-14, présent tous les jours de Q2).
        # Bloquer refuserait toutes les journées suivantes. Une résurrection
        # massive (fichier rejoué, colonnes décalées) reste bloquée par le
        # plafond de quarantaine (`max_quarantine_fraction`).
        suite.add_expectation(
            ExpectColumnValuesToNotBeKnownFailedDrive(
                column="serial_number",
                failed_serials=tuple(known_failed_serials),
                meta=dict(WARNING),
            )
        )

    # --- plages physiques (avertissements : capteurs SMART bruités) ----------
    if "smart_194_raw" in config.features.smart_columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="smart_194_raw",
                min_value=temperature_min,
                max_value=temperature_max,
                mostly=0.99,
                meta=dict(WARNING),
            )
        )
    if "smart_9_raw" in config.features.smart_columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="smart_9_raw", min_value=0, mostly=0.999, meta=dict(WARNING)
            )
        )
    # Règle non bloquante : les lignes fautives partent en quarantaine.
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="capacity_bytes", min_value=1, meta=dict(WARNING))
    )
    return suite


def _quarantine_mask(
    frame: pd.DataFrame,
    config: AppConfig,
    known_failed_serials: tuple[str, ...] = (),
) -> pd.Series:
    """Lignes à écarter sans rejeter le lot.

    Deux motifs : une valeur invalide dans `validation.quarantine_on`, ou un
    disque déjà déclaré en panne qui réapparaît.
    """
    mask = pd.Series(False, index=frame.index)
    for column in config.validation.quarantine_on:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        mask |= values.isna() | (values <= 0)
    if known_failed_serials and "serial_number" in frame.columns:
        mask |= frame["serial_number"].isin(set(known_failed_serials))
    return mask


def _outcomes_from_result(result: Any) -> list[ExpectationOutcome]:
    outcomes: list[ExpectationOutcome] = []
    for item in result.results:
        kwargs = item.expectation_config.kwargs or {}
        meta = item.expectation_config.meta or {}
        payload = dict(item.result or {})
        outcomes.append(
            ExpectationOutcome(
                expectation=item.expectation_config.type,
                column=kwargs.get("column"),
                severity=meta.get("severity", "blocking"),
                success=bool(item.success),
                observed=payload.get("observed_value"),
                unexpected_count=payload.get("unexpected_count"),
                unexpected_sample=list(payload.get("partial_unexpected_list") or []),
            )
        )
    return outcomes


def validate_batch(
    config: AppConfig,
    frame: pd.DataFrame,
    *,
    batch_name: str,
    expected_date: date | None = None,
    known_failed_serials: tuple[str, ...] = (),
    persist_report: bool = True,
) -> ValidationReport:
    """Applique le contrat à un lot journalier déjà chargé en mémoire.

    Ne lève pas : c'est à l'appelant (le flow) de décider d'arrêter sur
    `report.passed is False`, afin de pouvoir alerter et archiver le rapport.
    """
    context = gx.get_context(mode="ephemeral")
    suite = build_suite(
        context, config, expected_date=expected_date, known_failed_serials=known_failed_serials
    )
    source = context.data_sources.add_or_update_pandas("hdd_rul_batches")
    asset = source.add_dataframe_asset(name="daily_batch")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_batch")
    definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{SUITE_NAME}_definition", data=batch_definition, suite=suite)
    )

    # GX compare `date` à une chaîne ISO : on normalise pour éviter un faux
    # négatif lié au type (datetime64 vs date vs str) selon la source du lot.
    prepared = frame.copy()
    if "date" in prepared.columns:
        prepared["date"] = prepared["date"].astype("string")

    result = definition.run(batch_parameters={"dataframe": prepared}, result_format="SUMMARY")
    outcomes = _outcomes_from_result(result)

    quarantined = int(_quarantine_mask(frame, config, known_failed_serials).sum())
    total = len(frame)
    fraction = quarantined / total if total else 0.0
    if fraction > config.validation.max_quarantine_fraction:
        # Trop de lignes douteuses : la règle non bloquante devient bloquante.
        outcomes.append(
            ExpectationOutcome(
                expectation="expect_quarantine_fraction_to_be_below",
                column=",".join(config.validation.quarantine_on),
                severity="blocking",
                success=False,
                observed=round(fraction, 6),
                unexpected_count=quarantined,
            )
        )

    report = ValidationReport(
        batch_name=batch_name,
        batch_date=expected_date.isoformat() if expected_date else None,
        total_rows=total,
        quarantined_rows=quarantined,
        outcomes=outcomes,
        validated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    if persist_report:
        report.report_path = save_report(config, report)
    logger.info(report.summary())
    return report


def save_report(config: AppConfig, report: ValidationReport) -> Path:
    """Archive le rapport (preuve d'audit + artefact MLflow)."""
    directory = config.paths.artifacts / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.validated_at.replace(":", "").replace("-", "")
    path = directory / f"{Path(report.batch_name).stem}_{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def quarantine_rows(
    config: AppConfig,
    frame: pd.DataFrame,
    batch_name: str,
    known_failed_serials: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, int]:
    """Sépare les lignes saines des lignes en quarantaine et écrit ces dernières."""
    mask = _quarantine_mask(frame, config, known_failed_serials)
    if not mask.any():
        return frame, 0
    config.paths.quarantine.mkdir(parents=True, exist_ok=True)
    target = config.paths.quarantine / f"{Path(batch_name).stem}.parquet"
    frame.loc[mask].to_parquet(target, index=False)
    logger.warning("%d ligne(s) en quarantaine -> %s", int(mask.sum()), target)
    return frame.loc[~mask].copy(), int(mask.sum())
