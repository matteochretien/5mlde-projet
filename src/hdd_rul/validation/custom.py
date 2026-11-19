"""Expectations Great Expectations personnalisées, propres au domaine.

Les trois règles ci-dessous n'existent pas dans la bibliothèque standard de GX
parce qu'elles encodent une connaissance du jeu Drive Stats :

1. `ExpectColumnValuesToBeValidDriveSerial` — un numéro de série constructeur est
   alphanumérique, sans espace, de 5 à 32 caractères. Un champ tronqué ou décalé
   (erreur de parsing CSV) se détecte ici et non en aval.
2. `ExpectColumnValuesToNotBeKnownFailedDrive` — un disque déclaré en panne est
   retiré du parc : il ne doit plus réapparaître les jours suivants. Sa
   réapparition signifie soit un `failure` erroné, soit un lot déjà traité
   rejoué par erreur ; dans les deux cas le label RUL serait faux.
3. `ExpectDriveFailureRateToBeBelow` — le taux de pannes journalier du parc est
   de l'ordre de 0,1 pour mille. Un taux très supérieur signale un fichier
   corrompu ou un décalage de colonnes, pas une vague de pannes.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import BatchExpectation, ColumnMapExpectation
from great_expectations.expectations.metrics.map_metric_provider import (
    ColumnMapMetricProvider,
    column_condition_partial,
)
from great_expectations.expectations.metrics.metric_provider import metric_value
from great_expectations.expectations.metrics.table_metric_provider import TableMetricProvider

SERIAL_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._\-]{4,31}"


class ColumnValuesAreValidDriveSerial(ColumnMapMetricProvider):
    condition_metric_name = "column_values.valid_drive_serial"

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column, **kwargs):
        values = column.astype("string")
        return values.notna() & values.str.fullmatch(SERIAL_PATTERN).fillna(False)


class ExpectColumnValuesToBeValidDriveSerial(ColumnMapExpectation):
    """Attend des numéros de série plausibles (alphanumériques, 5 à 32 caractères)."""

    map_metric = "column_values.valid_drive_serial"


class ColumnValuesAreNotKnownFailedDrive(ColumnMapMetricProvider):
    condition_metric_name = "column_values.not_known_failed_drive"
    condition_value_keys = ("failed_serials",)

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column, failed_serials=None, **kwargs):
        known = set(failed_serials or ())
        if not known:
            return column.notna() | column.isna()  # tout vrai : rien à comparer
        return ~column.isin(known)


class ExpectColumnValuesToNotBeKnownFailedDrive(ColumnMapExpectation):
    """Attend qu'aucun disque déjà tombe en panne ne réapparaisse dans le lot."""

    failed_serials: tuple[str, ...] = ()
    map_metric = "column_values.not_known_failed_drive"
    success_keys: ClassVar[tuple[str, ...]] = ("mostly", "failed_serials")


class TableDriveFailureRate(TableMetricProvider):
    metric_name = "table.drive_failure_rate"
    value_keys = ("failure_column",)

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine,
        metric_domain_kwargs,
        metric_value_kwargs,
        metrics,
        runtime_configuration,
    ):
        frame, _, _ = execution_engine.get_compute_domain(metric_domain_kwargs, domain_type="table")
        column = metric_value_kwargs.get("failure_column") or "failure"
        if column not in frame.columns or len(frame) == 0:
            return 0.0
        numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return float((numeric >= 1).mean())


class ExpectDriveFailureRateToBeBelow(BatchExpectation):
    """Attend un taux de pannes du lot inférieur au seuil (anomalie de volume)."""

    max_rate: float = 0.01
    failure_column: str = "failure"
    metric_dependencies: ClassVar[tuple[str, ...]] = (
        "table.drive_failure_rate",
        "table.row_count",
    )
    success_keys: ClassVar[tuple[str, ...]] = ("max_rate", "failure_column")

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        rate = float(metrics["table.drive_failure_rate"])
        return {
            "success": rate <= self.max_rate,
            "result": {
                "observed_value": rate,
                "row_count": int(metrics["table.row_count"]),
                "max_rate": self.max_rate,
            },
        }


__all__ = [
    "ExpectColumnValuesToBeValidDriveSerial",
    "ExpectColumnValuesToNotBeKnownFailedDrive",
    "ExpectDriveFailureRateToBeBelow",
]
