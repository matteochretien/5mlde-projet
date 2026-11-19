"""Persistance des traces dans PostgreSQL (schéma `tracing`).

Sert deux usages :
  * exploitation — prédictions servies, feedbacks reçus, latences, erreurs ;
  * supervision — c'est la source unique des tableaux de bord Grafana.

Toutes les écritures sont **tolérantes à la panne** : si PostgreSQL est
indisponible, l'appel est journalisé et ignoré, sans faire échouer la
prédiction. L'API doit continuer à répondre même sans base de traces ; c'est
l'endpoint `/health` qui signale la dégradation (et le scénario d'incident
« dépendance indisponible » de la démonstration s'appuie dessus).
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from hdd_rul.common.logging import get_logger

if TYPE_CHECKING:  # pandas n'est importe que pour le typage
    import pandas as pd

logger = get_logger(__name__)

try:  # psycopg est une dépendance du projet, mais la chaîne doit tourner sans base
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


def to_json(value: Any) -> str:
    """JSON accepté par PostgreSQL : NaN et infinis deviennent `null`.

    `json.dumps` écrit `NaN` par défaut, ce que le type `jsonb` refusé. Or un
    attribut SMART absent arrive justement en NaN depuis pandas : sans ce
    nettoyage, toute une journée de prédictions est rejetée.
    """

    def clean(obj: Any) -> Any:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {key: clean(item) for key, item in obj.items()}
        if isinstance(obj, list | tuple):
            return [clean(item) for item in obj]
        return obj

    return json.dumps(clean(value), default=str, allow_nan=False)


def dsn_from_env() -> str:
    """Chaîne de connexion construite à partir des variables d'environnement."""
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'hdd_rul')} "
        f"user={os.environ.get('POSTGRES_USER', 'hdd')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'change-me')}"
    )


class TracingStore:
    """Accès au schéma `tracing`. Jamais bloquant pour l'appelant."""

    def __init__(self, dsn: str | None = None, *, enabled: bool = True) -> None:
        self.dsn = dsn or dsn_from_env()
        self.enabled = enabled and psycopg is not None
        self._last_error: str | None = None
        if enabled and psycopg is None:  # pragma: no cover
            logger.warning("psycopg absent : les traces ne seront pas persistées")

    # -- infrastructure ----------------------------------------------------
    def _note_error(self, exc: Exception, what: str) -> None:
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("%s (%s)", what, self._last_error)

    @contextmanager
    def connection(self):
        """Connexion courte (le pooling est délégué à l'appelant si besoin).

        Cède `None` si la base est injoignable. Une erreur levée *pendant*
        l'écriture est journalisée puis absorbée : une trace perdue ne doit
        jamais faire échouer l'appelant. (Céder une seconde fois après
        l'exception lèverait « generator didn't stop after throw() ».)
        """
        if not self.enabled:
            yield None
            return
        try:
            conn = psycopg.connect(self.dsn, connect_timeout=3, row_factory=dict_row)
        except Exception as exc:
            self._note_error(exc, "PostgreSQL indisponible")
            yield None
            return
        try:
            with conn:
                self._last_error = None
                yield conn
        except Exception as exc:
            self._note_error(exc, "écriture de trace ignorée")

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any | None:
        with self.connection() as conn:
            if conn is None:
                return None
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description is None:
                    return None
                return cursor.fetchone()
        return None

    def healthcheck(self) -> dict[str, Any]:
        """État de la base, exposé par `/health`."""
        if not self.enabled:
            return {"enabled": False, "reachable": False, "error": "desactive"}
        row = self.execute("SELECT 1 AS ok")
        return {
            "enabled": True,
            "reachable": row is not None,
            "error": self._last_error,
        }

    # -- écritures ---------------------------------------------------------
    def record_pipeline_run(
        self,
        *,
        flow_name: str,
        flow_run_id: str | None,
        status: str,
        trigger: str,
        batch_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Statut d'une exécution de flow, pour la vue technique de Grafana."""
        self.execute(
            """
            INSERT INTO tracing.pipeline_runs
                (flow_name, flow_run_id, status, trigger, batch_name, details, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                flow_name,
                flow_run_id,
                status,
                trigger,
                batch_name,
                to_json(details or {}),
                datetime.now(UTC),
            ),
        )

    def record_model_version(
        self,
        *,
        model_name: str,
        version: str,
        alias: str,
        run_id: str | None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO tracing.model_versions
                (model_name, version, alias, run_id, metrics, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_name, version, alias) DO UPDATE
                SET run_id = EXCLUDED.run_id,
                    metrics = EXCLUDED.metrics,
                    recorded_at = EXCLUDED.recorded_at
            """,
            (
                model_name,
                version,
                alias,
                run_id,
                to_json(metrics or {}),
                datetime.now(UTC),
            ),
        )

    def prune_alias(self, *, model_name: str, alias: str, keep_version: str | None) -> None:
        """Retire les lignes d'un alias qui ne désignent plus la bonne version.

        Un alias ne pointe que sur une version à la fois dans le registre : la
        table doit dire la même chose, sinon la vue croisée de Grafana affiche
        deux « challenger » (l'ancien et le nouveau).
        """
        self.execute(
            """
            DELETE FROM tracing.model_versions
            WHERE model_name = %s AND alias = %s AND (%s::text IS NULL OR version <> %s)
            """,
            (model_name, alias, keep_version, keep_version),
        )

    def record_request(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        model_version: str | None,
        n_items: int,
        error: str | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO tracing.api_requests
                (endpoint, method, status_code, duration_ms, model_version, n_items, error,
                 requested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                endpoint,
                method,
                status_code,
                duration_ms,
                model_version,
                n_items,
                error,
                datetime.now(UTC),
            ),
        )

    def record_predictions(self, rows: list[dict[str, Any]]) -> int:
        """Insère un lot de prédictions. Retourne le nombre de lignes écrites."""
        if not rows:
            return 0
        with self.connection() as conn:
            if conn is None:
                return 0
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO tracing.predictions
                        (prediction_id, observation_date, serial_number, model, capacity_bytes,
                         features, predicted_rul, alert, model_name, model_version, run_id,
                         source, predicted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id) DO NOTHING
                    """,
                    [
                        (
                            row["prediction_id"],
                            row.get("observation_date"),
                            row.get("serial_number"),
                            row.get("model"),
                            row.get("capacity_bytes"),
                            to_json(row.get("features") or {}),
                            row["predicted_rul"],
                            row.get("alert", False),
                            row.get("model_name"),
                            row.get("model_version"),
                            row.get("run_id"),
                            row.get("source", "api"),
                            row.get("predicted_at") or datetime.now(UTC),
                        )
                        for row in rows
                    ],
                )
                return len(rows)
        return 0

    def record_feedbacks(self, rows: list[dict[str, Any]]) -> int:
        """Enregistre les RUL réelles constatées (pannes ou survie à J+H)."""
        if not rows:
            return 0
        with self.connection() as conn:
            if conn is None:
                return 0
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO tracing.feedbacks
                        (serial_number, observation_date, actual_rul, failure_date, source,
                         received_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (serial_number, observation_date) DO UPDATE
                        SET actual_rul = EXCLUDED.actual_rul,
                            failure_date = EXCLUDED.failure_date,
                            source = EXCLUDED.source,
                            received_at = EXCLUDED.received_at
                    """,
                    [
                        (
                            row["serial_number"],
                            row["observation_date"],
                            row["actual_rul"],
                            row.get("failure_date"),
                            row.get("source", "api"),
                            row.get("received_at") or datetime.now(UTC),
                        )
                        for row in rows
                    ],
                )
                return len(rows)
        return 0

    # -- lectures ----------------------------------------------------------
    def fetch_predictions(self, *, serials: list[str], until: Any | None = None) -> pd.DataFrame:
        """Prédictions passées des disques donnes (pour dériver les feedbacks)."""
        import pandas as pd

        empty = pd.DataFrame(columns=["serial_number", "observation_date", "predicted_rul"])
        if not serials:
            return empty
        rows: list[dict[str, Any]] = []
        with self.connection() as conn:
            if conn is None:
                return empty
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT serial_number, observation_date, predicted_rul, model_version
                    FROM tracing.predictions
                    WHERE serial_number = ANY(%s)
                      AND (%s::date IS NULL OR observation_date <= %s::date)
                    """,
                    (list(serials), until, until),
                )
                rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else empty

    def record_quarantine(
        self, *, batch_name: str, batch_date: str | None, rows: int, reason: str
    ) -> None:
        self.execute(
            """
            INSERT INTO tracing.quarantine (batch_name, batch_date, rows, reason, recorded_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (batch_name, batch_date, rows, reason, datetime.now(UTC)),
        )


_store: TracingStore | None = None


def get_store() -> TracingStore:
    """Instance partagée (une connexion par écriture, pas de pool à maintenir)."""
    global _store
    if _store is None:
        enabled = os.environ.get("HDD_RUL_TRACING_ENABLED", "1") not in {"0", "false", "False"}
        _store = TracingStore(enabled=enabled)
    return _store
