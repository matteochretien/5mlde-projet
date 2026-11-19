"""Génère le tableau de bord Grafana (JSON) à partir d'une description lisible.

Écrire 1 000 lignes de JSON à la main est ingérable et impossible à relire en
revue : le dashboard est donc décrit ici en Python, et le JSON est un artefact
reproductible (`python scripts/build_dashboard.py`).

Trois vues, comme demandé : technique, métier, et croisée technico-métier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASOURCE = {"type": "postgres", "uid": "hdd-rul-postgres"}
OUTPUT = Path("grafana/dashboards/hdd-rul.json")


def panel(
    title: str,
    sql: str,
    *,
    kind: str = "timeseries",
    x: int = 0,
    y: int = 0,
    w: int = 12,
    h: int = 7,
    unit: str | None = None,
    description: str = "",
    legend_format: str | None = None,
    fmt: str | None = None,
    transformations: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "refId": "A",
        # `fmt` force le format : un barchart sans colonne de temps (histogramme)
        # doit etre en `table`, sinon Grafana repond « no time column found ».
        "format": fmt or ("table" if kind in {"table", "stat", "bargauge"} else "time_series"),
        "rawQuery": True,
        "rawSql": sql,
        "datasource": DATASOURCE,
        "editorMode": "code",
    }
    if legend_format:
        target["legendFormat"] = legend_format
    return {
        "type": kind,
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target],
        "fieldConfig": {
            "defaults": {"unit": unit or "short", "custom": {}},
            "overrides": [],
        },
        "transformations": transformations or [],
        "options": options
        or (
            {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}}
            if kind == "timeseries"
            else {}
        ),
    }


def by_simulated_day(
    title: str,
    sql: str,
    *,
    series: str,
    value: str,
    stacked: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Barres par jour SIMULÉ (`observation_date`), une série par valeur de `series`.

    Le rejeu traite plusieurs journées d'avril en quelques minutes : sur l'axe
    du temps réel (`predicted_at`), tout tombe sur un ou deux points et les
    courbes n'apparaissent pas. L'axe métier est le jour rejoué ; la requête
    renvoie (Jour, série, valeur) et Grafana pivote une colonne par série.
    """
    return panel(
        title,
        sql,
        kind="barchart",
        fmt="table",
        transformations=[
            {
                "id": "groupingToMatrix",
                "options": {
                    "rowField": "Jour",
                    "columnField": series,
                    "valueField": value,
                    "emptyValue": "zero",
                },
            }
        ],
        options={
            "stacking": "normal" if stacked else "none",
            "showValue": "never",
            "xTickLabelRotation": -45,
            "xTickLabelSpacing": 100,
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        **kwargs,
    )


def row(title: str, y: int) -> dict[str, Any]:
    return {
        "type": "row",
        "title": title,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "collapsed": False,
        "panels": [],
    }


def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []

    # ---------------- Vue technique ----------------------------------------
    panels.append(row("Vue technique — service et traitements", 0))
    panels.append(
        panel(
            "Trafic API (requêtes/min) par endpoint",
            """
SELECT $__timeGroupAlias(requested_at, '1m'), endpoint, count(*) AS requêtes
FROM tracing.api_requests
WHERE $__timeFilter(requested_at)
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            y=1,
            description="Charge réellement servie, par endpoint.",
        )
    )
    panels.append(
        panel(
            "Latence p95 (ms) par endpoint",
            """
SELECT $__timeGroupAlias(requested_at, '1m'), endpoint,
       percentile_disc(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
FROM tracing.api_requests
WHERE $__timeFilter(requested_at)
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            x=12,
            y=1,
            unit="ms",
            description="Objectif de dimensionnement : p95 < 200 ms (cf. loadtest/PROTOCOLE.md).",
        )
    )
    panels.append(
        panel(
            "Erreurs par code HTTP",
            """
SELECT $__timeGroupAlias(requested_at, '1m'), status_code::text AS code, count(*) AS erreurs
FROM tracing.api_requests
WHERE $__timeFilter(requested_at) AND status_code >= 400
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            kind="barchart",
            y=8,
            description="422 = entrée refusée par la validation, 503 = dépendance indisponible.",
        )
    )
    panels.append(
        panel(
            "Exécutions de flows (statut)",
            """
SELECT recorded_at AS "time", flow_name, status, trigger, batch_name,
       details->>'rows' AS lignes
FROM tracing.pipeline_runs
WHERE $__timeFilter(recorded_at)
ORDER BY recorded_at DESC
LIMIT 100
            """.strip(),
            kind="table",
            x=12,
            y=8,
            description="Suivi des flows Prefect : ingestion, lot journalier, entraînement.",
        )
    )
    panels.append(
        panel(
            "Version de modèle servie",
            """
SELECT model_version AS "Version", count(*) AS "Prédictions"
FROM tracing.predictions
WHERE $__timeFilter(predicted_at)
GROUP BY 1
ORDER BY 2 DESC
            """.strip(),
            kind="table",
            y=15,
            w=8,
            h=6,
            description="Permet de voir le basculement de version après une promotion.",
        )
    )
    panels.append(
        panel(
            "Lignes en quarantaine",
            """
SELECT $__timeGroupAlias(recorded_at, '1h'), sum(rows) AS lignes_quarantaine
FROM tracing.quarantine
WHERE $__timeFilter(recorded_at)
GROUP BY 1
ORDER BY 1
            """.strip(),
            x=8,
            y=15,
            w=16,
            h=6,
            description="Règles non bloquantes du contrat de données (ex. capacity_bytes <= 0).",
        )
    )

    # ---------------- Vue métier -------------------------------------------
    panels.append(row("Vue métier — parc de disques", 21))
    panels.append(
        by_simulated_day(
            "Disques à risque (RUL prédite < 30 j) par jour et par modèle",
            """
WITH top AS (
  SELECT model FROM tracing.predictions
  WHERE source = 'batch' AND alert
  GROUP BY model ORDER BY count(*) DESC LIMIT 8
)
SELECT observation_date::text AS "Jour",
       CASE WHEN model IN (SELECT model FROM top) THEN model ELSE 'autres' END AS modele,
       count(*) FILTER (WHERE alert) AS a_risque
FROM tracing.predictions
WHERE source = 'batch'
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            series="modele",
            value="a_risque",
            stacked=True,
            y=22,
            description=(
                "Ce que l'exploitant doit traiter : les disques à remplacer en priorité. "
                "Axe : jour rejoué (indépendant de la plage de temps). "
                "Les 8 modèles les plus alertés, le reste dans « autres »."
            ),
        )
    )
    panels.append(
        panel(
            "Distribution des RUL prédites",
            """
SELECT (LEAST(width_bucket(predicted_rul, 0, 90, 18), 18) * 5)::text AS "RUL prédite (jours)",
       count(*) AS "Disques"
FROM tracing.predictions
WHERE $__timeFilter(predicted_at)
GROUP BY LEAST(width_bucket(predicted_rul, 0, 90, 18), 18)
ORDER BY LEAST(width_bucket(predicted_rul, 0, 90, 18), 18)
            """.strip(),
            kind="barchart",
            fmt="table",
            x=12,
            y=22,
            description=(
                "Tranches de 5 j, étiquetées par leur borne haute. Un pic à 90 j est normal : "
                "c'est la valeur « pas de panne en vue » (RUL plafonnée à l'horizon)."
            ),
        )
    )
    panels.append(
        panel(
            "Pannes constatées par jour",
            # Une panne produit un feedback par prédiction passée du disque :
            # on compte les disques, pas les feedbacks.
            """
SELECT failure_date::text AS "Jour", count(DISTINCT serial_number) AS "Pannes"
FROM tracing.feedbacks
WHERE failure_date IS NOT NULL
GROUP BY 1
ORDER BY 1
            """.strip(),
            kind="barchart",
            fmt="table",
            options={
                "showValue": "auto",
                "xTickLabelRotation": -45,
                "legend": {"showLegend": False},
            },
            y=29,
            description=(
                "Disques tombés en panne, par jour rejoué (feedbacks automatiques du rejeu). "
                "Indépendant de la plage de temps."
            ),
        )
    )
    panels.append(
        panel(
            "Disques scorés et alertes (cumul)",
            """
SELECT count(*) AS "Disques scorés",
       count(*) FILTER (WHERE alert) AS "Alertes < 30 j",
       count(DISTINCT serial_number) AS "Disques distincts"
FROM tracing.predictions
WHERE $__timeFilter(predicted_at)
            """.strip(),
            kind="stat",
            x=12,
            y=29,
            description="Volume servi sur la période sélectionnée.",
        )
    )

    # ---------------- Vue croisée ------------------------------------------
    panels.append(row("Vue croisée — qualité du modèle en production", 36))
    panels.append(
        by_simulated_day(
            "MAE réelle en production, par version de modèle",
            """
SELECT p.observation_date::text AS "Jour", 'v' || p.model_version AS version,
       avg(abs(p.predicted_rul - f.actual_rul)) AS mae
FROM tracing.predictions p
JOIN tracing.feedbacks f
  ON f.serial_number = p.serial_number AND f.observation_date = p.observation_date
WHERE p.source = 'batch'
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            series="version",
            value="mae",
            y=37,
            unit="suffix: j",  # « d » serait converti en semaines par Grafana
            description=(
                "À comparer à la MAE de validation annoncée : dérive ou non ? "
                "Axe : jour rejoué (indépendant de la plage de temps)."
            ),
        )
    )
    panels.append(
        by_simulated_day(
            "MAE réelle sur la zone critique (RUL < 30 j)",
            """
SELECT p.observation_date::text AS "Jour", 'v' || p.model_version AS version,
       avg(abs(p.predicted_rul - f.actual_rul)) AS mae_lt30
FROM tracing.predictions p
JOIN tracing.feedbacks f
  ON f.serial_number = p.serial_number AND f.observation_date = p.observation_date
WHERE p.source = 'batch' AND f.actual_rul < 30
GROUP BY 1, 2
ORDER BY 1
            """.strip(),
            series="version",
            value="mae_lt30",
            x=12,
            y=37,
            unit="suffix: j",  # « d » serait converti en semaines par Grafana
            description="C'est la métrique sur laquelle repose la règle de mise en service.",
        )
    )
    panels.append(
        panel(
            "MAE de validation (MLflow) vs MAE de production, par version",
            """
SELECT v.version AS "Version",
       v.alias AS "Alias",
       round((v.metrics->>'mae')::numeric, 3) AS "MAE validation",
       round((v.metrics->>'mae_lt30')::numeric, 3) AS "MAE validation < 30 j",
       round(avg(abs(p.predicted_rul - f.actual_rul))::numeric, 3) AS "MAE production",
       count(f.id) AS "Feedbacks"
FROM tracing.model_versions v
LEFT JOIN tracing.predictions p ON p.model_version = v.version
LEFT JOIN tracing.feedbacks f
       ON f.serial_number = p.serial_number AND f.observation_date = p.observation_date
GROUP BY 1, 2, 3, 4
ORDER BY 1 DESC
            """.strip(),
            kind="table",
            y=44,
            w=24,
            description="Le tableau qui justifie (ou non) une promotion ou un retour arrière.",
        )
    )
    panels.append(
        panel(
            "Erreur par modèle de disque (version servie)",
            """
SELECT p.model AS "Modèle de disque",
       count(*) AS "Feedbacks",
       round(avg(abs(p.predicted_rul - f.actual_rul))::numeric, 3) AS "MAE"
FROM tracing.predictions p
JOIN tracing.feedbacks f
  ON f.serial_number = p.serial_number AND f.observation_date = p.observation_date
WHERE $__timeFilter(p.predicted_at)
GROUP BY 1
ORDER BY 3 DESC
            """.strip(),
            kind="table",
            y=51,
            w=24,
            h=8,
            description="Les attributs SMART n'ont pas la même sémantique selon le constructeur.",
        )
    )

    return {
        "uid": "hdd-rul-main",
        "title": "HDD RUL — technique, métier, croisé",
        "description": (
            "Supervision de la chaîne de prédiction de la RUL des disques durs. "
            "Source : schéma `tracing` de PostgreSQL."
        ),
        "tags": ["hdd-rul", "mlops"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        # 7 jours : les traitements du rejeu et des incidents restent visibles le
        # lendemain (soutenance). « Last 1 hour » masquerait tout sauf le trafic API.
        "time": {"from": "now-7d", "to": "now"},
        "editable": True,
        "panels": panels,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUTPUT} écrit ({len(build()['panels'])} panneaux)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
