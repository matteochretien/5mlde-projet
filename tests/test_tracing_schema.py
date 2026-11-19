"""Les écritures de traces visent des colonnes qui existent dans le schéma SQL.

Régression : un passage de remise des accents a renommé la colonne `details`
en `détails` dans l'INSERT de `pipeline_runs`. L'erreur SQL étant volontairement
avalée (une trace ne doit jamais faire échouer un lot), plus aucune exécution de
flow n'apparaissait dans Grafana, sans aucun test en échec.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from hdd_rul.common import tracing

SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "init-db.sql"


def _schema_columns() -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for name, body in re.findall(
        r"CREATE TABLE IF NOT EXISTS (tracing\.\w+)\s*\((.*?)\n\);", SCHEMA.read_text(), re.S
    ):
        columns = set()
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith(("--", "UNIQUE", "PRIMARY", "CONSTRAINT")):
                columns.add(line.split()[0])
        tables[name] = columns
    return tables


def test_les_insert_de_traces_correspondent_au_schema() -> None:
    schema = _schema_columns()
    assert "tracing.pipeline_runs" in schema, "schéma SQL introuvable ou format inattendu"

    source = inspect.getsource(tracing)
    inserts = re.findall(r"INSERT INTO (tracing\.\w+)\s*\(([^)]*)\)", source)
    assert inserts, "aucun INSERT trouvé dans hdd_rul.common.tracing"
    for table, columns in inserts:
        assert table in schema, f"table absente du schéma : {table}"
        inconnues = {c.strip() for c in columns.split(",")} - schema[table]
        assert not inconnues, f"{table} : colonnes absentes du schéma {sorted(inconnues)}"
