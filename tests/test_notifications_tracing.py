"""Alertes et persistance : les deux doivent être non bloquantes."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hdd_rul.common import tracing
from hdd_rul.common.config import AlertsConfig
from hdd_rul.common.notifications import Alert, build_payload, notify
from hdd_rul.common.tracing import TracingStore, to_json


def _alerte() -> Alert:
    return Alert(
        title="Titre",
        summary="Résumé",
        level="critical",
        status={"Statut": "Failed", "Étape": "validation", "Lot": None},
        sections={"Diagnostic": {"Nature": "donnée invalide", "Cause": "x_y_z * 2"}},
        links=[("Ouvrir le flow run dans Prefect", "http://localhost:4200/runs/flow-run/1")],
        commands={"Logs": "uv run prefect flow-run logs 1"},
    )


def test_charge_utile_de_l_alerte() -> None:
    payload = build_payload(_alerte(), AlertsConfig())
    assert payload["title"] == "[CRITIQUE] [hdd-rul] Titre"
    assert payload["themeColor"] == "CC3232"
    assert payload["sections"][0]["facts"] == [
        {"name": "Statut", "value": "Failed"},
        {"name": "Étape", "value": "validation"},
    ]  # les champs vides sont omis
    assert "Ouvrir le flow run dans Prefect : http://localhost:4200" in payload["text"]


def test_la_charge_utile_porte_une_adaptive_card_pour_teams() -> None:
    """Un workflow Teams (Power Automate) lit `attachments` : sans Adaptive Card,
    il répond 202 mais ne publie rien dans le canal."""
    payload = build_payload(_alerte(), AlertsConfig())
    attachment = payload["attachments"][0]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    card = attachment["content"]
    assert card["type"] == "AdaptiveCard"
    assert card["body"][0]["text"] == "[CRITIQUE] [hdd-rul] Titre"
    assert card["body"][0]["color"] == "Attention"
    # Le texte variable passe par des TextRun : un `_` ou `*` n'est pas du Markdown.
    runs = [i["text"] for b in card["body"] if b["type"] == "RichTextBlock" for i in b["inlines"]]
    assert "x_y_z * 2" in runs
    liens = [b["text"] for b in card["body"] if b["type"] == "TextBlock"]
    assert "◦ [Ouvrir le flow run dans Prefect](http://localhost:4200/runs/flow-run/1)" in liens
    json.dumps(payload)  # sérialisable tel quel


def test_sans_webhook_l_alerte_est_seulement_journalisee(caplog) -> None:
    """Un poste sans webhook doit rester utilisable : pas d'échec, mais une trace."""
    with caplog.at_level("WARNING"):
        assert notify(_alerte(), alerts=AlertsConfig()) is False
    assert "ALERTE[critical]" in caplog.text


def test_webhook_injoignable_n_echoue_pas(caplog) -> None:
    """Une alerte perdue ne doit jamais masquer l'incident qui l'a déclenchée."""
    alerts = AlertsConfig(webhook_url="http://127.0.0.1:1/hook", timeout_seconds=0.2)
    with caplog.at_level("WARNING"):
        assert notify(_alerte(), alerts=alerts) is False
    assert "envoi de l'alerte impossible" in caplog.text


def test_base_injoignable_n_echoue_pas(caplog) -> None:
    """L'API doit continuer à répondre même sans base de traces."""
    store = TracingStore(dsn="host=127.0.0.1 port=1 dbname=x user=x password=x connect_timeout=1")
    with caplog.at_level("WARNING"):
        store.record_request(
            endpoint="/predict",
            method="POST",
            status_code=200,
            duration_ms=12.0,
            model_version="1",
            n_items=1,
        )
        sante = store.healthcheck()
    assert sante["reachable"] is False
    assert sante["error"]
    assert store.record_predictions([]) == 0


def test_store_desactive_ne_tente_aucune_connexion() -> None:
    store = TracingStore(enabled=False)
    assert store.healthcheck() == {"enabled": False, "reachable": False, "error": "desactive"}
    assert (
        store.record_feedbacks(
            [{"serial_number": "X", "observation_date": None, "actual_rul": 1.0}]
        )
        == 0
    )


def test_to_json_remplace_nan_et_infinis_par_null() -> None:
    """`jsonb` refuse `NaN` : un attribut SMART absent ne doit pas rejeter le lot."""
    texte = to_json({"smart_187_raw": np.float64("nan"), "smart_5_raw": 3.0, "x": [math.inf]})
    assert "NaN" not in texte and "Infinity" not in texte
    assert json.loads(texte) == {"smart_187_raw": None, "smart_5_raw": 3.0, "x": [None]}


class _CurseurQuiEchoue:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, *args, **kwargs):
        raise RuntimeError("invalid input syntax for type json")

    execute = executemany


class _ConnexionFactice:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _CurseurQuiEchoue()


def test_ecriture_qui_echoue_n_echoue_pas_l_appelant(monkeypatch, caplog) -> None:
    """Régression : une erreur pendant l'écriture levait « generator didn't stop after
    throw() » et faisait échouer le scoring du lot entier."""
    if tracing.psycopg is None:  # pragma: no cover
        pytest.skip("psycopg absent")
    monkeypatch.setattr(tracing.psycopg, "connect", lambda *a, **k: _ConnexionFactice())
    store = TracingStore(dsn="factice")
    ligne = {"prediction_id": "p1", "predicted_rul": 12.0}
    with caplog.at_level("WARNING"):
        assert store.record_predictions([ligne]) == 0
        assert store.execute("SELECT 1") is None
        assert store.fetch_predictions(serials=["X"]).empty
    assert "écriture de trace ignorée" in caplog.text
    assert store.healthcheck()["error"]
