"""Alertes sortantes (webhook Teams / Slack) utilisées par les flows Prefect.

Une alerte est structurée comme un rapport d'incident court :

* un **titre** qui dit quoi et où (« Échec du flow daily_batch : Failed (étape
  validation) ») et une phrase de **résumé** ;
* une **ligne d'état** (statut, étape, lot, déclencheur…) ;
* des **sections** : Diagnostic (nature, cause, conséquence, action), Références
  (flow run, run MLflow, date, environnement) ;
* une section **Accès** : liens directs vers le service concerné (flow run
  Prefect, run ou version MLflow, tableau de bord Grafana) et commandes locales.

Sans URL configurée, l'alerte est seulement journalisée : le pipeline reste
exécutable sur un poste sans webhook, et les tests n'ont pas besoin de réseau.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib import error, request

from hdd_rul.common.config import AlertsConfig
from hdd_rul.common.logging import get_logger

logger = get_logger(__name__)

LEVEL_LABELS = {"info": "INFO", "warning": "AVERTISSEMENT", "critical": "CRITIQUE"}
_LEVEL_COLORS = {"info": "2DC937", "warning": "E7B416", "critical": "CC3232"}
_ADAPTIVE_COLORS = {"info": "Good", "warning": "Warning", "critical": "Attention"}
_BULLET = "◦ "


@dataclass
class Alert:
    """Contenu d'une alerte, indépendant du canal qui l'affiche."""

    title: str
    summary: str = ""
    level: str = "info"
    status: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    links: list[tuple[str, str]] = field(default_factory=list)
    commands: dict[str, str] = field(default_factory=dict)
    note: str | None = None

    def full_title(self, tag: str | None = None) -> str:
        level = LEVEL_LABELS.get(self.level, self.level.upper())
        prefix = f"[{level}]" + (f" [{tag}]" if tag else "")
        return f"{prefix} {self.title}"

    def status_line(self) -> str:
        return " | ".join(f"{k} : {v}" for k, v in self.status.items() if _filled(v))


def _filled(value: Any) -> bool:
    return value is not None and str(value) != ""


def _items(fields: dict[str, Any]) -> list[tuple[str, str]]:
    return [(str(k), str(v)) for k, v in fields.items() if _filled(v)]


def context_references(alerts: AlertsConfig, when: datetime | None = None) -> dict[str, str]:
    """Références communes à toutes les alertes : date et environnement."""
    moment = (when or datetime.now(UTC)).astimezone(UTC)
    return {
        "Date": moment.strftime("%d/%m/%Y %H:%M:%S UTC"),
        "Environnement": alerts.resolved_environment(),
    }


# ---------------------------------------------------------------------------
# Rendu Adaptive Card (workflow Teams)
# ---------------------------------------------------------------------------
def _rich(*inlines: dict[str, Any], spacing: str = "Small") -> dict[str, Any]:
    return {"type": "RichTextBlock", "spacing": spacing, "inlines": list(inlines)}


def _run(text: str, **style: Any) -> dict[str, Any]:
    return {"type": "TextRun", "text": text, **style}


def _heading(text: str) -> dict[str, Any]:
    return {
        "type": "TextBlock",
        "text": text,
        "weight": "Bolder",
        "size": "Medium",
        "spacing": "Medium",
        "wrap": True,
    }


def adaptive_card(alert: Alert, alerts: AlertsConfig | None = None) -> dict[str, Any]:
    """Adaptive Card : le format attendu par les workflows Teams (Power Automate).

    Le texte variable (messages d'erreur, noms de règles) passe par des
    `RichTextBlock` et non par du Markdown : un `_` ou un `*` dans un message
    d'exception ne doit pas changer la mise en forme. Seuls les liens, dont le
    libellé est maîtrisé ici, utilisent le Markdown.
    """
    config = alerts or AlertsConfig()
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": alert.full_title(config.tag),
            "weight": "Bolder",
            "size": "Large",
            "color": _ADAPTIVE_COLORS.get(alert.level, "Default"),
            "wrap": True,
        }
    ]
    if alert.summary:
        body.append(_rich(_run(alert.summary, italic=True), spacing="Medium"))
    if alert.status:
        inlines: list[dict[str, Any]] = []
        for key, value in _items(alert.status):
            if inlines:
                inlines.append(_run(" | "))
            inlines += [_run(f"{key} : ", weight="Bolder"), _run(value)]
        body.append(_rich(*inlines, spacing="Medium"))

    for name, fields in alert.sections.items():
        items = _items(fields)
        if not items:
            continue
        body.append(_heading(name))
        body += [_rich(_run(_BULLET), _run(f"{k} : ", weight="Bolder"), _run(v)) for k, v in items]

    if alert.links or alert.commands:
        body.append(_heading("Accès"))
        body += [
            {
                "type": "TextBlock",
                "text": f"{_BULLET}[{label}]({url})",
                "spacing": "Small",
                "wrap": True,
            }
            for label, url in alert.links
        ]
        body += [
            _rich(_run(_BULLET), _run(f"{label} : "), _run(cmd, fontType="Monospace"))
            for label, cmd in alert.commands.items()
        ]
    if alert.note:
        body.append(_rich(_run(alert.note, italic=True, isSubtle=True), spacing="Medium"))

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "msteams": {"width": "Full"},
        "body": body,
    }


# ---------------------------------------------------------------------------
# Rendu texte (Slack, ancien connecteur Teams, journaux)
# ---------------------------------------------------------------------------
def render_text(alert: Alert, alerts: AlertsConfig | None = None) -> str:
    config = alerts or AlertsConfig()
    lines = [alert.full_title(config.tag)]
    if alert.summary:
        lines.append(alert.summary)
    if alert.status:
        lines += ["", alert.status_line()]
    for name, fields in alert.sections.items():
        items = _items(fields)
        if items:
            lines += ["", name, *(f"- {k} : {v}" for k, v in items)]
    if alert.links or alert.commands:
        lines += ["", "Accès"]
        lines += [f"- {label} : {url}" for label, url in alert.links]
        lines += [f"- {label} : {cmd}" for label, cmd in alert.commands.items()]
    if alert.note:
        lines += ["", alert.note]
    return "\n".join(lines)


def build_payload(alert: Alert, alerts: AlertsConfig | None = None) -> dict[str, Any]:
    """Une seule charge utile, lisible par les trois cibles courantes.

    * workflow Teams (Power Automate, « requête webhook reçue ») : lit
      `attachments`, qui porte une Adaptive Card. Il répond 202 même si le
      format ne lui convient pas : sans cette pièce jointe, rien n'arrive dans
      le canal, et rien ne le signale côté pipeline ;
    * ancien connecteur Teams (Office 365) : lit les champs `MessageCard` ;
    * Slack : lit `text`.
    """
    config = alerts or AlertsConfig()
    title = alert.full_title(config.tag)
    sections: list[dict[str, Any]] = []
    if alert.status:
        sections.append({"facts": [{"name": k, "value": v} for k, v in _items(alert.status)]})
    for name, fields in alert.sections.items():
        if items := _items(fields):
            sections.append(
                {"activityTitle": name, "facts": [{"name": k, "value": v} for k, v in items]}
            )
    if alert.commands:
        sections.append(
            {
                "activityTitle": "Accès",
                "facts": [{"name": k, "value": v} for k, v in alert.commands.items()],
            }
        )
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": adaptive_card(alert, config),
            }
        ],
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": _LEVEL_COLORS.get(alert.level, "2DC937"),
        "summary": title,
        "title": title,
        "text": render_text(alert, config),
        "sections": sections,
        "potentialAction": [
            {"@type": "OpenUri", "name": label, "targets": [{"os": "default", "uri": url}]}
            for label, url in alert.links
        ],
    }


def notify(alert: Alert, *, alerts: AlertsConfig | None = None) -> bool:
    """Envoie une alerte. Retourne True si le webhook a accepté l'appel.

    N'échoue jamais : une alerte perdue ne doit pas masquer l'incident qui l'a
    déclenchée. L'échec d'envoi est journalisé en WARNING.
    """
    config = alerts or AlertsConfig()
    log = {"critical": logger.error, "warning": logger.warning}.get(alert.level, logger.info)
    log("ALERTE[%s] %s | %s", alert.level, alert.title, alert.status_line() or alert.summary)

    url = config.resolved_webhook_url()
    if not url:
        return False
    try:
        body = json.dumps(build_payload(alert, config), default=str).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            return 200 <= response.status < 300
    except (error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning("envoi de l'alerte impossible (%s): %s", type(exc).__name__, exc)
        return False
