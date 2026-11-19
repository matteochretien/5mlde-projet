"""Point d'entrée en ligne de commande : `python -m hdd_rul.cli <commande>`.

Les tâches `mise run ...` appellent ces commandes. Le projet ne dépend d'aucun
notebook : toute étape est reproductible depuis le terminal, avec la même
configuration que les flows Prefect.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from hdd_rul.common.config import AppConfig, load_config
from hdd_rul.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def cmd_synthetic(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.ingestion.synthetic import generate_synthetic_drive_stats

    files = generate_synthetic_drive_stats(config, force=args.force)
    print(f"{len(files)} fichiers journaliers dans {config.paths.raw}")
    return 0


def cmd_download(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.ingestion.download import ingest_quarters

    summary = ingest_quarters(config, args.quarters or None)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.ingestion.bronze import build_bronze

    manifest = build_bronze(config, source_dir=Path(args.source) if args.source else None)
    print(manifest.to_json())
    return 0


def cmd_validate(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.ingestion.landing import batch_date_from_name, list_pending_batches
    from hdd_rul.validation.contract import validate_batch

    targets = [Path(args.batch)] if args.batch else list_pending_batches(config)
    if not targets:
        print("aucun lot à valider (data/landing/ est vide)")
        return 0
    failures = 0
    for target in targets:
        frame = pd.read_csv(target, low_memory=False)
        report = validate_batch(
            config,
            frame,
            batch_name=target.name,
            expected_date=batch_date_from_name(target),
        )
        print(f"{target.name}: {'OK' if report.passed else 'REFUSE'} — {report.summary()}")
        print(f"  rapport: {report.report_path}")
        failures += 0 if report.passed else 1
    return 1 if failures else 0


def cmd_train(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.training.train import train_and_publish, training_summary_path

    outcome = train_and_publish(
        config,
        trigger=args.trigger,
        as_of=_parse_date(args.as_of),
        window_days=args.window_days,
    )
    print(json.dumps(outcome.as_dict(), indent=2, default=str))
    print(f"résumé: {training_summary_path(config)}")
    return 0


def cmd_rollback(_args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.training.registry import get_client, rollback_champion, sync_model_versions

    result = rollback_champion(get_client(config), config.mlflow.registered_model_name)
    sync_model_versions(config)  # la vue croisée Grafana suit le retour arrière
    print(json.dumps(result, indent=2))
    return 0


def cmd_model(_args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.training.registry import describe_champion

    print(json.dumps(describe_champion(config), indent=2, default=str))
    return 0


def cmd_replay(args: argparse.Namespace, config: AppConfig) -> int:
    from scripts.replay import replay_days

    deposited = replay_days(
        config,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        limit=args.limit,
        delay_seconds=args.delay,
        run_flow=not args.deposit_only,
    )
    print(f"{len(deposited)} lot(s) rejoué(s)")
    return 0


def cmd_batch(args: argparse.Namespace, _config: AppConfig) -> int:
    from hdd_rul.orchestration.flows import daily_batch

    state = daily_batch(config_path=args.config, batch_path=args.batch)
    print(json.dumps(state, indent=2, default=str))
    return 0


def cmd_alert_test(args: argparse.Namespace, config: AppConfig) -> int:
    from hdd_rul.common.notifications import build_payload, notify, render_text
    from hdd_rul.orchestration.alerts import sample_alert

    alert = sample_alert(config.alerts)
    if args.print:
        print(json.dumps(build_payload(alert, config.alerts), indent=2, ensure_ascii=False))
        return 0
    print(render_text(alert, config.alerts))
    if not config.alerts.resolved_webhook_url():
        print("\nHDD_RUL_WEBHOOK_URL absent : alerte journalisée seulement, rien n'est envoyé.")
        return 1
    sent = notify(alert, alerts=config.alerts)
    print("\nwebhook : " + ("accepté" if sent else "échec de l'envoi (voir le journal)"))
    return 0 if sent else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdd-rul",
        description="Chaîne MLOps de prédiction de la RUL des disques durs",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="fichier d'overlay de configuration (défaut: config/base.yaml)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING...")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="génère un jeu Drive Stats synthétique")
    synthetic.add_argument("--force", action="store_true", help="réécrit les fichiers existants")
    synthetic.set_defaults(handler=cmd_synthetic)

    download = subparsers.add_parser("download", help="téléchargé et décompresse les trimestres")
    download.add_argument("--quarters", nargs="*", help="ex: Q1_2026 Q2_2026")
    download.set_defaults(handler=cmd_download)

    ingest = subparsers.add_parser("ingest", help="construit la couche bronze (DuckDB)")
    ingest.add_argument("--source", default=None, help="répertoire de CSV (défaut: data/raw)")
    ingest.set_defaults(handler=cmd_ingest)

    validate = subparsers.add_parser("validate", help="applique le contrat de qualité")
    validate.add_argument("--batch", default=None, help="CSV à valider (défaut: data/landing/*)")
    validate.set_defaults(handler=cmd_validate)

    train = subparsers.add_parser("train", help="entraîne, évalue et publie")
    train.add_argument("--trigger", default="manual", help="manual, schedule ou replay")
    train.add_argument("--as-of", default=None, help="date de simulation (YYYY-MM-DD)")
    train.add_argument("--window-days", type=int, default=None, help="fenêtre glissante en jours")
    train.set_defaults(handler=cmd_train)

    rollback = subparsers.add_parser("rollback", help="retour arrière champion <- previous")
    rollback.set_defaults(handler=cmd_rollback)

    model = subparsers.add_parser("model", help="décrit la version servie")
    model.set_defaults(handler=cmd_model)

    replay = subparsers.add_parser("replay", help="rejoue le trimestre de simulation")
    replay.add_argument("--start", default=None)
    replay.add_argument("--end", default=None)
    replay.add_argument("--limit", type=int, default=5, help="nombre de jours à rejouer")
    replay.add_argument("--delay", type=float, default=0.0, help="pause entre deux jours")
    replay.add_argument(
        "--deposit-only",
        action="store_true",
        help="déposé dans landing/ sans déclencher le flow",
    )
    replay.set_defaults(handler=cmd_replay)

    batch = subparsers.add_parser("batch", help="exécute le flow daily_batch")
    batch.add_argument(
        "--batch", default=None, help="CSV à traiter (défaut: plus ancien en attente)"
    )
    batch.set_defaults(handler=cmd_batch)

    alert_test = subparsers.add_parser(
        "alert-test", help="envoie une alerte de test au webhook Teams / Slack"
    )
    alert_test.add_argument(
        "--print", action="store_true", help="affiche la charge utile JSON sans l'envoyer"
    )
    alert_test.set_defaults(handler=cmd_alert_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    config = load_config(args.config)
    try:
        return int(args.handler(args, config))
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        if args.log_level and args.log_level.upper() == "DEBUG":
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
