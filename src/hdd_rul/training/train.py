"""Entraînement, évaluation et publication : l'étape « training » du pipeline.

Chaque candidat est entraîné dans son **propre run MLflow de premier niveau**,
relié aux autres par le tag `training.group_id`. Ce choix (plutôt que des runs
imbriques) est volontaire : les deux entraînements tournent en parallèle dans
des tâches Prefect concurrentes, et l'état « run actif » de MLflow est global au
processus — des runs imbriques depuis plusieurs threads se mélangeraient.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # aucun affichage : on ne produit que des fichiers PNG
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from hdd_rul.common.config import AppConfig
from hdd_rul.common.logging import get_logger
from hdd_rul.features.build import feature_names, select_feature_frame
from hdd_rul.features.labels import LABEL_COLUMN, compute_labels, resolve_training_window
from hdd_rul.ingestion.bronze import load_bronze_samples, read_bronze_manifest
from hdd_rul.training.evaluate import EvaluationMetrics, evaluate_predictions
from hdd_rul.training.models import (
    BASELINE_NAME,
    CANDIDATE_NAME,
    build_pipeline,
    model_params,
)
from hdd_rul.training.registry import (
    ALIAS_CHAMPION,
    PromotionDecision,
    apply_decision,
    configure_mlflow,
    decide_promotion,
    get_client,
    get_version_by_alias,
    sync_model_versions,
    version_metrics,
)
from hdd_rul.training.split import SplitResult, temporal_group_split

logger = get_logger(__name__)


@dataclass
class TrainingContext:
    """Ce qui, en dehors des données, doit se retrouver dans la traçabilité."""

    group_id: str
    trigger: str = "manual"
    flow_run_id: str | None = None
    validation_passed: bool = True
    validation_report_path: Path | None = None
    data_window: dict[str, str] = field(default_factory=dict)
    data_manifest: dict[str, Any] = field(default_factory=dict)
    label_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainedModel:
    name: str
    run_id: str
    model_uri: str
    metrics: EvaluationMetrics
    pipeline: Pipeline

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "metrics": self.metrics.as_dict(),
        }


@dataclass
class TrainingOutcome:
    group_id: str
    models: dict[str, dict[str, Any]]
    decision: str
    reason: str
    registered_version: str | None
    aliases: dict[str, str]
    data_window: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_commit() -> str:
    """Commit courant, pour relier un modèle au code qui l'a produit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _residual_plot(truth: pd.Series, prediction: pd.Series, title: str, path: Path) -> None:
    """Résidus en fonction de la RUL réelle : lecture rapide des biais du modèle."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(truth, prediction, s=6, alpha=0.3)
    limits = [0, float(max(truth.max(), prediction.max()))]
    axes[0].plot(limits, limits, "r--", linewidth=1)
    axes[0].set_xlabel("RUL réelle (jours)")
    axes[0].set_ylabel("RUL prédite (jours)")
    axes[0].set_title(f"{title} — prédit vs réel")

    axes[1].scatter(truth, prediction - truth, s=6, alpha=0.3)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("RUL réelle (jours)")
    axes[1].set_ylabel("résidu (prédit - réel)")
    axes[1].set_title(f"{title} — résidus")

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _run_tags(config: AppConfig, context: TrainingContext, model_name: str) -> dict[str, str]:
    """Stratégie de tags de run (justifiée dans TRACABILITE.md).

    `data.*` répond à « sur quelles données ? », `git.commit` à « avec quel
    code ? », `trigger` et `prefect.flow_run_id` à « lancé par quoi ? », et
    `training.group_id` relie les candidats d'un même entraînement.
    """
    manifest = context.data_manifest
    return {
        "data.window_start": context.data_window.get("window_start", ""),
        "data.window_end": context.data_window.get("window_end", ""),
        "data.as_of": context.data_window.get("as_of", ""),
        "data.sha256": str(manifest.get("source_fingerprint", "")),
        "data.source_files": str(manifest.get("source_files", "")),
        "data.horizon_days": str(config.horizon_days),
        "git.commit": git_commit(),
        "trigger": context.trigger,
        "prefect.flow_run_id": context.flow_run_id or "",
        "training.group_id": context.group_id,
        "training.role": "baseline" if model_name == BASELINE_NAME else "candidate",
        "validation.status": "passed" if context.validation_passed else "failed",
        "config.source": str(config.source_path or ""),
    }


def train_single_model(
    model_name: str,
    config: AppConfig,
    split: SplitResult,
    context: TrainingContext,
) -> TrainedModel:
    """Entraîne un candidat, l'évalue et journalise tout dans un run MLflow.

    Appelable telle quelle depuis une tâche Prefect : c'est l'unité parallélisée.
    """
    configure_mlflow(config)
    pipeline = build_pipeline(model_name, config)

    x_train = select_feature_frame(split.train, config)
    y_train = split.train[LABEL_COLUMN]
    x_valid = select_feature_frame(split.valid, config)
    y_valid = split.valid[LABEL_COLUMN]

    with mlflow.start_run(run_name=f"{model_name}-{context.group_id[:8]}") as run:
        mlflow.set_tags(_run_tags(config, context, model_name))
        mlflow.log_params({f"model.{k}": v for k, v in model_params(model_name, config).items()})
        mlflow.log_params(
            {
                "horizon_days": config.horizon_days,
                "alert_threshold_days": config.split.alert_threshold_days,
                "top_k_models": config.features.top_k_models,
                "healthy_rate_pct": config.sampling.healthy_rate_pct,
                "healthy_stride_days": config.sampling.healthy_stride_days,
                "smart_raw": ",".join(str(a) for a in config.features.smart_raw),
                "split_valid_fraction": config.split.valid_fraction,
                "split_cutoff_date": split.cutoff_date,
                "rows_train": len(split.train),
                "rows_valid": len(split.valid),
            }
        )

        logger.info("entraînement de %s sur %d lignes", model_name, len(split.train))
        pipeline.fit(x_train, y_train)
        predictions = pd.Series(pipeline.predict(x_valid), index=x_valid.index)
        metrics = evaluate_predictions(
            y_valid, predictions, alert_threshold_days=config.split.alert_threshold_days
        )
        mlflow.log_metrics({k: v for k, v in metrics.as_dict().items() if v == v})  # NaN exclu
        mlflow.set_tags(
            {
                "eval.mae": f"{metrics.mae:.4f}",
                "eval.mae_lt30": f"{metrics.mae_lt30:.4f}",
                "split.cutoff_date": split.cutoff_date,
                "split.rows_train": str(len(split.train)),
                "split.rows_valid": str(len(split.valid)),
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            (workdir / "features.json").write_text(
                json.dumps(
                    {
                        "input_columns": list(x_train.columns),
                        "feature_names": feature_names(config),
                        "label": LABEL_COLUMN,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (workdir / "data_manifest.json").write_text(
                json.dumps(
                    {
                        "bronze": context.data_manifest,
                        "window": context.data_window,
                        "labels": context.label_stats,
                        "split": split.as_tags(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            _residual_plot(y_valid, predictions, model_name, workdir / "residus_rul.png")
            mlflow.log_artifacts(str(workdir), artifact_path="rapports")

        if context.validation_report_path and Path(context.validation_report_path).exists():
            # Le rapport de qualité voyage avec le modèle : on sait sur quelles
            # données validées il a été entraîné.
            mlflow.log_artifact(str(context.validation_report_path), artifact_path="rapports")

        signature = infer_signature(x_valid, predictions.to_numpy())
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="pipeline",  # prétraitement + modèle : un seul artefact déployable
            signature=signature,
            input_example=x_valid.head(3),
            # cloudpickle et non le format skops par défaut : le pipeline
            # embarque `derive_features`, une fonction du paquet, que skops
            # refuse de recharger sans liste blanche explicite.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        logger.info(
            "%s: MAE=%.3f | MAE(<%dj)=%.3f | rappel alerte=%.3f",
            model_name,
            metrics.mae,
            config.split.alert_threshold_days,
            metrics.mae_lt30,
            metrics.alert_recall,
        )
        return TrainedModel(
            name=model_name,
            run_id=run.info.run_id,
            model_uri=info.model_uri,
            metrics=metrics,
            pipeline=pipeline,
        )


def prepare_training_data(
    config: AppConfig,
    *,
    as_of: date | None = None,
    window_days: int | None = None,
) -> tuple[SplitResult, TrainingContext]:
    """Bronze -> labels -> fenêtre résolue -> split, avec le contexte de traçabilité."""
    samples = load_bronze_samples(config)
    labelled, stats = compute_labels(samples, config)
    window, bounds = resolve_training_window(
        labelled, as_of=as_of, window_days=window_days, horizon_days=config.horizon_days
    )
    if window.empty:
        raise ValueError(
            "aucune ligne dont le label soit résolu à cette date : "
            "élargir la fenêtre ou avancer `as_of`"
        )
    split = temporal_group_split(window, config)
    context = TrainingContext(
        group_id=uuid.uuid4().hex,
        data_window=bounds,
        data_manifest=read_bronze_manifest(config),
        label_stats=stats.as_dict(),
    )
    return split, context


def champion_metrics_on(
    config: AppConfig,
    client: Any,
    champion: Any | None,
    valid: pd.DataFrame | None,
) -> tuple[dict[str, float], str]:
    """Métriques du champion, mesurées si possible sur la validation du challenger.

    Comparer le challenger aux métriques *stockées* du champion compare deux
    modèles sur deux jeux de validation différents (fenêtres et disques
    différents) : l'écart mesuré mélange alors qualité du modèle et difficulté
    du jeu. On recharge donc le champion et on le note sur les mêmes lignes.
    Repli sur les métriques stockées si le rechargement échoue, avec la source
    tracée pour que la décision reste explicable.
    """
    if champion is None:
        return {}, "none"
    if valid is not None and not valid.empty:
        uri = f"models:/{config.mlflow.registered_model_name}@{ALIAS_CHAMPION}"
        try:
            pipeline = mlflow.sklearn.load_model(uri)
            predictions = pipeline.predict(select_feature_frame(valid, config))
            metrics = evaluate_predictions(
                valid[LABEL_COLUMN],
                predictions,
                alert_threshold_days=config.split.alert_threshold_days,
            )
            logger.info(
                "champion v%s réévalué sur la validation du challenger: MAE=%.3f | MAE(<%dj)=%.3f",
                champion.version,
                metrics.mae,
                config.split.alert_threshold_days,
                metrics.mae_lt30,
            )
            return metrics.as_dict(), "same_validation"
        except Exception as exc:
            logger.warning(
                "champion v%s non réévaluable (%s: %s) : repli sur ses métriques stockées",
                champion.version,
                type(exc).__name__,
                exc,
            )
    return version_metrics(client, champion), "stored_metrics"


def publish_candidate(
    config: AppConfig,
    candidate: TrainedModel,
    context: TrainingContext,
    valid: pd.DataFrame | None = None,
) -> tuple[PromotionDecision, str | None, dict[str, str]]:
    """Enregistre le candidat, compare au champion et applique la règle.

    `valid` est le jeu de validation du candidat : le champion y est réévalué
    pour que les deux modèles soient comparés sur les mêmes lignes.
    """
    client = get_client(config)
    name = config.mlflow.registered_model_name
    champion = get_version_by_alias(client, name, ALIAS_CHAMPION)
    champion_metrics, champion_source = champion_metrics_on(config, client, champion, valid)

    decision = decide_promotion(
        candidate.metrics.as_dict(),
        champion_metrics or None,
        config,
        validation_passed=context.validation_passed,
    )

    version = mlflow.register_model(candidate.model_uri, name)
    client.set_model_version_tag(
        name,
        version.version,
        "validation.status",
        "passed" if context.validation_passed else "failed",
    )
    client.set_model_version_tag(
        name, version.version, "eval.mae_lt30", f"{candidate.metrics.mae_lt30:.4f}"
    )
    client.set_model_version_tag(name, version.version, "eval.mae", f"{candidate.metrics.mae:.4f}")
    client.set_model_version_tag(
        name, version.version, "data.window_end", context.data_window.get("window_end", "")
    )
    client.set_model_version_tag(name, version.version, "trigger", context.trigger)
    client.set_model_version_tag(name, version.version, "promotion.champion_eval", champion_source)
    if champion is not None and champion_metrics:
        client.set_model_version_tag(
            name, version.version, "promotion.champion_version", str(champion.version)
        )
        for key in ("mae", "mae_lt30"):
            if key in champion_metrics:
                client.set_model_version_tag(
                    name,
                    version.version,
                    f"promotion.champion_{key}",
                    f"{champion_metrics[key]:.4f}",
                )

    aliases = apply_decision(client, name, version.version, decision)
    return decision, str(version.version), aliases


def train_and_publish(
    config: AppConfig,
    *,
    trigger: str = "manual",
    as_of: date | None = None,
    window_days: int | None = None,
    validation_passed: bool = True,
    validation_report_path: Path | None = None,
    flow_run_id: str | None = None,
    models: tuple[str, ...] = (BASELINE_NAME, CANDIDATE_NAME),
) -> TrainingOutcome:
    """Chaîne complète d'entraînement (version séquentielle, utilisée par la CLI).

    Le flow Prefect `train_pipeline` reprend les mêmes briques mais lance les
    entraînements en parallèle.
    """
    split, context = prepare_training_data(config, as_of=as_of, window_days=window_days)
    context.trigger = trigger
    context.flow_run_id = flow_run_id
    context.validation_passed = validation_passed
    context.validation_report_path = validation_report_path

    trained = {name: train_single_model(name, config, split, context) for name in models}
    return finalize_training(config, trained, context, valid=split.valid)


def finalize_training(
    config: AppConfig,
    trained: dict[str, TrainedModel],
    context: TrainingContext,
    valid: pd.DataFrame | None = None,
) -> TrainingOutcome:
    """Compare les candidats, publie et retourne le résultat exploitable.

    `valid` (jeu de validation des candidats) permet de réévaluer le champion
    sur les mêmes lignes avant d'appliquer la règle de mise en service.
    """
    candidate = trained.get(CANDIDATE_NAME)
    if candidate is None:
        raise ValueError(f"aucun candidat {CANDIDATE_NAME} entraîné : rien à publier")

    baseline = trained.get(BASELINE_NAME)
    if baseline is not None:
        configure_mlflow(config)
        with mlflow.start_run(run_id=candidate.run_id):
            # Comparaison à la baseline journalisée sur le run du candidat :
            # c'est la seule façon de dire si le modèle apporte quelque chose.
            mlflow.log_metrics(
                {
                    "baseline_mae": baseline.metrics.mae,
                    "baseline_mae_lt30": baseline.metrics.mae_lt30,
                    "gain_mae_vs_baseline": baseline.metrics.mae - candidate.metrics.mae,
                }
            )

    decision, version, aliases = publish_candidate(config, candidate, context, valid=valid)
    sync_model_versions(config)  # vue croisée Grafana : état des alias, CLI comme flow
    outcome = TrainingOutcome(
        group_id=context.group_id,
        models={name: model.as_dict() for name, model in trained.items()},
        decision=decision.decision,
        reason=decision.reason,
        registered_version=version,
        aliases=aliases,
        data_window=context.data_window,
    )
    save_outcome(config, outcome)
    logger.info(
        "entraînement %s termine: version %s %s (%s)",
        context.group_id[:8],
        version,
        decision.decision,
        decision.reason,
    )
    return outcome


def training_summary_path(config: AppConfig) -> Path:
    return config.paths.artifacts / "training" / "derniere_execution.json"


def save_outcome(config: AppConfig, outcome: TrainingOutcome) -> Path:
    """Garde une trace locale du dernier entraînement (pratique pour la démo)."""
    path = training_summary_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = outcome.as_dict() | {"saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
