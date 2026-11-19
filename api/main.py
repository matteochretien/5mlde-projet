"""API de prédiction FastAPI (port 8001).

Endpoints :
  * `POST /predict`        — prédiction en ligne, un disque ;
  * `POST /predict/batch`  — prédiction par lot (jusqu'à 5 000 disques) ;
  * `POST /feedback`       — RUL réellement constatée ;
  * `GET  /health`         — état du service et de ses dépendances ;
  * `GET  /model`          — identité et métriques de la version servie ;
  * `POST /admin/reload`   — recharge le champion après promotion ou rollback.

Principes :
  * le modèle est chargé **une fois par worker**, puis gardé en mémoire ;
  * chaque réponse porte la **version du modèle** qui l'a produite ;
  * toute requête est tracée dans PostgreSQL (endpoint, latence, code, version) ;
  * une dépendance indisponible donne un 503 explicite, jamais un 500 silencieux.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriveObservation,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelIdentity,
    ModelResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from hdd_rul import __version__
from hdd_rul.common.config import load_config
from hdd_rul.common.logging import get_logger, setup_logging
from hdd_rul.common.tracing import get_store
from hdd_rul.features.build import feature_names
from hdd_rul.serving.predictor import (
    ALERT_COLUMN,
    PREDICTION_COLUMN,
    LoadedModel,
    ModelUnavailableError,
    PredictorCache,
    predict_frame,
)
from hdd_rul.training.registry import registry_healthcheck

logger = get_logger(__name__)

KNOWN_MODELS_CACHE_KEY = "known_models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge la configuration et tente un premier chargement du modèle.

    L'échec de chargement n'empêche pas le démarrage : l'API doit pouvoir
    répondre `/health` en mode dégradé quand MLflow n'est pas joignable.
    """
    setup_logging()
    config = load_config()
    app.state.config = config
    app.state.predictor = PredictorCache(config)
    app.state.store = get_store()
    try:
        model = app.state.predictor.get()
        logger.info("modèle chargé au démarrage: version %s", model.version)
    except ModelUnavailableError as exc:
        logger.warning("démarrage sans modèle: %s", exc)
    yield
    logger.info("arrêt de l'API")


app = FastAPI(
    title="API RUL disques durs",
    description=(
        "Prédiction de la durée de vie restante (RUL, en jours) d'un disque dur "
        "à partir de ses statistiques SMART du jour."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _config(request: Request):
    return request.app.state.config


def _predictor(request: Request) -> PredictorCache:
    return request.app.state.predictor


def _identity(model: LoadedModel) -> ModelIdentity:
    return ModelIdentity(**model.identity())


def _require_model(request: Request) -> LoadedModel:
    """Récupère le modèle servi, ou répond 503 (et non 500) s'il est absent."""
    try:
        return _predictor(request).get()
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "aucun modèle 'champion' disponible : lancer un entraînement "
                f"ou vérifier MLflow. Détail: {exc}"
            ),
        ) from exc


def _known_models(request: Request, model: LoadedModel) -> set[str]:
    """Modèles de disque vus à l'entraînement, pour avertir sur les inconnus.

    Lu une fois depuis l'encodeur du pipeline, puis mis en cache : on ne veut
    pas inspecter le modèle à chaque requête.
    """
    cache: dict[str, Any] = request.app.state.__dict__.setdefault("_known", {})
    key = f"{model.model_name}:{model.version}"
    if key in cache:
        return cache[key]
    known: set[str] = set()
    try:
        encoder = model.pipeline.named_steps["encodage"].named_transformers_["categorielles"]
        for categories in encoder.categories_:
            known.update(str(value) for value in categories)
    except Exception:
        known = set()
    cache[key] = known
    return known


def _observations_to_frame(observations: list[DriveObservation]) -> pd.DataFrame:
    return pd.DataFrame([observation.model_dump() for observation in observations])


def _build_predictions(
    scored: pd.DataFrame,
    config,
    known_models: set[str],
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for row in scored.to_dict(orient="records"):
        warnings: list[str] = []
        if known_models and str(row.get("model")) not in known_models:
            warnings.append(
                f"modèle de disque inconnu à l'entraînement ({row.get('model')!r}) : "
                "traité dans la modalité 'autre', prédiction moins fiable"
            )
        capacity = row.get("capacity_bytes")
        if capacity is None or (isinstance(capacity, int | float) and capacity <= 0):
            warnings.append("capacité inconnue : feature traitée comme manquante")
        predictions.append(
            Prediction(
                prediction_id=str(uuid.uuid4()),
                serial_number=str(row["serial_number"]),
                predicted_rul_days=round(float(row[PREDICTION_COLUMN]), 3),
                alert=bool(row[ALERT_COLUMN]),
                horizon_days=config.horizon_days,
                alert_threshold_days=config.split.alert_threshold_days,
                warnings=warnings,
            )
        )
    return predictions


def _persist_predictions(
    request: Request,
    model: LoadedModel,
    scored: pd.DataFrame,
    predictions: list[Prediction],
    source: str,
) -> int:
    config = _config(request)
    rows = []
    for prediction, row in zip(predictions, scored.to_dict(orient="records"), strict=True):
        rows.append(
            {
                "prediction_id": prediction.prediction_id,
                "observation_date": row.get("date"),
                "serial_number": prediction.serial_number,
                "model": row.get("model"),
                "capacity_bytes": row.get("capacity_bytes"),
                "features": {
                    column: row.get(column)
                    for column in config.features.smart_columns
                    if column in scored.columns
                },
                "predicted_rul": prediction.predicted_rul_days,
                "alert": prediction.alert,
                "model_name": model.model_name,
                "model_version": model.version,
                "run_id": model.run_id,
                "source": source,
            }
        )
    return request.app.state.store.record_predictions(rows)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Trace chaque appel : latence, code, version servie. Vue technique Grafana."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _safe_record_request(request, 500, duration_ms, error=f"{type(exc).__name__}: {exc}")
        logger.exception("erreur non gérée sur %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "erreur interne", "path": request.url.path},
        )
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    if request.url.path not in {"/health", "/docs", "/openapi.json", "/redoc"}:
        _safe_record_request(request, response.status_code, duration_ms)
    return response


def _safe_record_request(
    request: Request, status_code: int, duration_ms: float, error: str | None = None
) -> None:
    store = getattr(request.app.state, "store", None)
    if store is None:
        return
    model = None
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is not None:
        loaded = predictor.try_get()
        model = loaded.version if loaded else None
    store.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=status_code,
        duration_ms=duration_ms,
        model_version=model,
        n_items=int(request.headers.get("X-Batch-Size", "1") or 1),
        error=error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Prédiction en ligne pour un disque."""
    started = time.perf_counter()
    config = _config(request)
    model = _require_model(request)

    frame = _observations_to_frame([payload.observation])
    scored = predict_frame(model, frame, config)
    predictions = _build_predictions(scored, config, _known_models(request, model))
    if payload.persist:
        _persist_predictions(request, model, scored, predictions, source="api")

    return PredictResponse(
        prediction=predictions[0],
        model=_identity(model),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Prédiction par lot : une liste de disques, typiquement une journée de parc."""
    started = time.perf_counter()
    config = _config(request)
    model = _require_model(request)

    frame = _observations_to_frame(payload.observations)
    scored = predict_frame(model, frame, config)
    predictions = _build_predictions(scored, config, _known_models(request, model))
    persisted = (
        _persist_predictions(request, model, scored, predictions, source="api_batch")
        if payload.persist
        else 0
    )

    return BatchPredictResponse(
        predictions=predictions,
        model=_identity(model),
        n_items=len(predictions),
        n_alerts=sum(1 for prediction in predictions if prediction.alert),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        persisted=persisted,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Enregistre la RUL réellement constatée (panne ou survie à l'horizon).

    Trois façons de la fournir, par ordre de priorité : `actual_rul_days`
    explicite, `failure_date` (la RUL est alors calculée et plafonnée), ou
    `still_alive_after_horizon` (la RUL vaut exactement H).
    """
    config = _config(request)
    horizon = config.horizon_days
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in payload.feedbacks:
        if item.actual_rul_days is not None:
            actual = min(float(item.actual_rul_days), float(horizon))
        elif item.failure_date is not None:
            delta = (item.failure_date - item.observation_date).days
            if delta < 0:
                rejected.append(
                    {
                        "serial_number": item.serial_number,
                        "raison": "failure_date antérieure à observation_date",
                    }
                )
                continue
            actual = float(min(delta, horizon))
        elif item.still_alive_after_horizon:
            actual = float(horizon)
        else:
            rejected.append(
                {
                    "serial_number": item.serial_number,
                    "raison": (
                        "fournir actual_rul_days, failure_date ou still_alive_after_horizon"
                    ),
                }
            )
            continue
        rows.append(
            {
                "serial_number": item.serial_number,
                "observation_date": item.observation_date,
                "actual_rul": actual,
                "failure_date": item.failure_date,
                "source": "api",
            }
        )

    persisted = request.app.state.store.record_feedbacks(rows)
    return FeedbackResponse(received=len(payload.feedbacks), persisted=persisted, rejected=rejected)


@app.get("/health", response_model=HealthResponse, tags=["exploitation"])
def health(request: Request) -> JSONResponse:
    """État du service.

    `503` si aucun modèle n'est servi. `200 degraded` si la base de traces ou le
    registre MLflow est injoignable : les prédictions continuent avec le modèle
    en mémoire, mais les traces ou un rechargement ne sont plus possibles.
    """
    predictor = _predictor(request)
    model = predictor.try_get()
    database = request.app.state.store.healthcheck()
    registry = registry_healthcheck(_config(request))

    if model is None:
        state = "unavailable"
    elif (database["enabled"] and not database["reachable"]) or (
        registry["enabled"] and not registry["reachable"]
    ):
        state = "degraded"
    else:
        state = "ok"

    payload = HealthResponse(
        status=state,
        api_version=__version__,
        model_loaded=model is not None,
        model=_identity(model) if model else None,
        model_error=predictor.error,
        database=database,
        registry=registry,
        checked_at=datetime.now(UTC),
    )
    code = status.HTTP_503_SERVICE_UNAVAILABLE if state == "unavailable" else status.HTTP_200_OK
    return JSONResponse(status_code=code, content=payload.model_dump(mode="json"))


@app.get("/model", response_model=ModelResponse, tags=["exploitation"])
def describe_model(request: Request) -> ModelResponse:
    """Version servie, ses métriques et la liste des features attendues."""
    config = _config(request)
    description = _predictor(request).describe()
    return ModelResponse(
        horizon_days=config.horizon_days,
        alert_threshold_days=config.split.alert_threshold_days,
        features=feature_names(config),
        **description,
    )


@app.post("/admin/reload", response_model=ModelResponse, tags=["exploitation"])
def reload_model(request: Request, alias: str | None = None) -> ModelResponse:
    """Recharge le modèle depuis le registre : à appeler après promotion/rollback.

    Évite de redémarrer l'API pour prendre en compte un changement d'alias ;
    c'est ce qui rend le retour arrière observable en direct dans `/model`.
    """
    predictor = _predictor(request)
    try:
        predictor.reload(alias=alias)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"rechargement impossible: {exc}",
        ) from exc
    return describe_model(request)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "API RUL disques durs",
        "version": __version__,
        "documentation": "/docs",
        "sante": "/health",
    }
