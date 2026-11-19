-- ---------------------------------------------------------------------------
-- Schéma de traces : prédictions servies, feedbacks, requêtes API, exécutions
-- de flows et quarantaine. Source unique des tableaux de bord Grafana.
-- Appliqué automatiquement au premier démarrage du conteneur PostgreSQL.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS tracing;

-- Versions de modèles vues en production, avec leurs métriques d'évaluation.
CREATE TABLE IF NOT EXISTS tracing.model_versions (
    id            BIGSERIAL PRIMARY KEY,
    model_name    TEXT        NOT NULL,
    version       TEXT        NOT NULL,
    alias         TEXT        NOT NULL,
    run_id        TEXT,
    metrics       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_name, version, alias)
);

-- Une ligne par appel HTTP : latence, code, version servie. Vue technique.
CREATE TABLE IF NOT EXISTS tracing.api_requests (
    id            BIGSERIAL PRIMARY KEY,
    endpoint      TEXT        NOT NULL,
    method        TEXT        NOT NULL,
    status_code   INTEGER     NOT NULL,
    duration_ms   DOUBLE PRECISION NOT NULL,
    model_version TEXT,
    n_items       INTEGER     NOT NULL DEFAULT 1,
    error         TEXT,
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS api_requests_requested_at_idx ON tracing.api_requests (requested_at);
CREATE INDEX IF NOT EXISTS api_requests_endpoint_idx ON tracing.api_requests (endpoint);

-- Une ligne par prédiction servie. `prediction_id` est fourni par l'API, ce qui
-- rend le rejeu d'un lot idempotent.
CREATE TABLE IF NOT EXISTS tracing.predictions (
    prediction_id    UUID PRIMARY KEY,
    observation_date DATE,
    serial_number    TEXT        NOT NULL,
    model            TEXT,
    capacity_bytes   BIGINT,
    features         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    predicted_rul    DOUBLE PRECISION NOT NULL,
    alert            BOOLEAN     NOT NULL DEFAULT false,
    model_name       TEXT,
    model_version    TEXT,
    run_id           TEXT,
    source           TEXT        NOT NULL DEFAULT 'api',
    predicted_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS predictions_predicted_at_idx ON tracing.predictions (predicted_at);
CREATE INDEX IF NOT EXISTS predictions_serial_date_idx
    ON tracing.predictions (serial_number, observation_date);
CREATE INDEX IF NOT EXISTS predictions_version_idx ON tracing.predictions (model_version);

-- RUL réellement constatée. Pendant le rejeu, les feedbacks arrivent
-- automatiquement : quand un disque passe à failure = 1, on renvoie sa vraie
-- RUL pour toutes ses prédictions passées.
CREATE TABLE IF NOT EXISTS tracing.feedbacks (
    id               BIGSERIAL PRIMARY KEY,
    serial_number    TEXT        NOT NULL,
    observation_date DATE        NOT NULL,
    actual_rul       DOUBLE PRECISION NOT NULL,
    failure_date     DATE,
    source           TEXT        NOT NULL DEFAULT 'api',
    received_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (serial_number, observation_date)
);
CREATE INDEX IF NOT EXISTS feedbacks_received_at_idx ON tracing.feedbacks (received_at);

-- Statut des exécutions de flows Prefect (ingestion, lot journalier,
-- entraînement) : permet de suivre les échecs depuis Grafana.
CREATE TABLE IF NOT EXISTS tracing.pipeline_runs (
    id           BIGSERIAL PRIMARY KEY,
    flow_name    TEXT        NOT NULL,
    flow_run_id  TEXT,
    status       TEXT        NOT NULL,
    trigger      TEXT        NOT NULL DEFAULT 'manual',
    batch_name   TEXT,
    details      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipeline_runs_recorded_at_idx ON tracing.pipeline_runs (recorded_at);

-- Lignes écartées par une règle non bloquante du contrat de données.
CREATE TABLE IF NOT EXISTS tracing.quarantine (
    id           BIGSERIAL PRIMARY KEY,
    batch_name   TEXT        NOT NULL,
    batch_date   DATE,
    rows         INTEGER     NOT NULL,
    reason       TEXT        NOT NULL,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Vues de supervision : la logique métier reste en SQL, les panneaux Grafana
-- se contentent de les interroger.
-- ---------------------------------------------------------------------------

-- Vue croisée : erreur réelle mesurée en production, par version de modèle.
CREATE OR REPLACE VIEW tracing.v_production_error AS
SELECT
    p.model_name,
    p.model_version,
    date_trunc('day', p.predicted_at)                    AS day,
    count(*)                                             AS n_feedbacks,
    avg(abs(p.predicted_rul - f.actual_rul))              AS mae,
    avg(abs(p.predicted_rul - f.actual_rul))
        FILTER (WHERE f.actual_rul < 30)                  AS mae_lt30
FROM tracing.predictions p
JOIN tracing.feedbacks f
  ON f.serial_number = p.serial_number
 AND f.observation_date = p.observation_date
GROUP BY 1, 2, 3;

-- Vue métier : disques à risque (RUL prédite < 30 j) par modèle de disque.
CREATE OR REPLACE VIEW tracing.v_drives_at_risk AS
SELECT
    date_trunc('day', predicted_at) AS day,
    model,
    count(*) FILTER (WHERE alert)   AS drives_at_risk,
    count(*)                        AS drives_scored,
    min(predicted_rul)              AS min_predicted_rul
FROM tracing.predictions
GROUP BY 1, 2;

-- Vue technique : trafic, latences et erreurs de l'API.
CREATE OR REPLACE VIEW tracing.v_api_health AS
SELECT
    date_trunc('minute', requested_at)                                   AS minute,
    endpoint,
    count(*)                                                             AS requests,
    avg(duration_ms)                                                     AS avg_ms,
    percentile_disc(0.95) WITHIN GROUP (ORDER BY duration_ms)            AS p95_ms,
    count(*) FILTER (WHERE status_code >= 500)                           AS errors_5xx,
    count(*) FILTER (WHERE status_code >= 400 AND status_code < 500)     AS errors_4xx
FROM tracing.api_requests
GROUP BY 1, 2;
