-- Phase 1: schema is defined but neither table is read or written by any service
-- in this phase. `predictions` becomes active in Phase 4 (shadow mode).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS complaints (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT now(),
    text            TEXT        NOT NULL,
    predicted_team  TEXT        NULL,
    tracking_id     TEXT        UNIQUE      NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT now(),
    request_id      TEXT        NOT NULL,
    complaint_id    UUID        NULL REFERENCES complaints(id) ON DELETE SET NULL,
    role            TEXT        NOT NULL    CHECK (role IN ('primary', 'shadow')),
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    team            TEXT        NOT NULL,
    team_confidence DOUBLE PRECISION NOT NULL,
    top_issues      JSONB       NOT NULL,
    auto_route      BOOLEAN     NOT NULL,
    latency_ms      INTEGER     NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_predictions_request_id ON predictions(request_id);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions(created_at);
