-- ============================================================
-- ChurnGuard AI — Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- Table 1: analyses — one row per upload session
CREATE TABLE IF NOT EXISTS analyses (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT        NOT NULL UNIQUE,
    best_model      TEXT,
    total_customers INTEGER,
    churn_yes       INTEGER,
    churn_no        INTEGER,
    high_risk_count INTEGER,
    churn_rate_pct  NUMERIC(5,2),
    potential_loss  NUMERIC(12,2),
    net_benefit     NUMERIC(12,2),
    roi             NUMERIC(8,2),
    seg_counts      JSONB,
    auc_scores      JSONB,
    model_metrics   JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Table 2: predictions — one row per customer per session
CREATE TABLE IF NOT EXISTS predictions (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    customer_id TEXT        NOT NULL,
    churn       TEXT        NOT NULL,
    probability NUMERIC(5,2),
    risk_level  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_analyses_session    ON analyses(session_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created    ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk    ON predictions(risk_level);

-- Enable Row Level Security (RLS) — open for service role key
ALTER TABLE analyses    ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow all operations via service role (used by backend)
DROP POLICY IF EXISTS "service_all_analyses"    ON analyses;
DROP POLICY IF EXISTS "service_all_predictions" ON predictions;
CREATE POLICY "service_all_analyses"    ON analyses    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all_predictions" ON predictions FOR ALL USING (true) WITH CHECK (true);
