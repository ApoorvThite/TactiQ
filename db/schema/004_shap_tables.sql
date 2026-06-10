-- TactiQ Phase 5 — SHAP values storage

CREATE TABLE IF NOT EXISTS matchup_shap_values (
    id                  SERIAL PRIMARY KEY,
    team_a_id           INTEGER,
    team_b_id           INTEGER,
    team_a_name         TEXT,
    team_b_name         TEXT,
    predicted_class     TEXT,
    p_win               FLOAT,
    p_draw              FLOAT,
    p_loss              FLOAT,
    shap_values_win     JSONB,
    shap_values_draw    JSONB,
    shap_values_loss    JSONB,
    top_feature_win     TEXT,
    top_feature_draw    TEXT,
    is_upset_candidate  BOOLEAN,
    upset_explanation   TEXT,
    computed_at         TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shap_teams ON matchup_shap_values(team_a_name, team_b_name);
