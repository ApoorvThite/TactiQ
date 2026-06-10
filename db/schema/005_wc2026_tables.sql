-- Phase 6: WC2026 group stage predictions and qualification probabilities

CREATE TABLE IF NOT EXISTS wc2026_group_predictions (
    id                    SERIAL PRIMARY KEY,
    group_label           CHAR(1)          NOT NULL,
    team_a_name           VARCHAR(100)     NOT NULL,
    team_b_name           VARCHAR(100)     NOT NULL,
    team_a_archetype      VARCHAR(50),
    team_b_archetype      VARCHAR(50),
    team_a_is_proxy       BOOLEAN          DEFAULT FALSE,
    team_b_is_proxy       BOOLEAN          DEFAULT FALSE,
    predicted_class       VARCHAR(10)      NOT NULL,  -- 'win' | 'draw' | 'loss'
    p_win                 FLOAT            NOT NULL,
    p_draw                FLOAT            NOT NULL,
    p_loss                FLOAT            NOT NULL,
    top_shap_feature_win  VARCHAR(60),
    top_shap_feature_draw VARCHAR(60),
    shap_values_win       JSONB,
    shap_values_draw      JSONB,
    shap_values_loss      JSONB,
    is_upset_candidate    BOOLEAN          DEFAULT FALSE,
    upset_explanation     TEXT,
    created_at            TIMESTAMPTZ      DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wc2026_qualification_probs (
    id                SERIAL PRIMARY KEY,
    team_name         VARCHAR(100)   NOT NULL,
    group_label       CHAR(1)        NOT NULL,
    archetype_name    VARCHAR(50),
    is_proxy          BOOLEAN        DEFAULT FALSE,
    p_first           FLOAT          NOT NULL DEFAULT 0,  -- finish 1st in group
    p_second          FLOAT          NOT NULL DEFAULT 0,  -- finish 2nd
    p_third           FLOAT          NOT NULL DEFAULT 0,  -- finish 3rd
    p_fourth          FLOAT          NOT NULL DEFAULT 0,  -- finish 4th / eliminated
    p_qualify_direct  FLOAT          NOT NULL DEFAULT 0,  -- top-2 auto-qualify
    p_best_third      FLOAT          NOT NULL DEFAULT 0,  -- chance of being best-8 3rd
    p_qualify_r32     FLOAT          NOT NULL DEFAULT 0,  -- overall chance to reach R32
    avg_sim_points    FLOAT,
    avg_sim_gd        FLOAT,
    sim_runs          INT            DEFAULT 10000,
    created_at        TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE(team_name, group_label)
);
