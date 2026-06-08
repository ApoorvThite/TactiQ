-- TactiQ Phase 2 — Feature tables
-- One row per team per match (granular) and aggregated style profiles

CREATE TABLE IF NOT EXISTS match_team_features (
    id                          SERIAL PRIMARY KEY,
    match_id                    INTEGER REFERENCES matches(match_id),
    team_id                     INTEGER REFERENCES teams(team_id),
    is_home                     BOOLEAN,
    match_date                  DATE,
    competition_name            TEXT,
    season_name                 TEXT,
    opponent_team_id            INTEGER,
    result                      TEXT,
    goals_scored                INTEGER,
    goals_conceded              INTEGER,

    -- Possession & tempo
    possession_pct              FLOAT,
    total_passes                INTEGER,
    pass_completion_pct         FLOAT,
    progressive_passes          INTEGER,
    passes_into_final_third     INTEGER,

    -- Pressing & defense
    total_pressures             INTEGER,
    pressure_success_rate       FLOAT,
    ppda                        FLOAT,
    defensive_actions           INTEGER,

    -- Attack & shooting
    total_shots                 INTEGER,
    shots_on_target             INTEGER,
    xg_created                  FLOAT,
    xg_conceded                 FLOAT,
    xg_ratio                    FLOAT,
    np_xg                       FLOAT,

    -- Carries & directness
    total_carries               INTEGER,
    progressive_carries         INTEGER,
    progressive_carry_pct       FLOAT,
    carries_into_final_third    INTEGER,

    -- Under pressure
    passes_under_pressure       INTEGER,
    pass_completion_under_pressure_pct FLOAT,

    -- Set pieces
    set_piece_shots             INTEGER,
    set_piece_shot_pct          FLOAT,

    -- Rolling form (populated by compute_rolling_form.sql)
    rolling_xg_created_5        FLOAT,
    rolling_xg_conceded_5       FLOAT,
    rolling_ppda_5              FLOAT,
    rolling_possession_5        FLOAT,
    form_points_5               INTEGER,

    UNIQUE (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_mtf_team_id  ON match_team_features(team_id);
CREATE INDEX IF NOT EXISTS idx_mtf_match_id ON match_team_features(match_id);
CREATE INDEX IF NOT EXISTS idx_mtf_date     ON match_team_features(match_date);

CREATE TABLE IF NOT EXISTS team_style_profiles (
    team_id                                     INTEGER PRIMARY KEY REFERENCES teams(team_id),
    team_name                                   TEXT,
    matches_played                              INTEGER,
    avg_possession_pct                          FLOAT,
    avg_pass_completion_pct                     FLOAT,
    avg_progressive_passes_p90                  FLOAT,
    avg_passes_final_third_p90                  FLOAT,
    avg_ppda                                    FLOAT,
    avg_pressure_success_rate                   FLOAT,
    avg_total_pressures_p90                     FLOAT,
    avg_xg_created_p90                          FLOAT,
    avg_xg_conceded_p90                         FLOAT,
    avg_xg_ratio                                FLOAT,
    avg_progressive_carry_pct                   FLOAT,
    avg_carries_final_third_p90                 FLOAT,
    avg_pass_completion_under_pressure_pct      FLOAT,
    avg_set_piece_shot_pct                      FLOAT,
    win_rate                                    FLOAT,
    updated_at                                  TIMESTAMP DEFAULT NOW()
);
