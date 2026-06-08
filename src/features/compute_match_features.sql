-- ============================================================
-- TactiQ: Match-Team Feature Engineering Pipeline
-- Computes all tactical metrics per team per match
-- All extra_data paths use flat StatsBomb structure:
--   e.g. extra_data->>'shot_statsbomb_xg', NOT extra_data->'shot'->>'statsbomb_xg'
-- ============================================================

INSERT INTO match_team_features (
    match_id, team_id, is_home, match_date, competition_name, season_name,
    opponent_team_id, result, goals_scored, goals_conceded,
    possession_pct, total_passes, pass_completion_pct,
    progressive_passes, passes_into_final_third,
    total_pressures, pressure_success_rate, ppda, defensive_actions,
    total_shots, shots_on_target, xg_created, xg_conceded, xg_ratio, np_xg,
    total_carries, progressive_carries, progressive_carry_pct,
    carries_into_final_third,
    passes_under_pressure, pass_completion_under_pressure_pct,
    set_piece_shots, set_piece_shot_pct
)
WITH

-- Base: join matches to competitions
match_base AS (
    SELECT
        m.match_id,
        m.match_date,
        c.competition_name,
        c.season_name,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score
    FROM matches m
    JOIN competitions c USING (competition_id, season_id)
),

-- Expand to one row per team per match
team_match_base AS (
    SELECT
        match_id, match_date, competition_name, season_name,
        home_team_id  AS team_id,
        away_team_id  AS opponent_team_id,
        TRUE          AS is_home,
        home_score    AS goals_scored,
        away_score    AS goals_conceded,
        CASE
            WHEN home_score > away_score THEN 'win'
            WHEN home_score = away_score THEN 'draw'
            ELSE 'loss'
        END AS result
    FROM match_base
    UNION ALL
    SELECT
        match_id, match_date, competition_name, season_name,
        away_team_id  AS team_id,
        home_team_id  AS opponent_team_id,
        FALSE         AS is_home,
        away_score    AS goals_scored,
        home_score    AS goals_conceded,
        CASE
            WHEN away_score > home_score THEN 'win'
            WHEN away_score = home_score THEN 'draw'
            ELSE 'loss'
        END AS result
    FROM match_base
),

-- ---- POSSESSION ----
possession_stats AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) FILTER (WHERE event_type IN (
            'Pass','Carry','Ball Receipt*','Dribble','Shot','Duel'
        )) AS team_possession_events
    FROM match_events
    GROUP BY match_id, team_id
),
possession_totals AS (
    SELECT match_id, SUM(team_possession_events) AS total_possession_events
    FROM possession_stats
    GROUP BY match_id
),
possession_pct_cte AS (
    SELECT
        p.match_id,
        p.team_id,
        ROUND(
            (100.0 * p.team_possession_events / NULLIF(t.total_possession_events, 0))::NUMERIC,
            2
        ) AS possession_pct
    FROM possession_stats p
    JOIN possession_totals t USING (match_id)
),

-- ---- PASSES ----
-- pass_outcome IS NULL = completed pass (StatsBomb convention)
-- pass_end_location is a flat top-level JSONB array [x, y]
pass_stats AS (
    SELECT
        match_id,
        team_id,
        COUNT(*)                                          AS total_passes,
        COUNT(*) FILTER (
            WHERE extra_data->>'pass_outcome' IS NULL
        )                                                 AS completed_passes,
        COUNT(*) FILTER (
            WHERE under_pressure = TRUE
        )                                                 AS passes_under_pressure,
        COUNT(*) FILTER (
            WHERE under_pressure = TRUE
              AND extra_data->>'pass_outcome' IS NULL
        )                                                 AS completed_passes_under_pressure,
        COUNT(*) FILTER (
            -- passes into final third: end x >= 80 (StatsBomb pitch is 120 long)
            WHERE (extra_data->'pass_end_location'->0)::FLOAT >= 80
              AND extra_data->>'pass_outcome' IS NULL
        )                                                 AS passes_into_final_third
    FROM match_events
    WHERE event_type = 'Pass'
    GROUP BY match_id, team_id
),

-- Progressive passes: completed, end_x - start_x >= 10, end_x >= 60
progressive_pass_cte AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) AS progressive_passes
    FROM match_events
    WHERE event_type = 'Pass'
      AND extra_data->>'pass_outcome' IS NULL
      AND location_x IS NOT NULL
      AND (extra_data->'pass_end_location'->0) IS NOT NULL
      AND (extra_data->'pass_end_location'->0)::FLOAT - location_x >= 10
      AND (extra_data->'pass_end_location'->0)::FLOAT >= 60
    GROUP BY match_id, team_id
),

-- ---- PRESSING / PPDA ----
pressure_stats AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) AS total_pressures,
        -- counterpress flag is stored flat as extra_data->>'counterpress'
        COUNT(*) FILTER (
            WHERE extra_data->>'counterpress' IS NOT NULL
        ) AS successful_pressures
    FROM match_events
    WHERE event_type = 'Pressure'
    GROUP BY match_id, team_id
),

-- Opponent passes anywhere in the match (for PPDA numerator)
opponent_passes_cte AS (
    SELECT
        me.match_id,
        tmb.team_id,
        COUNT(*) AS opponent_passes_allowed
    FROM match_events me
    JOIN team_match_base tmb
        ON me.match_id = tmb.match_id
       AND me.team_id  = tmb.opponent_team_id
    WHERE me.event_type = 'Pass'
    GROUP BY me.match_id, tmb.team_id
),

-- Defensive actions = pressures + blocks + clearances + interceptions
defensive_actions_cte AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) AS defensive_actions
    FROM match_events
    WHERE event_type IN ('Pressure','Block','Clearance','Ball Recovery',
                         'Interception','Duel')
    GROUP BY match_id, team_id
),

ppda_cte AS (
    SELECT
        o.match_id,
        o.team_id,
        ROUND(
            (o.opponent_passes_allowed::FLOAT / NULLIF(d.defensive_actions, 0))::NUMERIC,
            3
        ) AS ppda
    FROM opponent_passes_cte o
    JOIN defensive_actions_cte d USING (match_id, team_id)
),

-- ---- SHOTS & xG ----
-- shot_statsbomb_xg is a flat top-level key
-- shot_outcome: 'Goal', 'Saved', 'Saved to Post', 'Saved Off Target' = on target
-- shot_type: 'Free Kick', 'Corner', 'Penalty' = set piece
shot_stats AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) AS total_shots,
        COUNT(*) FILTER (
            WHERE extra_data->>'shot_outcome' IN ('Goal','Saved','Saved to Post')
        ) AS shots_on_target,
        COALESCE(
            SUM((extra_data->>'shot_statsbomb_xg')::FLOAT), 0
        ) AS xg_created,
        COALESCE(SUM(
            CASE WHEN extra_data->>'shot_type' != 'Penalty'
                 THEN (extra_data->>'shot_statsbomb_xg')::FLOAT
            END
        ), 0) AS np_xg,
        COUNT(*) FILTER (
            WHERE extra_data->>'shot_type' IN ('Free Kick','Corner','Penalty')
        ) AS set_piece_shots
    FROM match_events
    WHERE event_type = 'Shot'
    GROUP BY match_id, team_id
),

-- xG conceded = opponent's xG created in the same match
xg_conceded_cte AS (
    SELECT
        tmb.match_id,
        tmb.team_id,
        COALESCE(s.xg_created, 0) AS xg_conceded
    FROM team_match_base tmb
    LEFT JOIN shot_stats s
        ON s.match_id = tmb.match_id
       AND s.team_id  = tmb.opponent_team_id
),

-- ---- CARRIES ----
-- carry_end_location is a flat top-level JSONB array [x, y]
carry_stats AS (
    SELECT
        match_id,
        team_id,
        COUNT(*) AS total_carries,
        COUNT(*) FILTER (
            WHERE location_x IS NOT NULL
              AND (extra_data->'carry_end_location'->0) IS NOT NULL
              AND (extra_data->'carry_end_location'->0)::FLOAT - location_x >= 5
              AND (extra_data->'carry_end_location'->0)::FLOAT >= 60
        ) AS progressive_carries,
        COUNT(*) FILTER (
            WHERE (extra_data->'carry_end_location'->0) IS NOT NULL
              AND (extra_data->'carry_end_location'->0)::FLOAT >= 80
        ) AS carries_into_final_third
    FROM match_events
    WHERE event_type = 'Carry'
    GROUP BY match_id, team_id
)

-- ---- FINAL ASSEMBLY ----
SELECT
    tmb.match_id,
    tmb.team_id,
    tmb.is_home,
    tmb.match_date,
    tmb.competition_name,
    tmb.season_name,
    tmb.opponent_team_id,
    tmb.result,
    tmb.goals_scored,
    tmb.goals_conceded,

    COALESCE(pos.possession_pct, 50.0)                          AS possession_pct,
    COALESCE(ps.total_passes, 0)                                AS total_passes,
    ROUND(
        (100.0 * COALESCE(ps.completed_passes, 0)
        / NULLIF(ps.total_passes, 0))::NUMERIC, 2
    )                                                           AS pass_completion_pct,
    COALESCE(pp.progressive_passes, 0)                         AS progressive_passes,
    COALESCE(ps.passes_into_final_third, 0)                    AS passes_into_final_third,

    COALESCE(prs.total_pressures, 0)                           AS total_pressures,
    ROUND(
        (100.0 * COALESCE(prs.successful_pressures, 0)
        / NULLIF(prs.total_pressures, 0))::NUMERIC, 2
    )                                                           AS pressure_success_rate,
    COALESCE(ppda.ppda, 999)                                   AS ppda,
    COALESCE(da.defensive_actions, 0)                          AS defensive_actions,

    COALESCE(ss.total_shots, 0)                                AS total_shots,
    COALESCE(ss.shots_on_target, 0)                            AS shots_on_target,
    COALESCE(ss.xg_created, 0)                                 AS xg_created,
    COALESCE(xgc.xg_conceded, 0)                               AS xg_conceded,
    ROUND(
        (COALESCE(ss.xg_created, 0)
        / NULLIF(xgc.xg_conceded, 0))::NUMERIC, 3
    )                                                           AS xg_ratio,
    COALESCE(ss.np_xg, 0)                                      AS np_xg,

    COALESCE(cs.total_carries, 0)                              AS total_carries,
    COALESCE(cs.progressive_carries, 0)                        AS progressive_carries,
    ROUND(
        (100.0 * COALESCE(cs.progressive_carries, 0)
        / NULLIF(cs.total_carries, 0))::NUMERIC, 2
    )                                                           AS progressive_carry_pct,
    COALESCE(cs.carries_into_final_third, 0)                   AS carries_into_final_third,

    COALESCE(ps.passes_under_pressure, 0)                      AS passes_under_pressure,
    ROUND(
        (100.0 * COALESCE(ps.completed_passes_under_pressure, 0)
        / NULLIF(ps.passes_under_pressure, 0))::NUMERIC, 2
    )                                                           AS pass_completion_under_pressure_pct,

    COALESCE(ss.set_piece_shots, 0)                            AS set_piece_shots,
    ROUND(
        (100.0 * COALESCE(ss.set_piece_shots, 0)
        / NULLIF(ss.total_shots, 0))::NUMERIC, 2
    )                                                           AS set_piece_shot_pct

FROM team_match_base tmb
LEFT JOIN possession_pct_cte   pos  USING (match_id, team_id)
LEFT JOIN pass_stats            ps   USING (match_id, team_id)
LEFT JOIN progressive_pass_cte  pp   USING (match_id, team_id)
LEFT JOIN pressure_stats        prs  USING (match_id, team_id)
LEFT JOIN ppda_cte              ppda USING (match_id, team_id)
LEFT JOIN defensive_actions_cte da   USING (match_id, team_id)
LEFT JOIN shot_stats            ss   USING (match_id, team_id)
LEFT JOIN xg_conceded_cte       xgc  USING (match_id, team_id)
LEFT JOIN carry_stats           cs   USING (match_id, team_id)

ON CONFLICT (match_id, team_id) DO UPDATE SET
    possession_pct    = EXCLUDED.possession_pct,
    total_passes      = EXCLUDED.total_passes,
    pass_completion_pct = EXCLUDED.pass_completion_pct,
    progressive_passes = EXCLUDED.progressive_passes,
    passes_into_final_third = EXCLUDED.passes_into_final_third,
    total_pressures   = EXCLUDED.total_pressures,
    pressure_success_rate = EXCLUDED.pressure_success_rate,
    ppda              = EXCLUDED.ppda,
    defensive_actions = EXCLUDED.defensive_actions,
    total_shots       = EXCLUDED.total_shots,
    shots_on_target   = EXCLUDED.shots_on_target,
    xg_created        = EXCLUDED.xg_created,
    xg_conceded       = EXCLUDED.xg_conceded,
    xg_ratio          = EXCLUDED.xg_ratio,
    np_xg             = EXCLUDED.np_xg,
    total_carries     = EXCLUDED.total_carries,
    progressive_carries = EXCLUDED.progressive_carries,
    progressive_carry_pct = EXCLUDED.progressive_carry_pct,
    carries_into_final_third = EXCLUDED.carries_into_final_third,
    passes_under_pressure = EXCLUDED.passes_under_pressure,
    pass_completion_under_pressure_pct = EXCLUDED.pass_completion_under_pressure_pct,
    set_piece_shots   = EXCLUDED.set_piece_shots,
    set_piece_shot_pct = EXCLUDED.set_piece_shot_pct;
