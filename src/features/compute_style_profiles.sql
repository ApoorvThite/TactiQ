-- TactiQ Phase 2 — Team style profiles aggregation
-- Aggregates match_team_features into one style vector per team.
-- Requires >= 3 matches for a reliable profile.

TRUNCATE team_style_profiles;

INSERT INTO team_style_profiles
SELECT
    mtf.team_id,
    t.team_name,
    COUNT(DISTINCT mtf.match_id)                                        AS matches_played,
    ROUND(AVG(mtf.possession_pct)::NUMERIC,                        2)  AS avg_possession_pct,
    ROUND(AVG(mtf.pass_completion_pct)::NUMERIC,                   2)  AS avg_pass_completion_pct,
    ROUND(AVG(mtf.progressive_passes)::NUMERIC,                    2)  AS avg_progressive_passes_p90,
    ROUND(AVG(mtf.passes_into_final_third)::NUMERIC,               2)  AS avg_passes_final_third_p90,
    ROUND(AVG(NULLIF(mtf.ppda, 999))::NUMERIC,                     3)  AS avg_ppda,
    ROUND(AVG(mtf.pressure_success_rate)::NUMERIC,                 2)  AS avg_pressure_success_rate,
    ROUND(AVG(mtf.total_pressures)::NUMERIC,                       2)  AS avg_total_pressures_p90,
    ROUND(AVG(mtf.xg_created)::NUMERIC,                            3)  AS avg_xg_created_p90,
    ROUND(AVG(mtf.xg_conceded)::NUMERIC,                           3)  AS avg_xg_conceded_p90,
    ROUND(AVG(NULLIF(mtf.xg_ratio, NULL))::NUMERIC,                3)  AS avg_xg_ratio,
    ROUND(AVG(mtf.progressive_carry_pct)::NUMERIC,                 2)  AS avg_progressive_carry_pct,
    ROUND(AVG(mtf.carries_into_final_third)::NUMERIC,              2)  AS avg_carries_final_third_p90,
    ROUND(AVG(mtf.pass_completion_under_pressure_pct)::NUMERIC,    2)  AS avg_pass_completion_under_pressure_pct,
    ROUND(AVG(mtf.set_piece_shot_pct)::NUMERIC,                    2)  AS avg_set_piece_shot_pct,
    ROUND(
        (SUM(CASE mtf.result WHEN 'win' THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0))::NUMERIC, 3
    )                                                                   AS win_rate,
    NOW()                                                               AS updated_at
FROM match_team_features mtf
JOIN teams t USING (team_id)
GROUP BY mtf.team_id, t.team_name
HAVING COUNT(DISTINCT mtf.match_id) >= 3;
