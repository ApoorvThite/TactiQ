-- TactiQ Phase 2 — Rolling form features
-- Run AFTER compute_match_features.sql populates base rows.
-- Window: last 5 matches per team ordered by match_date.

WITH rolling AS (
    SELECT
        id,
        AVG(xg_created)     OVER w AS rolling_xg_created_5,
        AVG(xg_conceded)    OVER w AS rolling_xg_conceded_5,
        AVG(NULLIF(ppda, 999)) OVER w AS rolling_ppda_5,
        AVG(possession_pct) OVER w AS rolling_possession_5,
        SUM(CASE result
                WHEN 'win'  THEN 3
                WHEN 'draw' THEN 1
                ELSE 0
            END)            OVER w AS form_points_5
    FROM match_team_features
    WINDOW w AS (
        PARTITION BY team_id
        ORDER BY match_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )
)
UPDATE match_team_features mtf
SET
    rolling_xg_created_5  = ROUND(r.rolling_xg_created_5::NUMERIC,  3),
    rolling_xg_conceded_5 = ROUND(r.rolling_xg_conceded_5::NUMERIC, 3),
    rolling_ppda_5        = ROUND(r.rolling_ppda_5::NUMERIC,         3),
    rolling_possession_5  = ROUND(r.rolling_possession_5::NUMERIC,   2),
    form_points_5         = r.form_points_5::INTEGER
FROM rolling r
WHERE mtf.id = r.id;
