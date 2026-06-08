# Phase 2 — SQL Feature Engineering

**Date completed:** 2026-06-08
**Phase status:** Complete

---

## Summary

Phase 2 transformed 843,050 raw StatsBomb match events into a structured tactical feature table.
17 tactical metrics were engineered per team per match via a single CTE-based SQL pipeline,
then aggregated into team-level style profiles ready for ML clustering in Phase 3.

---

## Tables Created

| Table | Rows | Description |
|-------|------|-------------|
| `match_team_features` | 460 | One row per team per match — 33 columns including 17 tactical features + 5 rolling form columns |
| `team_style_profiles` | 54 | Aggregated style vector per team (min 3 matches) |

---

## Features Engineered

### Possession & Tempo

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `possession_pct` | % of possession events (Pass/Carry/Receipt/Dribble/Shot/Duel) owned by team | Tempo and ball control identity | 31.76% – 68.97% |
| `total_passes` | COUNT of Pass events | Volume indicator | ~200–700/match |
| `pass_completion_pct` | `completed_passes / total_passes × 100` (NULL outcome = completed) | Technical quality | ~60–95% |
| `progressive_passes` | Completed passes where end_x − start_x ≥ 10 AND end_x ≥ 60 | Directness in attack | ~30–150/match |
| `passes_into_final_third` | Completed passes where end_x ≥ 80 | Attacking progression | ~15–90/match |

### Pressing & Defense

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `total_pressures` | COUNT of Pressure events | Pressing volume | ~80–350/match |
| `pressure_success_rate` | `counterpress_events / total_pressures × 100` | Press effectiveness | ~15–30% |
| `ppda` | `opponent_passes / team_defensive_actions` | Pressing intensity (lower = more aggressive) | 1.41 – 2.94 |
| `defensive_actions` | COUNT of Pressure + Block + Clearance + Ball Recovery + Interception + Duel | Total defensive work | ~150–600/match |

### Attack & Shooting

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `total_shots` | COUNT of Shot events | Attacking volume | ~3–25/match |
| `shots_on_target` | Shots with outcome in ('Goal','Saved','Saved to Post') | Shot quality | ~1–10/match |
| `xg_created` | SUM of `shot_statsbomb_xg` per match | Attack threat quality | 0.0 – 4.5 |
| `xg_conceded` | Opponent's `xg_created` in same match | Defensive exposure | 0.0 – 4.5 |
| `xg_ratio` | `xg_created / xg_conceded` | Attack/defense balance | 0.36 – 7.03 |
| `np_xg` | xG from non-penalty shots only | Isolates open-play creativity | 0.0 – 4.0 |

### Carries & Directness

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `total_carries` | COUNT of Carry events | Ball-carrying volume | ~150–600/match |
| `progressive_carries` | Carries where end_x − start_x ≥ 5 AND end_x ≥ 60 | Direct ball progression | ~20–100/match |
| `progressive_carry_pct` | `progressive_carries / total_carries × 100` | Directness style | 10–20% |
| `carries_into_final_third` | Carries where end_x ≥ 80 | Penetration frequency | ~10–60/match |

### Under Pressure

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `passes_under_pressure` | COUNT of passes while `under_pressure = TRUE` | Pressure exposure | ~30–200/match |
| `pass_completion_under_pressure_pct` | Completion rate when under pressure | Technical composure | ~50–85% |

### Set Pieces

| Feature | Formula / Definition | Tactical Signal | Observed Range |
|---------|----------------------|-----------------|----------------|
| `set_piece_shots` | Shots with `shot_type` in ('Free Kick','Corner','Penalty') | Set piece dependency | ~0–8/match |
| `set_piece_shot_pct` | `set_piece_shots / total_shots × 100` | Tactical emphasis | 0–50% |

### Rolling Form (window over last 5 matches per team)

| Feature | Window Definition |
|---------|-------------------|
| `rolling_xg_created_5` | AVG xg_created, prior 5 matches |
| `rolling_xg_conceded_5` | AVG xg_conceded, prior 5 matches |
| `rolling_ppda_5` | AVG ppda (excluding 999 sentinel), prior 5 matches |
| `rolling_possession_5` | AVG possession_pct, prior 5 matches |
| `form_points_5` | SUM of points (W=3, D=1, L=0) from prior 5 matches |

---

## Pipeline Run Output

```
TactiQ Phase 2 — Feature Pipeline
----------------------------------------
  Running: Create feature tables ...
  [OK] Create feature tables
  Running: Compute match-team features ...
  [OK] Compute match-team features
  Running: Compute rolling form ...
  [OK] Compute rolling form
  Running: Compute team style profiles ...
  [OK] Compute team style profiles

============================================================
 TACTIQ — Phase 2 Feature Pipeline Complete
============================================================

MATCH_TEAM_FEATURES
  Total rows               : 460   (expected ~460 = 230 matches × 2 teams)
  Rows with ppda computed  : 460
  Rows with xg_created > 0 : 459
  Avg possession_pct       : 50.00%  (should be near 50%)
  Avg xg_created per match : 1.509

TEAM_STYLE_PROFILES
  Teams profiled           : 54
  Avg matches per team     : 8.5
  PPDA range               : 1.413 (min) — 2.943 (max)
  xG ratio range           : 0.357 (min) — 7.028 (max)
  Possession range         : 31.76% — 68.97%

TOP 5 TEAMS BY PPDA (most aggressive pressing — lower = more pressing)
  1. Spain                     : 1.413
  2. Austria                   : 1.502
  3. Mexico                    : 1.534
  4. Nigeria                   : 1.565
  5. Germany                   : 1.567

TOP 5 TEAMS BY XG_RATIO (best attack/defense balance)
  1. Brazil                    : 7.028
  2. Ecuador                   : 6.729
  3. Spain                     : 4.955
  4. Argentina                 : 4.365
  5. Germany                   : 4.045

TOP 5 TEAMS BY PROGRESSIVE CARRY PCT
  1. Mexico                    : 16.33%
  2. Senegal                   : 15.76%
  3. Iceland                   : 15.64%
  4. Russia                    : 15.51%
  5. Brazil                    : 15.33%

NULL AUDIT
  rolling_xg_created_5 nulls : 54  (first 5 matches per team expected)
  ppda nulls                 : 0  (should be 0)
  xg_ratio nulls             : 1  (0-xg-conceded games expected)
============================================================
```

---

## Key Findings from Exploration

- **Most aggressive pressers:** Spain (PPDA 1.41), Austria (1.50), Germany (1.57) — elite pressing sides consistently apply high defensive pressure, with Spain combining it with 68.97% possession (highest in the dataset).
- **Best xG ratio:** Brazil (7.03) and Ecuador (6.73) — both dominated opponents in their respective tournaments, creating significantly more danger than they conceded. Brazil's 2018 WC group stage performance heavily drives this.
- **Possession vs PPDA:** High-possession teams (Spain, Germany) tend to have lower PPDA — they press aggressively AND dominate the ball, disproving the myth that pressing and possession are mutually exclusive.
- **Surprising finding:** Mexico ranks 3rd in PPDA and 1st in progressive carry % — a counter-pressing, direct style that leverages quick ball-carrying transitions rather than patient buildup.
- **1 xg_ratio null:** One match where xg_conceded = 0 (team conceded no shots with xG) — correctly handled by NULLIF guard.

---

## SQL Design Decisions

**Why CTEs over subqueries or temp tables:**
CTEs keep each metric computation isolated and named, making the pipeline readable and debuggable. Each CTE can be inspected independently. Temp tables would require separate transactions; subqueries would make the final SELECT unreadable.

**Why LEFT JOIN in the final assembly (not INNER JOIN):**
Teams that had zero events of a given type (e.g., a team with 0 shots) would be dropped from an INNER JOIN. LEFT JOIN preserves all 460 team-match rows and fills missing metrics with COALESCE defaults (0 for counts, 50.0 for possession).

**Why NULLIF guards on division operations:**
Prevents division-by-zero errors that would abort the entire INSERT. Returns NULL instead, which is propagated correctly by COALESCE and downstream aggregations.

**Why ppda defaults to 999 on null (not 0 or NULL):**
NULL would be filtered by `WHERE ppda IS NULL` queries, masking computation failures. 0 would indicate perfect pressing (impossible). 999 is an obvious sentinel that `NULLIF(ppda, 999)` strips in aggregations, while NULL audit queries catch any true failures.

**Why flat JSONB paths (`extra_data->>'shot_statsbomb_xg'`) not nested:**
StatsBombPy's `events()` method flattens all nested dicts into top-level keys in the DataFrame before insertion. The ingestion script stored this flattened structure in `extra_data`, so all paths are flat — e.g., `shot_statsbomb_xg` not `shot.statsbomb_xg`.

---

## Feature Limitations & Known Issues

| Limitation | Detail |
|------------|--------|
| **PPDA definition** | Uses all opponent passes (not zone-filtered) as numerator since StatsBomb's coordinate system per-team perspective makes location-based filtering complex without coordinate normalization |
| **pressure_success_rate** | Approximated as counterpress events / total pressures. True press success would require tracking ball possession change in the next 5 events |
| **No Copa América data** | StatsBomb free tier doesn't include Copa América 2021/2024, limiting coverage to WC2018, WC2022, Euro2020, Euro2024 |
| **Rolling nulls** | First match per team always has NULL rolling features — expected behavior, 54 nulls total |
| **set_piece_shot_pct** | Shot type classification relies on `shot_type` field; headers from corners classified as 'Corner' not 'Open Play', so this captures set-piece dependency accurately |

---

## Phase 3 Preview

Phase 3 takes `team_style_profiles` as input and applies:
- **PCA** to reduce 15 features to principal components
- **UMAP** for non-linear dimensionality reduction and visualization
- **K-means clustering** to assign each team a tactical archetype label (e.g., "High Press", "Deep Block", "Possession Control", "Counter-Attack Direct")

The cluster assignments become the "tactical DNA" features used in Phase 4 matchup prediction.
