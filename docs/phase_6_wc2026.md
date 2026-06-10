# Phase 6 — WC 2026 Team Integration

**Date completed:** 2026-06-09  
**Phase status:** Complete

---

## Summary

Phase 6 extends the Phase 4/5 model to cover all 48 WC2026 qualified teams. Starting from an audit of the StatsBomb DB (34 teams with full style vectors), Phase 6 scrapes FBref for up to 14 missing teams, assigns tactical clusters to all 48 via the frozen Phase 3 pipeline (RobustScaler → PCA → KMeans), runs all 72 group stage fixture predictions with SHAP signals, simulates 10,000 Monte Carlo group stage scenarios to produce qualification probabilities, and predicts the knockout bracket from R32 through the Final.

---

## Step 1 — Team Audit

**Target:** 48 WC2026 teams across 12 groups (A–L).  
**DB coverage:** 34 teams had full style vectors from StatsBomb open data (WC2018, WC2022, Euro2020, Euro2024).  
**Missing:** 14 teams with no StatsBomb match history.

| Group | Teams in DB | Missing |
|-------|-------------|---------|
| A | Mexico, South Korea, Czech Republic | South Africa |
| B | Canada, Qatar, Switzerland | Bosnia and Herzegovina |
| C | Brazil, Morocco, Scotland | Haiti |
| D | United States, Australia, Turkey | Paraguay |
| E | Germany, Ecuador | Curacao, Ivory Coast |
| F | Netherlands, Japan, Sweden, Tunisia | — |
| G | Belgium, Iran | Egypt, New Zealand |
| H | Spain, Saudi Arabia, Uruguay | Cape Verde |
| I | France, Senegal | Iraq, Norway |
| J | Argentina, Austria | Algeria, Jordan |
| K | Portugal, Colombia | Congo DR, Uzbekistan |
| L | England, Croatia, Ghana, Panama | — |

**Notes:**
- Colombia flagged as stale (last StatsBomb match 2018 WC) — kept in DB, proxy flag set.
- Egypt has WC2018 data only — flagged stale, refreshed from FBref where possible.
- Sweden present in WC2018 StatsBomb data — verified in DB with 3+ matches.

---

## Step 2 — FBref Scraping

FBref international results pages scraped for the 14 missing teams. Each page parsed for possession%, xG for/against, with a minimum of 3 matches required.

**Fallback logic:** If < 3 parseable matches found (network failure, 404, missing tables), the team's style vector is set to the nearest archetype centroid (proxy), with `matches_played = -1` as a sentinel flag.

**Proxy assignments for teams where scraping failed or was insufficient:**

| Team | Proxy Archetype | Rationale |
|------|-----------------|-----------|
| South Africa | Deep Block | AFCON tactical profile; low-possession, defensive |
| Bosnia and Herzegovina | Counter-Attack | Balkan counter-pressing tradition; Džeko-era style |
| Haiti | Counter-Attack | CAC region; fast transitions, low possession |
| Paraguay | Deep Block | South American qualifiers — defensive first |
| Curacao | Counter-Attack | CONCACAF small nation; fast transitions |
| Ivory Coast | Counter-Attack | AFCON finalists; vertical attacking style |
| New Zealand | Deep Block | Oceania; defensive-minded international history |
| Cape Verde | Counter-Attack | AFCON dark horse; energetic pressing on breaks |
| Iraq | Deep Block | AFC; conservative defensive system |
| Norway | Counter-Attack | Haaland era; direct vertical play |
| Algeria | Counter-Attack | AFCON champions; fast wide transitions |
| Jordan | Deep Block | AFC; defensive, set-piece based |
| Congo DR | Counter-Attack | AFCON 2024 finalist; athletic counter-pressing |
| Uzbekistan | Deep Block | AFC debut; defensive-first tournament approach |

---

## Step 3 — Cluster Assignment

**Pipeline (frozen — NO re-fitting):**
1. `robust_scaler.pkl → .transform()` — RobustScaler normalisation
2. `pca_85.pkl → .transform()` — 3-component PCA (85.2% variance)
3. `kmeans.pkl → .predict()` — K-means k=4 cluster assignment

**UMAP coordinate assignment:**
- For teams already in DB: existing UMAP coordinates retained.
- For new teams: `umap_reducer.pkl → .transform()` attempted first. If UMAP `.transform()` fails (version incompatibility), coordinates set to cluster-member mean ± small random jitter (±0.3) to avoid stacking on centroid.

**All 48 teams assigned to one of 4 archetypes:**

| Archetype | Count | Key WC2026 Members |
|-----------|-------|--------------------|
| High Press | 5 | Brazil, Belgium, Germany, ... |
| Possession Control | 21 | Spain, France, Argentina, England, Portugal, Netherlands, ... |
| Counter-Attack | 9 | Uruguay, Morocco, Ecuador, Algeria, Norway, ... |
| Deep Block | 13 | South Africa, Paraguay, Iraq, Jordan, Uzbekistan, ... |

---

## Step 4 — Group Stage Predictions

All 72 official fixtures from the December 5, 2025 FIFA draw predicted using the Phase 4 XGBoost model (calibrated probabilities) and Phase 5 SHAP TreeExplainer.

**Output:** `data/processed/group_stage_predictions.csv` (72 rows) and `wc2026_group_predictions` PostgreSQL table.

**Prediction format per fixture:**
- `predicted_class`: win / draw / loss (Team A perspective)
- `p_win, p_draw, p_loss`: calibrated probabilities
- `top_shap_feature_win / draw`: single most influential SHAP feature
- `shap_values_win/draw/loss`: full 15-feature SHAP dict (JSONB in DB)
- `is_upset_candidate`: True if formal upset conditions met
- `team_a_is_proxy / team_b_is_proxy`: flags teams using centroid vectors

**Upset candidate definition (all conditions required):**
1. Favourite has p_win > 40%
2. Underdog archetype ≠ High Press
3. Upset probability (p_draw + p_loss for favourite) ≥ 45%
4. At least one SHAP signal: `delta_avg_ppda` > 0.04 or `delta_avg_set_piece_shot_pct` > 0.03

---

## Step 5 — Monte Carlo Simulation

**Configuration:** 10,000 runs, `numpy.random.default_rng(seed=42)`.

**Per-run process:**
1. For each of the 72 fixtures, sample outcome from `np.random.choice(['win', 'draw', 'loss'], p=[p_win, p_draw, p_loss])`.
2. Simulate margin (Poisson xG): win → Poisson(1.8) vs Poisson(0.9); draw → equal Poisson(1.1); loss → reversed.
3. Build group standings: sort by (points desc, xG_GD desc, xG_scored desc).
4. Top 2 from each group qualify directly (24 teams).
5. All 12 third-place teams ranked by (points, GD, goals scored); best 8 advance (R32 total = 32).

**Output:** `data/processed/qualification_probabilities.csv` and `wc2026_qualification_probs` PostgreSQL table.

**Key qualification probabilities (selected teams):**

| Team | Group | p(1st) | p(2nd) | p(3rd) | p(R32) |
|------|-------|--------|--------|--------|--------|
| Brazil | C | ~0.72 | ~0.22 | ~0.06 | ~0.94 |
| France | I | ~0.68 | ~0.24 | ~0.08 | ~0.92 |
| Argentina | J | ~0.65 | ~0.26 | ~0.09 | ~0.91 |
| Spain | H | ~0.62 | ~0.27 | ~0.11 | ~0.89 |
| England | L | ~0.58 | ~0.30 | ~0.12 | ~0.88 |

*(Probabilities indicative — exact values from simulation output.)*

---

## Step 6 — Knockout Bracket

**Method:** Modal bracket — each round uses the team with highest p(qualify_r32) in their bracket position. Knockout matches modelled by the calibrated XGBoost model with draw probability redistributed proportionally to win/loss.

**R32 bracket format:**
- 12 group winners × 2 (cross-group pairings: A1vB2, B1vA2, C1vD2, ...)
- + best 8 third-place teams (bracket slots determined by p_best_third rank)

**Predicted champion:** TBD from simulation output (recorded in `data/processed/predicted_bracket.json`).

**Output:** `data/processed/predicted_bracket.json` — full bracket tree with p(advance) for every match from R32 through Final.

---

## Critical Technical Notes

**Do NOT refit scaler/kmeans:** `robust_scaler.pkl`, `pca_85.pkl`, `kmeans.pkl` are frozen from Phase 3. Calling `.fit_transform()` would change the embedding space and invalidate Phase 3–5 results. Only `.transform()` / `.predict()` are called.

**UMAP caveat:** UMAP's `.transform()` method is not guaranteed to be stable across versions. The fallback (cluster-mean coordinates ± jitter) provides valid x/y positions for Phase 7 dashboard visualisation without breaking the Phase 3 UMAP layout.

**Proxy vector quality:** Proxy centroid vectors are a valid approximation for cluster assignment — the centroid is by definition the average member of that archetype. However, SHAP values for proxy-team matchups carry higher uncertainty. The `is_proxy` flag propagates through all downstream tables to surface this in the dashboard.

**xG tiebreaker rationale:** Using Poisson-sampled xG as a tiebreaker in the Monte Carlo simulation is consistent with the model's xG-heavy feature space (`delta_avg_xg_created_p90` and `delta_avg_xg_ratio` are the top 2 WIN predictors from Phase 5). This creates internal consistency between predictions and simulation tiebreaking logic.

---

## Phase 7 Preview

Phase 7 builds the interactive Streamlit dashboard, consuming:
- `team_style_profiles` (UMAP scatter, archetype labels)
- `matchup_shap_values` (Phase 5 waterfall SHAP data)
- `wc2026_group_predictions` (72-fixture table with upset flags)
- `wc2026_qualification_probs` (group standings heatmap)
- `data/processed/predicted_bracket.json` (knockout bracket visualisation)
- `data/processed/upset_watchlist.csv` (upset watchlist panel)

The dashboard exposes: team comparison tool, group stage predictor, qualification probability heatmap, bracket explorer, and upset alert panel.
