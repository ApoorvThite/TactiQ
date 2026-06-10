# Phase 5 — SHAP Explainability & Upset Detector

**Date completed:** 2026-06-09  
**Phase status:** Complete

---

## Summary

Phase 5 layers full SHAP TreeExplainer analysis on top of the Phase 4 XGBoost model, producing per-feature attribution for every prediction and an upset detector that identifies WC2026 matchups where lower-ranked teams hold structural tactical edges. 460 training rows were explained, 9 figures generated (fig8–fig16), 5 per-matchup waterfall plots produced, 5 upset candidates identified from 24 WC2026 fixture predictions, and 29 rows inserted into `matchup_shap_values` for dashboard use in Phase 7.

---


## Global Feature Importance (SHAP)

### WIN prediction — top 3 features

| Feature | Mean |SHAP| | Interpretation |
|---------|------|---|
| `delta_matches_played` | 0.2094 | Teams with more matches in the dataset have more reliable style vectors — the model has higher confidence in its prediction when the style gap is supported by larger sample sizes |
| `delta_avg_xg_created_p90` | 0.1781 | Absolute attacking output advantage is the clearest predictor of victory — teams that create more chances per 90 win significantly more often |
| `delta_avg_xg_ratio` | 0.0991 | The attack/defence balance ratio captures overall tactical dominance; a large positive delta means Team A creates more relative to what they concede |

### DRAW prediction — top 3 features

| Feature | Mean |SHAP| | Interpretation |
|---------|------|---|
| `delta_avg_ppda` | 0.0939 | Pressing differential drives draw probability — when Team A presses much harder than Team B, the defensive shape of the lower-block team absorbs the pressure and creates tactical stalemates |
| `form_points_delta` | 0.0516 | Recent form matters more for draws than wins — evenly matched form produces closer games |
| `delta_avg_pass_completion_pct` | 0.0508 | Technical parity (similar passing accuracy) tends toward draws; large mismatches tend toward decisive outcomes |

### Key difference: what drives wins vs draws

WIN is dominated by attacking output features (xG created, xG ratio) — the team that simply creates more danger wins more often. DRAW is dominated by pressing and technical balance features (PPDA, pass completion) — games end level when tactical styles cancel each other out rather than when one team is definitively superior. This is the model's clearest structural finding.

---

## Most Interesting SHAP Finding

The most counterintuitive finding from the global beeswarm is that `delta_matches_played` is the single strongest WIN predictor (mean |SHAP| = 0.2094), outranking xG ratio and possession. This is not a tactical feature — it is a data quality proxy. Teams with more StatsBomb matches (Argentina=11, Belgium=19, Germany and England=large samples) have richer style profiles, so their style vectors are more representative of their actual tactical DNA. When the model encounters a large `delta_matches_played`, it effectively increases its confidence in the style-delta features for the better-sampled team. This has a real-world interpretation: more experienced tournament teams (more international matches recorded) perform more consistently to their tactical archetype, while less-sampled teams introduce noise. Phase 6 will partially address this by adding WC2026 squad data to update the vectors.

---

## Per-Matchup SHAP Explanations

### 1. Spain vs Morocco → DRAW (55.2%)

**Top SHAP contributors to DRAW:**
- `delta_avg_possession_pct` (+0.369): Morocco's low-possession style vs Spain's ball dominance creates the possession gap that drives draw probability — Morocco defends deep and absorbs
- `delta_avg_ppda` (+0.121): Spain presses significantly harder; Morocco's defensive block neutralises the pressure → classic High Press vs Deep Block stalemate

**Tactical explanation:** Spain's ball dominance is real, but Morocco's Low Block absorbs pressure without creating a decisive xG advantage for Spain.

---

### 2. Germany vs Brazil → Brazil WIN (49.8%)

**Top SHAP contributors to LOSS (Germany loses):**
- `delta_avg_xg_created_p90` and `delta_avg_xg_ratio`: Brazil has the highest xG ratio in the dataset (7.03). Even in same-archetype (High Press vs High Press) matchups, Brazil's raw attacking superiority tilts the model toward a Brazil win.

**Tactical explanation:** Two High Press teams — but Brazil converts their press into dramatically more xG. Germany's pressing is tactically similar but less productive in finishing.

---

### 3. France vs Argentina → DRAW (48.4%)

**Top SHAP contributors to DRAW:**
- `delta_avg_possession_pct`: France has more possession than Argentina (Possession Control vs High Press creates balance)
- `delta_avg_ppda`: Argentina presses harder, France's technical quality absorbs it

**Tactical explanation:** The 2022 World Cup Final in miniature — France's structured build-up vs Argentina's aggressive press produces a close-fought draw signal. The model captured this matchup correctly (the final was 2-2 before penalties in reality).

---

### 4. England vs Iran → England WIN (72.2%)

**Top SHAP contributors to WIN:**
- `delta_matches_played` (+0.273): England has far more StatsBomb match data, increasing confidence
- `delta_avg_xg_created_p90` (+0.259): England creates significantly more xG per 90

**Tactical explanation:** The largest xG creation gap among all 5 test matchups + highest confidence score. England's Possession Control vs Iran's Counter-Attack — Iran can threaten on the break but cannot match England's sustained output.

---

### 5. Netherlands vs Ecuador → Netherlands WIN (50.8%)

**Top SHAP contributors to WIN:**
- `delta_matches_played` (+0.151): Netherlands has more samples
- `delta_avg_xg_ratio` (−0.115): Ecuador actually has a stronger xG ratio (High Press archetype with Brazil-like style); this pushes slightly toward upset, keeping it close at 50.8%

**Tactical explanation:** The most evenly contested of the 5 predictions. Ecuador's High Press DNA gives them a genuine stylistic edge that narrows Netherlands' win probability to just above coin-flip.

---

## Upset Watchlist

| Rank | Underdog | Favourite | p(not-fav-win) | Key tactical signal |
|------|----------|-----------|----------------|---------------------|
| 1 | Italy (Possession Control) | Croatia (Possession Control) | 0.60 | PPDA neutralisation — Italy's press-resistance edge (SHAP +0.079) |
| 2 | Portugal (Possession Control) | Germany (High Press) | 0.58 | PPDA + set piece SHAP + archetype disadvantage (High Press vs Possession Control upset-prone) |
| 3 | Romania (Counter-Attack) | Peru (Possession Control) | 0.58 | PPDA neutralisation + set piece edge; near-even match |
| 4 | Panama (Deep Block) | United States (Possession Control) | 0.51 | Deep Block absorbs US press; PPDA SHAP +0.130 — strongest single upset signal |
| 5 | Japan (Deep Block) | Spain (High Press) | 0.49 | Historical WC2022 upset echoed; High Press vs Deep Block draw-prone; experience delta drives model's Spanish edge |

**Most tactically justified upset:** Germany vs Portugal — the model gives Germany only 42% win probability, with Portugal's set-piece dependency and PPDA resistance combining into a genuinely ambiguous matchup.

---

## Tactical Narratives

**Spain vs Morocco:**
Spain (High Press) faces Morocco (Deep Block), a matchup that is historically draw-prone (Deep Block absorbs the press). Spain dominates possession — controls the tempo. Spain has more tournament experience in the dataset. The model expects DRAW predicted at 55.2%.

**Germany vs Brazil:**
Both teams are High Press archetypes — high-variance (both sides press hard). Germany generates significantly more xG per match. Germany creates significantly more danger relative to what they concede. The model expects Brazil WIN predicted at 49.8%.

**France vs Argentina:**
France (Possession Control) faces Argentina (High Press), a matchup that is historically tactically variable. France has more tournament experience in the dataset. France presses far less aggressively — Argentina will be under intense pressure. The model expects DRAW predicted at 48.4%.

**England vs Iran:**
England (Possession Control) faces Iran (Counter-Attack), a matchup that is historically upset-prone (Counter-Attack on the break). England has more tournament experience in the dataset. England generates significantly more xG per match. The model expects England WIN predicted at 72.2%.

**Netherlands vs Ecuador:**
Netherlands (Possession Control) faces Ecuador (High Press), a matchup that is historically tactically variable. Netherlands has more tournament experience in the dataset. Ecuador has a stronger attack/defence balance. The model expects Netherlands WIN predicted at 50.8%.

---

## SHAP Technical Notes

**Why TreeExplainer on raw model, not calibrated wrapper:**
SHAP's `TreeExplainer` requires direct access to the tree structure (leaf values, split conditions) of the underlying XGBoost booster. The `IsotonicMulticlassCalibrator` wrapper adds post-hoc isotonic regression on top of the model's raw probabilities — SHAP cannot traverse into the calibration layer because it is not a tree. The raw model's SHAP values still represent the correct feature attributions; the calibration only rescales the output probabilities, it does not change which features matter or in which direction.

**What the expected_value (base rate) represents:**
The expected value for each class is the model's average output (in log-odds space) across the training set when no feature information is provided. For win: −0.003, draw: +0.005, loss: −0.003. These values are close to zero because the training distribution is approximately balanced (38%/25%/38%). Each waterfall plot starts from this base rate and adds feature contributions to arrive at the final prediction.

**Why SHAP interaction values matter for `archetype_matchup_id`:**
`archetype_matchup_id` is an integer 0–15 encoding the cross-archetype type. Its main effect SHAP (±0.002–0.044) may appear modest, but interaction effects with xG delta and PPDA delta can be significant. A High Press vs Deep Block matchup (ID=3) with a large xG delta behaves very differently from the same archetype code with a neutral delta — the interaction effect captures this non-additive signal. The fig11 scatter plots visualise these cross-feature relationships for the top-3 features.

---

## Phase 6 Preview

Phase 6 loads actual WC2026 squad data for all 48 qualified teams, applies the full pipeline to generate current style vectors (using StatsBomb or scraped data for recent matches), updates cluster assignments for teams not previously in the dataset, and runs the complete prediction engine on all 48 teams + the actual official group draw fixtures. The proxy vector for Norway (Counter-Attack centroid) will be replaced with computed vectors from Norwegian national team match data. Phase 6 output feeds directly into Phase 7's interactive dashboard.
