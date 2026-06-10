# Phase 4 — Matchup Model Training

**Date completed:** 2026-06-09  
**Phase status:** Complete

---

## Summary

Phase 4 builds the core XGBoost classifier that predicts win/draw/loss probabilities from the style-vector delta between two teams. A 460-row training set (230 matches × 2 perspectives) was assembled, 15 features engineered, Optuna hyperparameter tuning run for 50 trials, and the final model evaluated via 5-fold grouped cross-validation. Isotonic regression calibration reduces Brier scores across all three classes. The calibrated model is live and produces valid probability outputs for all 5 test matchups.

---

## Feature Engineering

### Style-delta vector (10 features)

The model's primary input is not raw style vectors, but the **element-wise difference** between two teams' Phase 3 scaled style vectors:

```
delta = style_vector_A − style_vector_B
```

This 10-dimensional delta encodes which tactical dimensions diverge between teams. A large negative `delta_avg_ppda` means Team A presses far more aggressively than Team B — a signal XGBoost can learn correlates with win probability.

### Why both perspectives (460 rows, not 230)

Each match produces two rows: the home team's perspective (delta = home − away) and the away team's mirror (delta = away − home = −1 × row 1). This is valid because:
- The model must generalise to both sides of every matchup
- It doubles training data without adding new information (symmetry constraint)
- **Match-level grouped CV was essential**: both rows from the same match always land in the same fold, preventing data leakage (the mirror row of a test match would otherwise appear in training with its exact negative delta)

### Context features (5 features)

| Feature | Rationale |
|---|---|
| `is_home` | Home advantage is real in football; ~60% of wins are home wins historically |
| `form_points_delta` | Recent form (5-match rolling points) captures momentum beyond historical style |
| `archetype_matchup_id` | Integer 0–15 encoding all 16 inter-archetype combinations; lets XGBoost learn tactical type advantages without one-hot explosion on 460 rows |
| `delta_matches_played` | Experience proxy; teams with more matches have more stable style profiles |
| `competition_weight` | Used as `sample_weight` in training (not a feature column): World Cup=1.0, Euros=0.8, others=0.7 — tells XGBoost to weight high-quality matches more in gradient updates |

`form_points_delta` had 66 null rows (teams with < 5 prior matches) → imputed to 0 (neutral prior, consistent with Phase 6 treatment of WC2026 debut nations).

---

## Model Configuration

### Final XGBoost hyperparameters (from Optuna, 50 trials, minimize CV log loss)

| Parameter | Value |
|---|---|
| `n_estimators` | 201 |
| `max_depth` | 2 |
| `learning_rate` | 0.0149 |
| `subsample` | 0.9513 |
| `colsample_bytree` | 0.7221 |
| `reg_alpha` | 0.0322 |
| `reg_lambda` | 0.0028 |
| `min_child_weight` | 6 |
| `objective` | `multi:softprob` |
| `num_class` | 3 |

**Why XGBoost over alternatives:**
- **vs Logistic Regression:** LR assumes linear decision boundaries. The interaction between archetype matchup and delta features (e.g., High Press vs Deep Block behaves differently from Counter-Attack vs Counter-Attack) is non-linear.
- **vs Random Forest:** XGBoost's sequential boosting corrects residuals from prior trees, making better use of 460 rows. RF requires more data to achieve similar bias-variance balance.
- **vs Neural Networks:** With n=460 and 15 features, neural nets overfit without complex regularisation. XGBoost's built-in `reg_alpha`/`reg_lambda` and shallow trees (max_depth=2) achieve natural regularisation. Neural nets also lack interpretability — critical for SHAP in Phase 5.
- **vs SVM:** XGBoost's `predict_proba` output calibrates well to actual probabilities; SVM probabilities are notoriously poorly calibrated.

The shallow tree depth (max_depth=2) reflects Optuna's preference for low-complexity trees given the 460-row dataset — this is a healthy bias-variance trade-off.

---

## CV Results

5-fold grouped cross-validation (match-level splits, both perspectives per match in same fold):

```
                   Mean      Std
  Log Loss       : 1.012  ± 0.072   (baseline: 1.081)
  Accuracy       : 48.0%  ± 10.2%   (baseline: 37.6%)
  F1 Macro       : 0.441  ± 0.095
  F1 Win         : 0.548  ± 0.099
  F1 Draw        : 0.218  ± 0.125
  F1 Loss        : 0.558  ± 0.100

Beats majority-class baseline?
  Log Loss : YES  (1.012 vs 1.081)
  Accuracy : YES  (48.0% vs 37.6%)
```

The model beats the majority-class baseline on both metrics. The ±10.2% accuracy std reflects the small per-fold sample size (~92 rows per fold) — expected with n=460.

---

## Confusion Matrix

OOF predictions across all 460 rows:

```
Predicted →    Win   Draw   Loss
Actual Win  :   98     38     37
Actual Draw :   44     23     47
Actual Loss :   41     32    100
```

Win and Loss recall are strong (~57% each). Draw is the hardest class: only 23/114 draws predicted correctly (F1=0.218). This is expected — draws are high-entropy outcomes that tactical DNA alone cannot reliably identify. Phase 5's upset detector will focus specifically on misclassified draws and upsets.

---

## Calibration

Isotonic regression calibration was applied using OOF probabilities (one IsotonicRegression fitted per class, then renormalised).

| Class | Brier Before | Brier After |
|-------|---|---|
| Win   | 0.207 | 0.170 |
| Draw  | 0.193 | 0.156 |
| Loss  | 0.207 | 0.172 |

Brier scores improve substantially for all three classes after calibration. The reliability diagram (fig6_calibration.png) shows the raw probabilities slightly overconfident (curves bow away from the diagonal); calibration pulls them toward the perfect-calibration line. This matters for Phase 6's match simulation, where probability accuracy directly affects predicted group-stage outcomes.

---

## Top Features

| Rank | Feature | Gain % | Tactical interpretation |
|------|---------|--------|------------------------|
| 1 | `delta_avg_xg_created_p90` | 12.1% | Absolute attacking output difference — the team generating more xG per 90 wins significantly more often |
| 2 | `delta_avg_xg_ratio` | 10.1% | Attack/defence balance ratio delta — captures overall tactical dominance beyond just attacking output |
| 3 | `delta_matches_played` | 9.6% | Experience proxy — teams with more StatsBomb matches have richer, more representative style profiles; the model trusts their vectors more |
| 4 | `form_points_delta` | 7.1% | Recent form momentum — complements style DNA with recency bias |
| 5 | `delta_avg_set_piece_shot_pct` | 7.1% | Set-piece dependency difference — matches where one team heavily relies on set pieces while the other doesn't can swing probability |

Notable: `is_home` ranks last (2.9%) — perhaps surprising, but the StatsBomb dataset is tournament data (World Cup / Euros) often played at neutral venues, reducing home advantage signal.

---

## Sample Predictions Output

```
───────────────────────────────────────────
 TactiQ Matchup Prediction
───────────────────────────────────────────
 Spain (High Press)  vs  Morocco (Deep Block)
───────────────────────────────────────────
 Win     36.3%  ███████░░░░░░░░░░░░░
 Draw    55.2%  ███████████░░░░░░░░░
 Loss     8.5%  ██░░░░░░░░░░░░░░░░░░
───────────────────────────────────────────
 Predicted: DRAW  [High confidence]
───────────────────────────────────────────

───────────────────────────────────────────
 TactiQ Matchup Prediction
───────────────────────────────────────────
 Germany (High Press)  vs  Brazil (High Press)
───────────────────────────────────────────
 Win     24.9%  █████░░░░░░░░░░░░░░░
 Draw    25.3%  █████░░░░░░░░░░░░░░░
 Loss    49.8%  ██████████░░░░░░░░░░
───────────────────────────────────────────
 Predicted: Brazil WIN  [Medium confidence]
───────────────────────────────────────────

───────────────────────────────────────────
 TactiQ Matchup Prediction
───────────────────────────────────────────
 France (Possession Control)  vs  Argentina (High Press)
───────────────────────────────────────────
 Win     33.3%  ███████░░░░░░░░░░░░░
 Draw    48.4%  ██████████░░░░░░░░░░
 Loss    18.3%  ████░░░░░░░░░░░░░░░░
───────────────────────────────────────────
 Predicted: DRAW  [Medium confidence]
───────────────────────────────────────────

───────────────────────────────────────────
 TactiQ Matchup Prediction
───────────────────────────────────────────
 England (Possession Control)  vs  Iran (Counter-Attack)
───────────────────────────────────────────
 Win     72.2%  ██████████████░░░░░░
 Draw    17.4%  ███░░░░░░░░░░░░░░░░░
 Loss    10.3%  ██░░░░░░░░░░░░░░░░░░
───────────────────────────────────────────
 Predicted: England WIN  [High confidence]
───────────────────────────────────────────

───────────────────────────────────────────
 TactiQ Matchup Prediction
───────────────────────────────────────────
 Netherlands (Possession Control)  vs  Ecuador (High Press)
───────────────────────────────────────────
 Win     50.8%  ██████████░░░░░░░░░░
 Draw    37.3%  ███████░░░░░░░░░░░░░
 Loss    11.9%  ██░░░░░░░░░░░░░░░░░░
───────────────────────────────────────────
 Predicted: Netherlands WIN  [Medium confidence]
───────────────────────────────────────────
```

All 5 predictions sum to 1.0 ✓. All probabilities are non-negative ✓.

---

## Limitations

**Small training set:** 230 matches is a lean dataset for a 3-class problem. XGBoost with shallow trees (max_depth=2) and strong regularisation mitigates overfitting, but confidence intervals on CV metrics are wide (±10.2% accuracy std). Phase 6's simulation will account for this uncertainty.

**Draws are the hardest class:** F1=0.218 for draws reflects the fundamental unpredictability of this outcome — draws result from close, evenly-matched tactical battles where small in-game decisions (referee calls, individual moments) dominate over structural style differences. The model correctly assigns moderate draw probability when teams are similarly profiled but cannot distinguish "close draw" from "one-sided win."

**Static style vectors:** Vectors are averaged across all historical matches. A team peaking in form (Spain under new management, Morocco at 2022 WC) vs their historical average creates systematic bias. `form_points_delta` partially corrects this, but cannot fully capture squad changes, tactical evolution, or injuries.

**Neutral venue data:** Most StatsBomb matches are World Cup/Euros at neutral venues. The `is_home` feature has reduced power (2.9% gain) because true home advantage is rare in the training data. This is appropriate for WC2026 predictions but would need recalibration for domestic league use.

---

## Phase 5 Preview

Phase 5 adds full SHAP TreeExplainer analysis — per-prediction feature attribution so every output explains *which tactical factors* drove the probability. For example, "Spain wins because their PPDA advantage (+1.2 scaled delta) and xG creation edge (+0.8) outweigh Morocco's defensive organisation." Phase 5 also builds the **upset detector**: a secondary classifier that flags matches where the model's confident prediction is likely wrong, based on historical patterns of tactical upsets.
