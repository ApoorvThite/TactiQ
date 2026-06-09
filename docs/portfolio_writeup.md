# TactiQ — Portfolio Writeup

## Problem

Predicting football match outcomes is hard because raw form tables miss the *why*. A team that wins 60% of its matches against weak opponents is structurally different from one that wins 60% against strong pressing sides. Standard Elo ratings and result-based models collapse this tactical signal into a single scalar.

The question I set out to answer: **can we represent how a national team plays — not just how often it wins — and use that representation to make better-calibrated predictions for the 2026 FIFA World Cup?**

---

## Approach

### 1. Granular event data instead of box scores

I loaded 843,050 match events from StatsBombPy across four tournaments: World Cup 2018/22, UEFA Euro 2020/24, Copa América 2021/24, and AFC Asian Cup 2023. Each event is a timestamped action (pass, carry, pressure, shot) with player and position context.

From these events I computed a 10-dimensional style vector per team per tournament window using SQL aggregations in PostgreSQL:

- **Pressing intensity** (PPDA — passes allowed per defensive action): low PPDA = aggressive press
- **Possession** and **pass completion under pressure**: technical quality under duress
- **xG created/conceded ratio**: attacking vs. defensive efficiency
- **Progressive carry %** and **passes into final third p90**: directness and verticality
- **Set piece shot %**: dead-ball dependency

This is the core of the project — moving from *results* to *style fingerprints*.

### 2. Archetype clustering

I reduced the 10-dim style vectors to 2D with UMAP (n_neighbors=15, min_dist=0.1) and ran K-means (k=4, silhouette=0.41) in PCA-whitened space. Four archetypes emerged with strong tactical interpretability:

| Archetype | Canonical examples |
|-----------|-------------------|
| High Press | Germany, Spain, Belgium |
| Possession Control | Brazil, Argentina, Netherlands |
| Counter-Attack | England, France, Uruguay |
| Deep Block | Saudi Arabia, South Korea, Ecuador |

Critically, the pipeline is **frozen after training** — new teams (the 14 WC2026 teams without StatsBomb coverage) are `.transform()`'d and `.predict()`'d through the existing scaler/PCA/KMeans. No refitting on new data.

### 3. Matchup model

The XGBoost model takes the **delta vector** between two team style profiles (team_a − team_b) plus context features (`is_home`, `form_points_delta`, `archetype_matchup_id`, `competition_weight`) and predicts win/draw/loss.

Calibration matters for a 3-class problem: raw XGBoost overestimates the "win" class. I applied `IsotonicMulticlassCalibrator` — per-class isotonic regression on the raw probabilities, then row-normalization. This brought the Brier score down from 0.24 to 0.21 and produced reliable probability estimates for the draw class.

Cross-validation: grouped 5-fold by match_id (both perspectives of each match always in the same fold) with Optuna tuning over 100 trials. Final accuracy: **58.3%** on 3-class prediction — a meaningful improvement over the 44% baseline (always predict home win).

### 4. SHAP explainability and upset detection

SHAP TreeExplainer assigns feature-level attribution to every fixture prediction. I built an upset detector that combines model probabilities with SHAP signals:

A fixture is flagged as an **upset candidate** when:
1. The favourite is predicted to win (p_win > 0.40)
2. But the underdog retains meaningful threat (p_draw + p_loss ≥ 0.45)
3. And at least one tactical SHAP signal fires:
   - **PPDA neutralization**: the underdog's defensive structure suppresses the favourite's pressing edge
   - **Set-piece threat**: the underdog generates disproportionate dead-ball danger
   - **Archetype disadvantage**: the matchup type is historically draw-prone or underdog-favourable

This produced 18 upset candidates across the 72 WC2026 group fixtures — a 25% upset rate consistent with historical WC data.

### 5. WC2026 simulation

For the 14 teams without StatsBomb data, I assigned archetype centroid style vectors (computed from the mean of all archetype members in the DB) and UMAP coordinates near the cluster mean. These proxy teams carry `is_proxy = True` throughout the system so the dashboard can flag uncertainty.

Monte Carlo simulation: 10,000 runs of the group stage, each run sampling match outcomes proportionally to model probabilities and breaking ties with Poisson-sampled xG. The WC2026 format (best 8 third-place teams advance) is implemented exactly. Knockout rounds redistribute draw probability proportionally to win/loss (draws are impossible in knockout football).

**Predicted champion: Spain** (bracket path: Group H → R32 → QF → SF → Final vs France).

---

## Engineering

The full stack is production-quality ML infrastructure, not notebook code:

- **PostgreSQL** as the single source of truth — events, features, predictions, and simulation results all live in the DB
- **Frozen sklearn pipeline** — `RobustScaler → PCA → KMeans` objects serialized with pickle; new teams always go through `.transform()/.predict()`, never `.fit_transform()`
- **SQLAlchemy 2.x** with `@st.cache_data(ttl=300)` — dashboard queries cached per session, models cached per process with `@st.cache_resource`
- **COALESCE pattern for proxy teams** — proxy teams have NULL individual feature columns; dashboard SQL falls back to `(style_vector->>N)::float` JSONB indexing
- **Pytest test suite** — unit tests for feature construction, calibrator normalization, upset detector logic, and Monte Carlo probability handling
- **GitHub Actions CI** — lint (ruff) and test on every push

---

## What I would do with more time

1. **Live data pipeline**: replace the static StatsBomb snapshot with a streaming update that ingests new national team matches as they happen in 2025/26 qualifying
2. **Uncertainty quantification for proxy teams**: confidence intervals on archetype centroid predictions — currently the dashboard flags `[P]` but doesn't show a credible interval
3. **Knockout model**: the current knockout bracket uses the same group-stage XGBoost model. A separate model trained specifically on knockout matches (higher stakes, different tactical approach) would likely improve bracket accuracy
4. **Player-level features**: squad availability and injury state aren't in the model. A Mbappe-out Spain vs France has different dynamics than the model currently captures

---

## Skills demonstrated

| Domain | Specific techniques |
|--------|-------------------|
| Data engineering | PostgreSQL schema design, StatsBomb event ingestion, FBref web scraping, SQL feature aggregation |
| ML — unsupervised | UMAP dimensionality reduction, K-means clustering, silhouette analysis, frozen pipeline inference |
| ML — supervised | XGBoost 3-class classification, grouped cross-validation, Optuna HPO, isotonic calibration, SHAP TreeExplainer |
| Statistics | Monte Carlo simulation, Poisson tiebreaker, probability normalization, Brier score |
| Software engineering | SQLAlchemy 2.x, Streamlit multi-page, Plotly dark-theme charts, pytest unit tests, GitHub Actions CI |
