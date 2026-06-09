# TactiQ ($TDNA)

**Tactical DNA matchup prediction engine for FIFA World Cup 2026.**

TactiQ ingests 843,050 match events from 4 major international tournaments, engineers 10-dimensional tactical style vectors per national team, clusters teams into playing archetypes using UMAP + K-means, and predicts World Cup 2026 match outcomes from style-vector matchups using a calibrated XGBoost model — with SHAP explanations and an upset detector surfaced in an interactive Streamlit dashboard.

---

## Architecture

```
StatsBomb events (WC18/22, Euro20/24, Copa21/24, AFC23)
        │
        ▼
PostgreSQL (raw_events, match_results)
        │
        ▼
SQL feature pipeline → team_style_profiles (10-dim vectors)
        │
        ├──► UMAP + K-means → 4 archetypes (54 StatsBomb teams)
        │        High Press · Possession Control · Counter-Attack · Deep Block
        │
        ├──► XGBoost matchup model (delta vectors → win/draw/loss)
        │        Calibrated with IsotonicRegression per class
        │
        ├──► SHAP TreeExplainer → feature attribution per fixture
        │        Upset detector: 3 tactical signals
        │
        └──► Monte Carlo (10,000 sims) → WC2026 qualification probabilities
                 → Knockout bracket prediction (Spain champion)
                 → Streamlit dashboard (5 pages)
```

---

## Results

| Metric | Value |
|--------|-------|
| Cross-val accuracy | 58.3% (3-class) |
| Log loss (calibrated) | 0.97 |
| Brier score | 0.21 |
| Archetypes | 4 (K-means, silhouette = 0.41) |
| WC2026 teams covered | 48 (34 StatsBomb + 14 proxy) |
| Group stage fixtures predicted | 72 |
| Upset candidates detected | ~18 (25% rate) |
| Predicted champion | **Spain** |

---

## Setup

### Prerequisites

- Python 3.10+ (Anaconda recommended)
- PostgreSQL 14+
- StatsBomb free data access (via `statsbombpy`)

### 1. Clone and install

```bash
git clone https://github.com/yourname/tactiq.git
cd tactiq
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tactiq
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
```

### 3. Create the database

```bash
createdb tactiq
python db/schema/create_schema.py
```

### 4. Ingest data

```bash
# StatsBomb event data (WC 2018/22, Euro 2020/24, Copa 2021/24, AFC 2023) — ~10 min
python src/ingestion/load_statsbomb.py

# Kaggle historical results (2010–2024) — download results.csv first
# Place at: data/raw/results.csv
python src/ingestion/load_kaggle_results.py

# FBref tactical stats (WC2026 proxy teams)
python src/ingestion/scrape_fbref.py

# Validate data integrity
python src/ingestion/validate_data.py
```

### 5. Run the full pipeline

```bash
# Phase 2 — SQL feature engineering
python src/features/run_feature_pipeline.py

# Phase 3 — UMAP clustering
python src/models/run_style_profiler.py

# Phase 4 — XGBoost model
python src/models/run_matchup_model.py

# Phase 5 — SHAP + upset detector
python src/models/run_explainability.py

# Phase 6 — WC2026 integration (48 teams, Monte Carlo, bracket)
python src/models/run_wc2026_integration.py
```

### 6. Launch the dashboard

```bash
python -m streamlit run src/dashboard/app.py --server.port=8501
```

Open `http://localhost:8501`

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | KPI overview, UMAP scatter, Spain champion card, upset alerts |
| **Team DNA Explorer** | Radar chart, style rankings, UMAP comparison for any WC2026 team |
| **Matchup Predictor** | Live XGBoost + SHAP prediction for any head-to-head |
| **Group Stage** | 72 fixtures across 12 group tabs with qualification probabilities |
| **Upset Watchlist** | Tactical mismatches where lower-ranked teams hold structural edges |

---

## Feature Engineering

Each team is represented as a 10-dimensional style vector:

| Feature | Description |
|---------|-------------|
| `avg_possession_pct` | Mean ball possession percentage |
| `avg_ppda` | Passes allowed per defensive action (press intensity proxy) |
| `avg_pressure_success_rate` | % of pressing sequences that win the ball |
| `avg_xg_created_p90` | Expected goals created per 90 minutes |
| `avg_xg_ratio` | xG created / xG conceded ratio |
| `avg_progressive_carry_pct` | % of carries that advance ball 10+ yards toward goal |
| `avg_pass_completion_pct` | Overall pass accuracy |
| `avg_passes_final_third_p90` | Passes into attacking third per 90 |
| `avg_pass_completion_under_pressure_pct` | Pass accuracy when under pressure |
| `avg_set_piece_shot_pct` | % of shots from set pieces |

Matchup features are delta vectors (team_a − team_b) plus context: `is_home`, `form_points_delta`, `archetype_matchup_id`, `delta_matches_played`, `competition_weight`.

---

## Model

**XGBoost** with Optuna hyperparameter tuning (100 trials, grouped 5-fold CV).

Calibrated with `IsotonicMulticlassCalibrator` — per-class isotonic regression applied to raw XGBoost probabilities, then row-normalized. This corrects the systematic overconfidence on the "win" class.

SHAP `TreeExplainer` provides feature attribution for every prediction. The upset detector flags fixtures where:
1. Team A is predicted to win (p_win > 0.40)
2. Team B is not High Press
3. Team B's p_draw + p_loss ≥ 0.45
4. At least one SHAP signal is triggered (PPDA neutralization, set-piece threat, archetype disadvantage)

---

## Proxy Teams

14 WC2026 teams lack sufficient StatsBomb coverage (<3 qualifying matches). These teams are assigned the centroid style vector of their designated archetype (computed from all archetype members in the DB), a UMAP position near archetype cluster mean, and `is_proxy = True` throughout the system.

| Team | Archetype |
|------|-----------|
| South Africa | Deep Block |
| Bosnia and Herzegovina | Counter-Attack |
| Haiti | Counter-Attack |
| Paraguay | Deep Block |
| Curacao | Counter-Attack |
| Ivory Coast | Counter-Attack |
| New Zealand | Deep Block |
| Cape Verde | Counter-Attack |
| Iraq | Deep Block |
| Norway | Counter-Attack |
| Algeria | Counter-Attack |
| Jordan | Deep Block |
| Congo DR | Counter-Attack |
| Uzbekistan | Deep Block |

---

## Project Structure

```
tactiq/
├── data/
│   ├── raw/                   Kaggle results.csv, FBref CSVs
│   └── processed/             team_similarity_matrix.csv, upset_watchlist.csv
├── db/
│   └── schema/                001–005 SQL schema files + create_schema.py
├── docs/
│   ├── figures/               All phase output figures (fig1–fig21)
│   └── phase_*.md             Per-phase technical documentation
├── models/                    xgboost_calibrated.pkl, shap_explainer.pkl,
│                              scaler.pkl, pca.pkl, kmeans.pkl, umap_model.pkl
├── src/
│   ├── ingestion/             StatsBomb, Kaggle, FBref loaders + validator
│   ├── features/              SQL feature pipeline runner
│   ├── models/                Style profiler, matchup model, explainability,
│   │                          WC2026 integration scripts
│   └── dashboard/             Streamlit app (app.py + pages/ + utils/)
└── tests/                     Pytest unit tests
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Feature vector construction (delta computation, shape, bounds)
- `IsotonicMulticlassCalibrator` probability normalization
- Upset detector logic (all signal conditions)
- Monte Carlo probability normalization
- Dashboard query result shapes (mocked DB)

---

## Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| StatsBomb Open Data | WC 2018/22, Euro 2020/24, Copa América 2021/24, AFC Asian Cup 2023 | `statsbombpy` (free) |
| Kaggle Football Results | International matches 1872–2024 | `results.csv` download required |
| FBref | WC2026 proxy team tactical stats | Web scraping (`requests` + `BeautifulSoup`) |

---

## License

MIT
