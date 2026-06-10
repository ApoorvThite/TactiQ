# Phase 7 — Streamlit Dashboard

**Date completed:** 2026-06-09  
**Phase status:** Complete

---

## Summary

Phase 7 delivers a portfolio-grade interactive Streamlit dashboard with 5 pages, a unified dark-theme design system, and live matchup prediction backed by the Phase 4 XGBoost model and Phase 5 SHAP explainer. All pages load from PostgreSQL via SQLAlchemy with `@st.cache_data(ttl=300)` caching. The dashboard runs at `http://localhost:8501`.

---

## Pages Overview

### 1. Home (`app.py`)

**Purpose:** Executive overview of the tournament prediction engine.

**Key elements:**
- 4 KPI metrics: 48 teams, 843K match events, 72 fixtures, 4 archetypes
- Full UMAP scatter of all 48 WC2026 teams colored by archetype
- Predicted champion callout card (Spain) with bracket path summary
- Spain vs France Final probability bar chart
- Top-3 upset candidate alert cards pulled from `wc2026_group_predictions`
- Archetype legend with team counts

**Data sources:** `team_style_profiles`, `wc2026_group_predictions` (upset flag)

---

### 2. Team DNA Explorer (`pages/1_Team_DNA.py`)

**Purpose:** Explore individual team tactical profiles and compare styles.

**Key interactions:**
- Archetype multi-select filter + proxy team toggle in sidebar
- Team selection via dropdown → updates radar chart and stats panel
- Optional comparison team overlay on radar chart
- Most Similar Teams panel: pulls from `team_similarity_matrix.csv` (54×54 cosine similarity from Phase 3); falls back to same-archetype StatsBomb teams for proxy entries

**Visualisations:**
- Filtered UMAP scatter with highlighted team marker (enlarged, label shown)
- 8-axis radar chart (normalised 0–1 across all 48 WC2026 teams)
- Key stats with per-feature rankings (#X / 48)

**Data sources:** `team_style_profiles`, `data/processed/team_similarity_matrix.csv`

---

### 3. Matchup Predictor (`pages/2_Matchup_Predictor.py`)

**Purpose:** Predict any WC2026 head-to-head matchup with full SHAP explainability.

**Key interactions:**
- Two dropdown selectors (default: Spain vs Morocco)
- Live prediction runs on every pair change
- Pre-computed SHAP data from `matchup_shap_values` checked first (faster)
- Falls back to live XGBoost + SHAP inference via `utils/predict.py`

**Visualisations:**
- Full-width result banner colored by predicted outcome
- Horizontal stacked probability bar (win/draw/loss)
- SHAP attribution bar chart (Plotly — interactive, dark-themed)
- Head-to-head stats comparison table
- Dual-team radar overlay (both archetypes, archetype colors)
- Upset alert card if `is_upset_candidate` conditions met

**Data sources:** `team_style_profiles`, `matchup_shap_values`, live models (`xgboost_calibrated.pkl`, `shap_explainer.pkl`)

---

### 4. Group Stage (`pages/3_Group_Stage.py`)

**Purpose:** Browse all 72 WC2026 group stage fixture predictions.

**Key interactions:**
- 12 group tabs (A–L) using `st.tabs()`
- Each tab shows 6 fixture cards with probability bars
- Upset signal badge on relevant fixtures
- Qualification probability mini-bars per team in right column

**Visualisations:**
- 72 probability bars (one per fixture, color-coded win/draw/loss)
- Per-group horizontal qualification bar chart (p_qualify_r32)
- Full-width 12×4 qualification heatmap at bottom (RdYlGn colorscale)

**Data sources:** `wc2026_group_predictions`, `wc2026_qualification_probs`

---

### 5. Upset Watchlist (`pages/4_Upset_Watchlist.py`)

**Purpose:** Surface all matchups where tactical SHAP signals indicate structural upset risk.

**Key interactions:**
- Group filter (multi-select) and minimum p(upset) slider
- Sort by p(upset), group, or underdog archetype
- Upset candidate cards with inline probability bars

**Visualisations:**
- Upset candidate cards with `upset-alert` CSS class (amber left border)
- Per-card probability bar
- Bottom scatter: p(favourite wins) × p(upset) — identifies highest-risk matches
- Summary KPI row: total fixtures, candidates, upset rate, highest-risk matchup

**Data sources:** `wc2026_group_predictions` (is_upset_candidate = TRUE)

---

## Architecture

```
src/dashboard/
├── app.py                     Home page + entry point
├── run.py                     subprocess launcher
├── pages/
│   ├── 1_Team_DNA.py          UMAP explorer + radar
│   ├── 2_Matchup_Predictor.py XGBoost + SHAP live prediction
│   ├── 3_Group_Stage.py       72 fixtures across 12 tabs
│   └── 4_Upset_Watchlist.py   Structural upset detection
├── utils/
│   ├── db.py                  SQLAlchemy + @st.cache_data queries
│   ├── charts.py              7 reusable Plotly functions
│   ├── predict.py             Live prediction + SHAP + narrative
│   ├── styles.py              Color constants + design tokens
│   └── sidebar.py             Shared sidebar renderer
└── assets/
    └── custom.css             Dark theme + badge styles
```

---

## Design Decisions

**Streamlit over Dash or React:**
Streamlit requires zero frontend code — the entire UI is Python. For a portfolio project demonstrating ML engineering (not web engineering), Streamlit maximises the signal-to-noise ratio. Dash requires layout DSL; React requires a full JS build pipeline. Streamlit 1.58 has native multi-page support, tabs, and `st.plotly_chart` interactivity.

**`@st.cache_data` for DB queries:**
Each page load re-runs the Python script from scratch. Without caching, every interaction would re-execute 5+ SQL queries. `ttl=300` provides a 5-minute cache — long enough for a demo session, short enough that a manual `Rerun` (top-right) will pick up live DB changes.

**`@st.cache_resource` for models:**
The XGBoost calibrated model and SHAP explainer are ~5MB pickle files. `@st.cache_resource` loads them once per Streamlit process (not per session), shared across all users. This makes live prediction fast (<0.5s after warmup).

**Plotly over Altair/matplotlib:**
Plotly charts are interactive (hover, zoom, pan) without additional JS. Altair is elegant but has limited dark-theme support. Matplotlib renders as static PNG in Streamlit — fine for SHAP waterfall images but cannot be styled dynamically. Plotly's `go.Figure` with `paper_bgcolor=TACTIQ_DARK` produces a seamless dark experience.

**Shared sidebar via `render_sidebar()`:**
Streamlit multi-page apps run each page as a completely isolated script. To avoid copy-pasting the sidebar content across 5 files, `utils/sidebar.py` exports a single `render_sidebar()` function imported by every page.

---

## Known Limitations

- **SHAP waterfall is a bar chart approximation:** Native `shap.plots.waterfall()` renders via matplotlib as a static image — it works in Streamlit but cannot be styled to match the dark theme and is not interactive. The Plotly bar chart in `shap_waterfall_table()` is fully interactive and theme-consistent.
- **Proxy team uncertainty not quantified:** The 14 proxy teams use archetype centroid vectors. Their style vectors are less representative than StatsBomb-derived vectors. The `is_proxy` flag surfaces this in the UI, but SHAP confidence intervals are not shown.
- **No live data connection:** Predictions are computed from Phase 6 pre-run data. There is no live StatsBomb feed or real-time squad update pipeline.
- **UMAP click interactivity limited:** Streamlit does not natively map Plotly click events to session state without `streamlit-plotly-events`. A dropdown below the scatter provides reliable team selection instead.
- **Similarity matrix covers 54 teams only:** The `team_similarity_matrix.csv` from Phase 3 covers StatsBomb teams. The 14 WC2026 proxy teams fall back to same-archetype comparisons.

---

## Phase 8 Preview

Phase 8 completes the project portfolio:
- `README.md` — full project description, setup instructions, architecture diagram
- `tests/` — unit tests for feature pipeline, model calibration, and DB queries
- `docs/architecture.png` — system architecture diagram
- `docs/portfolio_writeup.md` — narrative for job applications (problem → approach → results)
- `requirements.txt` / `pyproject.toml` — pinned dependency manifest
- GitHub Actions CI for linting and test runs
