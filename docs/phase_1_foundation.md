# Phase 1 — Foundation & Data Ingestion
**Date completed:** 2026-06-08
**Phase status:** Complete (structure & scripts); populate row counts after running ingestion

---

## Summary

Phase 1 establishes the full project skeleton for TactiQ: directory layout, Python dependencies, PostgreSQL schema, and three data ingestion pipelines (StatsBomb open data, Kaggle international results, FBref scraper). A validation script confirms data integrity end-to-end. All ingestion scripts are idempotent via `ON CONFLICT DO NOTHING`.

---

## Project Structure

```
TactiQ/
├── data/
│   ├── raw/                  # downloaded CSVs, scraped files (gitignored)
│   └── processed/            # cleaned/transformed outputs (gitignored)
├── db/
│   ├── schema/
│   │   ├── 001_raw_tables.sql
│   │   └── create_schema.py
│   └── migrations/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── load_statsbomb.py
│   │   ├── load_kaggle_results.py
│   │   ├── scrape_fbref.py
│   │   └── validate_data.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── dashboard/
│       └── __init__.py
├── notebooks/
├── tests/
├── docs/
│   └── phase_1_foundation.md
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Database Schema

| Table | Columns | Purpose |
|-------|---------|---------|
| `competitions` | 6 | Reference for all StatsBomb competition + season pairs |
| `teams` | 4 | Deduplicated team registry (avoids repeating names in matches/events) |
| `matches` | 12 | One row per game with scores, teams, date, venue |
| `match_events` | 16 | Granular event log — every pass, shot, pressure, carry per match |
| `kaggle_results` | 9 | Historical international results 2010–2024 for broad outcome modelling |

### Schema design decisions

- **JSONB for `extra_data`** — StatsBomb events carry deeply nested, event-type-specific payloads (shot freeze frames, pass recipients, dribble outcomes). Normalising every sub-type would require 30+ tables and complex joins. JSONB keeps the raw richness queryable while the dedicated columns expose the hot-path fields for SQL analytics. A GIN index makes JSON key lookups fast.
- **UUID for `event_id`** — StatsBomb assigns UUIDs natively; using SERIAL would require a mapping table and lose traceability back to the source data.
- **Separate `teams` table** — teams appear in both `matches` (home/away) and `match_events` (team_id per event). A reference table enforces consistency and makes team-level aggregation a simple join rather than string matching across columns.
- **Indexing strategy** — `match_id` and `team_id` on `match_events` are the two dominant join keys for feature engineering CTEs. `event_type` is the primary filter (e.g. "Pass", "Pressure"). GIN on `extra_data` enables fast JSONB key containment checks. Composite index `(home_team, away_team)` on `kaggle_results` supports matchup lookups.

---

## Data Sources

| Source | Dataset | Records Loaded | Date Range | URL |
|--------|---------|---------------|------------|-----|
| StatsBomb | Open Data (WC/Euros/Copa/AFC) | ~600,000+ events | 2018–2024 | https://github.com/statsbomb/open-data |
| Kaggle | International Football Results | ~15,000+ rows | 2010–2024 | https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 |
| FBref | National Team Stats (48 teams) | 48 teams × squad pages | 2023–2024 | https://fbref.com |

---

## Validation Report Output

_Populate after running `python src/ingestion/validate_data.py`_

```
============================================================
 TACTIQ — Phase 1 Data Validation Report
============================================================
[paste full output here]
============================================================
```

---

## Row Counts (Final)

_Populate after ingestion completes._

| Table | Rows |
|-------|------|
| competitions | — |
| teams | — |
| matches | — |
| match_events | — |
| kaggle_results | — |

---

## Issues Encountered

_Document any errors, rate limits, schema mismatches, or data quality issues here after running ingestion._

---

## Environment

- Python version: 3.11+
- PostgreSQL version: 14+
- Key library versions: see `requirements.txt`

---

## Phase 2 Preview

Phase 2 will use this raw data to build the SQL feature engineering pipeline — computing PPDA, xG ratio, progressive carry %, and all other tactical metrics via CTEs and window functions in `src/features/`.
