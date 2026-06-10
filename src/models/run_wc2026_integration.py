"""Phase 6 orchestration — WC 2026 Team Integration.

Execution order:
  1. Audit 48 WC2026 teams against DB
  2. Scrape FBref for missing teams (fallback to proxy centroid if < 3 matches)
  3. Assign clusters to all 48 teams, upsert into team_style_profiles
  4. Run 72 group stage predictions with SHAP signals
  5. Monte Carlo 10,000-run group stage simulation → qualification probabilities
  6. Predict knockout bracket from R32 through Final
  7. Save all artifacts to PostgreSQL + CSV/JSON

Run:
    python src/models/run_wc2026_integration.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'
MODELS_DIR    = ROOT / 'models'


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def main():
    print('\n' + '='*62)
    print(' TACTIQ — Phase 6: WC 2026 Team Integration')
    print('='*62)

    # ── Step 1: Audit ────────────────────────────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 1 — Audit WC2026 Teams Against DB')
    print('─'*62)
    from src.models.audit_wc2026_teams import audit_teams
    in_db, missing, stale = audit_teams()

    # ── Step 2: Scrape FBref for missing teams ────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 2 — FBref Scraping for Missing Teams')
    print('─'*62)
    scraped_results = {}
    if missing:
        from src.ingestion.scrape_fbref_national import scrape_missing_teams
        missing_names = [e['team_name'] for e in missing]
        print(f'  Scraping {len(missing_names)} missing teams: {missing_names}')
        scraped_results = scrape_missing_teams(missing_names)
    else:
        print('  All 48 teams already in DB — no scraping required.')

    # Also scrape stale teams if any
    if stale:
        stale_names = [e['team_name'] for e in stale
                       if e['team_name'] not in scraped_results]
        if stale_names:
            print(f'\n  Refreshing {len(stale_names)} stale teams: {stale_names}')
            from src.ingestion.scrape_fbref_national import scrape_missing_teams
            stale_scraped = scrape_missing_teams(stale_names)
            scraped_results.update(stale_scraped)

    scraped_count  = sum(1 for v in scraped_results.values() if not v['is_proxy'])
    proxy_count    = sum(1 for v in scraped_results.values() if v['is_proxy'])
    print(f'\n  FBref scraped (live data) : {scraped_count}')
    print(f'  Proxy centroids used      : {proxy_count}')

    # ── Step 3: Assign clusters ───────────────────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 3 — Cluster Assignment (RobustScaler → PCA → KMeans)')
    print('─'*62)
    from src.models.assign_wc2026_clusters import process_teams
    all_profiles = process_teams(scraped_results, in_db)

    profiles_by_name = {p['team_name']: p for p in all_profiles}

    # ── Step 4: Group stage predictions ──────────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 4 — Group Stage Predictions (72 fixtures)')
    print('─'*62)
    import pickle
    from src.models.predict_group_stage import (
        run_group_stage_predictions, save_predictions_to_db, save_predictions_csv
    )

    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        model_calib = pickle.load(f)
    with open(MODELS_DIR / 'shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)

    conn = _get_conn()
    cur  = conn.cursor()

    predictions = run_group_stage_predictions(profiles_by_name, model_calib, explainer)
    n_pred = save_predictions_to_db(predictions, cur, conn)
    save_predictions_csv(predictions)

    # ── Step 5: Monte Carlo simulation ───────────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 5 — Monte Carlo Group Stage Simulation (10,000 runs)')
    print('─'*62)
    from src.models.simulate_group_stage import (
        run_simulations, print_qualification_table, save_to_db as save_qual_to_db, save_csv
    )

    qual_df = pd.read_csv(PROCESSED_DIR / 'group_stage_predictions.csv')
    qual_results = run_simulations(qual_df)
    print_qualification_table(qual_results)
    save_qual_to_db(qual_results, profiles_by_name)
    save_csv(qual_results)

    # ── Step 6: Knockout bracket ─────────────────────────────────────────────
    print('\n' + '─'*62)
    print(' Step 6 — Knockout Bracket Prediction (R32 → Final)')
    print('─'*62)
    from src.models.predict_knockout_bracket import (
        build_predicted_bracket, print_bracket_summary
    )

    qual_probs_df = pd.read_csv(PROCESSED_DIR / 'qualification_probabilities.csv')
    bracket = build_predicted_bracket(qual_probs_df, model_calib, cur)
    print_bracket_summary(bracket)

    out_bracket = PROCESSED_DIR / 'predicted_bracket.json'
    with open(out_bracket, 'w') as f:
        json.dump(bracket, f, indent=2)
    print(f'\n  Saved → data/processed/predicted_bracket.json')

    cur.close()
    conn.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    n_in_db   = len(in_db)
    n_missing = len(missing)
    n_proxy   = proxy_count + sum(1 for e in missing if e['team_name'] not in scraped_results)
    n_direct  = n_pred
    upset_count = sum(1 for p in predictions if p.get('is_upset_candidate'))

    top_qual = sorted(qual_results.values(), key=lambda x: -x['p_qualify_r32'])[:5]

    print('\n\n' + '='*62)
    print(' TACTIQ — Phase 6 WC2026 Integration Complete')
    print('='*62)

    print('\nTEAM COVERAGE')
    print(f'  WC2026 qualified teams   : 48')
    print(f'  Teams in DB (StatsBomb)  : {n_in_db}')
    print(f'  Teams scraped (FBref)    : {scraped_count}')
    print(f'  Teams using proxy vector : {proxy_count}')

    print('\nGROUP STAGE PREDICTIONS')
    print(f'  Fixtures predicted       : {n_direct}')
    print(f'  Upset candidates         : {upset_count}')

    print('\nMONTE CARLO SIMULATION')
    print(f'  Simulation runs          : 10,000')
    print(f'  Top-5 qualification odds :')
    for r in top_qual:
        print(f'    {r["team_name"]:<25} (Group {r["group"]}) '
              f'p(R32) = {r["p_qualify_r32"]*100:.1f}%')

    print(f'\nKNOCKOUT BRACKET')
    print(f'  Predicted champion       : {bracket["Champion"]}')

    print('\nSAVED ARTIFACTS')
    artifacts = [
        ('data/processed/wc2026_teams_in_db.json',        PROCESSED_DIR / 'wc2026_teams_in_db.json'),
        ('data/processed/wc2026_teams_missing.json',       PROCESSED_DIR / 'wc2026_teams_missing.json'),
        ('data/processed/group_stage_predictions.csv',     PROCESSED_DIR / 'group_stage_predictions.csv'),
        ('data/processed/qualification_probabilities.csv', PROCESSED_DIR / 'qualification_probabilities.csv'),
        ('data/processed/predicted_bracket.json',          PROCESSED_DIR / 'predicted_bracket.json'),
    ]
    for label, path in artifacts:
        exists = path.exists() and path.stat().st_size > 10
        print(f'  {label:<52} {"✓" if exists else "✗"}')

    print('\nDATABASE')
    print(f'  wc2026_group_predictions rows      : {n_direct}')
    print(f'  wc2026_qualification_probs rows    : {len(qual_results)}')

    print('\n' + '='*62)
    print(' Ready for Phase 7: Interactive Dashboard')
    print('='*62)


if __name__ == '__main__':
    main()
