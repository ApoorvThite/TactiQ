"""Phase 5 orchestration — SHAP explainability, upset detector, DB storage."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
MODELS_DIR    = ROOT / 'models'
FIGURES_DIR   = ROOT / 'docs' / 'figures'
PROCESSED_DIR = ROOT / 'data' / 'processed'

from src.models.shap_explainer import (
    fit_explainer, global_analysis, waterfall_and_summary,
    FEATURE_NAMES,
)
from src.models.upset_detector  import run_upset_detector, WC2026_GROUP_FIXTURES, LABEL_NAMES
from src.models.explain_matchup import run_all_narratives


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_team_db(name, cur):
    cur.execute(
        "SELECT team_id, team_name, archetype_name, style_vector, matches_played "
        "FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)", (name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    return {'team_id': row[0], 'team_name': row[1], 'archetype_name': row[2],
            'style_vector': np.array(row[3]), 'matches_played': row[4]}


def save_shap_to_db(shap_rows):
    """Insert SHAP results for all matchups into matchup_shap_values."""
    conn = _get_conn()
    cur  = conn.cursor()

    # Apply schema
    ddl = open(ROOT / 'db' / 'schema' / '004_shap_tables.sql').read()
    cur.execute(ddl)
    conn.commit()

    cur.execute('DELETE FROM matchup_shap_values')

    for r in shap_rows:
        cur.execute(
            """INSERT INTO matchup_shap_values
               (team_a_id, team_b_id, team_a_name, team_b_name, predicted_class,
                p_win, p_draw, p_loss,
                shap_values_win, shap_values_draw, shap_values_loss,
                top_feature_win, top_feature_draw,
                is_upset_candidate, upset_explanation)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (r.get('team_a_id'), r.get('team_b_id'),
             r['team_a_name'], r['team_b_name'],
             r['predicted_class'], r['p_win'], r['p_draw'], r['p_loss'],
             json.dumps(r['shap_win']), json.dumps(r['shap_draw']), json.dumps(r['shap_loss']),
             r.get('top_feature_win'), r.get('top_feature_draw'),
             r.get('is_upset_candidate', False),
             r.get('upset_explanation', ''))
        )

    conn.commit()
    count = len(shap_rows)
    cur.close()
    conn.close()
    print(f'Inserted {count} rows into matchup_shap_values')
    return count


def main():
    print('\n' + '='*60)
    print(' TACTIQ — Phase 5: SHAP Explainability & Upset Detector')
    print('='*60)

    # ── Step 1: Fit SHAP explainer ───────────────────────────────────────
    print('\n' + '─'*60)
    print(' Step 1 — Fit SHAP TreeExplainer')
    print('─'*60)
    model_raw, explainer, shap_values, X, df = fit_explainer()

    # ── Step 2: Global analysis ──────────────────────────────────────────
    print('\n' + '─'*60)
    print(' Step 2 — Global SHAP Analysis')
    print('─'*60)
    mean_abs, top3_feat = global_analysis(explainer, shap_values, X)

    # ── Step 3: Waterfall plots for 5 test matchups ──────────────────────
    print('\n' + '─'*60)
    print(' Step 3 — Per-Matchup Waterfall Plots')
    print('─'*60)

    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        model_calib = pickle.load(f)

    conn = _get_conn()
    cur  = conn.cursor()

    test_matchups = [
        ('Spain',       'Morocco',    'fig12_waterfall_spain_morocco.png'),
        ('Germany',     'Brazil',     'fig13_waterfall_germany_brazil.png'),
        ('France',      'Argentina',  'fig14_waterfall_france_argentina.png'),
        ('England',     'Iran',       'fig15_waterfall_england_iran.png'),
        ('Netherlands', 'Ecuador',    'fig16_waterfall_netherlands_ecuador.png'),
    ]

    waterfall_results = []
    for a_name, b_name, fig_name in test_matchups:
        team_a = _load_team_db(a_name, cur)
        team_b = _load_team_db(b_name, cur)
        if team_a is None or team_b is None:
            print(f'Skipping {a_name} vs {b_name} — team not found')
            continue
        result = waterfall_and_summary(
            explainer, model_raw, model_calib,
            team_a, team_b,
            FIGURES_DIR / fig_name,
        )
        waterfall_results.append(result)
        print(f'Saved → docs/figures/{fig_name}')

    cur.close()
    conn.close()

    # ── Step 4: Upset detector ───────────────────────────────────────────
    print('\n' + '─'*60)
    print(' Step 4 — Upset Detector')
    print('─'*60)
    upset_rows = run_upset_detector(explainer, model_calib)

    # ── Step 5: Store SHAP values to PostgreSQL ──────────────────────────
    print('\n' + '─'*60)
    print(' Step 5 — Storing SHAP Values to PostgreSQL')
    print('─'*60)

    # Combine waterfall results + upset_rows (de-duplicate by team pair)
    all_shap_rows = []

    for r in waterfall_results:
        all_shap_rows.append({
            'team_a_id':       r.get('team_a_id'),
            'team_b_id':       r.get('team_b_id'),
            'team_a_name':     r['team_a_name'],
            'team_b_name':     r['team_b_name'],
            'predicted_class': r['predicted_class'],
            'p_win':           r['p_win'],
            'p_draw':          r['p_draw'],
            'p_loss':          r['p_loss'],
            'shap_win':        r['shap_win'],
            'shap_draw':       r['shap_draw'],
            'shap_loss':       r['shap_loss'],
            'top_feature_win':  max(r['shap_win'],  key=lambda k: abs(r['shap_win'][k])),
            'top_feature_draw': max(r['shap_draw'], key=lambda k: abs(r['shap_draw'][k])),
            'is_upset_candidate': False,
            'upset_explanation':  '',
        })

    for r in upset_rows:
        probs  = [r['p_win_a'], r['p_draw'], r['p_loss_a']]
        pred_c = LABEL_NAMES[int(np.argmax(probs))]
        all_shap_rows.append({
            'team_a_id':       r.get('team_a_id'),
            'team_b_id':       r.get('team_b_id'),
            'team_a_name':     r['team_a'],
            'team_b_name':     r['team_b'],
            'predicted_class': pred_c,
            'p_win':           r['p_win_a'],
            'p_draw':          r['p_draw'],
            'p_loss':          r['p_loss_a'],
            'shap_win':        r['shap_win'],
            'shap_draw':       r['shap_draw'],
            'shap_loss':       r['shap_loss'],
            'top_feature_win':  r.get('top_feature_win', ''),
            'top_feature_draw': r.get('top_feature_draw', ''),
            'is_upset_candidate': r['is_upset_candidate'],
            'upset_explanation':  r.get('upset_explanation', ''),
        })

    db_count = save_shap_to_db(all_shap_rows)

    # ── Step 6: Tactical narratives ──────────────────────────────────────
    print('\n' + '─'*60)
    print(' Step 6 — Tactical Explanation Engine')
    print('─'*60)
    narratives = run_all_narratives()

    # ── Final summary ────────────────────────────────────────────────────
    upset_candidates = [r for r in upset_rows if r['is_upset_candidate']]
    top_upset = max(upset_candidates, key=lambda r: r['p_not_fav_win'], default=None)

    print('\n')
    print('=' * 60)
    print(' TACTIQ — Phase 5 Explainability & Upset Detector Complete')
    print('=' * 60)

    top_win_feat  = FEATURE_NAMES[int(np.argmax(mean_abs['win']))]
    top_draw_feat = FEATURE_NAMES[int(np.argmax(mean_abs['draw']))]
    print('\nSHAP ANALYSIS')
    print(f'  Training rows explained  : 460')
    win_idx  = int(np.argmax(mean_abs['win']))
    draw_idx = int(np.argmax(mean_abs['draw']))
    print(f'  Top global feature (win) : {top_win_feat}  '
          f'(mean |SHAP| = {mean_abs["win"][win_idx]:.4f})')
    print(f'  Top global feature (draw): {top_draw_feat}  '
          f'(mean |SHAP| = {mean_abs["draw"][draw_idx]:.4f})')

    print(f'\nWATERFALL PLOTS GENERATED: {len(waterfall_results)} / 5 '
          f'{"✓" if len(waterfall_results) == 5 else "!"}')

    print(f'\nUPSET DETECTOR')
    print(f'  WC2026 matchups analysed : {len(upset_rows)}')
    print(f'  Upset candidates found   : {len(upset_candidates)}')
    if top_upset:
        print(f'  Highest upset risk       : {top_upset["team_a"]} vs {top_upset["team_b"]} '
              f'(p_not_favourite_win = {top_upset["p_not_fav_win"]:.2f})')

    print(f'\nTACTICAL NARRATIVES: {sum(1 for n in narratives if n)} / 5 '
          f'{"✓" if sum(1 for n in narratives if n) == 5 else "!"}')

    print(f'\nDATABASE')
    print(f'  matchup_shap_values rows : {db_count}')

    print('\nSAVED ARTIFACTS')
    artifacts = [
        ('models/shap_explainer.pkl',                   MODELS_DIR / 'shap_explainer.pkl'),
        ('data/processed/upset_watchlist.csv',          PROCESSED_DIR / 'upset_watchlist.csv'),
        ('docs/figures/fig8_shap_bar_global.png',       FIGURES_DIR / 'fig8_shap_bar_global.png'),
        ('docs/figures/fig9_shap_beeswarm_win.png',     FIGURES_DIR / 'fig9_shap_beeswarm_win.png'),
        ('docs/figures/fig10_shap_heatmap.png',         FIGURES_DIR / 'fig10_shap_heatmap.png'),
        ('docs/figures/fig11_shap_interactions.png',    FIGURES_DIR / 'fig11_shap_interactions.png'),
        ('docs/figures/fig12_waterfall_spain_morocco.png',      FIGURES_DIR / 'fig12_waterfall_spain_morocco.png'),
        ('docs/figures/fig13_waterfall_germany_brazil.png',     FIGURES_DIR / 'fig13_waterfall_germany_brazil.png'),
        ('docs/figures/fig14_waterfall_france_argentina.png',   FIGURES_DIR / 'fig14_waterfall_france_argentina.png'),
        ('docs/figures/fig15_waterfall_england_iran.png',       FIGURES_DIR / 'fig15_waterfall_england_iran.png'),
        ('docs/figures/fig16_waterfall_netherlands_ecuador.png', FIGURES_DIR / 'fig16_waterfall_netherlands_ecuador.png'),
    ]
    for label, path in artifacts:
        exists = path.exists() and path.stat().st_size > 100
        print(f'  {label:<52} {"✓" if exists else "✗"}')

    print('\n' + '=' * 60)
    print(' Ready for Phase 6: WC 2026 Team Integration')
    print('=' * 60)


if __name__ == '__main__':
    import sys as _sys
    if '--upset-only' in _sys.argv:
        # Skip SHAP fitting / global plots / waterfall — reload saved artifacts
        import pickle as _pkl
        print('\n[--upset-only] Loading saved models...')
        with open(MODELS_DIR / 'xgboost_matchup.pkl',    'rb') as _f: _raw = _pkl.load(_f)
        with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as _f: _cal = _pkl.load(_f)
        with open(MODELS_DIR / 'shap_explainer.pkl',     'rb') as _f: _exp = _pkl.load(_f)
        _rows = run_upset_detector(_exp, _cal)
        # Refresh DB
        _db_rows = []
        for _r in _rows:
            _probs  = [_r['p_win_a'], _r['p_draw'], _r['p_loss_a']]
            _pred_c = LABEL_NAMES[int(np.argmax(_probs))]
            _db_rows.append({
                'team_a_id': _r.get('team_a_id'), 'team_b_id': _r.get('team_b_id'),
                'team_a_name': _r['team_a'], 'team_b_name': _r['team_b'],
                'predicted_class': _pred_c,
                'p_win': _r['p_win_a'], 'p_draw': _r['p_draw'], 'p_loss': _r['p_loss_a'],
                'shap_win': _r['shap_win'], 'shap_draw': _r['shap_draw'], 'shap_loss': _r['shap_loss'],
                'top_feature_win': _r.get('top_feature_win', ''),
                'top_feature_draw': _r.get('top_feature_draw', ''),
                'is_upset_candidate': _r['is_upset_candidate'],
                'upset_explanation': _r.get('upset_explanation', ''),
            })
        _n = save_shap_to_db(_db_rows)
        print(f'DB updated: {_n} rows in matchup_shap_values')
    else:
        main()
