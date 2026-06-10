"""Phase 6 Step 4 — Run all 72 WC2026 group stage predictions with SHAP signals.

Loads all 48 team profiles, runs the calibrated model on every fixture,
computes SHAP signals for each, stores results in wc2026_group_predictions.
"""

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
PROCESSED_DIR = ROOT / 'data' / 'processed'

from src.models.upset_detector import WC2026_GROUP_FIXTURES

FEATURE_NAMES = [
    'delta_avg_possession_pct', 'delta_avg_ppda', 'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90', 'delta_avg_xg_ratio', 'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct', 'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct', 'delta_avg_set_piece_shot_pct',
    'is_home', 'form_points_delta', 'archetype_matchup_id', 'delta_matches_played',
    'competition_weight',
]

ARCHETYPES  = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}
LABEL_NAMES = ['win', 'draw', 'loss']


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_models():
    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        model_calib = pickle.load(f)
    with open(MODELS_DIR / 'shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return model_calib, explainer


def _load_all_team_profiles(cur):
    """Load team style profiles for all 48 WC2026 teams from DB."""
    # Get all unique team names from the fixtures
    all_names = set()
    for a, b, _ in WC2026_GROUP_FIXTURES:
        all_names.update([a, b])

    profiles = {}
    missing_from_db = []

    for name in all_names:
        cur.execute(
            "SELECT team_id, team_name, archetype_name, style_vector, matches_played "
            "FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)",
            (name,)
        )
        row = cur.fetchone()
        if row:
            profiles[name] = {
                'team_id':        row[0],
                'team_name':      row[1],
                'archetype_name': row[2],
                'style_vector':   np.array(row[3]),
                'matches_played': row[4],
                'is_proxy':       row[4] < 0,
            }
        else:
            missing_from_db.append(name)

    if missing_from_db:
        print(f'  [WARN] {len(missing_from_db)} teams not found in DB: {missing_from_db}')
        print('         Run assign_wc2026_clusters.py first.')

    return profiles


def _build_feature_vector(team_a, team_b, is_home=True, form_delta=0.0):
    sv_a      = np.array(team_a['style_vector'])
    sv_b      = np.array(team_b['style_vector'])
    delta     = sv_a - sv_b
    arch_a_id = ARCHETYPES.get(team_a['archetype_name'], 0)
    arch_b_id = ARCHETYPES.get(team_b['archetype_name'], 0)
    matchup_id = arch_a_id * 4 + arch_b_id
    comp_w    = 1.0  # FIFA World Cup
    delta_mp  = team_a['matches_played'] - team_b['matches_played']
    return np.concatenate([delta, [float(is_home), float(form_delta),
                                   float(matchup_id), float(delta_mp), float(comp_w)]])


def _is_upset_candidate(p_win, p_draw, p_loss, arch_a, arch_b,
                         shap_draw, shap_loss):
    if p_win <= 0.40:
        return False
    if arch_b == 'High Press':
        return False
    if (p_draw + p_loss) < 0.45:
        return False
    ppda_d = shap_draw.get('delta_avg_ppda', 0)
    ppda_l = shap_loss.get('delta_avg_ppda', 0)
    sp_d   = shap_draw.get('delta_avg_set_piece_shot_pct', 0)
    sp_l   = shap_loss.get('delta_avg_set_piece_shot_pct', 0)
    if ppda_d > 0.04 or ppda_l > 0.04 or sp_d > 0.03 or sp_l > 0.03:
        return True
    return False


def run_group_stage_predictions(profiles, model_calib, explainer):
    """Run predictions for all 72 fixtures. Returns list of result dicts."""
    results = []
    missing_teams = []

    print(f'\n  Running {len(WC2026_GROUP_FIXTURES)} group stage predictions...')

    for team_a_name, team_b_name, group in WC2026_GROUP_FIXTURES:
        team_a = profiles.get(team_a_name)
        team_b = profiles.get(team_b_name)

        if team_a is None or team_b is None:
            for n in [team_a_name, team_b_name]:
                if profiles.get(n) is None and n not in missing_teams:
                    missing_teams.append(n)
            results.append({
                'group': group, 'team_a_name': team_a_name, 'team_b_name': team_b_name,
                'predicted_class': 'draw', 'p_win': 0.33, 'p_draw': 0.34, 'p_loss': 0.33,
                'error': 'team_missing',
            })
            continue

        x_vec    = _build_feature_vector(team_a, team_b)
        X_single = x_vec.reshape(1, -1)

        calib = model_calib.predict_proba(X_single)[0]
        p_win, p_draw, p_loss = float(calib[0]), float(calib[1]), float(calib[2])
        pred_idx = int(np.argmax(calib))
        pred_cls = LABEL_NAMES[pred_idx]

        sv_single = explainer(X_single)
        shap_win  = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 0].tolist()))
        shap_draw = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 1].tolist()))
        shap_loss = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 2].tolist()))

        top_feat_win  = max(shap_win,  key=lambda k: abs(shap_win[k]))
        top_feat_draw = max(shap_draw, key=lambda k: abs(shap_draw[k]))

        upset = _is_upset_candidate(
            p_win, p_draw, p_loss,
            team_a['archetype_name'], team_b['archetype_name'],
            shap_draw, shap_loss,
        )
        upset_exp = ''
        if upset:
            arch_key = f'{team_a["archetype_name"]} vs {team_b["archetype_name"]}'
            upset_exp = (
                f'{team_b_name} ({team_b["archetype_name"]}) holds structural edge '
                f'— {arch_key} matchup; pressing signal (PPDA) points toward upset'
            )

        results.append({
            'group':               group,
            'team_a_name':         team_a_name,
            'team_b_name':         team_b_name,
            'team_a_archetype':    team_a['archetype_name'],
            'team_b_archetype':    team_b['archetype_name'],
            'team_a_is_proxy':     team_a.get('is_proxy', False),
            'team_b_is_proxy':     team_b.get('is_proxy', False),
            'predicted_class':     pred_cls,
            'p_win':               round(p_win,  4),
            'p_draw':              round(p_draw, 4),
            'p_loss':              round(p_loss, 4),
            'top_shap_feature_win':  top_feat_win,
            'top_shap_feature_draw': top_feat_draw,
            'shap_values_win':     shap_win,
            'shap_values_draw':    shap_draw,
            'shap_values_loss':    shap_loss,
            'is_upset_candidate':  upset,
            'upset_explanation':   upset_exp,
            'team_a_id':           team_a.get('team_id'),
            'team_b_id':           team_b.get('team_id'),
        })

    if missing_teams:
        print(f'  [WARN] Skipped due to missing profiles: {missing_teams}')

    upset_count = sum(1 for r in results if r.get('is_upset_candidate'))
    print(f'  Predictions complete: {len(results)} fixtures | '
          f'{upset_count} upset candidates')

    return results


def save_predictions_to_db(predictions, cur, conn):
    """Apply schema and insert group stage predictions."""
    ddl = open(ROOT / 'db' / 'schema' / '005_wc2026_tables.sql').read()
    cur.execute(ddl)
    conn.commit()

    cur.execute('DELETE FROM wc2026_group_predictions')

    for r in predictions:
        if r.get('error'):
            continue
        cur.execute(
            """INSERT INTO wc2026_group_predictions
               (group_label, team_a_name, team_b_name, team_a_archetype, team_b_archetype,
                team_a_is_proxy, team_b_is_proxy, predicted_class,
                p_win, p_draw, p_loss,
                top_shap_feature_win, top_shap_feature_draw,
                shap_values_win, shap_values_draw, shap_values_loss,
                is_upset_candidate, upset_explanation)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (r['group'], r['team_a_name'], r['team_b_name'],
             r.get('team_a_archetype'), r.get('team_b_archetype'),
             r.get('team_a_is_proxy', False), r.get('team_b_is_proxy', False),
             r['predicted_class'], r['p_win'], r['p_draw'], r['p_loss'],
             r.get('top_shap_feature_win'), r.get('top_shap_feature_draw'),
             json.dumps(r.get('shap_values_win', {})),
             json.dumps(r.get('shap_values_draw', {})),
             json.dumps(r.get('shap_values_loss', {})),
             r.get('is_upset_candidate', False),
             r.get('upset_explanation', ''))
        )

    conn.commit()
    n = sum(1 for r in predictions if not r.get('error'))
    print(f'  Inserted {n} rows into wc2026_group_predictions')
    return n


def save_predictions_csv(predictions):
    """Save group stage predictions to CSV (without SHAP blobs)."""
    skip = {'shap_values_win', 'shap_values_draw', 'shap_values_loss',
            'team_a_id', 'team_b_id'}
    rows = [{k: v for k, v in r.items() if k not in skip} for r in predictions]
    out = PROCESSED_DIR / 'group_stage_predictions.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('  Saved → data/processed/group_stage_predictions.csv')


if __name__ == '__main__':
    conn = _get_conn()
    cur  = conn.cursor()
    model_calib, explainer = _load_models()
    profiles = _load_all_team_profiles(cur)
    preds = run_group_stage_predictions(profiles, model_calib, explainer)
    save_predictions_to_db(preds, cur, conn)
    save_predictions_csv(preds)
    cur.close()
    conn.close()
