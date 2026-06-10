"""Phase 5 Step 6 — Tactical explanation engine for matchup predictions."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models'

FEATURE_NAMES = [
    'delta_avg_possession_pct', 'delta_avg_ppda', 'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90', 'delta_avg_xg_ratio', 'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct', 'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct', 'delta_avg_set_piece_shot_pct',
    'is_home', 'form_points_delta', 'archetype_matchup_id', 'delta_matches_played',
    'competition_weight',
]

LABEL_NAMES = ['win', 'draw', 'loss']

ARCHETYPES = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}

ARCHETYPE_OUTCOME = {
    'High Press vs Deep Block':             'draw-prone (Deep Block absorbs the press)',
    'High Press vs Counter-Attack':         'upset-prone (Counter-Attack exploits space)',
    'Possession Control vs Deep Block':     'draw-prone (Deep Block frustrates build-up)',
    'Possession Control vs Counter-Attack': 'upset-prone (Counter-Attack on the break)',
    'High Press vs High Press':             'high-variance (both sides press hard)',
    'Counter-Attack vs Counter-Attack':     'tactical stalemate (neither controls the ball)',
    'Deep Block vs High Press':             'underdog-favourable (sitting deep and countering)',
}

FEATURE_NARRATIVES = {
    'delta_avg_xg_ratio': {
        'positive': '{a} creates significantly more danger relative to what they concede',
        'negative': '{b} has a stronger attack/defence balance',
    },
    'delta_avg_ppda': {
        'positive': '{a} presses far less aggressively — {b} will be under intense pressure',
        'negative': '{a} presses harder — {b} will struggle to build from the back',
    },
    'delta_avg_possession_pct': {
        'positive': '{a} dominates possession — controls the tempo',
        'negative': '{b} controls the ball — {a} will defend deep',
    },
    'delta_avg_xg_created_p90': {
        'positive': '{a} generates significantly more xG per match',
        'negative': '{b} creates more dangerous chances per 90',
    },
    'delta_avg_progressive_carry_pct': {
        'positive': '{a} carries the ball forward more directly — dangerous on transitions',
        'negative': '{b} transitions quickly — counter-attack threat',
    },
    'delta_avg_set_piece_shot_pct': {
        'positive': '{a} relies more on dead-ball situations for chances',
        'negative': "{b}'s set-piece dependency is a threat",
    },
    'delta_avg_pressure_success_rate': {
        'positive': '{a} wins the ball back more successfully under pressure',
        'negative': '{b} presses more effectively when they do press',
    },
    'delta_avg_pass_completion_pct': {
        'positive': '{a} is the more technically precise side',
        'negative': '{b} has better technical quality in possession',
    },
    'delta_avg_passes_final_third_p90': {
        'positive': '{a} attacks with higher frequency into the final third',
        'negative': '{b} penetrates more often into dangerous zones',
    },
    'delta_avg_pass_completion_under_pressure_pct': {
        'positive': '{a} is composed under a high press — hard to dispossess',
        'negative': '{b} keeps possession better when pressed',
    },
    'form_points_delta': {
        'positive': '{a} arrives in better recent form',
        'negative': '{b} is the in-form side',
    },
    'archetype_matchup_id': {
        'any': 'The {arch_a} vs {arch_b} matchup is {outcome}',
    },
    'is_home': {
        'positive': '{a} has home advantage',
        'negative': '{a} is playing away from home',
    },
    'delta_matches_played': {
        'positive': '{a} has more tournament experience in the dataset',
        'negative': '{b} has more high-level match data behind them',
    },
    'competition_weight': {
        'any': 'This is a {comp_tier} competition — weighted accordingly in training',
    },
}


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_team(name, cur):
    cur.execute(
        "SELECT team_id, team_name, archetype_name, style_vector, matches_played "
        "FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)", (name,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Team '{name}' not found in team_style_profiles")
    return {
        'team_id': row[0], 'team_name': row[1],
        'archetype_name': row[2],
        'style_vector': np.array(row[3]),
        'matches_played': row[4],
    }


def _narrative_for_feature(feat, shap_val, team_a_name, team_b_name,
                             arch_a, arch_b, x_val=None):
    templates = FEATURE_NARRATIVES.get(feat)
    if templates is None:
        return None

    a, b = team_a_name, team_b_name

    if 'any' in templates:
        arch_key = f'{arch_a} vs {arch_b}'
        outcome  = ARCHETYPE_OUTCOME.get(arch_key, 'variable — no strong historical pattern')
        comp_tier = 'World Cup-level'
        text = templates['any'].format(arch_a=arch_a, arch_b=arch_b,
                                        outcome=outcome, comp_tier=comp_tier)
    elif shap_val > 0:
        text = templates['positive'].format(a=a, b=b)
    else:
        text = templates['negative'].format(a=a, b=b)

    direction = '+' if shap_val > 0 else ''
    return f'{direction}{shap_val:.3f}: {text}'


def explain_matchup(team_a_name: str, team_b_name: str,
                    is_home: bool = True, form_delta: float = 0.0,
                    competition: str = 'FIFA World Cup') -> str:
    """Return a 2-3 sentence tactical narrative for a matchup."""
    conn = _get_conn()
    cur  = conn.cursor()
    team_a = _load_team(team_a_name, cur)
    team_b = _load_team(team_b_name, cur)
    cur.close()
    conn.close()

    COMP_WEIGHTS = {'FIFA World Cup': 1.0, 'UEFA Euro': 0.8}
    sv_a  = team_a['style_vector']
    sv_b  = team_b['style_vector']
    delta = sv_a - sv_b
    arch_a_id  = ARCHETYPES.get(team_a['archetype_name'], 0)
    arch_b_id  = ARCHETYPES.get(team_b['archetype_name'], 0)
    matchup_id = arch_a_id * 4 + arch_b_id
    comp_w     = COMP_WEIGHTS.get(competition, 0.7)
    delta_mp   = team_a['matches_played'] - team_b['matches_played']
    x_vec = np.concatenate([delta, [float(is_home), float(form_delta),
                                    float(matchup_id), float(delta_mp), float(comp_w)]])

    with open(MODELS_DIR / 'xgboost_matchup.pkl', 'rb') as f:
        pickle.load(f)
    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        cal_model = pickle.load(f)
    with open(MODELS_DIR / 'shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)

    X_single = x_vec.reshape(1, -1)
    calib_proba = cal_model.predict_proba(X_single)[0]
    p_win, p_draw, p_loss = float(calib_proba[0]), float(calib_proba[1]), float(calib_proba[2])
    pred_idx = int(np.argmax(calib_proba))
    pred_cls = LABEL_NAMES[pred_idx]

    sv_single = explainer(X_single)
    shap_cls  = sv_single.values[0, :, pred_idx]

    # Top 3 features by |SHAP|
    top3_idx  = np.argsort(np.abs(shap_cls))[::-1][:3]
    top3_feats = [(FEATURE_NAMES[i], shap_cls[i], x_vec[i]) for i in top3_idx]

    # Build narrative sentences
    a, b = team_a_name, team_b_name
    arch_a, arch_b = team_a['archetype_name'], team_b['archetype_name']
    arch_key = f'{arch_a} vs {arch_b}'
    outcome_desc = ARCHETYPE_OUTCOME.get(arch_key, 'tactically variable')

    # Sentence 1: archetype framing
    sent1 = (f'Both teams are {arch_a} archetypes — {outcome_desc}.'
             if arch_a == arch_b else
             f'{a} ({arch_a}) faces {b} ({arch_b}), a matchup that is historically {outcome_desc}.')

    # Sentence 2: top tactical driver
    feat0, sv0, xv0 = top3_feats[0]
    s2_raw = _narrative_for_feature(feat0, sv0, a, b, arch_a, arch_b, xv0)
    sent2 = s2_raw.split(': ', 1)[1] if s2_raw and ': ' in s2_raw else s2_raw or ''

    # Sentence 3: secondary factor + verdict
    feat1, sv1, xv1 = top3_feats[1]
    s3_raw = _narrative_for_feature(feat1, sv1, a, b, arch_a, arch_b, xv1)
    s3_body = s3_raw.split(': ', 1)[1] if s3_raw and ': ' in s3_raw else s3_raw or ''
    winner_str = (f'{a} WIN predicted at {p_win*100:.1f}%' if pred_cls == 'win' else
                  f'{b} WIN predicted at {p_loss*100:.1f}%' if pred_cls == 'loss' else
                  f'DRAW predicted at {p_draw*100:.1f}%')
    sent3 = f'{s3_body}. The model expects {winner_str}.'

    narrative = f'{sent1} {sent2}. {sent3}'

    # Print formatted
    width = 55
    print(f'{team_a_name} vs {team_b_name} — Tactical Breakdown')
    print('─' * width)
    # Word-wrap to 55 chars
    words = narrative.split()
    line, lines = '', []
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + ' ' + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    for line in lines:
        print(line)
    print('─' * width)
    print()

    return narrative


def run_all_narratives():
    test_cases = [
        ('Spain',       'Morocco'),
        ('Germany',     'Brazil'),
        ('France',      'Argentina'),
        ('England',     'Iran'),
        ('Netherlands', 'Ecuador'),
    ]
    results = []
    for a, b in test_cases:
        try:
            r = explain_matchup(a, b)
            results.append(r)
        except Exception as e:
            print(f'ERROR {a} vs {b}: {e}')
            results.append(None)
    return results


if __name__ == '__main__':
    run_all_narratives()
