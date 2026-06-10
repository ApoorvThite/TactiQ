"""Phase 4 Step 7 — Matchup prediction function."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models'

STYLE_FEATURES = [
    'avg_possession_pct',
    'avg_ppda',
    'avg_pressure_success_rate',
    'avg_xg_created_p90',
    'avg_xg_ratio',
    'avg_progressive_carry_pct',
    'avg_pass_completion_pct',
    'avg_passes_final_third_p90',
    'avg_pass_completion_under_pressure_pct',
    'avg_set_piece_shot_pct',
]

ARCHETYPES     = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}
LABEL_NAMES    = ['win', 'draw', 'loss']
BAR_WIDTH      = 20
BAR_CHAR       = '█'
BAR_EMPTY_CHAR = '░'


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_team(name: str, cur) -> dict:
    cur.execute(
        """SELECT team_id, team_name, archetype_name, style_vector, matches_played
           FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)""",
        (name,)
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Team '{name}' not found in team_style_profiles")
    return {
        'team_id':        row[0],
        'team_name':      row[1],
        'archetype_name': row[2],
        'style_vector':   np.array(row[3]),
        'matches_played': row[4],
    }


def _bar(prob: float) -> str:
    filled = round(prob * BAR_WIDTH)
    return BAR_CHAR * filled + BAR_EMPTY_CHAR * (BAR_WIDTH - filled)


def predict_matchup(team_a_name: str, team_b_name: str, is_home: bool = True,
                    form_delta: float = 0.0, competition: str = 'FIFA World Cup') -> dict:
    """
    Predict win/draw/loss probabilities for team_a vs team_b.

    Parameters
    ----------
    team_a_name   : Name of team A (the "home" or focal team)
    team_b_name   : Name of team B (the opponent)
    is_home       : True if team_a is the home side
    form_delta    : team_a form_points_5 minus team_b form_points_5 (0 = unknown)
    competition   : Competition name for weight lookup (not used in prediction feature —
                    kept for API consistency; weight only matters at training time)
    """
    conn = _get_conn()
    cur  = conn.cursor()
    a    = _load_team(team_a_name, cur)
    b    = _load_team(team_b_name, cur)
    cur.close()
    conn.close()

    sv_a = a['style_vector']
    sv_b = b['style_vector']
    delta = sv_a - sv_b

    arch_a_id  = ARCHETYPES.get(a['archetype_name'], 0)
    arch_b_id  = ARCHETYPES.get(b['archetype_name'], 0)
    matchup_id = arch_a_id * 4 + arch_b_id

    comp_weights = {'FIFA World Cup': 1.0, 'UEFA Euro': 0.8}
    comp_w = comp_weights.get(competition, 0.7)

    delta_matches = a['matches_played'] - b['matches_played']

    feature_vec = np.concatenate([
        delta,                                     # 10 style delta features
        [float(is_home),                           # is_home
         float(form_delta),                        # form_points_delta
         float(matchup_id),                        # archetype_matchup_id
         float(delta_matches),                     # delta_matches_played
         float(comp_w)],                           # competition_weight
    ]).reshape(1, -1)

    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        model = pickle.load(f)

    proba = model.predict_proba(feature_vec)[0]
    p_win, p_draw, p_loss = float(proba[0]), float(proba[1]), float(proba[2])

    predicted_idx = int(np.argmax(proba))
    predicted     = LABEL_NAMES[predicted_idx]
    max_prob      = float(proba[predicted_idx])

    if max_prob > 0.55:
        confidence = 'high'
    elif max_prob > 0.45:
        confidence = 'medium'
    else:
        confidence = 'low'

    result = {
        'team_a':          a['team_name'],
        'team_b':          b['team_name'],
        'archetype_a':     a['archetype_name'],
        'archetype_b':     b['archetype_name'],
        'matchup_type':    f'{a["archetype_name"]} vs {b["archetype_name"]}',
        'p_win':           round(p_win, 4),
        'p_draw':          round(p_draw, 4),
        'p_loss':          round(p_loss, 4),
        'predicted_result': predicted,
        'confidence':      confidence,
    }

    # Print formatted output
    print('─' * 43)
    print(' TactiQ Matchup Prediction')
    print('─' * 43)
    print(f' {a["team_name"]} ({a["archetype_name"]})  vs  {b["team_name"]} ({b["archetype_name"]})')
    print('─' * 43)
    print(f' Win    {p_win*100:5.1f}%  {_bar(p_win)}')
    print(f' Draw   {p_draw*100:5.1f}%  {_bar(p_draw)}')
    print(f' Loss   {p_loss*100:5.1f}%  {_bar(p_loss)}')
    print('─' * 43)
    winner_label = f'{a["team_name"]} WIN' if predicted == 'win' else \
                   f'{b["team_name"]} WIN' if predicted == 'loss' else 'DRAW'
    print(f' Predicted: {winner_label}  [{confidence.capitalize()} confidence]')
    print('─' * 43)
    print()

    return result


def run_test_predictions():
    """Run the 5 mandatory test matchups."""
    test_cases = [
        ('Spain',       'Morocco'),
        ('Germany',     'Brazil'),
        ('France',      'Argentina'),
        ('England',     'Iran'),
        ('Netherlands', 'Ecuador'),
    ]
    results = []
    for team_a, team_b in test_cases:
        try:
            r = predict_matchup(team_a, team_b)
            results.append(r)
        except Exception as e:
            print(f'ERROR predicting {team_a} vs {team_b}: {e}')
    return results


if __name__ == '__main__':
    run_test_predictions()
