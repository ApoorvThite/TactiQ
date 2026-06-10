"""Phase 6 Step 6 — Predict WC2026 knockout bracket from R32 through Final.

Uses the modal bracket from Monte Carlo qualification probabilities.
For each stage, picks the team most likely to reach that round and runs
the matchup model to determine the predicted winner.
Saves predicted_bracket.json for Phase 7 dashboard.
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

LABEL_NAMES = ['win', 'draw', 'loss']
ARCHETYPES  = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}

FEATURE_NAMES = [
    'delta_avg_possession_pct', 'delta_avg_ppda', 'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90', 'delta_avg_xg_ratio', 'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct', 'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct', 'delta_avg_set_piece_shot_pct',
    'is_home', 'form_points_delta', 'archetype_matchup_id', 'delta_matches_played',
    'competition_weight',
]

# WC2026 R32 bracket seeding: Group winner vs best-ranked 3rd-place or runner-up
# Standard FIFA bracket (ordered by group, alternating 1st vs 2nd)
# Format: (position_a, position_b) — the two groups whose qualifiers meet
# WC2026 bracket TBD but using a plausible cross-group format:
# 48-team → 32-team → 16-team → 8-team QF → 4-team SF → Final

# Modal group finishers (1st or 2nd) are used for R32.
# The bracket pairing by FIFA convention (groups alternate):
R32_PAIRINGS_BY_GROUP = [
    # Standard cross-group bracket pairs (FIFA WC2026 tentative format)
    ('A1', 'B2'), ('C1', 'D2'), ('E1', 'F2'), ('G1', 'H2'),
    ('I1', 'J2'), ('K1', 'L2'), ('A2', 'B1'), ('C2', 'D1'),
    ('E2', 'F1'), ('G2', 'H1'), ('I2', 'J1'), ('K2', 'L1'),
    # 4 best-third-place slots fill in (simplified assignment)
    ('T1', 'T2'), ('T3', 'T4'), ('T5', 'T6'), ('T7', 'T8'),
]


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
    return model_calib


def _load_team_profile(name, cur):
    cur.execute(
        "SELECT team_id, team_name, archetype_name, style_vector, matches_played "
        "FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)",
        (name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        'team_id':        row[0],
        'team_name':      row[1],
        'archetype_name': row[2],
        'style_vector':   np.array(row[3]),
        'matches_played': row[4],
    }


def _build_feature_vector(team_a, team_b):
    sv_a   = np.array(team_a['style_vector'])
    sv_b   = np.array(team_b['style_vector'])
    delta  = sv_a - sv_b
    aid    = ARCHETYPES.get(team_a['archetype_name'], 0)
    bid    = ARCHETYPES.get(team_b['archetype_name'], 0)
    return np.concatenate([delta, [1.0, 0.0, float(aid * 4 + bid),
                                   float(team_a['matches_played'] - team_b['matches_played']),
                                   1.0]])


def predict_match(team_a, team_b, model_calib):
    """
    Run matchup model for a single knockout game.
    Knockout: no draws — if p_draw is highest, add it to p_win/p_loss proportionally.
    Returns (winner_team, p_team_a_advances, match_info_dict).
    """
    x_vec = _build_feature_vector(team_a, team_b).reshape(1, -1)
    proba = model_calib.predict_proba(x_vec)[0]
    p_win, p_draw, p_loss = float(proba[0]), float(proba[1]), float(proba[2])

    # In knockout, redistribute draw probability
    total_decisive = p_win + p_loss
    if total_decisive > 0:
        p_a = p_win + p_draw * (p_win / total_decisive)
        p_b = p_loss + p_draw * (p_loss / total_decisive)
    else:
        p_a, p_b = 0.5, 0.5

    winner = team_a if p_a >= p_b else team_b
    loser  = team_b if p_a >= p_b else team_a

    return winner, loser, {
        'team_a': team_a['team_name'],
        'team_b': team_b['team_name'],
        'p_a_advances': round(p_a, 4),
        'p_b_advances': round(p_b, 4),
        'predicted_winner': winner['team_name'],
        'winner_archetype': winner['archetype_name'],
        'p_win_raw': round(p_win, 4),
        'p_draw_raw': round(p_draw, 4),
        'p_loss_raw': round(p_loss, 4),
    }


def build_predicted_bracket(qual_probs_df, model_calib, cur):
    """
    Build full predicted bracket by picking most-likely qualifiers and
    simulating each knockout round deterministically.
    """
    # Determine modal group qualifiers (top p_qualify_r32 per group)
    qualifiers_by_group = {}
    for group in sorted(qual_probs_df['group'].unique()):
        g_df = qual_probs_df[qual_probs_df['group'] == group].sort_values(
            'p_qualify_r32', ascending=False
        )
        qualifiers_by_group[group] = g_df['team_name'].tolist()

    # Build R32 bracket from group qualifiers
    bracket = {
        'R32':  [],
        'R16':  [],
        'QF':   [],
        'SF':   [],
        'Final': [],
        'Champion': None,
    }

    # R32: Group 1st vs cross-group 2nd
    group_list = sorted(qualifiers_by_group.keys())  # A-L

    r32_teams = []
    # Standard WC pairing: A1vB2, B1vA2, C1vD2, D1vC2, ...
    for i in range(0, len(group_list), 2):
        if i + 1 >= len(group_list):
            break
        ga = group_list[i]
        gb = group_list[i + 1]
        teams_a = qualifiers_by_group[ga]
        teams_b = qualifiers_by_group[gb]
        if len(teams_a) >= 1 and len(teams_b) >= 2:
            r32_teams.append((teams_a[0], teams_b[1]))   # 1st_A vs 2nd_B
        if len(teams_a) >= 2 and len(teams_b) >= 1:
            r32_teams.append((teams_b[0], teams_a[1]))   # 1st_B vs 2nd_A

    # Add best 8 third-place teams (simplified: take top-8 by p_best_third)
    third_place = (
        qual_probs_df
        .sort_values('p_best_third', ascending=False)
        .head(8)['team_name']
        .tolist()
    )
    for j in range(0, len(third_place), 2):
        if j + 1 < len(third_place):
            r32_teams.append((third_place[j], third_place[j + 1]))

    print(f'\n  R32: {len(r32_teams)} matches')

    def _play_round(matchups, round_name):
        round_results = []
        winners       = []
        for a_name, b_name in matchups:
            team_a = _load_team_profile(a_name, cur)
            team_b = _load_team_profile(b_name, cur)
            if team_a is None or team_b is None:
                missing = a_name if team_a is None else b_name
                print(f'    [WARN] {missing} not in DB — advancing opponent by default')
                adv = b_name if team_a is None else a_name
                winners.append(adv)
                round_results.append({'team_a': a_name, 'team_b': b_name,
                                      'predicted_winner': adv, 'note': 'missing_profile'})
                continue

            winner, loser, info = predict_match(team_a, team_b, model_calib)
            winners.append(winner['team_name'])
            round_results.append(info)
            conf = max(info['p_a_advances'], info['p_b_advances'])
            print(f'    {a_name} vs {b_name} → {winner["team_name"]} '
                  f'({conf*100:.0f}%)')

        bracket[round_name] = round_results
        return winners

    # Run all rounds
    r32_winners = _play_round(r32_teams, 'R32')

    r16_pairs = list(zip(r32_winners[::2], r32_winners[1::2]))
    print(f'\n  R16: {len(r16_pairs)} matches')
    r16_winners = _play_round(r16_pairs, 'R16')

    qf_pairs = list(zip(r16_winners[::2], r16_winners[1::2]))
    print(f'\n  QF: {len(qf_pairs)} matches')
    qf_winners = _play_round(qf_pairs, 'QF')

    sf_pairs = list(zip(qf_winners[::2], qf_winners[1::2]))
    print(f'\n  SF: {len(sf_pairs)} matches')
    sf_winners = _play_round(sf_pairs, 'SF')

    if len(sf_winners) >= 2:
        print(f'\n  Final:')
        final_winners = _play_round([(sf_winners[0], sf_winners[1])], 'Final')
        champion = final_winners[0] if final_winners else sf_winners[0]
    elif sf_winners:
        champion = sf_winners[0]
        bracket['Final'] = []
    else:
        champion = 'Unknown'

    bracket['Champion'] = champion

    return bracket


def print_bracket_summary(bracket):
    print('\n' + '='*60)
    print(' TactiQ — WC2026 Predicted Bracket')
    print('='*60)

    for stage in ['R32', 'R16', 'QF', 'SF', 'Final']:
        matches = bracket.get(stage, [])
        print(f'\n{stage} ({len(matches)} matches):')
        for m in matches:
            if m.get('note') == 'missing_profile':
                print(f'  {m["team_a"]} vs {m["team_b"]} → {m["predicted_winner"]} (default)')
            else:
                w  = m['predicted_winner']
                pa = m.get('p_a_advances', 0)
                pb = m.get('p_b_advances', 0)
                conf = max(pa, pb)
                print(f'  {m["team_a"]} vs {m["team_b"]} → {w} ({conf*100:.0f}%)')

    print(f'\n  🏆 Predicted Champion: {bracket["Champion"]}')
    print('='*60)


if __name__ == '__main__':
    conn = _get_conn()
    cur  = conn.cursor()
    model_calib = _load_models()

    qual_csv = PROCESSED_DIR / 'qualification_probabilities.csv'
    if not qual_csv.exists():
        print('ERROR: qualification_probabilities.csv not found. Run simulate_group_stage.py first.')
        sys.exit(1)

    qual_df  = pd.read_csv(qual_csv)
    bracket  = build_predicted_bracket(qual_df, model_calib, cur)

    print_bracket_summary(bracket)

    out = PROCESSED_DIR / 'predicted_bracket.json'
    with open(out, 'w') as f:
        json.dump(bracket, f, indent=2)
    print(f'\nSaved → data/processed/predicted_bracket.json')

    cur.close()
    conn.close()
