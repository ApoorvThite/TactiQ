"""Live matchup prediction wrapper for the Streamlit dashboard."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

ROOT       = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / 'models'

ARCHETYPES  = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}
LABEL_NAMES = ['win', 'draw', 'loss']

FEATURE_NAMES = [
    'delta_avg_possession_pct', 'delta_avg_ppda', 'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90', 'delta_avg_xg_ratio', 'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct', 'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct', 'delta_avg_set_piece_shot_pct',
    'is_home', 'form_points_delta', 'archetype_matchup_id', 'delta_matches_played',
    'competition_weight',
]

STYLE_FEATURE_COLS = [
    'avg_possession_pct', 'avg_ppda', 'avg_pressure_success_rate',
    'avg_xg_created_p90', 'avg_xg_ratio', 'avg_progressive_carry_pct',
    'avg_pass_completion_pct', 'avg_passes_final_third_p90',
    'avg_pass_completion_under_pressure_pct', 'avg_set_piece_shot_pct',
]

ARCHETYPE_OUTCOME = {
    'High Press vs Deep Block':             'draw-prone (Deep Block absorbs the press)',
    'High Press vs Counter-Attack':         'upset-prone (Counter-Attack exploits the space)',
    'Possession Control vs Deep Block':     'draw-prone (Deep Block frustrates build-up)',
    'Possession Control vs Counter-Attack': 'upset-prone (Counter-Attack on the break)',
    'High Press vs High Press':             'high-variance (both sides press hard)',
    'Counter-Attack vs Counter-Attack':     'tactical stalemate (neither controls the ball)',
    'Deep Block vs High Press':             'underdog-favourable (sitting deep and countering)',
    'Deep Block vs Counter-Attack':         'compact and cautious (both defend first)',
    'Counter-Attack vs Possession Control': 'upset-prone (fast breaks vs controlled play)',
}

FEATURE_NARRATIVES = {
    'delta_avg_xg_ratio':                          ('creates significantly more danger relative to what they concede',
                                                    'has a stronger attack/defence balance'),
    'delta_avg_ppda':                              ('presses far less aggressively',
                                                    'presses harder — opponent struggles to build from the back'),
    'delta_avg_possession_pct':                    ('dominates possession — controls the tempo',
                                                    'cedes the ball — will defend deep'),
    'delta_avg_xg_created_p90':                   ('generates significantly more xG per match',
                                                    'creates more dangerous chances per 90'),
    'delta_avg_progressive_carry_pct':             ('carries the ball forward more directly',
                                                    'transitions quickly — counter-attack threat'),
    'delta_avg_set_piece_shot_pct':               ('relies heavily on set-piece situations',
                                                    'has a set-piece dependency edge'),
    'delta_avg_pressure_success_rate':             ('wins the ball back more successfully',
                                                    'presses more effectively when they do press'),
    'delta_avg_pass_completion_pct':               ('is the more technically precise side',
                                                    'has better technical quality in possession'),
    'delta_avg_passes_final_third_p90':           ('attacks with higher frequency into the final third',
                                                    'penetrates more often into dangerous zones'),
    'delta_avg_pass_completion_under_pressure_pct': ('is composed under a high press',
                                                     'keeps possession better when pressed'),
    'form_points_delta':                           ('arrives in better recent form',
                                                    'is the in-form side'),
    'delta_matches_played':                        ('has more tournament experience in the dataset',
                                                    'has more high-level match data behind them'),
}


@st.cache_resource
def _load_models():
    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'rb') as f:
        model_calib = pickle.load(f)
    with open(MODELS_DIR / 'shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return model_calib, explainer


def _build_feature_vector(row_a, row_b):
    sv_a = np.array([float(row_a.get(c, 0) or 0) for c in STYLE_FEATURE_COLS])
    sv_b = np.array([float(row_b.get(c, 0) or 0) for c in STYLE_FEATURE_COLS])
    delta = sv_a - sv_b
    arch_a_id  = ARCHETYPES.get(row_a.get('archetype_name', ''), 0)
    arch_b_id  = ARCHETYPES.get(row_b.get('archetype_name', ''), 0)
    matchup_id = arch_a_id * 4 + arch_b_id
    delta_mp   = float(row_a.get('matches_played', 0) or 0) - float(row_b.get('matches_played', 0) or 0)
    return np.concatenate([delta, [1.0, 0.0, float(matchup_id), delta_mp, 1.0]])


def _build_narrative(row_a, row_b, shap_cls, pred_cls, p_win, p_draw, p_loss):
    a = row_a['team_name']
    b = row_b['team_name']
    arch_a = row_a.get('archetype_name', '')
    arch_b = row_b.get('archetype_name', '')
    arch_key = f'{arch_a} vs {arch_b}'
    outcome = ARCHETYPE_OUTCOME.get(arch_key, 'tactically variable')

    sent1 = (f'Both teams are {arch_a} archetypes — {outcome}.'
             if arch_a == arch_b else
             f'{a} ({arch_a}) faces {b} ({arch_b}), a matchup that is historically {outcome}.')

    top3_idx = np.argsort(np.abs(shap_cls))[::-1][:3]
    sents = [sent1]
    for i in top3_idx:
        feat = FEATURE_NAMES[i]
        val  = shap_cls[i]
        if feat not in FEATURE_NARRATIVES:
            continue
        pos_tmpl, neg_tmpl = FEATURE_NARRATIVES[feat]
        if val > 0:
            sents.append(f'{a} {pos_tmpl}.')
        else:
            sents.append(f'{b} {neg_tmpl}.')
        if len(sents) >= 4:
            break

    verdict = (f'The model expects {a} WIN ({p_win*100:.0f}%).' if pred_cls == 'win' else
               f'The model expects {b} WIN ({p_loss*100:.0f}%).' if pred_cls == 'loss' else
               f'The model expects a DRAW ({p_draw*100:.0f}%).')
    sents.append(verdict)
    return ' '.join(sents)


@st.cache_data(ttl=300, show_spinner=False)
def predict_matchup(team_a_name: str, team_b_name: str,
                    team_a_data: dict, team_b_data: dict) -> dict:
    """
    Run live matchup prediction.
    team_a_data / team_b_data: dicts with style features and archetype_name.
    Returns dict with probabilities, predicted class, shap values, narrative.
    """
    model_calib, explainer = _load_models()

    x_vec    = _build_feature_vector(team_a_data, team_b_data)
    X_single = x_vec.reshape(1, -1)

    proba    = model_calib.predict_proba(X_single)[0]
    p_win, p_draw, p_loss = float(proba[0]), float(proba[1]), float(proba[2])
    pred_idx = int(np.argmax(proba))
    pred_cls = LABEL_NAMES[pred_idx]

    sv_single = explainer(X_single)
    shap_win  = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 0].tolist()))
    shap_draw = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 1].tolist()))
    shap_loss = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 2].tolist()))
    shap_cls  = sv_single.values[0, :, pred_idx]

    narrative = _build_narrative(
        team_a_data, team_b_data, shap_cls, pred_cls, p_win, p_draw, p_loss
    )

    is_upset = (
        p_win > 0.40
        and team_b_data.get('archetype_name') != 'High Press'
        and (p_draw + p_loss) >= 0.45
        and (abs(shap_draw.get('delta_avg_ppda', 0)) > 0.04
             or abs(shap_loss.get('delta_avg_ppda', 0)) > 0.04)
    )

    return {
        'team_a_name':     team_a_name,
        'team_b_name':     team_b_name,
        'predicted_class': pred_cls,
        'p_win':           p_win,
        'p_draw':          p_draw,
        'p_loss':          p_loss,
        'shap_win':        shap_win,
        'shap_draw':       shap_draw,
        'shap_loss':       shap_loss,
        'narrative':       narrative,
        'is_upset':        is_upset,
        'arch_a':          team_a_data.get('archetype_name', ''),
        'arch_b':          team_b_data.get('archetype_name', ''),
    }
