"""Page 3 — Matchup Predictor: pick any two WC2026 teams, get prediction + SHAP."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json

import streamlit as st

css_path = Path(__file__).parents[1] / 'assets' / 'custom.css'
st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

from src.dashboard.utils.db import load_team_profiles, load_shap_values
from src.dashboard.utils.charts import prob_bars, shap_waterfall_table, radar_chart
from src.dashboard.utils.predict import predict_matchup
from src.dashboard.utils.styles import ARCHETYPE_COLORS, RESULT_COLORS, confidence_label
from src.dashboard.utils.sidebar import render_sidebar
from src.dashboard.utils.flags import flag as get_flag

render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin:0 0 0.25rem; font-size:1.7rem; font-weight:800;">⚔️ Matchup Predictor</h2>'
    '<p style="color:#8B92A8; margin:0 0 1rem;">Select any two WC2026 teams for a '
    'full tactical prediction powered by XGBoost + SHAP.</p>',
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df_teams  = load_team_profiles()
df_shap   = load_shap_values()
all_teams = sorted(df_teams['team_name'].tolist())


def _team_row(name: str) -> dict | None:
    r = df_teams[df_teams['team_name'] == name]
    return r.iloc[0].to_dict() if not r.empty else None


# ── Team selectors ────────────────────────────────────────────────────────────
c_a, c_vs, c_b, c_btn = st.columns([4, 1, 4, 2])

with c_a:
    idx_a = all_teams.index('Spain') if 'Spain' in all_teams else 0
    team_a = st.selectbox('Team A', all_teams, index=idx_a, key='team_a_sel')

with c_vs:
    st.markdown(
        '<div style="text-align:center; padding-top:1.8rem; '
        'font-size:1.1rem; color:#8B92A8; font-weight:600;">vs</div>',
        unsafe_allow_html=True,
    )

with c_b:
    idx_b = all_teams.index('Morocco') if 'Morocco' in all_teams else 1
    team_b = st.selectbox('Team B', all_teams, index=idx_b, key='team_b_sel')

with c_btn:
    st.markdown('<div style="height:1.65rem;"></div>', unsafe_allow_html=True)
    predict_btn = st.button('⚡ Predict Matchup', use_container_width=True, type='primary')

if team_a == team_b:
    st.warning('Please select two different teams.')
    st.stop()

# ── Auto-predict on load or button click ─────────────────────────────────────
row_a = _team_row(team_a)
row_b = _team_row(team_b)

if row_a is None or row_b is None:
    st.error('Team profile not found in database.')
    st.stop()

# Check pre-computed SHAP values first
precomputed = None
if not df_shap.empty:
    mask = (
        (df_shap['team_a_name'].str.lower() == team_a.lower()) &
        (df_shap['team_b_name'].str.lower() == team_b.lower())
    )
    if mask.any():
        row_shap = df_shap[mask].iloc[0]
        shap_dict = row_shap.get('shap_values_win') or {}
        if isinstance(shap_dict, str):
            try:
                shap_dict = json.loads(shap_dict)
            except Exception:
                shap_dict = {}
        precomputed = {
            'predicted_class': row_shap['predicted_class'],
            'p_win':  float(row_shap['p_win']),
            'p_draw': float(row_shap['p_draw']),
            'p_loss': float(row_shap['p_loss']),
            'shap_win':  shap_dict,
            'shap_draw': row_shap.get('shap_values_draw') or {},
            'shap_loss': row_shap.get('shap_values_loss') or {},
            'is_upset':  bool(row_shap.get('is_upset_candidate', False)),
            'narrative': str(row_shap.get('upset_explanation', '')),
            'arch_a': row_a.get('archetype_name', ''),
            'arch_b': row_b.get('archetype_name', ''),
            'team_a_name': team_a,
            'team_b_name': team_b,
        }

# Run live prediction
with st.spinner(f'Computing {team_a} vs {team_b}…'):
    result = predict_matchup(team_a, team_b, row_a, row_b)

# Merge pre-computed narrative if available and live narrative is short
if precomputed and len(precomputed.get('narrative', '')) > 50:
    result['narrative'] = precomputed['narrative']

pred_cls = result['predicted_class']
p_win    = result['p_win']
p_draw   = result['p_draw']
p_loss   = result['p_loss']
p_max    = max(p_win, p_draw, p_loss)
conf     = confidence_label(p_max)

# ── Result banner ─────────────────────────────────────────────────────────────
result_class = {'win': 'result-win', 'draw': 'result-draw', 'loss': 'result-loss'}[pred_cls]
result_color = RESULT_COLORS[pred_cls]
arch_a = result['arch_a']
arch_b = result['arch_b']
color_a = ARCHETYPE_COLORS.get(arch_a, '#888')
color_b = ARCHETYPE_COLORS.get(arch_b, '#888')

st.markdown(
    f"""
    <div class="result-banner {result_class}" style="margin-top:1rem;">
        <div style="display:flex; justify-content:center; align-items:center; gap:1.5rem;">
            <div style="text-align:right;">
                <div style="font-size:1.4rem; font-weight:800; color:#E8EDF8;">
                    {get_flag(team_a)} {team_a}</div>
                <div style="font-size:0.78rem; color:{color_a};">{arch_a}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:900; color:{result_color};">
                    {"WIN" if pred_cls == "win" else ("DRAW" if pred_cls == "draw" else "LOSS")}
                </div>
                <div style="font-size:0.8rem; color:#6B7394;">{p_max*100:.0f}%
                    &nbsp;·&nbsp;
                    <span style="color:{'#00D4A0' if conf == 'HIGH' else '#FBBF24' if conf == 'MEDIUM' else '#EF4444'};">
                        {conf} confidence
                    </span>
                </div>
            </div>
            <div style="text-align:left;">
                <div style="font-size:1.4rem; font-weight:800; color:#E8EDF8;">
                    {get_flag(team_b)} {team_b}</div>
                <div style="font-size:0.78rem; color:{color_b};">{arch_b}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Probability bar ───────────────────────────────────────────────────────────
fig_bar = prob_bars(team_a, team_b, p_win, p_draw, p_loss, arch_a, arch_b)
st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

st.markdown(
    f'<div style="display:flex; justify-content:space-around; '
    f'font-size:0.82rem; color:#8B92A8; margin:-0.5rem 0 0.5rem;">'
    f'<span>🟢 {team_a} WIN: {p_win*100:.0f}%</span>'
    f'<span>🟡 DRAW: {p_draw*100:.0f}%</span>'
    f'<span>🔴 {team_b} WIN: {p_loss*100:.0f}%</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# ── SHAP + Narrative ──────────────────────────────────────────────────────────
col_shap, col_narrative = st.columns([5, 5])

shap_key = {'win': 'shap_win', 'draw': 'shap_draw', 'loss': 'shap_loss'}[pred_cls]
shap_for_class = result.get(shap_key, result.get('shap_win', {}))

with col_shap:
    if shap_for_class:
        fig_shap = shap_waterfall_table(shap_for_class, pred_cls.upper())
        st.plotly_chart(fig_shap, use_container_width=True,
                        config={'displayModeBar': False})
    else:
        st.info('SHAP data not available for this matchup.')

with col_narrative:
    st.markdown('<div class="section-header">Tactical Breakdown</div>',
                unsafe_allow_html=True)
    narrative = result.get('narrative', '')
    if narrative:
        # Split into sentences for readability
        sentences = narrative.replace('. ', '.\n\n').split('\n\n')
        for s in sentences:
            if s.strip():
                st.markdown(f'<p style="color:#C8CAD0; font-size:0.9rem; '
                            f'line-height:1.6;">{s.strip()}</p>',
                            unsafe_allow_html=True)
    else:
        st.caption('Run prediction to see tactical analysis.')

    # Quick stats comparison
    st.markdown('<div class="section-header" style="margin-top:1rem;">Head-to-Head Stats</div>',
                unsafe_allow_html=True)
    stats_rows = [
        ('Possession %',  'avg_possession_pct',   '{:.1f}%'),
        ('PPDA',          'avg_ppda',              '{:.2f}'),
        ('xG Ratio',      'avg_xg_ratio',          '{:.2f}'),
        ('xG p90',        'avg_xg_created_p90',    '{:.2f}'),
    ]
    for label, col_name, fmt in stats_rows:
        va = row_a.get(col_name) or 0
        vb = row_b.get(col_name) or 0
        better_a = (va > vb) if col_name != 'avg_ppda' else (va < vb)
        ca = '#1D9E75' if better_a else '#8B92A8'
        cb = '#1D9E75' if not better_a else '#8B92A8'
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; '
            f'font-size:0.82rem; padding:3px 0;">'
            f'<span style="color:{ca}; font-weight:{"600" if better_a else "400"};">'
            f'{fmt.format(va)}</span>'
            f'<span style="color:#8B92A8;">{label}</span>'
            f'<span style="color:{cb}; font-weight:{"600" if not better_a else "400"};">'
            f'{fmt.format(vb)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Radar comparison ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Style DNA Comparison</div>',
            unsafe_allow_html=True)
fig_radar = radar_chart(team_a, df_teams, compare_team=team_b)
st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

# ── Upset alert ───────────────────────────────────────────────────────────────
if result.get('is_upset'):
    st.markdown(
        f"""
        <div class="upset-alert" style="margin-top:1rem;">
            <h4>⚠ Upset Alert</h4>
            <div style="font-size:0.9rem; color:#E8EAF0; margin-bottom:0.35rem;">
                This matchup meets the structural upset criteria.
            </div>
            <div style="font-size:0.83rem; color:#8B92A8;">
                {team_b}'s {arch_b} archetype holds a pressing
                (PPDA) structural edge over {team_a}'s {arch_a} system.
                The SHAP attribution shows meaningful uncertainty — this
                result is not as settled as the headline probability suggests.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
