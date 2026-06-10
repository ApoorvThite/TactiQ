"""TactiQ Dashboard — Home page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(
    page_title='TactiQ | WC2026 Tactical Intelligence',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded',
)

css_path = Path(__file__).parent / 'assets' / 'custom.css'
st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

from src.dashboard.utils.db import (
    load_team_profiles, load_qualification_probs, load_upset_watchlist,
)
from src.dashboard.utils.charts import tournament_contenders_chart, prob_bars
from src.dashboard.utils.sidebar import render_sidebar
from src.dashboard.utils.flags import flag as get_flag

render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_date = st.columns([3, 1])
with col_title:
    st.markdown(
        '<h1 style="margin:0; font-size:2.6rem; font-weight:900; letter-spacing:-0.04em;'
        'background:linear-gradient(135deg,#00D4A0 0%,#7C6FE0 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;">⚽ TactiQ</h1>'
        '<p style="margin:0.3rem 0 0; color:#6B7394; font-size:0.85rem; font-weight:500;'
        'letter-spacing:0.06em; text-transform:uppercase;">'
        'Tactical DNA Engine &nbsp;·&nbsp; FIFA World Cup 2026</p>',
        unsafe_allow_html=True,
    )
with col_date:
    st.markdown(
        '<div style="text-align:right; padding-top:0.4rem;">'
        '<div style="font-size:0.68rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin-bottom:4px;">Tournament Starts</div>'
        '<div style="color:#E8EDF8; font-size:1.2rem; font-weight:800; letter-spacing:-0.02em;">'
        'June 11, 2026</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

# ── Project brief + glossary ──────────────────────────────────────────────────
st.markdown(
    '<div style="background:#0A0F1C; border:1px solid rgba(255,255,255,0.07);'
    'border-radius:14px; padding:1.4rem 1.75rem; margin-bottom:1.5rem;">'
    '<p style="color:#C0C8E8; font-size:0.95rem; line-height:1.75; margin:0 0 1.1rem;">'
    '<b style="color:#E8EDF8;">TactiQ</b> is a tactical intelligence engine built on '
    '<b style="color:#E8EDF8;">843,000+ StatsBomb match events</b> across four major tournaments. '
    'It clusters all 48 FIFA World Cup 2026 nations into tactical archetypes using unsupervised learning, '
    'then powers an XGBoost model to predict match outcomes with SHAP-attributed explanations — '
    'showing not just <em>who</em> wins, but <em>why</em>.'
    '</p>'
    '<div style="font-size:0.68rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
    'letter-spacing:0.12em; margin-bottom:0.75rem; padding-bottom:0.4rem;'
    'border-bottom:1px solid rgba(255,255,255,0.06);">Key Metrics Glossary</div>'
    '</div>',
    unsafe_allow_html=True,
)

_terms = [
    ('#00D4A0', 'PPDA',             'Passes Allowed Per Defensive Action',
     'Measures pressing intensity. Lower = more aggressive press. '
     'A PPDA of 6 means a team allows only 6 opposition passes before winning the ball back.'),
    ('#7C6FE0', 'xG',               'Expected Goals',
     'The probability that a shot results in a goal, based on location, angle, and assist type. '
     'xG &gt; 1.5 per game signals a dangerous attack.'),
    ('#3B98EA', 'xG Ratio',         'xG For &divide; (xG For + xG Against)',
     "A team's share of the combined xG in their matches. "
     'Above 0.5 = creating more danger than conceding.'),
    ('#FBBF24', 'SHAP',             'SHapley Additive exPlanations',
     'A game-theoretic method that explains each feature\'s contribution to a model prediction. '
     'Used here to show why the model favours a particular outcome.'),
    ('#F07050', 'UMAP',             'Uniform Manifold Approximation &amp; Projection',
     'A dimensionality-reduction algorithm that maps 10-dimensional style vectors into 2D space. '
     'Teams closer together play more similarly.'),
    ('#5EA9E8', 'Press Success %',  '% of pressing actions that win possession',
     'Of all high-intensity defensive actions, how many result in regaining the ball. '
     'High press intensity with low success = a risky strategy.'),
]

g1, g2, g3 = st.columns(3)
for i, (color, term, full, desc) in enumerate(_terms):
    col = [g1, g2, g3][i % 3]
    with col:
        st.markdown(
            f'<div style="background:#0A0F1C; border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:10px; padding:0.85rem 1rem; margin-bottom:0.6rem;">'
            f'<div style="margin-bottom:0.3rem;">'
            f'<span style="color:{color}; font-weight:800; font-size:0.92rem;">{term}</span>'
            f'</div>'
            f'<div style="color:#5A6490; font-size:0.74rem; font-weight:600; margin-bottom:0.35rem;">{full}</div>'
            f'<div style="color:#7B83A8; font-size:0.78rem; line-height:1.5;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── KPI cards ─────────────────────────────────────────────────────────────────
kpis = [
    ('48',   'Teams Analyzed',     '#00D4A0'),
    ('843K', 'Match Events',       '#7C6FE0'),
    ('72',   'Fixtures Predicted', '#3B98EA'),
    ('4',    'Tactical Archetypes','#FBBF24'),
]
k1, k2, k3, k4 = st.columns(4)
for col, (val, label, color) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(
            f'<div style="background:#0F1520; border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:12px; padding:1.2rem 1.4rem; position:relative; overflow:hidden;">'
            f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
            f'background:{color};"></div>'
            f'<div style="font-size:2.2rem; font-weight:900; color:{color};'
            f'letter-spacing:-0.03em; line-height:1.1;">{val}</div>'
            f'<div style="font-size:0.76rem; color:#7B83A8; font-weight:600;'
            f'text-transform:uppercase; letter-spacing:0.1em; margin-top:0.3rem;">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="height:1.75rem;"></div>', unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_teams  = load_team_profiles()
df_qual   = load_qualification_probs()
df_upsets = load_upset_watchlist()

# ── Main row: Contenders chart + Champion card ────────────────────────────────
col_chart, col_champion = st.columns([6, 4])

with col_chart:
    st.markdown(
        '<div style="font-size:0.75rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin-bottom:0.2rem;">Top 16 Tournament Contenders</div>'
        '<div style="font-size:0.82rem; color:#6B7394; margin-bottom:0.6rem;">'
        'Ranked by Monte Carlo R32 qualification probability &nbsp;·&nbsp; '
        '<span style="color:#FBBF24;">◆</span>&nbsp;= p(win group)'
        '</div>',
        unsafe_allow_html=True,
    )
    fig_contenders = tournament_contenders_chart(df_qual, n=16)
    st.plotly_chart(fig_contenders, use_container_width=True,
                    config={'displayModeBar': False}, key='contenders_main')

with col_champion:
    spain = df_teams[df_teams['team_name'] == 'Spain']
    spain_ppda = float(spain['avg_ppda'].iloc[0])    if not spain.empty else 0
    spain_xg   = float(spain['avg_xg_ratio'].iloc[0]) if not spain.empty else 0

    st.markdown(
        f"""
        <div class="champion-card">
            <h2>🏆 &nbsp;Predicted Champion</h2>
            <h1>{get_flag('Spain')} Spain</h1>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1rem;">
                <span class="badge badge-green">High Press</span>
                <span style="background:rgba(255,255,255,0.06); color:#8B93B8;
                             padding:3px 10px; border-radius:999px; font-size:0.74rem;
                             font-weight:600;">PPDA {spain_ppda:.2f}</span>
                <span style="background:rgba(255,255,255,0.06); color:#8B93B8;
                             padding:3px 10px; border-radius:999px; font-size:0.74rem;
                             font-weight:600;">xG {spain_xg:.2f}</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.07); padding-top:0.85rem;">
                <div style="font-size:0.7rem; color:#5A6490; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                    Predicted Final
                </div>
                <div style="font-size:1.1rem; color:#E8EDF8; font-weight:700;">
                    {get_flag('Spain')} Spain <span style="color:#3A4060; font-weight:400;">vs</span> {get_flag('France')} France
                </div>
                <div style="margin-top:0.6rem;">
                    <span class="badge badge-green">Spain WIN &nbsp;·&nbsp; 60%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_final = prob_bars('Spain', 'France', 0.60, 0.22, 0.18,
                          'High Press', 'Possession Control')
    st.plotly_chart(fig_final, use_container_width=True,
                    config={'displayModeBar': False}, key='final_bar')

    st.markdown(
        '<div style="font-size:0.7rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin:0.9rem 0 0.5rem; padding-bottom:0.35rem;'
        'border-bottom:1px solid rgba(255,255,255,0.06);">Bracket Path to Final</div>',
        unsafe_allow_html=True,
    )
    bracket_steps = [
        ('R32',   'Spain', 'Iran',    '82%'),
        ('QF',    'Spain', 'Germany', '67%'),
        ('SF',    'Spain', 'Brazil',  '66%'),
        ('Final', 'Spain', 'France',  '60%'),
    ]
    rows_html = ''.join(
        f'<div class="bracket-row">'
        f'<span><b style="color:#C0C8E8; font-size:0.8rem;">{stage}</b>'
        f'<span style="color:#3A4060;"> &nbsp;—&nbsp; </span>'
        f'<span style="color:#7B83A8; font-size:0.82rem;">'
        f'{get_flag(t_a)}&nbsp;{t_a} <span style="color:#2A3050;">vs</span> {get_flag(t_b)}&nbsp;{t_b}'
        f'</span></span>'
        f'<span style="color:#00D4A0; font-weight:700; font-size:0.85rem;">{pct}</span>'
        f'</div>'
        for stage, t_a, t_b, pct in bracket_steps
    )
    st.markdown(rows_html, unsafe_allow_html=True)

# ── Upset Alerts ──────────────────────────────────────────────────────────────
st.markdown('<div style="height:1.75rem;"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:0.75rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
    'letter-spacing:0.1em; margin-bottom:0.75rem; padding-bottom:0.4rem;'
    'border-bottom:1px solid rgba(255,255,255,0.06);">'
    '⚠&nbsp; Upset Alerts &nbsp;—&nbsp; Underdogs with structural tactical edges'
    '</div>',
    unsafe_allow_html=True,
)

if df_upsets.empty:
    st.info('No upset candidates detected under current thresholds.')
else:
    top3 = df_upsets.head(3)
    uc1, uc2, uc3 = st.columns(3)
    for col, (_, row) in zip([uc1, uc2, uc3], top3.iterrows()):
        p_upset = row.get('p_not_fav_win', 0)
        explanation = str(row.get('upset_explanation', ''))[:110]
        with col:
            st.markdown(
                f"""
                <div class="upset-alert">
                    <h4>Group {row['group_label']} &nbsp;·&nbsp; Upset Candidate</h4>
                    <div style="font-size:1.05rem; font-weight:800; color:#E8EDF8;
                                margin-bottom:0.15rem;">
                        {get_flag(row['underdog'])} {row['underdog']}
                    </div>
                    <div style="font-size:0.84rem; color:#5A6490; margin-bottom:0.65rem;">
                        vs&nbsp;<span style="color:#7B83A8;">{get_flag(row['favourite'])} {row['favourite']}</span>
                    </div>
                    <div style="font-size:1.7rem; font-weight:900; color:#FBBF24;
                                line-height:1; margin-bottom:0.4rem;">
                        {p_upset*100:.0f}%
                        <span style="font-size:0.74rem; color:#5A6490; font-weight:500;">
                            &nbsp;p(upset)
                        </span>
                    </div>
                    <div style="font-size:0.78rem; color:#6B7394; line-height:1.5;">
                        {explanation}…
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Archetype legend ──────────────────────────────────────────────────────────
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
st.divider()
st.markdown(
    '<div style="font-size:0.75rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
    'letter-spacing:0.1em; margin-bottom:0.75rem;">Tactical Archetypes</div>',
    unsafe_allow_html=True,
)
arch_info = {
    'High Press':         ('Aggressive press, high PPDA',   '#00D4A0'),
    'Possession Control': ('Dominant ball retention',        '#9F8FF0'),
    'Counter-Attack':     ('Vertical transitions on breaks', '#F07050'),
    'Deep Block':         ('Compact low-block defence',      '#5EA9E8'),
}
ac1, ac2, ac3, ac4 = st.columns(4)
for col, (arch, (desc, color)) in zip([ac1, ac2, ac3, ac4], arch_info.items()):
    count = len(df_teams[df_teams['archetype_name'] == arch])
    with col:
        st.markdown(
            f'<div class="arch-card">'
            f'<div style="display:flex; align-items:center; margin-bottom:0.45rem;">'
            f'<span class="arch-dot" style="background:{color}; box-shadow:0 0 6px {color}80;"></span>'
            f'<span style="font-weight:700; color:{color}; font-size:0.92rem;">{arch}</span>'
            f'</div>'
            f'<div style="color:#6B7394; font-size:0.78rem; line-height:1.4;">{desc}</div>'
            f'<div style="color:#C0C8E8; font-size:0.88rem; font-weight:700; margin-top:0.5rem;">'
            f'{count} <span style="color:#5A6490; font-weight:400;">teams</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
