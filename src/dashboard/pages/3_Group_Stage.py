"""Page 4 — Group Stage: 12 tabs with fixture predictions + qualification heatmap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

css_path = Path(__file__).parents[1] / 'assets' / 'custom.css'
st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

from src.dashboard.utils.db import load_group_predictions, load_qualification_probs
from src.dashboard.utils.charts import prob_bars, group_qual_bars, qualification_heatmap
from src.dashboard.utils.styles import ARCHETYPE_COLORS, RESULT_COLORS
from src.dashboard.utils.sidebar import render_sidebar

render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin:0 0 0.25rem; font-size:1.7rem; font-weight:800;">📊 Group Stage</h2>'
    '<p style="color:#8B92A8; margin:0 0 1rem;">All 72 official WC2026 group fixtures '
    'with predicted outcomes and qualification probabilities.</p>',
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df_preds = load_group_predictions()
df_qual  = load_qualification_probs()

groups = sorted(df_preds['group_label'].unique())

# ── 12 group tabs ─────────────────────────────────────────────────────────────
tabs = st.tabs([f'Group {g}' for g in groups])

for tab, group_label in zip(tabs, groups):
    with tab:
        g_preds = df_preds[df_preds['group_label'] == group_label]
        g_qual  = df_qual[df_qual['group_label'] == group_label].sort_values(
            'p_qualify_r32', ascending=False
        )

        col_fixtures, col_qual = st.columns([6, 4])

        with col_fixtures:
            # Group header — list teams
            teams_in_group = g_qual['team_name'].tolist()
            st.markdown(
                f'<div style="font-weight:700; font-size:0.9rem; color:#8B92A8; '
                f'margin-bottom:0.5rem;">'
                f'Group {group_label} — {" · ".join(teams_in_group)}</div>',
                unsafe_allow_html=True,
            )

            for _, row in g_preds.iterrows():
                team_a = row['team_a_name']
                team_b = row['team_b_name']
                p_win  = float(row['p_win'])
                p_draw = float(row['p_draw'])
                p_loss = float(row['p_loss'])
                pred   = row['predicted_class']
                is_upset = bool(row['is_upset_candidate'])
                arch_a = row.get('team_a_archetype', '')
                arch_b = row.get('team_b_archetype', '')
                color_a = ARCHETYPE_COLORS.get(arch_a, '#888')
                color_b = ARCHETYPE_COLORS.get(arch_b, '#888')

                p_max = max(p_win, p_draw, p_loss)
                if p_max >= 0.60:
                    conf_label = '● High'
                    conf_color = '#22C55E'
                elif p_max >= 0.45:
                    conf_label = '● Med'
                    conf_color = '#F59E0B'
                else:
                    conf_label = '● Low'
                    conf_color = '#EF4444'

                result_color = RESULT_COLORS.get(pred, '#888')
                upset_badge = (
                    '&nbsp;<span style="background:rgba(245,158,11,0.2);color:#F59E0B;'
                    'font-size:0.65rem;padding:1px 6px;border-radius:4px;">⚠ UPSET SIGNAL</span>'
                    if is_upset else ''
                )

                st.markdown(
                    f'<div style="border:1px solid #2D3250; border-radius:6px; '
                    f'padding:0.6rem 0.8rem; margin-bottom:0.5rem; '
                    f'background:#1E2130;">'
                    f'<div style="display:flex; justify-content:space-between; '
                    f'align-items:center; margin-bottom:0.35rem;">'
                    f'<span style="font-size:0.9rem; font-weight:700; color:#E8EAF0;">'
                    f'<span style="color:{color_a};">●</span> {team_a}</span>'
                    f'<span style="font-size:0.75rem; color:{result_color}; font-weight:600;">'
                    f'{pred.upper()}</span>'
                    f'<span style="font-size:0.9rem; font-weight:700; color:#E8EAF0;">'
                    f'{team_b} <span style="color:{color_b};">●</span></span>'
                    f'</div>'
                    f'<div style="font-size:0.7rem; color:#8B92A8; margin-bottom:0.3rem;">'
                    f'{arch_a} vs {arch_b}'
                    f'<span style="margin-left:0.5rem; color:{conf_color};">{conf_label}</span>'
                    f'{upset_badge}</div>',
                    unsafe_allow_html=True,
                )

                fig_bar = prob_bars(team_a, team_b, p_win, p_draw, p_loss)
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'bar_{group_label}_{team_a}_{team_b}')

                st.markdown('</div>', unsafe_allow_html=True)

        with col_qual:
            # Qualification probability bars
            if not g_qual.empty:
                fig_qual = group_qual_bars(g_qual, group_label)
                st.plotly_chart(fig_qual, use_container_width=True,
                                config={'displayModeBar': False},
                                key=f'qual_{group_label}')

            # Group standings table
            st.markdown('<div class="section-header">Qualification Odds</div>',
                        unsafe_allow_html=True)

            for _, qr in g_qual.iterrows():
                arch  = qr.get('archetype_name', '')
                color = ARCHETYPE_COLORS.get(arch, '#888')
                proxy_flag = ' [P]' if qr.get('is_proxy') else ''
                p_r32 = float(qr['p_qualify_r32'])
                bar_w = int(p_r32 * 100)

                st.markdown(
                    f'<div style="margin-bottom:0.5rem;">'
                    f'<div style="display:flex; justify-content:space-between; '
                    f'font-size:0.82rem;">'
                    f'<span style="color:#E8EAF0; font-weight:600;">'
                    f'{qr["team_name"]}{proxy_flag}</span>'
                    f'<span style="color:{color};">{p_r32*100:.0f}%</span>'
                    f'</div>'
                    f'<div style="height:4px; background:#2D3250; border-radius:2px; '
                    f'margin-top:3px;">'
                    f'<div style="width:{bar_w}%; height:4px; background:{color}; '
                    f'border-radius:2px;"></div>'
                    f'</div>'
                    f'<div style="font-size:0.72rem; color:#8B92A8; margin-top:2px;">'
                    f'1st: {qr["p_first"]*100:.0f}% · '
                    f'2nd: {qr["p_second"]*100:.0f}% · '
                    f'3rd: {qr["p_third"]*100:.0f}%'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Most likely winner
            if not g_qual.empty:
                winner_row = g_qual.iloc[0]
                st.markdown(
                    f'<div style="margin-top:0.75rem; padding:0.5rem 0.75rem; '
                    f'background:rgba(29,158,117,0.1); border-radius:6px; '
                    f'border-left:3px solid #1D9E75; font-size:0.83rem;">'
                    f'<span style="color:#8B92A8;">Most likely winner:</span> '
                    f'<b style="color:#1D9E75;">{winner_row["team_name"]}</b> '
                    f'({winner_row["p_first"]*100:.0f}%)</div>',
                    unsafe_allow_html=True,
                )

# ── Full heatmap ──────────────────────────────────────────────────────────────
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Overall Qualification Probability Heatmap</div>',
            unsafe_allow_html=True)
fig_heat = qualification_heatmap(df_qual)
st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False},
                key='heatmap_main')
st.caption('Color: probability of qualifying for Round of 32 (top-2 per group + best 8 third-place). '
           'Based on 10,000 Monte Carlo simulations.')
