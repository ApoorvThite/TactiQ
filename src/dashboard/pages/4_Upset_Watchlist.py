"""Page 5 — Upset Watchlist: tactical mismatches where underdogs hold structural edges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

css_path = Path(__file__).parents[1] / 'assets' / 'custom.css'
st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

from src.dashboard.utils.db import load_upset_watchlist, load_group_predictions
from src.dashboard.utils.charts import upset_scatter, prob_bars
from src.dashboard.utils.styles import ARCHETYPE_COLORS, RESULT_COLORS
from src.dashboard.utils.sidebar import render_sidebar

render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin:0 0 0.25rem; font-size:1.7rem; font-weight:800;">⚠ Upset Watchlist</h2>'
    '<p style="color:#8B92A8; margin:0 0 0.25rem;">Matches where lower-ranked teams hold '
    'structural tactical edges — detected via SHAP attribution analysis.</p>'
    '<p style="color:#8B92A8; margin:0 0 1rem; font-size:0.8rem;">Based on 843,050 match '
    'events across 4 major tournaments · 10,000 Monte Carlo simulations</p>',
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df_upset = load_upset_watchlist()
df_all   = load_group_predictions()

# ── Filters ───────────────────────────────────────────────────────────────────
all_groups = sorted(df_all['group_label'].unique())

f_col1, f_col2, f_col3 = st.columns([3, 3, 2])
with f_col1:
    group_filter = st.multiselect('Filter by group', all_groups, default=all_groups)
with f_col2:
    min_p_upset = st.slider('Min p(upset)', 0.30, 0.70, 0.40, 0.01, format='%.0f%%',
                             help='Minimum probability that the favourite does NOT win.')
with f_col3:
    sort_by = st.selectbox('Sort by', ['p(upset) ↓', 'Group', 'Underdog archetype'])

# ── Apply filters ─────────────────────────────────────────────────────────────
df_filtered = df_upset.copy()
if group_filter:
    df_filtered = df_filtered[df_filtered['group_label'].isin(group_filter)]
df_filtered = df_filtered[df_filtered['p_not_fav_win'] >= min_p_upset]

if sort_by == 'Group':
    df_filtered = df_filtered.sort_values(['group_label', 'p_not_fav_win'], ascending=[True, False])
elif sort_by == 'Underdog archetype':
    df_filtered = df_filtered.sort_values(['arch_underdog', 'p_not_fav_win'], ascending=[True, False])
else:
    df_filtered = df_filtered.sort_values('p_not_fav_win', ascending=False)

st.markdown(
    f'<div style="font-size:0.82rem; color:#8B92A8; margin-bottom:1rem;">'
    f'Showing <b style="color:#F59E0B;">{len(df_filtered)}</b> upset candidates '
    f'(of {len(df_upset)} total)</div>',
    unsafe_allow_html=True,
)

if df_filtered.empty:
    st.info('No upset candidates match the current filters. Adjust the group filter or lower the threshold.')
    st.stop()

# ── Upset candidate cards ─────────────────────────────────────────────────────
for _, row in df_filtered.iterrows():
    fav       = row['favourite']
    underdog  = row['underdog']
    arch_fav  = row.get('arch_favourite', '')
    arch_und  = row.get('arch_underdog', '')
    p_fav     = float(row['p_win'])
    p_draw    = float(row['p_draw'])
    p_loss    = float(row['p_loss'])
    p_upset   = float(row['p_not_fav_win'])
    signal    = row.get('top_upset_signal', '')
    expl      = str(row.get('upset_explanation', ''))
    group     = row['group_label']
    pred_cls  = row.get('predicted_class', 'draw')

    color_und = ARCHETYPE_COLORS.get(arch_und, '#888')
    color_fav = ARCHETYPE_COLORS.get(arch_fav, '#888')
    result_col = RESULT_COLORS.get(pred_cls, '#888')

    with st.container():
        st.markdown(
            f'<div class="upset-alert" style="padding:1rem 1.25rem; margin-bottom:1rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">'
            f'<span style="background:rgba(245,158,11,0.15); color:#F59E0B; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase; letter-spacing:0.05em;">'
            f'Group {group} · ⚠ Upset Candidate</span>'
            f'<span style="color:#F59E0B; font-size:1.2rem; font-weight:800;">{p_upset*100:.0f}%'
            f'<span style="font-size:0.72rem; font-weight:400; color:#8B92A8;"> p(upset)</span></span>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-around; align-items:center; margin-bottom:0.75rem;">'
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.3rem; font-weight:800; color:#E8EAF0;">{underdog}</div>'
            f'<div style="font-size:0.75rem; color:{color_und};">{arch_und}</div>'
            f'<div style="font-size:0.7rem; color:#8B92A8; margin-top:2px;">underdog</div>'
            f'</div>'
            f'<div style="font-size:1.2rem; color:#8B92A8; font-weight:300;">vs</div>'
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.3rem; font-weight:800; color:#E8EAF0;">{fav}</div>'
            f'<div style="font-size:0.75rem; color:{color_fav};">{arch_fav}</div>'
            f'<div style="font-size:0.7rem; color:#8B92A8; margin-top:2px;">favourite</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Prob bar inline
        col_bar, col_signal = st.columns([6, 4])
        with col_bar:
            fig_bar = prob_bars(fav, underdog, p_fav, p_draw, p_loss)
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={'displayModeBar': False},
                            key=f'upset_bar_{fav}_{underdog}')
            st.markdown(
                f'<div style="font-size:0.72rem; color:#8B92A8; text-align:center;">'
                f'Predicted: <b style="color:{result_col};">{pred_cls.upper()}</b>'
                f' ({max(p_fav, p_draw, p_loss)*100:.0f}%)</div>',
                unsafe_allow_html=True,
            )

        with col_signal:
            st.markdown(
                '<div style="font-size:0.7rem; color:#8B92A8; text-transform:uppercase; '
                'letter-spacing:0.06em; font-weight:600; margin-bottom:0.4rem;">'
                'Key Tactical Signal</div>',
                unsafe_allow_html=True,
            )
            # Top signal feature
            if signal:
                human_signal = signal.replace('delta_avg_', '').replace('_', ' ').title()
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#F59E0B; font-weight:600; '
                    f'margin-bottom:0.35rem;">📌 {human_signal}</div>',
                    unsafe_allow_html=True,
                )
            # Explanation
            expl_short = expl[:200] + '…' if len(expl) > 200 else expl
            st.markdown(
                f'<div style="font-size:0.8rem; color:#8B92A8; line-height:1.5;">'
                f'{expl_short}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr style="border-color:#2D3250; margin:0.25rem 0 1rem;">', unsafe_allow_html=True)

# ── Upset scatter ─────────────────────────────────────────────────────────────
if len(df_filtered) >= 2:
    st.markdown('<div class="section-header">Upset Risk Overview</div>',
                unsafe_allow_html=True)
    fig_scatter = upset_scatter(df_filtered)
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False},
                    key='upset_scatter')
    st.caption(
        'X = probability favourite wins. Y = probability of upset (draw or underdog win). '
        'Top-right = highest upset risk. Color = underdog archetype.'
    )

# ── Summary stats ─────────────────────────────────────────────────────────────
st.divider()
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric('Total Fixtures', len(df_all))
with s2:
    st.metric('Upset Candidates', len(df_upset))
with s3:
    pct = len(df_upset) / max(len(df_all), 1) * 100
    st.metric('Upset Rate', f'{pct:.0f}%')
with s4:
    if not df_filtered.empty:
        top = df_filtered.iloc[0]
        st.metric('Highest Risk', f'{top["underdog"]} vs {top["favourite"]}')
