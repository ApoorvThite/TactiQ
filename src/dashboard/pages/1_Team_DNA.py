"""Page 2 — Team DNA Explorer: UMAP scatter + radar chart + similarity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

css_path = Path(__file__).parents[1] / 'assets' / 'custom.css'
st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

from src.dashboard.utils.db import load_team_profiles, load_similarity_matrix
from src.dashboard.utils.charts import umap_scatter, radar_chart
from src.dashboard.utils.styles import ARCHETYPE_COLORS
from src.dashboard.utils.sidebar import render_sidebar
from src.dashboard.utils.flags import flag as get_flag

render_sidebar()

if 'selected_team' not in st.session_state:
    st.session_state.selected_team = 'Spain'

df         = load_team_profiles()
sim_matrix = load_similarity_matrix()
all_teams  = sorted(df['team_name'].tolist())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin:0 0 0.2rem; font-size:1.8rem; font-weight:900;'
    'letter-spacing:-0.03em; color:#E8EDF8;">🧬 Team DNA Explorer</h2>'
    '<p style="color:#6B7394; margin:0 0 1rem; font-size:0.85rem;">'
    'Explore tactical positioning of all 48 WC2026 teams. Select a team to see its full style profile.</p>',
    unsafe_allow_html=True,
)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem; color:#5A6490; font-weight:700;'
        'text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.5rem;">Filter</div>',
        unsafe_allow_html=True,
    )
    arch_filter  = st.multiselect('Archetypes', options=list(ARCHETYPE_COLORS.keys()),
                                   default=list(ARCHETYPE_COLORS.keys()))
    show_proxy   = st.toggle('Show proxy teams', value=True)
    compare_team = st.selectbox('Compare radar with', ['None'] + all_teams)

df_filtered = df[df['archetype_name'].isin(arch_filter)]
if not show_proxy:
    df_filtered = df_filtered[~df_filtered['is_proxy']]

# ── Row 1: Team selector + UMAP ───────────────────────────────────────────────
sel_col, _ = st.columns([4, 8])
with sel_col:
    selected = st.selectbox(
        'Select team',
        options=all_teams,
        index=all_teams.index(st.session_state.selected_team)
              if st.session_state.selected_team in all_teams else 0,
        key='team_select_main',
    )
    if selected != st.session_state.selected_team:
        st.session_state.selected_team = selected
        st.rerun()

team = st.session_state.selected_team

fig_umap = umap_scatter(df_filtered, highlight_teams=[team], size=480)
st.plotly_chart(fig_umap, use_container_width=True,
                config={'displayModeBar': False}, key='umap_main')
st.markdown(
    '<div style="font-size:0.74rem; color:#5A6490; margin-top:-0.5rem; margin-bottom:1rem;">'
    'Each dot = a team &nbsp;·&nbsp; Position = tactical style &nbsp;·&nbsp;'
    ' Hover any dot for details &nbsp;·&nbsp; Open markers = proxy vector</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Row 2: Team detail panel ───────────────────────────────────────────────────
row  = df[df['team_name'] == team]
if row.empty:
    st.warning(f'{team} not found in database.')
    st.stop()

r        = row.iloc[0]
arch     = r['archetype_name']
color    = ARCHETYPE_COLORS.get(arch, '#888888')
is_proxy = bool(r['is_proxy'])

badge_cls = {
    'High Press':         'badge-highpress',
    'Possession Control': 'badge-possession',
    'Counter-Attack':     'badge-counter',
    'Deep Block':         'badge-deep',
}.get(arch, 'badge-proxy')

proxy_badge = (
    '<span class="badge badge-proxy" style="margin-left:6px;">PROXY</span>'
    if is_proxy else ''
)

# Team name header (full width)
st.markdown(
    f'<div style="margin-bottom:1rem;">'
    f'<div style="font-size:2.2rem; font-weight:900; color:#E8EDF8; letter-spacing:-0.03em; line-height:1.1;">'
    f'{get_flag(team)} {team}</div>'
    f'<div style="margin-top:0.5rem; display:flex; gap:7px; flex-wrap:wrap;">'
    f'<span class="badge {badge_cls}">{arch}</span>'
    f'{proxy_badge}'
    f'</div></div>',
    unsafe_allow_html=True,
)

col_radar, col_stats, col_similar = st.columns([4, 4, 4])

# ── Radar ─────────────────────────────────────────────────────────────────────
with col_radar:
    st.markdown(
        '<div style="font-size:0.72rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin-bottom:0.5rem;">Style DNA Radar</div>',
        unsafe_allow_html=True,
    )
    compare = compare_team if compare_team != 'None' and compare_team != team else None
    fig_radar = radar_chart(team, df, compare_team=compare)
    st.plotly_chart(fig_radar, use_container_width=True,
                    config={'displayModeBar': False}, key='radar_main')

# ── Key stats ─────────────────────────────────────────────────────────────────
with col_stats:
    st.markdown(
        '<div style="font-size:0.72rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin-bottom:0.75rem;">Key Stats</div>',
        unsafe_allow_html=True,
    )

    stats_to_show = [
        ('PPDA',           'avg_ppda',                  '{:.2f}',  True),
        ('xG Ratio',       'avg_xg_ratio',               '{:.2f}',  False),
        ('Possession %',   'avg_possession_pct',         '{:.1f}%', False),
        ('xG Created p90', 'avg_xg_created_p90',         '{:.2f}',  False),
        ('Pass Accuracy',  'avg_pass_completion_pct',    '{:.1f}%', False),
        ('Press Success',  'avg_pressure_success_rate',  '{:.1f}%', False),
    ]

    for label, col_name, fmt, lower_better in stats_to_show:
        val = r.get(col_name)
        if val is None or (isinstance(val, float) and val != val):
            st.markdown(
                f'<div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<div style="display:flex; justify-content:space-between;">'
                f'<span style="color:#7B83A8; font-size:0.82rem;">{label}</span>'
                f'<span style="color:#4A5270; font-size:0.82rem;">N/A</span></div></div>',
                unsafe_allow_html=True,
            )
            continue

        val      = float(val)
        col_data = df[col_name].dropna()
        rank     = int((col_data < val).sum() if lower_better else (col_data > val).sum()) + 1
        total    = len(col_data)
        pct      = (total - rank) / max(total - 1, 1)

        st.markdown(
            f'<div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.3rem;">'
            f'<span style="color:#8B93B8; font-size:0.83rem; font-weight:500;">{label}</span>'
            f'<span style="display:flex; align-items:baseline; gap:6px;">'
            f'<b style="color:#E8EDF8; font-size:1rem; font-weight:800; letter-spacing:-0.01em;">{fmt.format(val)}</b>'
            f'<span style="color:#3A4060; font-size:0.7rem;">#{rank}/{total}</span>'
            f'</span></div>'
            f'<div style="height:3px; background:rgba(255,255,255,0.05); border-radius:2px;">'
            f'<div style="height:100%; width:{pct*100:.1f}%; background:{color}; border-radius:2px; opacity:0.7;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    mp     = int(r.get('matches_played', 0) or 0)
    mp_str = 'Proxy vector' if mp < 0 else f'{mp} StatsBomb matches'
    wr     = r.get('win_rate')
    wr_str = f'{wr*100:.0f}%' if wr is not None and wr == wr else 'N/A'
    st.markdown(
        f'<div style="font-size:0.74rem; color:#4A5270; margin-top:0.75rem;">'
        f'📊 {mp_str} &nbsp;·&nbsp; Win rate: <b style="color:#7B83A8;">{wr_str}</b></div>',
        unsafe_allow_html=True,
    )

# ── Similar teams ─────────────────────────────────────────────────────────────
with col_similar:
    st.markdown(
        '<div style="font-size:0.72rem; color:#5A6490; font-weight:700; text-transform:uppercase;'
        'letter-spacing:0.1em; margin-bottom:0.75rem;">Most Similar Teams</div>',
        unsafe_allow_html=True,
    )

    if sim_matrix is not None and team in sim_matrix.columns:
        sims = (
            sim_matrix[team]
            .drop(index=team, errors='ignore')
            .sort_values(ascending=False)
        )
        top_sims = sims.head(6)
        for i, (sim_team, sim_val) in enumerate(top_sims.items(), 1):
            sim_arch  = df.loc[df['team_name'] == sim_team, 'archetype_name']
            sim_arch  = sim_arch.iloc[0] if not sim_arch.empty else '—'
            sim_color = ARCHETYPE_COLORS.get(sim_arch, '#888888')
            bar_w     = sim_val * 100
            st.markdown(
                f'<div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">'
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'<span style="color:#3A4060; font-size:0.74rem; font-weight:700; min-width:16px;">{i}.</span>'
                f'<div>'
                f'<div style="font-size:0.9rem; font-weight:700; color:#D0D8F0;">'
                f'{get_flag(sim_team)} {sim_team}</div>'
                f'<div style="font-size:0.7rem; color:{sim_color}; margin-top:1px;">{sim_arch}</div>'
                f'</div></div>'
                f'<span style="color:#00D4A0; font-weight:800; font-size:0.88rem;">{sim_val:.3f}</span>'
                f'</div>'
                f'<div style="height:2px; background:rgba(255,255,255,0.04); border-radius:1px;">'
                f'<div style="height:100%; width:{bar_w:.0f}%; background:{sim_color}; border-radius:1px; opacity:0.5;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        same_arch = df[
            (df['archetype_name'] == arch) &
            (df['team_name'] != team) &
            (~df['is_proxy'])
        ].head(5)
        if same_arch.empty:
            st.caption('Similarity data not available.')
        else:
            st.caption('Showing same-archetype teams.')
            for i, (_, sr) in enumerate(same_arch.iterrows(), 1):
                st.markdown(
                    f'<div style="font-size:0.88rem; padding:0.4rem 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.05);">'
                    f'{i}. <b style="color:#D0D8F0;">'
                    f'{get_flag(sr["team_name"])} {sr["team_name"]}</b>'
                    f'<span style="color:#5A6490; font-size:0.76rem;"> — {arch}</span></div>',
                    unsafe_allow_html=True,
                )
