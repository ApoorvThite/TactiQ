"""Reusable Plotly chart functions for TactiQ dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.utils.styles import (
    ARCHETYPE_COLORS, ARCHETYPE_COLORS_MUTED,
    PLOTLY_LAYOUT, PROXY_COLOR,
    TACTIQ_CARD, TACTIQ_DARK, TACTIQ_MUTED, TACTIQ_TEXT, TACTIQ_TEAL, TACTIQ_AMBER,
)

# ─── Human-readable feature labels ────────────────────────────────────────────
FEATURE_LABELS = {
    'delta_avg_possession_pct':                    'Possession Advantage',
    'delta_avg_ppda':                              'Pressing Advantage (PPDA)',
    'delta_avg_pressure_success_rate':             'Press Success Rate',
    'delta_avg_xg_created_p90':                   'xG Creation Advantage',
    'delta_avg_xg_ratio':                          'xG Ratio Advantage',
    'delta_avg_progressive_carry_pct':             'Progressive Carries',
    'delta_avg_pass_completion_pct':               'Pass Accuracy',
    'delta_avg_passes_final_third_p90':           'Final Third Entries',
    'delta_avg_pass_completion_under_pressure_pct': 'Composure Under Press',
    'delta_avg_set_piece_shot_pct':               'Set Piece Threat',
    'is_home':                                     'Home Advantage',
    'form_points_delta':                           'Form Advantage',
    'archetype_matchup_id':                        'Archetype Matchup',
    'delta_matches_played':                        'Experience Edge',
    'competition_weight':                          'Competition Level',
}

RADAR_AXES = [
    ('avg_possession_pct',      'Possession %'),
    ('press_intensity',         'Press Intensity'),
    ('avg_pressure_success_rate','Press Success'),
    ('avg_xg_created_p90',      'xG Created p90'),
    ('avg_xg_ratio',            'xG Ratio'),
    ('avg_progressive_carry_pct','Progressive Carries'),
    ('avg_pass_completion_pct', 'Pass Accuracy'),
    ('avg_set_piece_shot_pct',  'Set Pieces'),
]


def _norm_col(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ─── 1. UMAP Scatter ──────────────────────────────────────────────────────────
def umap_scatter(
    df: pd.DataFrame,
    highlight_teams: list[str] | None = None,
    size: int = 550,
) -> go.Figure:
    """
    UMAP scatter of all WC2026 teams, colored by archetype.
    Labels shown for highlighted teams only; all others use hover.
    Proxy teams shown with open markers.
    """
    from src.dashboard.utils.flags import flag as get_flag

    fig = go.Figure()
    hl  = set(highlight_teams or [])

    for arch, color in ARCHETYPE_COLORS.items():
        sub = df[df['archetype_name'] == arch]
        if sub.empty:
            continue

        real  = sub[~sub['is_proxy']]
        proxy = sub[sub['is_proxy']]

        for df_seg, is_proxy in [(real, False), (proxy, True)]:
            if df_seg.empty:
                continue

            symbol = 'circle-open' if is_proxy else 'circle'

            # Split highlighted vs regular so we can control labels separately
            is_hl = df_seg['team_name'].isin(hl)
            for hl_group, df_sub in [('hl', df_seg[is_hl]), ('reg', df_seg[~is_hl])]:
                if df_sub.empty:
                    continue

                highlighted = hl_group == 'hl'
                dot_sizes   = [20 if highlighted else 9] * len(df_sub)
                opacity     = 0.5 if is_proxy else (1.0 if highlighted else 0.75)

                hover_text = [
                    (f"<b>{get_flag(row.team_name)} {row.team_name}</b><br>"
                     f"<span style='color:#aaa'>{row.archetype_name}</span><br>"
                     f"PPDA: {row.avg_ppda:.2f} &nbsp;·&nbsp; "
                     f"xG: {row.avg_xg_ratio:.2f} &nbsp;·&nbsp; "
                     f"Poss: {row.avg_possession_pct:.1f}%"
                     + ("<br><i>Proxy vector</i>" if is_proxy else ""))
                    for row in df_sub.itertuples()
                ]

                # Show text label only for highlighted teams
                label_text = (
                    [f"{get_flag(t)} {t}" for t in df_sub['team_name']]
                    if highlighted else [''] * len(df_sub)
                )

                fig.add_trace(go.Scatter(
                    x=df_sub['umap_x'],
                    y=df_sub['umap_y'],
                    mode='markers+text' if highlighted else 'markers',
                    name=arch + (' (proxy)' if is_proxy else ''),
                    legendgroup=arch,
                    showlegend=(not is_proxy and not highlighted),
                    marker=dict(
                        size=dot_sizes,
                        color=color,
                        symbol=symbol,
                        line=dict(color=color, width=1.5 if highlighted else 0.8),
                        opacity=opacity,
                    ),
                    text=label_text,
                    textposition='top center',
                    textfont=dict(
                        size=11,
                        color=TACTIQ_TEXT,
                        family='Inter, system-ui, sans-serif',
                    ),
                    hovertext=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(
                        bgcolor='#0F1520',
                        bordercolor=color,
                        font=dict(size=12, color=TACTIQ_TEXT),
                    ),
                    customdata=df_sub['team_name'].values,
                ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=size,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
        hovermode='closest',
    )
    return fig


# ─── 2. Radar / Spider chart ──────────────────────────────────────────────────
def radar_chart(
    team_name: str,
    df: pd.DataFrame,
    compare_team: str | None = None,
) -> go.Figure:
    """
    Radar chart for one team (with optional comparison overlay).
    Features normalised 0–1 across all teams in df.
    """
    df = df.copy()
    if 'press_intensity' not in df.columns:
        df['press_intensity'] = 1.0 / df['avg_ppda'].clip(lower=0.5)

    feature_cols = [f for f, _ in RADAR_AXES]
    labels       = [l for _, l in RADAR_AXES]

    for col in feature_cols:
        df[col + '_n'] = _norm_col(df[col].fillna(df[col].median()))

    norm_cols = [c + '_n' for c in feature_cols]

    def _get_row(name):
        r = df[df['team_name'] == name]
        return r.iloc[0] if not r.empty else None

    row_a = _get_row(team_name)
    if row_a is None:
        return go.Figure()

    arch_a = row_a['archetype_name']
    color_a = ARCHETYPE_COLORS.get(arch_a, TACTIQ_TEAL)
    values_a = [float(row_a[c]) for c in norm_cols] + [float(row_a[norm_cols[0]])]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_a,
        theta=labels + [labels[0]],
        fill='toself',
        name=team_name,
        line=dict(color=color_a, width=2),
        fillcolor=ARCHETYPE_COLORS_MUTED.get(arch_a, 'rgba(29,158,117,0.2)'),
    ))

    if compare_team:
        row_b = _get_row(compare_team)
        if row_b is not None:
            arch_b   = row_b['archetype_name']
            color_b  = ARCHETYPE_COLORS.get(arch_b, '#888888')
            values_b = [float(row_b[c]) for c in norm_cols] + [float(row_b[norm_cols[0]])]
            fig.add_trace(go.Scatterpolar(
                r=values_b,
                theta=labels + [labels[0]],
                fill='toself',
                name=compare_team,
                line=dict(color=color_b, width=2, dash='dash'),
                fillcolor='rgba(136,136,136,0.15)',
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        polar=dict(
            bgcolor=TACTIQ_CARD,
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=8, color=TACTIQ_MUTED),
                gridcolor='#2D3250',
                linecolor='#2D3250',
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=TACTIQ_TEXT),
                gridcolor='#2D3250',
                linecolor='#2D3250',
            ),
        ),
        legend=dict(font=dict(size=11), orientation='h', y=-0.05),
        title=dict(text=f'Style DNA — {team_name}', font=dict(size=13)),
    )
    return fig


# ─── 3. Probability bar (win/draw/loss) ───────────────────────────────────────
def prob_bars(
    team_a: str,
    team_b: str,
    p_win: float,
    p_draw: float,
    p_loss: float,
    arch_a: str = '',
    arch_b: str = '',
) -> go.Figure:
    """Horizontal stacked bar with percentage text perfectly centered in each segment."""
    colors = ['#00D4A0', '#FBBF24', '#EF4444']
    values = [p_win, p_draw, p_loss]
    texts  = [
        f'{p_win*100:.0f}%'  if p_win  > 0.07 else '',
        f'{p_draw*100:.0f}%' if p_draw > 0.07 else '',
        f'{p_loss*100:.0f}%' if p_loss > 0.07 else '',
    ]

    fig = go.Figure()
    for val, color, txt in zip(values, colors, texts):
        fig.add_trace(go.Bar(
            x=[val], y=[''], orientation='h',
            marker_color=color,
            marker_line_width=0,
            text=txt,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(
                size=14,
                color='white',
                family='Inter, system-ui, sans-serif',
            ),
            hoverinfo='skip',
            showlegend=False,
        ))

    fig.update_layout(
        barmode='stack',
        height=56,
        bargap=0,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif', size=12),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


# ─── 4. Qualification heatmap ─────────────────────────────────────────────────
def qualification_heatmap(df_qual: pd.DataFrame) -> go.Figure:
    """12-group × 4-rank heatmap of p(qualify for R32)."""
    groups = sorted(df_qual['group_label'].unique())
    ranks  = ['1st', '2nd', '3rd', '4th']

    z_vals   = []
    text_vals = []

    for rank_i in range(4):
        row_z, row_t = [], []
        for g in groups:
            g_df = df_qual[df_qual['group_label'] == g].reset_index(drop=True)
            if rank_i < len(g_df):
                r    = g_df.iloc[rank_i]
                prob = r['p_qualify_r32']
                name = r['team_name']
                row_z.append(prob)
                row_t.append(f'{name}<br>{prob*100:.0f}%')
            else:
                row_z.append(0)
                row_t.append('')
        z_vals.append(row_z)
        text_vals.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=groups,
        y=ranks,
        text=text_vals,
        texttemplate='%{text}',
        colorscale='RdYlGn',
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='p(R32)', font=dict(color=TACTIQ_TEXT, size=11)),
            tickformat='.0%',
            tickfont=dict(color=TACTIQ_TEXT, size=10),
        ),
        hoverongaps=False,
    ))

    fig.update_layout(
        paper_bgcolor=TACTIQ_DARK,
        plot_bgcolor=TACTIQ_CARD,
        font=dict(color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif', size=11),
        margin=dict(l=16, r=16, t=40, b=16),
        height=300,
        title=dict(text='Qualification Probability by Group', font=dict(size=13)),
        xaxis=dict(side='top', tickfont=dict(size=11, color=TACTIQ_TEXT)),
        yaxis=dict(tickfont=dict(size=11, color=TACTIQ_TEXT), autorange='reversed'),
    )
    return fig


# ─── 5. SHAP waterfall bar chart ─────────────────────────────────────────────
def shap_waterfall_table(
    shap_dict: dict,
    predicted_class: str,
    n_features: int = 10,
) -> go.Figure:
    """
    Interactive SHAP attribution bar chart (replaces matplotlib waterfall).
    Positive = teal, negative = coral.
    """
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n_features]
    feats  = [FEATURE_LABELS.get(k, k.replace('delta_', '').replace('_', ' ').title())
              for k, _ in items]
    values = [v for _, v in items]
    colors = [TACTIQ_TEAL if v >= 0 else '#EF4444' for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=feats,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.3f}' for v in values],
        textposition='outside',
        textfont=dict(size=10, color=TACTIQ_TEXT),
        hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>',
    ))

    fig.add_vline(x=0, line_width=1, line_color='#8B92A8')

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(280, n_features * 32),
        title=dict(
            text=f'Why {predicted_class.upper()}? — Feature Contributions',
            font=dict(size=13),
        ),
        xaxis=dict(
            title='SHAP Value', gridcolor='#2D3250',
            tickfont=dict(color=TACTIQ_MUTED),
            zerolinecolor='#2D3250',
        ),
        yaxis=dict(
            autorange='reversed', tickfont=dict(size=10, color=TACTIQ_TEXT),
        ),
    )
    return fig


# ─── 6. Upset scatter plot ────────────────────────────────────────────────────
def upset_scatter(df_upset: pd.DataFrame) -> go.Figure:
    """Scatter: p(favourite wins) vs upset risk. One dot per upset candidate."""
    fig = go.Figure()

    for arch in df_upset['arch_underdog'].unique():
        sub = df_upset[df_upset['arch_underdog'] == arch]
        color = ARCHETYPE_COLORS.get(arch, '#888888')
        fig.add_trace(go.Scatter(
            x=sub['p_win'],
            y=sub['p_not_fav_win'],
            mode='markers+text',
            name=arch,
            marker=dict(size=14, color=color, opacity=0.85,
                        line=dict(color='white', width=0.8)),
            text=sub['underdog'],
            textposition='top center',
            textfont=dict(size=9, color=TACTIQ_TEXT),
            hovertemplate=(
                '<b>%{text}</b> vs %{customdata}<br>'
                'p(fav wins): %{x:.1%}<br>'
                'p(upset): %{y:.1%}<extra></extra>'
            ),
            customdata=sub['favourite'],
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title=dict(text='Upset Risk vs Favourite Win Probability', font=dict(size=13)),
        xaxis=dict(
            title='p(Favourite Wins)', tickformat='.0%',
            gridcolor='#2D3250', tickfont=dict(color=TACTIQ_MUTED),
        ),
        yaxis=dict(
            title='p(Upset — Fav does not win)', tickformat='.0%',
            gridcolor='#2D3250', tickfont=dict(color=TACTIQ_MUTED),
        ),
        legend=dict(font=dict(size=11), orientation='h', y=-0.15),
    )
    return fig


# ─── 7. Group qualification bar chart (per group) ─────────────────────────────
def group_qual_bars(df_group: pd.DataFrame, group_label: str) -> go.Figure:
    """Horizontal bars for qualification probability of 4 teams in one group."""
    df = df_group.sort_values('p_qualify_r32', ascending=True)

    colors = [ARCHETYPE_COLORS.get(a, '#888888') for a in df['archetype_name']]

    fig = go.Figure(go.Bar(
        x=df['p_qualify_r32'],
        y=df['team_name'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0%}' for v in df['p_qualify_r32']],
        textposition='outside',
        textfont=dict(size=11, color=TACTIQ_TEXT),
        hovertemplate='%{y}: %{x:.1%}<extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor=TACTIQ_DARK,
        plot_bgcolor=TACTIQ_CARD,
        font=dict(color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif', size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=220,
        title=dict(text=f'Group {group_label} — p(Qualify for R32)', font=dict(size=12)),
        xaxis=dict(range=[0, 1.12], tickformat='.0%', showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11, color=TACTIQ_TEXT)),
    )
    return fig


# ─── 8. Tournament contenders chart ──────────────────────────────────────────
def tournament_contenders_chart(
    df_qual: pd.DataFrame,
    n: int = 16,
) -> go.Figure:
    """
    Horizontal bar chart of top-N teams by p(qualify for R32),
    with a diamond marker for p(win group). Colored by archetype.
    """
    from src.dashboard.utils.flags import flag as get_flag

    top = df_qual.nlargest(n, 'p_qualify_r32').sort_values('p_qualify_r32', ascending=True)

    colors = [ARCHETYPE_COLORS.get(a, '#888888') for a in top['archetype_name']]
    labels = [f"{get_flag(t)}  {t}" for t in top['team_name']]

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        x=[1.0] * len(top),
        y=labels,
        orientation='h',
        marker_color='rgba(255,255,255,0.03)',
        marker_line_width=0,
        showlegend=False,
        hoverinfo='skip',
    ))

    # Main bar: p(qualify R32)
    fig.add_trace(go.Bar(
        x=top['p_qualify_r32'],
        y=labels,
        orientation='h',
        marker_color=colors,
        marker_opacity=0.82,
        marker_line_width=0,
        name='p(Qualify R32)',
        text=[f'{v:.0%}' for v in top['p_qualify_r32']],
        textposition='outside',
        textfont=dict(size=11, color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif'),
        hovertemplate='<b>%{y}</b><br>p(Qualify): %{x:.1%}<extra></extra>',
    ))

    # Diamond marker: p(win group)
    fig.add_trace(go.Scatter(
        x=top['p_first'],
        y=labels,
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=10,
            color=TACTIQ_AMBER,
            line=dict(color='rgba(0,0,0,0.4)', width=1),
        ),
        name='p(Win Group)',
        hovertemplate='<b>%{y}</b><br>p(Win Group): %{x:.1%}<extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor=TACTIQ_DARK,
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif', size=12),
        margin=dict(l=16, r=70, t=10, b=16),
        height=max(400, n * 30),
        barmode='overlay',
        legend=dict(
            orientation='h', y=1.06, x=0,
            font=dict(size=11, color=TACTIQ_MUTED),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(
            range=[0, 1.18],
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.04)',
            tickfont=dict(color=TACTIQ_MUTED, size=10),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11.5, color=TACTIQ_TEXT),
            gridcolor='rgba(255,255,255,0.02)',
        ),
    )
    return fig
