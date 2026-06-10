"""TactiQ design system — color constants and CSS injection."""

ARCHETYPE_COLORS = {
    'High Press':         '#00D4A0',
    'Possession Control': '#9F8FF0',
    'Counter-Attack':     '#F07050',
    'Deep Block':         '#5EA9E8',
}

ARCHETYPE_COLORS_MUTED = {
    'High Press':         'rgba(0,212,160,0.18)',
    'Possession Control': 'rgba(159,143,240,0.18)',
    'Counter-Attack':     'rgba(240,112,80,0.18)',
    'Deep Block':         'rgba(94,169,232,0.18)',
}

PROXY_COLOR   = '#4A5270'
RESULT_COLORS = {'win': '#00D4A0', 'draw': '#FBBF24', 'loss': '#EF4444'}

CONFIDENCE_COLORS = {'high': '#00D4A0', 'medium': '#FBBF24', 'low': '#EF4444'}

TACTIQ_DARK   = '#080C14'
TACTIQ_CARD   = '#0F1520'
TACTIQ_BORDER = '#1A2035'
TACTIQ_TEXT   = '#E8EDF8'
TACTIQ_MUTED  = '#6B7394'
TACTIQ_TEAL   = '#00D4A0'
TACTIQ_AMBER  = '#FBBF24'
TACTIQ_CORAL  = '#EF4444'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=TACTIQ_DARK,
    plot_bgcolor='#0A0F1C',
    font=dict(color=TACTIQ_TEXT, family='Inter, system-ui, sans-serif', size=12),
    margin=dict(l=16, r=16, t=40, b=16),
)


def confidence_label(p_max: float) -> str:
    if p_max >= 0.60:
        return 'HIGH'
    if p_max >= 0.45:
        return 'MEDIUM'
    return 'LOW'


def confidence_color(p_max: float) -> str:
    return CONFIDENCE_COLORS[confidence_label(p_max).lower()]
