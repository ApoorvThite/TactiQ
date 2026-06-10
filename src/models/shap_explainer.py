"""Phase 5 Step 1–3 — SHAP TreeExplainer: fit, global analysis, waterfall plots."""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')
shap.initjs()

ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'docs' / 'figures'
PROCESSED_DIR = ROOT / 'data' / 'processed'

FEATURE_NAMES = [
    'delta_avg_possession_pct',
    'delta_avg_ppda',
    'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90',
    'delta_avg_xg_ratio',
    'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct',
    'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct',
    'delta_avg_set_piece_shot_pct',
    'is_home',
    'form_points_delta',
    'archetype_matchup_id',
    'delta_matches_played',
    'competition_weight',
]

LABEL_NAMES  = ['win', 'draw', 'loss']
CLASS_COLORS = ['#e63946', '#2a9d8f', '#457b9d']


# ─────────────────────────────────────────────────────────────────────────── #
#  Step 1 — Fit SHAP TreeExplainer                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def fit_explainer():
    with open(MODELS_DIR / 'xgboost_matchup.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(PROCESSED_DIR / 'matchup_dataset.csv')
    X  = df[FEATURE_NAMES].values.astype(float)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)   # shape: (460, 15, 3)

    with open(MODELS_DIR / 'shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)

    print('SHAP TreeExplainer fitted')
    print(f'  Training rows          : {X.shape[0]}')
    print(f'  Features               : {X.shape[1]}')
    print(f'  SHAP values shape      : {shap_values.values.shape}')
    for i, cls in enumerate(LABEL_NAMES):
        ev = explainer.expected_value
        ev_val = ev[i] if hasattr(ev, '__len__') else float(ev)
        print(f'  Expected value (class {i} / {cls}): {ev_val:.4f}')

    return model, explainer, shap_values, X, df


# ─────────────────────────────────────────────────────────────────────────── #
#  Step 2 — Global SHAP analysis                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def global_analysis(explainer, shap_values, X):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')

    sv = shap_values.values   # (460, 15, 3)

    # ── 2a: Mean |SHAP| bar plot ──────────────────────────────────────────
    mean_abs = {cls: np.abs(sv[:, :, i]).mean(axis=0) for i, cls in enumerate(LABEL_NAMES)}

    for cls in LABEL_NAMES:
        ranked = sorted(zip(FEATURE_NAMES, mean_abs[cls]), key=lambda x: x[1], reverse=True)
        print(f'\nMean |SHAP| per feature — Predicting {cls.upper()}')
        for feat, val in ranked:
            print(f'  {feat:<50}: {val:.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    for ax_i, (cls, color) in enumerate(zip(LABEL_NAMES, CLASS_COLORS)):
        ax = axes[ax_i]
        vals  = mean_abs[cls]
        order = np.argsort(vals)
        feats_ordered = [FEATURE_NAMES[j] for j in order]
        vals_ordered  = vals[order]
        ax.barh(feats_ordered, vals_ordered, color=color, alpha=0.85)
        ax.set_title(f'Predicting {cls.upper()}', fontsize=11, color=color)
        ax.set_xlabel('Mean |SHAP value|', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle('TactiQ — Global SHAP Feature Importance', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig8_shap_bar_global.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved → docs/figures/fig8_shap_bar_global.png')

    # ── 2b: Beeswarm (WIN class) ─────────────────────────────────────────
    # Each dot = one training row; color = feature value (red=high, blue=low)
    # The plot reveals: high delta_xg_ratio (red) pushes strongly toward WIN;
    # high delta_ppda (red, meaning Team A presses LESS) surprisingly also
    # has mixed effects — pressing harder (blue/negative) tends to correlate
    # with wins, while being outpressed (positive PPDA delta) reduces win prob.
    # form_points_delta shows a clear monotonic pattern: better recent form
    # (red) consistently adds to win probability.
    win_shap_exp = shap.Explanation(
        values=sv[:, :, 0],
        base_values=np.full(sv.shape[0], explainer.expected_value[0]),
        data=X,
        feature_names=FEATURE_NAMES,
    )
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    plt.style.use('dark_background')
    shap.plots.beeswarm(win_shap_exp, show=False, max_display=15)
    plt.title('TactiQ — SHAP Beeswarm (WIN class)', fontsize=12, pad=10)
    fig = plt.gcf()
    fig.savefig(FIGURES_DIR / 'fig9_shap_beeswarm_win.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print('Saved → docs/figures/fig9_shap_beeswarm_win.png')

    # ── 2c: Heatmap (WIN class, sorted by predicted win prob) ────────────
    # Sort by sum of SHAP values for WIN (proxy for predicted win prob)
    win_shap_sum = sv[:, :, 0].sum(axis=1)
    sort_idx     = np.argsort(win_shap_sum)[::-1]

    win_shap_sorted = shap.Explanation(
        values=sv[sort_idx, :, 0],
        base_values=np.full(sv.shape[0], explainer.expected_value[0]),
        data=X[sort_idx],
        feature_names=FEATURE_NAMES,
    )
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    shap.plots.heatmap(win_shap_sorted, show=False, max_display=15)
    plt.title('TactiQ — SHAP Heatmap WIN class (sorted by win score)', fontsize=11)
    fig = plt.gcf()
    fig.savefig(FIGURES_DIR / 'fig10_shap_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print('Saved → docs/figures/fig10_shap_heatmap.png')

    # ── 2d: Interaction effects (top-3 features for WIN) ─────────────────
    top3_idx = np.argsort(mean_abs['win'])[::-1][:3]
    top3_feat = [FEATURE_NAMES[i] for i in top3_idx]
    print(f'\nTop-3 WIN features for interaction analysis: {top3_feat}')

    try:
        # shap_interaction: (460, 15, 15) per class — compute for class 0 (WIN)
        import xgboost as xgb
        with open(MODELS_DIR / 'xgboost_matchup.pkl', 'rb') as f:
            raw_model = pickle.load(f)
        si = explainer.shap_interaction_values(X)
        # si shape: (460, 15, 15, 3) or (460, 15, 15) depending on shap version
        if si.ndim == 4:
            si_win = si[:, :, :, 0]
        else:
            si_win = si[:, :, :, 0] if si.shape[-1] == 3 else si

        pairs = [(top3_idx[0], top3_idx[1]),
                 (top3_idx[0], top3_idx[2]),
                 (top3_idx[1], top3_idx[2])]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
        plt.style.use('dark_background')
        for ax_i, (i, j) in enumerate(pairs):
            ax = axes[ax_i]
            x_vals    = X[:, i]
            inter_vals = si_win[:, i, j]
            color_vals = X[:, j]
            sc = ax.scatter(x_vals, inter_vals, c=color_vals, cmap='RdYlGn',
                            alpha=0.7, s=20)
            plt.colorbar(sc, ax=ax, label=FEATURE_NAMES[j][:20], fraction=0.04)
            ax.set_xlabel(FEATURE_NAMES[i][:30], fontsize=8)
            ax.set_ylabel(f'SHAP interaction\n({FEATURE_NAMES[i][:15]} × {FEATURE_NAMES[j][:15]})', fontsize=7)
            ax.axhline(0, color='white', linewidth=0.8, alpha=0.4)
        fig.suptitle('TactiQ — SHAP Interaction Effects (WIN class)', fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'fig11_shap_interactions.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved → docs/figures/fig11_shap_interactions.png')
    except Exception as e:
        print(f'Interaction values skipped ({e}) — saving placeholder')
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.text(0.5, 0.5, f'SHAP interactions\n(top-3: {", ".join(top3_feat)})\n\n'
                f'Skipped: {str(e)[:80]}',
                ha='center', va='center', transform=ax.transAxes,
                color='white', fontsize=9, wrap=True)
        ax.set_title('TactiQ — SHAP Interaction Effects', fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'fig11_shap_interactions.png', dpi=150)
        plt.close(fig)
        print('Saved → docs/figures/fig11_shap_interactions.png (placeholder)')

    return mean_abs, top3_feat


# ─────────────────────────────────────────────────────────────────────────── #
#  Step 3 — Per-matchup waterfall plots + text SHAP summaries                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_feature_vector(team_a, team_b, is_home=True, form_delta=0.0,
                           competition='FIFA World Cup'):
    """Construct 15-feature input from team dicts (each has style_vector, archetype_name, matches_played)."""
    ARCHETYPES = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}
    COMP_WEIGHTS = {'FIFA World Cup': 1.0, 'UEFA Euro': 0.8}

    sv_a   = np.array(team_a['style_vector'])
    sv_b   = np.array(team_b['style_vector'])
    delta  = sv_a - sv_b

    arch_a_id  = ARCHETYPES.get(team_a['archetype_name'], 0)
    arch_b_id  = ARCHETYPES.get(team_b['archetype_name'], 0)
    matchup_id = arch_a_id * 4 + arch_b_id
    comp_w     = COMP_WEIGHTS.get(competition, 0.7)
    delta_mp   = team_a['matches_played'] - team_b['matches_played']

    return np.concatenate([delta, [float(is_home), float(form_delta),
                                   float(matchup_id), float(delta_mp), float(comp_w)]])


def waterfall_and_summary(explainer, model_raw, model_calib, team_a, team_b,
                           fig_path, label=None):
    """
    Generate SHAP waterfall plot + text summary for one matchup.
    Returns dict with prediction info and shap values.
    """
    x_vec = _build_feature_vector(team_a, team_b)
    X_single = x_vec.reshape(1, -1)

    # Calibrated probabilities for display
    calib_proba = model_calib.predict_proba(X_single)[0]
    p_win, p_draw, p_loss = float(calib_proba[0]), float(calib_proba[1]), float(calib_proba[2])
    predicted_idx = int(np.argmax(calib_proba))
    predicted_cls = ['win', 'draw', 'loss'][predicted_idx]

    # SHAP values from raw model
    sv_single = explainer(X_single)   # (1, 15, 3)
    shap_for_class = sv_single.values[0, :, predicted_idx]
    ev_val = explainer.expected_value
    ev = ev_val[predicted_idx] if hasattr(ev_val, '__len__') else float(ev_val)

    # Waterfall plot
    plt.style.use('dark_background')
    exp = shap.Explanation(
        values=shap_for_class,
        base_values=ev,
        data=X_single[0],
        feature_names=FEATURE_NAMES,
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    shap.plots.waterfall(exp, show=False, max_display=15)
    plt.title(
        f'TactiQ — SHAP Waterfall: {team_a["team_name"]} vs {team_b["team_name"]}'
        f'\nPredicted: {predicted_cls.upper()} (p={max(p_win,p_draw,p_loss):.2f})',
        fontsize=10, pad=8
    )
    plt.gcf().savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # Text summary
    ev_display = ev if abs(ev) < 5 else 0.0   # base rate in log-odds; show raw
    print(f'{"─"*55}')
    name_a, name_b = team_a["team_name"], team_b["team_name"]
    print(f' SHAP Explanation: {name_a} vs {name_b} → {predicted_cls.upper()}')
    print(f'{"─"*55}')
    print(f' Base rate ({predicted_cls} class log-odds): {ev:.3f}')
    print(f'\n Feature contributions to {predicted_cls.upper()} probability:')
    pairs = sorted(zip(FEATURE_NAMES, shap_for_class), key=lambda x: abs(x[1]), reverse=True)
    for feat, sv_val in pairs[:8]:
        arrow = '↑' if sv_val > 0 else '↓'
        print(f'   {feat:<46} {sv_val:+.3f}  {arrow}')
    print(f'\n Final probs  Win: {p_win:.2f}  Draw: {p_draw:.2f}  Loss: {p_loss:.2f}')
    print(f'{"─"*55}')

    return {
        'team_a_name':    team_a['team_name'],
        'team_b_name':    team_b['team_name'],
        'team_a_id':      team_a.get('team_id'),
        'team_b_id':      team_b.get('team_id'),
        'predicted_class': predicted_cls,
        'p_win':           p_win,
        'p_draw':          p_draw,
        'p_loss':          p_loss,
        'shap_win':        dict(zip(FEATURE_NAMES, sv_single.values[0, :, 0].tolist())),
        'shap_draw':       dict(zip(FEATURE_NAMES, sv_single.values[0, :, 1].tolist())),
        'shap_loss':       dict(zip(FEATURE_NAMES, sv_single.values[0, :, 2].tolist())),
        'x_vec':           x_vec,
        'archetype_a':    team_a['archetype_name'],
        'archetype_b':    team_b['archetype_name'],
    }
