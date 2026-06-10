"""Phase 4 Steps 2–6 — Train, evaluate, calibrate, and save the XGBoost matchup model."""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'docs' / 'figures'
PROCESSED_DIR = ROOT / 'data' / 'processed'

LABEL_MAP    = {'win': 0, 'draw': 1, 'loss': 2}

from src.models.calibrator import IsotonicMulticlassCalibrator  # noqa: E402
LABEL_NAMES  = ['win', 'draw', 'loss']

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


def _grouped_cv_split(match_ids: np.ndarray, labels_home: np.ndarray, n_splits=5):
    """
    Split unique match_ids into n_splits folds using stratified k-fold on the
    home-perspective label. Both rows of each match go to the same fold.
    Returns list of (train_match_ids, test_match_ids).
    """
    unique_matches = np.unique(match_ids)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_idx, test_idx in skf.split(unique_matches, labels_home):
        splits.append((unique_matches[train_idx], unique_matches[test_idx]))
    return splits


def train_matchup_model():
    # ------------------------------------------------------------------ #
    #  Load dataset                                                         #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(PROCESSED_DIR / 'matchup_dataset.csv')
    X  = df[FEATURE_NAMES].values.astype(float)
    y  = df['label'].map(LABEL_MAP).values
    w  = df['competition_weight'].values
    match_ids = df['match_id'].values

    # Home-perspective label for each unique match (for stratification)
    home_df = df[df['is_home'] == 1].set_index('match_id')
    unique_matches = np.unique(match_ids)
    home_labels = home_df.loc[unique_matches, 'label'].map(LABEL_MAP).values

    # ------------------------------------------------------------------ #
    #  Step 2 — Class weights & baseline                                   #
    # ------------------------------------------------------------------ #
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y)
    print('\n' + '='*60)
    print(' Step 2 — Class Weights & Baseline')
    print('='*60)
    print('Class counts and weights:')
    for i, name in enumerate(LABEL_NAMES):
        cnt = (y == i).sum()
        print(f'  {name:<6}: {cnt} rows  weight={class_weights[i]:.3f}')

    # Majority class baseline
    majority_class = np.bincount(y).argmax()
    baseline_preds = np.full(len(y), majority_class)
    baseline_acc   = accuracy_score(y, baseline_preds)
    # Uniform probability over non-majority gives worst log loss — use
    # the actual class frequency vector as the baseline predicted proba
    class_freq      = np.bincount(y) / len(y)
    baseline_proba  = np.tile(class_freq, (len(y), 1))
    baseline_ll     = log_loss(y, baseline_proba)
    print(f'\nBaseline (predict most-frequent class = {LABEL_NAMES[majority_class]}):')
    print(f'  Accuracy : {baseline_acc*100:.1f}%')
    print(f'  Log loss : {baseline_ll:.3f}')

    # ------------------------------------------------------------------ #
    #  Step 3 — Grouped CV splits                                          #
    # ------------------------------------------------------------------ #
    cv_splits = _grouped_cv_split(unique_matches, home_labels, n_splits=5)

    # Map match_id → row indices in df for fast lookup
    match_to_rows = {}
    for row_idx, mid in enumerate(match_ids):
        match_to_rows.setdefault(mid, []).append(row_idx)

    def get_row_indices(match_id_set):
        rows = []
        for mid in match_id_set:
            rows.extend(match_to_rows[mid])
        return np.array(rows)

    # ------------------------------------------------------------------ #
    #  Step 4 — Optuna hyperparameter tuning                               #
    # ------------------------------------------------------------------ #
    print('\n' + '='*60)
    print(' Step 4 — Optuna Hyperparameter Tuning (50 trials)')
    print('='*60)

    sample_weights_per_class = {i: class_weights[i] for i in range(3)}

    def make_sample_weights(y_arr, comp_w_arr):
        """Combine class weight and competition weight."""
        cw = np.array([sample_weights_per_class[yi] for yi in y_arr])
        return cw * comp_w_arr

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
            'max_depth':        trial.suggest_int('max_depth', 2, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective':        'multi:softprob',
            'num_class':        3,
            'eval_metric':      'mlogloss',
            'random_state':     42,
            'verbosity':        0,
        }
        fold_lls = []
        for train_mids, test_mids in cv_splits:
            tr_idx = get_row_indices(train_mids)
            te_idx = get_row_indices(test_mids)
            X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], w[tr_idx]
            X_te, y_te       = X[te_idx], y[te_idx]
            sw_tr = make_sample_weights(y_tr, w_tr)
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            proba = model.predict_proba(X_te)
            fold_lls.append(log_loss(y_te, proba))
        return float(np.mean(fold_lls))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        'objective':   'multi:softprob',
        'num_class':   3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'verbosity':   0,
    })

    print('Optuna: 50 trials complete')
    print(f'Best log loss (CV)  : {study.best_value:.3f}')
    print('Best params:')
    for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
              'colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']:
        v = best_params[k]
        fmt = f'{v:.4f}' if isinstance(v, float) else str(v)
        print(f'  {k:<20}: {fmt}')

    # ------------------------------------------------------------------ #
    #  Step 5 — Cross-validated evaluation with best params                #
    # ------------------------------------------------------------------ #
    print('\n' + '='*60)
    print(' Step 5 — Model Evaluation (5-Fold Grouped CV)')
    print('='*60)

    cv_lls, cv_accs, cv_f1_macros = [], [], []
    cv_f1_win, cv_f1_draw, cv_f1_loss = [], [], []
    oof_proba = np.zeros((len(y), 3))
    oof_pred  = np.zeros(len(y), dtype=int)

    for train_mids, test_mids in cv_splits:
        tr_idx = get_row_indices(train_mids)
        te_idx = get_row_indices(test_mids)
        X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], w[tr_idx]
        X_te, y_te       = X[te_idx], y[te_idx]
        sw_tr = make_sample_weights(y_tr, w_tr)

        model = XGBClassifier(**best_params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        proba = model.predict_proba(X_te)
        preds = proba.argmax(axis=1)

        oof_proba[te_idx] = proba
        oof_pred[te_idx]  = preds

        cv_lls.append(log_loss(y_te, proba))
        cv_accs.append(accuracy_score(y_te, preds))
        f1s = f1_score(y_te, preds, labels=[0, 1, 2], average=None, zero_division=0)
        cv_f1_macros.append(f1_score(y_te, preds, average='macro', zero_division=0))
        cv_f1_win.append(f1s[0])
        cv_f1_draw.append(f1s[1])
        cv_f1_loss.append(f1s[2])

    print(f'\n{"":22} {"Mean":>8}  {"Std":>6}')
    print(f'  Log Loss       : {np.mean(cv_lls):.3f}  ± {np.std(cv_lls):.3f}'
          f'   (baseline: {baseline_ll:.3f})')
    print(f'  Accuracy       : {np.mean(cv_accs)*100:.1f}%  ± {np.std(cv_accs)*100:.1f}%'
          f'    (baseline: {baseline_acc*100:.1f}%)')
    print(f'  F1 Macro       : {np.mean(cv_f1_macros):.3f}  ± {np.std(cv_f1_macros):.3f}')
    print(f'  F1 Win         : {np.mean(cv_f1_win):.3f}  ± {np.std(cv_f1_win):.3f}')
    print(f'  F1 Draw        : {np.mean(cv_f1_draw):.3f}  ± {np.std(cv_f1_draw):.3f}')
    print(f'  F1 Loss        : {np.mean(cv_f1_loss):.3f}  ± {np.std(cv_f1_loss):.3f}')

    beats_ll  = np.mean(cv_lls) < baseline_ll
    beats_acc = np.mean(cv_accs) > baseline_acc
    print('\nBeats majority-class baseline?')
    print(f'  Log Loss : {"YES" if beats_ll  else "NO"}  (model: {np.mean(cv_lls):.3f} vs baseline: {baseline_ll:.3f})')
    print(f'  Accuracy : {"YES" if beats_acc else "NO"}  (model: {np.mean(cv_accs)*100:.1f}% vs baseline: {baseline_acc*100:.1f}%)')

    # Confusion matrix (OOF)
    cm = confusion_matrix(y, oof_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix (OOF, all 460 rows):')
    print(f'{"Predicted →":>15}  {"Win":>6}  {"Draw":>6}  {"Loss":>6}')
    for i, name in enumerate(LABEL_NAMES):
        print(f'  {"Actual " + name.capitalize():<13}: {cm[i,0]:>6}  {cm[i,1]:>6}  {cm[i,2]:>6}')

    # ------------------------------------------------------------------ #
    #  Brier scores & calibration                                          #
    # ------------------------------------------------------------------ #
    brier_pre = {}
    for i, name in enumerate(LABEL_NAMES):
        y_bin = (y == i).astype(int)
        brier_pre[name] = brier_score_loss(y_bin, oof_proba[:, i])

    print('\nBrier Scores (before calibration):')
    for name in LABEL_NAMES:
        print(f'  Brier Score — {name:<5}: {brier_pre[name]:.3f}')

    # Train final model on full data
    sw_full = make_sample_weights(y, w)
    final_model = XGBClassifier(**best_params)
    final_model.fit(X, y, sample_weight=sw_full)

    # --- Isotonic calibration using OOF probabilities ---
    calibrators = []
    for i in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_proba[:, i], (y == i).astype(float))
        calibrators.append(iso)

    calibrated_model = IsotonicMulticlassCalibrator(final_model, calibrators)

    # Brier scores after calibration — evaluated on OOF set
    calib_proba = calibrated_model.predict_proba(X)
    brier_post = {}
    for i, name in enumerate(LABEL_NAMES):
        y_bin = (y == i).astype(int)
        brier_post[name] = brier_score_loss(y_bin, calib_proba[:, i])

    # For reliability diagram — use OOF raw vs calibrated

    print('\nBrier Scores (after calibration):')
    for name in LABEL_NAMES:
        print(f'  Brier Score — {name:<5}: {brier_pre[name]:.3f} → {brier_post[name]:.3f}')

    # Calibration reliability diagram
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for ax_idx, (cls_idx, cls_name) in enumerate([(0, 'win'), (1, 'draw')]):
        ax = axes[ax_idx]
        # Before calibration (OOF raw)
        frac_pos_b, mean_pred_b = calibration_curve(
            (y == cls_idx).astype(int), oof_proba[:, cls_idx], n_bins=8, strategy='quantile')
        # After calibration (OOF calibrated)
        frac_pos_a, mean_pred_a = calibration_curve(
            (y == cls_idx).astype(int), calib_proba[:, cls_idx], n_bins=8, strategy='quantile')
        ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.5, label='Perfect calibration')
        ax.plot(mean_pred_b, frac_pos_b, 'o-', color='#e63946', linewidth=2, markersize=5, label='Before calibration')
        ax.plot(mean_pred_a, frac_pos_a, 's-', color='#2a9d8f', linewidth=2, markersize=5, label='After calibration')
        ax.set_xlabel('Mean predicted probability', fontsize=10)
        ax.set_ylabel('Fraction of positives', fontsize=10)
        ax.set_title(f'Reliability Diagram — {cls_name.capitalize()} class', fontsize=11)
        ax.legend(fontsize=8, framealpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle('TactiQ — Probability Calibration', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_calibration.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved → docs/figures/fig6_calibration.png')

    # ------------------------------------------------------------------ #
    #  Step 6 — Feature importance                                         #
    # ------------------------------------------------------------------ #
    print('\n' + '='*60)
    print(' Step 6 — Feature Importance (by Gain)')
    print('='*60)

    gain_scores = final_model.get_booster().get_score(importance_type='gain')
    fi_df = pd.DataFrame({'feature': list(gain_scores.keys()),
                           'gain':    list(gain_scores.values())})
    # Map f0, f1... back to feature names
    fi_df['feature'] = fi_df['feature'].apply(
        lambda f: FEATURE_NAMES[int(f[1:])] if f.startswith('f') and f[1:].isdigit() else f
    )
    fi_df = fi_df.sort_values('gain', ascending=False).reset_index(drop=True)
    total_gain = fi_df['gain'].sum()
    fi_df['pct'] = fi_df['gain'] / total_gain * 100

    print('Feature Importance (by Gain):')
    for _, row in fi_df.iterrows():
        print(f'  {row["feature"]:<50}: {row["gain"]:>8.3f}  ({row["pct"]:>5.1f}%)')

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = ['#e63946' if i < 3 else '#457b9d' for i in range(len(fi_df))]
    ax.barh(fi_df['feature'][::-1], fi_df['gain'][::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Gain', fontsize=11)
    ax.set_title('TactiQ — Feature Importance (XGBoost Gain)', fontsize=13, pad=12)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7_feature_importance.png', dpi=150)
    plt.close(fig)
    print('Saved → docs/figures/fig7_feature_importance.png')

    # ------------------------------------------------------------------ #
    #  Save models                                                          #
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / 'xgboost_matchup.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    final_model.save_model(str(MODELS_DIR / 'xgboost_matchup.json'))

    with open(MODELS_DIR / 'xgboost_calibrated.pkl', 'wb') as f:
        pickle.dump(calibrated_model, f)

    print('\nSaved → models/xgboost_matchup.pkl')
    print('Saved → models/xgboost_matchup.json')
    print('Saved → models/xgboost_calibrated.pkl')

    return {
        'cv_ll_mean':   float(np.mean(cv_lls)),
        'cv_ll_std':    float(np.std(cv_lls)),
        'cv_acc_mean':  float(np.mean(cv_accs)),
        'cv_acc_std':   float(np.std(cv_accs)),
        'cv_f1_macro':  float(np.mean(cv_f1_macros)),
        'cv_f1_std':    float(np.std(cv_f1_macros)),
        'baseline_ll':  baseline_ll,
        'baseline_acc': baseline_acc,
        'beats_ll':     beats_ll,
        'brier_pre':    brier_pre,
        'brier_post':   brier_post,
        'fi_df':        fi_df,
        'best_params':  best_params,
        'class_dist':   {LABEL_NAMES[i]: int((y == i).sum()) for i in range(3)},
    }


if __name__ == '__main__':
    train_matchup_model()
