"""Phase 4 orchestration — build dataset, train model, run predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.build_matchup_dataset import build_matchup_dataset
from src.models.train_matchup_model   import train_matchup_model
from src.models.predict_matchup       import run_test_predictions

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'docs' / 'figures'
PROCESSED_DIR = ROOT / 'data' / 'processed'


def main():
    print('\n' + '='*60)
    print(' TACTIQ — Phase 4: Matchup Model')
    print('='*60)

    # Step 1
    build_matchup_dataset()

    # Steps 2–6
    metrics = train_matchup_model()

    # Step 7 — Sample predictions
    print('\n' + '='*60)
    print(' Step 7 — Sample Predictions')
    print('='*60 + '\n')
    predictions = run_test_predictions()

    # Final summary
    print('\n')
    print('=' * 60)
    print(' TACTIQ — Phase 4 Matchup Model Complete')
    print('=' * 60)

    dist = metrics['class_dist']
    total = sum(dist.values())
    print('\nTRAINING DATA')
    print(f'  Matches used     : 230')
    print(f'  Training rows    : 460')
    print(f'  Features         : 15 (10 style delta + 5 context)')
    dist_str = ' / '.join(f'{k} {v/total*100:.0f}%' for k, v in dist.items())
    print(f'  Class balance    : {dist_str}')

    print('\nMODEL PERFORMANCE (5-fold grouped CV)')
    print(f'  Log loss         : {metrics["cv_ll_mean"]:.3f} ± {metrics["cv_ll_std"]:.3f}'
          f'  (baseline: {metrics["baseline_ll"]:.3f})')
    print(f'  Accuracy         : {metrics["cv_acc_mean"]*100:.1f}% ± {metrics["cv_acc_std"]*100:.1f}%'
          f'   (baseline: {metrics["baseline_acc"]*100:.1f}%)')
    print(f'  F1 Macro         : {metrics["cv_f1_macro"]:.3f} ± {metrics["cv_f1_std"]:.3f}')
    print(f'  Beats baseline?  : {"YES" if metrics["beats_ll"] else "NO"}')

    print('\nCALIBRATION')
    for cls in ['win', 'draw']:
        print(f'  Brier ({cls}) before/after  : '
              f'{metrics["brier_pre"][cls]:.3f} → {metrics["brier_post"][cls]:.3f}')

    fi_df = metrics['fi_df']
    print('\nTOP PREDICTIVE FEATURES (by gain)')
    for _, row in fi_df.head(3).iterrows():
        print(f'  {row["feature"]:<48}: {row["pct"]:.1f}%')

    print('\nSAVED ARTIFACTS')
    artifacts = [
        ('models/xgboost_matchup.pkl',          MODELS_DIR / 'xgboost_matchup.pkl'),
        ('models/xgboost_matchup.json',         MODELS_DIR / 'xgboost_matchup.json'),
        ('models/xgboost_calibrated.pkl',       MODELS_DIR / 'xgboost_calibrated.pkl'),
        ('data/processed/matchup_dataset.csv',  PROCESSED_DIR / 'matchup_dataset.csv'),
        ('docs/figures/fig6_calibration.png',   FIGURES_DIR / 'fig6_calibration.png'),
        ('docs/figures/fig7_feature_importance.png', FIGURES_DIR / 'fig7_feature_importance.png'),
    ]
    for label, path in artifacts:
        print(f'  {label:<44} {"✓" if path.exists() else "✗"}')

    print('\nSAMPLE PREDICTIONS')
    test_names = [
        ('Spain', 'Morocco'),
        ('Germany', 'Brazil'),
        ('France', 'Argentina'),
        ('England', 'Iran'),
        ('Netherlands', 'Ecuador'),
    ]
    for pred, (a, b) in zip(predictions, test_names):
        result = pred['predicted_result'].upper()
        winner = f'{a} {result}' if result != 'DRAW' else 'DRAW'
        if result == 'LOSS':
            winner = f'{b} WIN'
        p_max = max(pred['p_win'], pred['p_draw'], pred['p_loss'])
        print(f'  {a} vs {b:<15} → {winner:<18} (p={p_max:.2f})')

    print('\n' + '=' * 60)
    print(' Ready for Phase 5: SHAP Explainability & Upset Detector')
    print('=' * 60)


if __name__ == '__main__':
    main()
