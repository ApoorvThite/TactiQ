"""Phase 4 Step 1 — Build the matchup training dataset from historical matches."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'

STYLE_FEATURES = [
    'avg_possession_pct',
    'avg_ppda',
    'avg_pressure_success_rate',
    'avg_xg_created_p90',
    'avg_xg_ratio',
    'avg_progressive_carry_pct',
    'avg_pass_completion_pct',
    'avg_passes_final_third_p90',
    'avg_pass_completion_under_pressure_pct',
    'avg_set_piece_shot_pct',
]

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

ARCHETYPES = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}

COMPETITION_WEIGHTS = {
    'FIFA World Cup': 1.0,
    'UEFA Euro':      0.8,
}

MIRROR_LABEL = {'win': 'loss', 'draw': 'draw', 'loss': 'win'}


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def build_matchup_dataset() -> pd.DataFrame:
    conn = _get_conn()

    # Load both rows per match joined with style profiles
    query = """
        SELECT
            m.match_id,
            m.team_id,
            m.opponent_team_id,
            m.is_home,
            m.match_date,
            m.competition_name,
            m.goals_scored,
            m.goals_conceded,
            m.form_points_5,
            t.team_name,
            t.archetype_name,
            t.style_vector,
            t.matches_played
        FROM match_team_features m
        JOIN team_style_profiles t ON m.team_id = t.team_id
        ORDER BY m.match_id, m.is_home DESC
    """
    df = pd.read_sql(query, conn)

    # Load opponent style info
    opp_query = """
        SELECT team_id, team_name AS opp_team_name,
               archetype_name AS opp_archetype,
               style_vector AS opp_style_vector,
               matches_played AS opp_matches_played
        FROM team_style_profiles
    """
    opp_df = pd.read_sql(opp_query, conn).set_index('team_id')
    conn.close()

    rows = []

    # Process each match: find home row and away row, then create both perspectives
    match_groups = df.groupby('match_id')

    for match_id, grp in match_groups:
        home_rows = grp[grp['is_home'] == True]
        away_rows = grp[grp['is_home'] == False]

        if len(home_rows) != 1 or len(away_rows) != 1:
            continue

        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        comp_weight = COMPETITION_WEIGHTS.get(home['competition_name'], 0.7)

        # Parse style vectors (psycopg2 auto-deserialises JSONB to Python list)
        home_sv = np.array(home['style_vector'])
        away_sv = np.array(away['style_vector'])

        # Determine home-perspective label
        if home['goals_scored'] > home['goals_conceded']:
            home_label = 'win'
        elif home['goals_scored'] == home['goals_conceded']:
            home_label = 'draw'
        else:
            home_label = 'loss'

        def make_row(team_a, team_b, sv_a, sv_b, label, is_home_flag, form_a, form_b):
            delta = sv_a - sv_b

            arch_a = team_a['archetype_name']
            arch_b = team_b['archetype_name']
            arch_a_id = ARCHETYPES.get(arch_a, 0)
            arch_b_id = ARCHETYPES.get(arch_b, 0)
            matchup_id = arch_a_id * 4 + arch_b_id

            form_delta = None
            if pd.notna(form_a) and pd.notna(form_b):
                form_delta = float(form_a) - float(form_b)

            matches_delta = int(team_a['matches_played']) - int(team_b['matches_played'])

            return {
                'match_id':        match_id,
                'team_a_id':       int(team_a['team_id']),
                'team_b_id':       int(team_b['team_id']),
                'team_a_name':     team_a['team_name'],
                'team_b_name':     team_b['team_name'],
                'match_date':      team_a['match_date'],
                'competition_name': team_a['competition_name'],
                'label':           label,
                # 10 delta features
                **{f'delta_{f}': float(delta[i]) for i, f in enumerate(STYLE_FEATURES)},
                # 5 context features
                'is_home':              int(is_home_flag),
                'form_points_delta':    form_delta,
                'archetype_matchup_id': matchup_id,
                'delta_matches_played': matches_delta,
                'competition_weight':   comp_weight,
            }

        # Row 1: home perspective
        rows.append(make_row(home, away, home_sv, away_sv,
                             home_label, True,
                             home['form_points_5'], away['form_points_5']))

        # Row 2: away perspective (mirror)
        rows.append(make_row(away, home, away_sv, home_sv,
                             MIRROR_LABEL[home_label], False,
                             away['form_points_5'], home['form_points_5']))

    dataset = pd.DataFrame(rows)

    # Impute form_points_delta nulls with 0
    null_form_count = dataset['form_points_delta'].isna().sum()
    dataset['form_points_delta'] = dataset['form_points_delta'].fillna(0.0)

    # Verify no remaining nulls in feature columns
    feat_null = dataset[FEATURE_NAMES].isna().sum().sum()
    assert feat_null == 0, f"Unexpected nulls in features: {feat_null}"

    # Save
    PROCESSED_DIR.mkdir(exist_ok=True)
    dataset.to_csv(PROCESSED_DIR / 'matchup_dataset.csv', index=False)

    # Print summary
    print('=' * 60)
    print(' Matchup Dataset Summary')
    print('=' * 60)
    print(f'Total rows             : {len(dataset)}   (230 matches × 2 perspectives)')
    print(f'Unique matches         : {dataset["match_id"].nunique()}')
    print('Class distribution:')
    for lbl in ['win', 'draw', 'loss']:
        n = (dataset['label'] == lbl).sum()
        print(f'  {lbl:<6}: {n}  ({100*n/len(dataset):.1f}%)')
    print()
    print('Feature ranges (spot check):')
    print(f'  delta_avg_ppda           : {dataset["delta_avg_ppda"].min():.2f} to {dataset["delta_avg_ppda"].max():.2f}')
    print(f'  delta_avg_possession_pct : {dataset["delta_avg_possession_pct"].min():.2f} to {dataset["delta_avg_possession_pct"].max():.2f}')
    print(f'  form_points_delta        : {int(dataset["form_points_delta"].min())} to {int(dataset["form_points_delta"].max())}')
    print(f'  archetype_matchup_id     : {int(dataset["archetype_matchup_id"].min())} to {int(dataset["archetype_matchup_id"].max())}')
    print()
    print(f'Missing values           : {dataset[FEATURE_NAMES].isna().sum().sum()}')
    print(f'Rows with form_delta null (before impute): {null_form_count} → imputed with 0')
    print()
    print('Saved → data/processed/matchup_dataset.csv')
    print('=' * 60)

    return dataset


if __name__ == '__main__':
    build_matchup_dataset()
