"""Phase 6 Step 5 — Monte Carlo group stage simulation (10,000 runs).

Each run samples a win/draw/loss result for every match using the model
probabilities, builds group standings with xG-based tiebreakers, determines
top-2 qualifiers and third-place finishers, then selects the best 8 third-place
teams to complete the 32-team R32 bracket.

Stores per-team qualification probabilities in wc2026_qualification_probs.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'

N_SIMS = 10_000
RNG    = np.random.default_rng(42)

# WC2026 uses 32-team R32: top-2 from 12 groups (24) + best-8 third-place (8)
N_BEST_THIRD = 8


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_predictions():
    """Load group stage predictions from CSV."""
    csv = PROCESSED_DIR / 'group_stage_predictions.csv'
    if not csv.exists():
        raise FileNotFoundError(
            'group_stage_predictions.csv not found — run predict_group_stage.py first'
        )
    df = pd.read_csv(csv)
    return df


def _sample_result(p_win, p_draw, p_loss, rng):
    """
    Sample one match result.
    Returns (pts_a, pts_b, xg_a, xg_b) where xg is used for tiebreaking.
    Probabilities are re-normalised to handle floating-point drift from CSV.
    """
    total = p_win + p_draw + p_loss
    p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total
    outcome = rng.choice(['win', 'draw', 'loss'], p=[p_win, p_draw, p_loss])

    if outcome == 'win':
        xg_a = rng.poisson(1.8)
        xg_b = rng.poisson(0.9)
        pts_a, pts_b = 3, 0
    elif outcome == 'draw':
        g = rng.poisson(1.1)
        xg_a, xg_b = g, g
        pts_a, pts_b = 1, 1
    else:  # loss
        xg_a = rng.poisson(0.9)
        xg_b = rng.poisson(1.8)
        pts_a, pts_b = 0, 3

    return pts_a, pts_b, int(xg_a), int(xg_b)


def _build_group_standings(group_fixtures, results_this_run):
    """
    Compute final group standings from sampled results.
    Returns list of (team, pts, gd, gf) sorted by: pts desc, gd desc, gf desc.
    """
    stats = defaultdict(lambda: {'pts': 0, 'gd': 0, 'gf': 0})

    for (a, b), (pts_a, pts_b, xg_a, xg_b) in zip(group_fixtures, results_this_run):
        stats[a]['pts'] += pts_a
        stats[a]['gd']  += xg_a - xg_b
        stats[a]['gf']  += xg_a
        stats[b]['pts'] += pts_b
        stats[b]['gd']  += xg_b - xg_a
        stats[b]['gf']  += xg_b

    return sorted(
        stats.items(),
        key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']),
        reverse=True,
    )


def run_simulations(predictions_df):
    """
    Run N_SIMS Monte Carlo group stage simulations.
    Returns per-team aggregated stats dict.
    """
    # Group fixtures by group label
    groups = {}
    fixture_probs = {}

    for _, row in predictions_df.iterrows():
        g = row['group']
        key = (row['team_a_name'], row['team_b_name'])
        groups.setdefault(g, []).append(key)
        fixture_probs[key] = (float(row['p_win']), float(row['p_draw']), float(row['p_loss']))

    group_labels   = sorted(groups.keys())
    all_teams      = {t for pairs in groups.values() for pair in pairs for t in pair}

    # Counters per team
    counters = {t: {'pts_sum': 0, 'gd_sum': 0,
                    'finish_1': 0, 'finish_2': 0,
                    'finish_3': 0, 'finish_4': 0,
                    'qualify_direct': 0,
                    'best_third_counts': 0}
                for t in all_teams}

    team_to_group = {}
    for g, pairs in groups.items():
        for a, b in pairs:
            team_to_group[a] = g
            team_to_group[b] = g

    print(f'\n  Running {N_SIMS:,} Monte Carlo simulations...')

    for sim_i in range(N_SIMS):
        standings_by_group = {}
        third_place_records = []  # (pts, gd, gf, team)

        for g in group_labels:
            fixtures = groups[g]
            # Sample all 6 matches in this group
            sampled = [
                _sample_result(*fixture_probs[key], RNG)
                for key in fixtures
            ]
            standings = _build_group_standings(fixtures, sampled)

            standings_by_group[g] = standings

            for rank, (team, stats) in enumerate(standings, 1):
                counters[team]['pts_sum'] += stats['pts']
                counters[team]['gd_sum']  += stats['gd']
                if rank == 1:
                    counters[team]['finish_1'] += 1
                    counters[team]['qualify_direct'] += 1
                elif rank == 2:
                    counters[team]['finish_2'] += 1
                    counters[team]['qualify_direct'] += 1
                elif rank == 3:
                    counters[team]['finish_3'] += 1
                    third_place_records.append((stats['pts'], stats['gd'], stats['gf'], team))
                else:
                    counters[team]['finish_4'] += 1

        # Determine best 8 third-place teams
        third_place_records.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_8_third = [rec[3] for rec in third_place_records[:N_BEST_THIRD]]
        for team in best_8_third:
            counters[team]['best_third_counts'] += 1

        if (sim_i + 1) % 2000 == 0:
            print(f'    ... {sim_i + 1:,} / {N_SIMS:,} done')

    print(f'  Simulations complete.')

    # Aggregate into probabilities
    results = {}
    for team, c in counters.items():
        p_qual_direct = c['qualify_direct'] / N_SIMS
        p_best_third  = c['best_third_counts'] / N_SIMS
        results[team] = {
            'team_name':        team,
            'group':            team_to_group.get(team, '?'),
            'p_first':          round(c['finish_1'] / N_SIMS, 4),
            'p_second':         round(c['finish_2'] / N_SIMS, 4),
            'p_third':          round(c['finish_3'] / N_SIMS, 4),
            'p_fourth':         round(c['finish_4'] / N_SIMS, 4),
            'p_qualify_direct': round(p_qual_direct,  4),
            'p_best_third':     round(p_best_third,   4),
            'p_qualify_r32':    round(min(p_qual_direct + p_best_third, 1.0), 4),
            'avg_sim_points':   round(c['pts_sum'] / N_SIMS, 3),
            'avg_sim_gd':       round(c['gd_sum']  / N_SIMS, 3),
        }

    return results


def print_qualification_table(results):
    """Print formatted qualification probability table by group."""
    groups = defaultdict(list)
    for t, r in results.items():
        groups[r['group']].append(r)

    print('\n' + '='*72)
    print(' TactiQ — WC2026 Qualification Probabilities (10,000 simulations)')
    print('='*72)
    print(f'  {"Team":<28} {"Grp"} {"1st%":>5} {"2nd%":>5} {"3rd%":>5} '
          f'{"4th%":>5} {"p(R32)":>7} {"Avg Pts":>8}')
    print('-'*72)

    for g in sorted(groups.keys()):
        group_teams = sorted(groups[g], key=lambda x: -x['p_qualify_r32'])
        for r in group_teams:
            flag = '✓' if r['p_qualify_r32'] >= 0.50 else ' '
            print(f'  {flag} {r["team_name"]:<27} {r["group"]}  '
                  f'{r["p_first"]*100:5.1f} {r["p_second"]*100:5.1f} '
                  f'{r["p_third"]*100:5.1f} {r["p_fourth"]*100:5.1f}  '
                  f'{r["p_qualify_r32"]*100:6.1f}%  {r["avg_sim_points"]:7.2f}')
        print()

    print('='*72)


def save_to_db(results, profiles_by_name=None):
    """Upsert qualification probabilities into wc2026_qualification_probs."""
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute('DELETE FROM wc2026_qualification_probs')

    for team_name, r in results.items():
        arch  = None
        proxy = False
        if profiles_by_name:
            p = profiles_by_name.get(team_name, {})
            arch  = p.get('archetype_name')
            proxy = p.get('is_proxy', False)

        cur.execute(
            """INSERT INTO wc2026_qualification_probs
               (team_name, group_label, archetype_name, is_proxy,
                p_first, p_second, p_third, p_fourth,
                p_qualify_direct, p_best_third, p_qualify_r32,
                avg_sim_points, avg_sim_gd, sim_runs)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (team_name, group_label) DO UPDATE SET
                 p_first = EXCLUDED.p_first, p_second = EXCLUDED.p_second,
                 p_third = EXCLUDED.p_third, p_fourth = EXCLUDED.p_fourth,
                 p_qualify_direct = EXCLUDED.p_qualify_direct,
                 p_best_third = EXCLUDED.p_best_third,
                 p_qualify_r32 = EXCLUDED.p_qualify_r32,
                 avg_sim_points = EXCLUDED.avg_sim_points,
                 avg_sim_gd = EXCLUDED.avg_sim_gd""",
            (team_name, r['group'], arch, proxy,
             r['p_first'], r['p_second'], r['p_third'], r['p_fourth'],
             r['p_qualify_direct'], r['p_best_third'], r['p_qualify_r32'],
             r['avg_sim_points'], r['avg_sim_gd'], N_SIMS)
        )

    conn.commit()
    n = len(results)
    cur.close()
    conn.close()
    print(f'  Inserted {n} rows into wc2026_qualification_probs')
    return n


def save_csv(results):
    out = PROCESSED_DIR / 'qualification_probabilities.csv'
    pd.DataFrame(list(results.values())).sort_values(
        ['group', 'p_qualify_r32'], ascending=[True, False]
    ).to_csv(out, index=False)
    print(f'  Saved → data/processed/qualification_probabilities.csv')


if __name__ == '__main__':
    df = _load_predictions()
    results = run_simulations(df)
    print_qualification_table(results)
    save_to_db(results)
    save_csv(results)
