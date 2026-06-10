"""Phase 6 Step 1 — Audit WC2026 teams against team_style_profiles in DB."""

import difflib
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'

# All 48 WC2026 teams, keyed by group
WC2026_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Congo DR', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# Canonical name → DB name aliases
NAME_ALIASES = {
    'Türkiye':          'Turkey',
    'Czechia':          'Czech Republic',
    'DR Congo':         'Congo DR',
    "Côte d'Ivoire":   'Ivory Coast',
    'USA':              'United States',
    'Korea Republic':   'South Korea',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def audit_teams():
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute(
        "SELECT team_id, team_name, archetype_name, matches_played "
        "FROM team_style_profiles ORDER BY team_name"
    )
    db_rows = cur.fetchall()
    cur.close()
    conn.close()

    db_teams = {r[1]: {'team_id': r[0], 'archetype_name': r[2], 'matches_played': r[3]}
                for r in db_rows}
    db_names_lower = {n.lower(): n for n in db_teams}

    all_wc_teams = []
    for group, teams in WC2026_GROUPS.items():
        for t in teams:
            all_wc_teams.append((group, t))

    in_db    = []
    missing  = []
    stale    = []

    print('\n' + '='*62)
    print(' TactiQ — Phase 6: WC2026 Team Audit')
    print(f' Checking {len(all_wc_teams)} qualified teams against DB')
    print('='*62)

    for group, team_name in all_wc_teams:
        canonical = NAME_ALIASES.get(team_name, team_name)
        lower = canonical.lower()

        # Exact match
        if lower in db_names_lower:
            db_name = db_names_lower[lower]
            info = db_teams[db_name]
            mp = info['matches_played']
            flag = '[STALE]' if mp < 3 else ''
            status = 'IN_DB'
            entry = {
                'team_name':      canonical,
                'group':          group,
                'db_name':        db_name,
                'team_id':        info['team_id'],
                'archetype_name': info['archetype_name'],
                'matches_played': mp,
                'status':         status,
                'stale':          mp < 3,
            }
            if mp < 3:
                stale.append(entry)
            in_db.append(entry)
            print(f'  Group {group} | ✓  {canonical:<30} → {info["archetype_name"]:20} '
                  f'({mp} matches) {flag}')
            continue

        # Fuzzy match — high cutoff (0.85) to avoid Iraq→Iran / Algeria→Nigeria false positives
        close = difflib.get_close_matches(lower, db_names_lower.keys(), n=1, cutoff=0.85)
        if close:
            db_name = db_names_lower[close[0]]
            info = db_teams[db_name]
            mp = info['matches_played']
            entry = {
                'team_name':      canonical,
                'group':          group,
                'db_name':        db_name,
                'team_id':        info['team_id'],
                'archetype_name': info['archetype_name'],
                'matches_played': mp,
                'status':         'IN_DB_FUZZY',
                'stale':          mp < 3,
            }
            in_db.append(entry)
            if mp < 3:
                stale.append(entry)
            print(f'  Group {group} | ~  {canonical:<30} → {db_name} (fuzzy) '
                  f'| {info["archetype_name"]:20} ({mp} matches)')
            continue

        # Not found
        entry = {
            'team_name': canonical,
            'group':     group,
            'db_name':   None,
            'team_id':   None,
            'status':    'MISSING',
            'stale':     False,
        }
        missing.append(entry)
        print(f'  Group {group} | ✗  {canonical:<30} → NOT IN DB')

    print('\n' + '-'*62)
    print(f'  Total WC2026 teams   : {len(all_wc_teams)}')
    print(f'  In DB                : {len(in_db)}')
    print(f'  Missing (need scrape): {len(missing)}')
    print(f'  Stale (<3 matches)   : {len(stale)}')

    if missing:
        print('\n  Missing teams:')
        for e in missing:
            print(f'    Group {e["group"]} — {e["team_name"]}')

    if stale:
        print('\n  Stale entries:')
        for e in stale:
            print(f'    Group {e["group"]} — {e["team_name"]} ({e["matches_played"]} matches)')

    print('='*62)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / 'wc2026_teams_in_db.json', 'w') as f:
        json.dump(in_db, f, indent=2)
    with open(PROCESSED_DIR / 'wc2026_teams_missing.json', 'w') as f:
        json.dump(missing, f, indent=2)

    print(f'\nSaved → data/processed/wc2026_teams_in_db.json')
    print(f'Saved → data/processed/wc2026_teams_missing.json')

    return in_db, missing, stale


if __name__ == '__main__':
    audit_teams()
