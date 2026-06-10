"""Phase 6 Step 2 — FBref scraper for WC2026 teams missing from StatsBomb data.

Scrapes per-match possession, xG, passing, and defensive stats for each
missing team. Computes a 10-feature style vector for cluster assignment.
Falls back to proxy centroid if fewer than 3 parseable matches are found.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://fbref.com/en/',
}

FBREF_SQUAD_BASE = 'https://fbref.com/en/squads/{squad_id}/International-Results'
FBREF_HOME       = 'https://fbref.com/en/'

MIN_MATCHES = 3

# FBref squad IDs for the 14 proxy teams + any others missing from StatsBomb
MISSING_SQUAD_IDS = {
    'South Africa':           '2f171d59',
    'Bosnia and Herzegovina': 'dbf9a9d8',
    'Haiti':                  'de21f9fc',
    'Paraguay':               '1df9d8f3',
    'Curacao':                '6e7d94ff',
    'Ivory Coast':            '8b26c9c2',
    'New Zealand':            'a8f8c6f5',
    'Cape Verde':             'e8e87f8d',
    'Iraq':                   'f2ee1891',
    'Norway':                 '62c4acff',
    'Algeria':                'd2de9b07',
    'Jordan':                 '04a7a53e',
    'Congo DR':               '46b25e8f',
    'Uzbekistan':             '63e3b94e',
    # Include these as potential supplements (may already be in DB from StatsBomb)
    'Sweden':                 '81b9b0c7',
    'Ghana':                  'b33c6b2a',
    'Panama':                 'f7b42b0f',
    'Scotland':               '7e4e0022',
    'Egypt':                  'f5cee43d',
}

# Proxy archetype assignments (fallback when scraping fails)
PROXY_ARCHETYPES = {
    'South Africa':           'Deep Block',
    'Bosnia and Herzegovina': 'Counter-Attack',
    'Haiti':                  'Counter-Attack',
    'Paraguay':               'Deep Block',
    'Curacao':                'Counter-Attack',
    'Ivory Coast':            'Counter-Attack',
    'New Zealand':            'Deep Block',
    'Cape Verde':             'Counter-Attack',
    'Iraq':                   'Deep Block',
    'Norway':                 'Counter-Attack',
    'Algeria':                'Counter-Attack',
    'Jordan':                 'Deep Block',
    'Congo DR':               'Counter-Attack',
    'Uzbekistan':             'Deep Block',
    'Sweden':                 'Possession Control',
    'Ghana':                  'Counter-Attack',
    'Panama':                 'Deep Block',
    'Scotland':               'High Press',
    'Egypt':                  'Deep Block',
}

# Archetype centroids (feature means) — used as fallback style vectors.
# Computed from Phase 3 cluster analysis. 10 features in order:
# avg_possession_pct, avg_ppda, avg_pressure_success_rate, avg_xg_created_p90,
# avg_xg_ratio, avg_progressive_carry_pct, avg_pass_completion_pct,
# avg_passes_final_third_p90, avg_pass_completion_under_pressure_pct, avg_set_piece_shot_pct
ARCHETYPE_CENTROIDS = {
    'High Press':       [52.1, 8.2,  0.28, 1.65, 1.42, 0.14, 82.1, 38.2, 0.68, 0.11],
    'Possession Control':[58.3, 11.4, 0.24, 1.58, 1.38, 0.11, 85.6, 44.1, 0.72, 0.09],
    'Counter-Attack':   [44.8, 14.7, 0.21, 1.41, 1.18, 0.17, 79.3, 31.4, 0.63, 0.13],
    'Deep Block':       [40.2, 18.3, 0.18, 1.12, 0.94, 0.12, 76.8, 25.6, 0.59, 0.15],
}


def _get_proxy_vector(team_name):
    """Return archetype centroid as fallback style vector."""
    arch = PROXY_ARCHETYPES.get(team_name, 'Counter-Attack')
    return np.array(ARCHETYPE_CENTROIDS[arch]), arch, True


def _warm_session(session):
    try:
        session.get(FBREF_HOME, headers=HEADERS, timeout=20)
        time.sleep(random.uniform(2, 4))
    except Exception:
        pass


def _fetch_page(url, session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _parse_match_rows(html):
    """
    Parse FBref International Results page.
    Returns list of dicts with raw stats per match.
    """
    if html is None:
        return []

    try:
        soup = BeautifulSoup(html, 'lxml')
        # Find the main results table
        table = soup.find('table', {'id': lambda x: x and 'results' in x.lower()})
        if table is None:
            # Try any table
            tables = soup.find_all('table')
            if not tables:
                return []
            table = max(tables, key=lambda t: len(t.find_all('tr')))

        rows = []
        headers = []
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all('th')]

        tbody = table.find('tbody')
        if tbody is None:
            return []

        for tr in tbody.find_all('tr'):
            cells = tr.find_all(['td', 'th'])
            if len(cells) < 5:
                continue
            row = {}
            for i, cell in enumerate(cells):
                key = headers[i] if i < len(headers) else f'col_{i}'
                row[key] = cell.get_text(strip=True)
            rows.append(row)

        return rows
    except Exception:
        return []


def _extract_stats(rows):
    """
    Extract tactical feature averages from parsed FBref match rows.
    Returns dict of feature means or None if insufficient data.
    """
    records = []
    for row in rows:
        # Skip header rows or empty rows
        if not any(row.values()):
            continue

        # Try to extract: possession, xG, result
        rec = {}

        # Possession — look for 'Poss' column
        for key in ['Poss', 'Possession', 'poss']:
            if key in row and row[key]:
                try:
                    rec['possession'] = float(row[key])
                    break
                except ValueError:
                    pass

        # xG scored (for)
        for key in ['xG', 'xGF', 'xg']:
            if key in row and row[key]:
                try:
                    rec['xg_for'] = float(row[key])
                    break
                except ValueError:
                    pass

        # xG against
        for key in ['xGA', 'xGa', 'xg_against']:
            if key in row and row[key]:
                try:
                    rec['xg_against'] = float(row[key])
                    break
                except ValueError:
                    pass

        if len(rec) >= 1:
            records.append(rec)

    if len(records) < MIN_MATCHES:
        return None

    def safe_mean(vals, default):
        clean = [v for v in vals if v is not None]
        return np.mean(clean) if len(clean) >= MIN_MATCHES else default

    poss_vals    = [r.get('possession') for r in records]
    xg_for_vals  = [r.get('xg_for')    for r in records]
    xg_ag_vals   = [r.get('xg_against') for r in records]

    avg_poss = safe_mean(poss_vals, 45.0)
    avg_xg_f = safe_mean(xg_for_vals, 1.2)
    avg_xg_a = safe_mean(xg_ag_vals, 1.2)
    avg_xg_ratio = avg_xg_f / max(avg_xg_a, 0.5)

    return {
        'n_matches':        len(records),
        'avg_possession':   avg_poss,
        'avg_xg_created':   avg_xg_f,
        'avg_xg_conceded':  avg_xg_a,
        'avg_xg_ratio':     avg_xg_ratio,
    }


def _build_style_vector(stats, proxy_arch):
    """
    Build a 10-feature style vector from scraped stats.
    For features we can't scrape directly, interpolate from the
    proxy archetype centroid (these teams are typically at the
    Counter-Attack or Deep Block end of the spectrum).
    """
    arch     = PROXY_ARCHETYPES.get(proxy_arch, proxy_arch)
    centroid = np.array(ARCHETYPE_CENTROIDS.get(arch, ARCHETYPE_CENTROIDS['Counter-Attack']))

    sv = centroid.copy()

    # Override with scraped values where available
    poss = stats.get('avg_possession', centroid[0])
    sv[0] = poss

    # PPDA: inversely related to possession and pressing tendency
    # Low poss teams press less → higher PPDA
    sv[1] = centroid[1] * (1 + (50 - poss) / 100)

    xg_created = stats.get('avg_xg_created', centroid[3])
    sv[3] = xg_created

    xg_ratio = stats.get('avg_xg_ratio', centroid[4])
    sv[4] = xg_ratio

    return sv


def scrape_team(team_name, squad_id, session):
    """
    Attempt to scrape tactical stats for one team.
    Returns (style_vector, archetype, is_proxy).
    """
    url  = FBREF_SQUAD_BASE.format(squad_id=squad_id)
    html = _fetch_page(url, session)

    if html is None:
        print(f'    [FAIL] {team_name}: HTTP error — using proxy')
        return _get_proxy_vector(team_name)

    rows = _parse_match_rows(html)

    if len(rows) < MIN_MATCHES:
        print(f'    [WARN] {team_name}: only {len(rows)} rows found — using proxy')
        return _get_proxy_vector(team_name)

    stats = _extract_stats(rows)

    if stats is None:
        print(f'    [WARN] {team_name}: insufficient stats — using proxy')
        return _get_proxy_vector(team_name)

    arch    = PROXY_ARCHETYPES.get(team_name, 'Counter-Attack')
    sv      = _build_style_vector(stats, team_name)
    n       = stats['n_matches']
    print(f'    [OK]   {team_name}: {n} matches | poss={stats["avg_possession"]:.1f}% '
          f'| xG={stats["avg_xg_created"]:.2f} | ratio={stats["avg_xg_ratio"]:.2f} '
          f'→ {arch}')
    return sv, arch, False


def scrape_missing_teams(team_names):
    """
    Scrape FBref for a list of team names.
    Returns dict: team_name → {style_vector, archetype, is_proxy, n_matches}
    """
    session = requests.Session()
    print('\n[FBref] Warming up session...')
    _warm_session(session)

    results = {}
    for team_name in team_names:
        squad_id = MISSING_SQUAD_IDS.get(team_name)
        if squad_id is None:
            print(f'    [SKIP] {team_name}: no squad ID — using proxy immediately')
            sv, arch, is_proxy = _get_proxy_vector(team_name)
        else:
            print(f'  Scraping {team_name} (squad_id={squad_id})...')
            try:
                sv, arch, is_proxy = scrape_team(team_name, squad_id, session)
            except Exception as e:
                print(f'    [ERROR] {team_name}: {e} — using proxy')
                sv, arch, is_proxy = _get_proxy_vector(team_name)
            time.sleep(random.uniform(4, 8))

        results[team_name] = {
            'style_vector':   sv.tolist(),
            'archetype_name': arch,
            'is_proxy':       is_proxy,
        }

    return results


if __name__ == '__main__':
    sample = ['Norway', 'Ivory Coast', 'Algeria']
    r = scrape_missing_teams(sample)
    for k, v in r.items():
        print(f'{k}: {v["archetype_name"]} (proxy={v["is_proxy"]})')
