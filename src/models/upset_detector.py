"""Phase 5 Step 4 — Upset detector for WC2026 official group stage fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'

ARCHETYPES  = {'High Press': 0, 'Possession Control': 1, 'Counter-Attack': 2, 'Deep Block': 3}
LABEL_NAMES = ['win', 'draw', 'loss']

FEATURE_NAMES = [
    'delta_avg_possession_pct', 'delta_avg_ppda', 'delta_avg_pressure_success_rate',
    'delta_avg_xg_created_p90', 'delta_avg_xg_ratio', 'delta_avg_progressive_carry_pct',
    'delta_avg_pass_completion_pct', 'delta_avg_passes_final_third_p90',
    'delta_avg_pass_completion_under_pressure_pct', 'delta_avg_set_piece_shot_pct',
    'is_home', 'form_points_delta', 'archetype_matchup_id', 'delta_matches_played',
    'competition_weight',
]

# Official 2026 FIFA World Cup group stage fixtures
# Source: FIFA draw, December 5 2025 — Washington D.C.
# Format: (team_a, team_b, group)
WC2026_GROUP_FIXTURES = [

    # ── GROUP A ──────────────────────────────────────────
    # Mexico | South Africa | South Korea | Czech Republic
    ('Mexico',        'South Africa',   'A'),
    ('South Korea',   'Czech Republic', 'A'),
    ('Mexico',        'South Korea',    'A'),
    ('South Africa',  'Czech Republic', 'A'),
    ('Mexico',        'Czech Republic', 'A'),
    ('South Africa',  'South Korea',    'A'),

    # ── GROUP B ──────────────────────────────────────────
    # Canada | Bosnia-Herzegovina | Qatar | Switzerland
    ('Canada',                'Bosnia and Herzegovina', 'B'),
    ('Qatar',                 'Switzerland',            'B'),
    ('Canada',                'Qatar',                  'B'),
    ('Bosnia and Herzegovina','Switzerland',            'B'),
    ('Canada',                'Switzerland',            'B'),
    ('Bosnia and Herzegovina','Qatar',                  'B'),

    # ── GROUP C ──────────────────────────────────────────
    # Brazil | Morocco | Haiti | Scotland
    ('Brazil',   'Morocco',  'C'),
    ('Haiti',    'Scotland', 'C'),
    ('Brazil',   'Haiti',    'C'),
    ('Morocco',  'Scotland', 'C'),
    ('Brazil',   'Scotland', 'C'),
    ('Haiti',    'Morocco',  'C'),

    # ── GROUP D ──────────────────────────────────────────
    # United States | Paraguay | Australia | Türkiye
    ('United States', 'Paraguay',  'D'),
    ('Australia',     'Turkey',    'D'),
    ('United States', 'Australia', 'D'),
    ('Paraguay',      'Turkey',    'D'),
    ('United States', 'Turkey',    'D'),
    ('Paraguay',      'Australia', 'D'),

    # ── GROUP E ──────────────────────────────────────────
    # Germany | Curaçao | Ivory Coast | Ecuador
    ('Germany',     'Curacao',     'E'),
    ('Ivory Coast', 'Ecuador',     'E'),
    ('Germany',     'Ivory Coast', 'E'),
    ('Curacao',     'Ecuador',     'E'),
    ('Germany',     'Ecuador',     'E'),
    ('Curacao',     'Ivory Coast', 'E'),

    # ── GROUP F ──────────────────────────────────────────
    # Netherlands | Japan | Sweden | Tunisia
    ('Netherlands', 'Japan',    'F'),
    ('Sweden',      'Tunisia',  'F'),
    ('Netherlands', 'Sweden',   'F'),
    ('Japan',       'Tunisia',  'F'),
    ('Netherlands', 'Tunisia',  'F'),
    ('Japan',       'Sweden',   'F'),

    # ── GROUP G ──────────────────────────────────────────
    # Belgium | Egypt | Iran | New Zealand
    ('Belgium',     'Egypt',       'G'),
    ('Iran',        'New Zealand', 'G'),
    ('Belgium',     'Iran',        'G'),
    ('Egypt',       'New Zealand', 'G'),
    ('Belgium',     'New Zealand', 'G'),
    ('Egypt',       'Iran',        'G'),

    # ── GROUP H ──────────────────────────────────────────
    # Spain | Cape Verde | Saudi Arabia | Uruguay
    ('Spain',        'Cape Verde',   'H'),
    ('Saudi Arabia', 'Uruguay',      'H'),
    ('Spain',        'Saudi Arabia', 'H'),
    ('Cape Verde',   'Uruguay',      'H'),
    ('Spain',        'Uruguay',      'H'),
    ('Cape Verde',   'Saudi Arabia', 'H'),

    # ── GROUP I ──────────────────────────────────────────
    # France | Senegal | Iraq | Norway
    ('France',  'Senegal', 'I'),
    ('Iraq',    'Norway',  'I'),
    ('France',  'Iraq',    'I'),
    ('Senegal', 'Norway',  'I'),
    ('France',  'Norway',  'I'),
    ('Iraq',    'Senegal', 'I'),

    # ── GROUP J ──────────────────────────────────────────
    # Argentina | Algeria | Austria | Jordan
    ('Argentina', 'Algeria', 'J'),
    ('Austria',   'Jordan',  'J'),
    ('Argentina', 'Austria', 'J'),
    ('Algeria',   'Jordan',  'J'),
    ('Argentina', 'Jordan',  'J'),
    ('Algeria',   'Austria', 'J'),

    # ── GROUP K ──────────────────────────────────────────
    # Portugal | Congo DR | Uzbekistan | Colombia
    ('Portugal',   'Congo DR',   'K'),
    ('Uzbekistan', 'Colombia',   'K'),
    ('Portugal',   'Uzbekistan', 'K'),
    ('Congo DR',   'Colombia',   'K'),
    ('Portugal',   'Colombia',   'K'),
    ('Congo DR',   'Uzbekistan', 'K'),

    # ── GROUP L ──────────────────────────────────────────
    # England | Croatia | Ghana | Panama
    ('England',  'Croatia', 'L'),
    ('Ghana',    'Panama',  'L'),
    ('England',  'Ghana',   'L'),
    ('Croatia',  'Panama',  'L'),
    ('England',  'Panama',  'L'),
    ('Croatia',  'Ghana',   'L'),
]

assert len(WC2026_GROUP_FIXTURES) == 72, f"Expected 72 fixtures, got {len(WC2026_GROUP_FIXTURES)}"

# Teams not in StatsBomb free data → use archetype centroid as proxy
PROXY_ARCHETYPES = {
    # Group A
    'South Africa':           'Deep Block',
    # Group B
    'Bosnia and Herzegovina': 'Counter-Attack',
    # Group C
    'Haiti':                  'Counter-Attack',
    # Group D
    'Paraguay':               'Deep Block',
    # Group E
    'Curacao':                'Counter-Attack',
    'Ivory Coast':            'Counter-Attack',
    # Group G
    'New Zealand':            'Deep Block',
    # Group H
    'Cape Verde':             'Counter-Attack',
    # Group I
    'Iraq':                   'Deep Block',
    'Norway':                 'Counter-Attack',
    # Group J
    'Algeria':                'Counter-Attack',
    'Jordan':                 'Deep Block',
    # Group K
    'Congo DR':               'Counter-Attack',
    'Uzbekistan':             'Deep Block',
}

ARCHETYPE_TACTICAL_DESC = {
    'High Press vs Deep Block':             'draw-prone',
    'High Press vs Counter-Attack':         'upset-prone',
    'Possession Control vs Deep Block':     'draw-prone',
    'Possession Control vs Counter-Attack': 'upset-prone',
    'Counter-Attack vs High Press':         'underdog-favourable',
    'Deep Block vs High Press':             'underdog-favourable',
    'High Press vs High Press':             'high-variance',
    'Counter-Attack vs Counter-Attack':     'high-variance',
}


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_team_from_db(name, cur, cluster_centroids):
    """Load team from DB; fall back to proxy centroid if not found."""
    cur.execute(
        "SELECT team_id, team_name, archetype_name, style_vector, matches_played "
        "FROM team_style_profiles WHERE LOWER(team_name) = LOWER(%s)",
        (name,)
    )
    row = cur.fetchone()
    if row:
        return {
            'team_id':        row[0],
            'team_name':      row[1],
            'archetype_name': row[2],
            'style_vector':   np.array(row[3]),
            'matches_played': row[4],
            'is_proxy':       False,
        }

    # Fall back to proxy
    if name not in PROXY_ARCHETYPES:
        raise ValueError(f"Team '{name}' not in DB and no proxy defined in PROXY_ARCHETYPES")
    arch = PROXY_ARCHETYPES[name]
    sv   = cluster_centroids[arch]
    print(f'  ⚠  Proxy: {name} → {arch} centroid')
    return {
        'team_id':        None,
        'team_name':      name,
        'archetype_name': arch,
        'style_vector':   sv,
        'matches_played': 8,
        'is_proxy':       True,
    }


def _build_feature_vector(team_a, team_b, is_home=True, form_delta=0.0,
                           competition='FIFA World Cup'):
    COMP_WEIGHTS = {'FIFA World Cup': 1.0, 'UEFA Euro': 0.8}
    sv_a      = np.array(team_a['style_vector'])
    sv_b      = np.array(team_b['style_vector'])
    delta     = sv_a - sv_b
    arch_a_id = ARCHETYPES.get(team_a['archetype_name'], 0)
    arch_b_id = ARCHETYPES.get(team_b['archetype_name'], 0)
    matchup_id = arch_a_id * 4 + arch_b_id
    comp_w    = COMP_WEIGHTS.get(competition, 0.7)
    delta_mp  = team_a['matches_played'] - team_b['matches_played']
    return np.concatenate([delta, [float(is_home), float(form_delta),
                                   float(matchup_id), float(delta_mp), float(comp_w)]])


def is_upset_candidate(pred, shap_win_dict, shap_draw_dict, shap_loss_dict):
    """
    All conditions must be met:
    1. Team A predicted to win (p_win > 0.40)
    2. Team B NOT in 'High Press' archetype
    3. At least one upset indicator triggered
    4. Team B's p_win + p_draw >= 0.45
    """
    if pred['p_win'] <= 0.40:
        return False, []
    if pred['archetype_b'] == 'High Press':
        return False, []
    if (pred['p_draw'] + pred['p_loss']) < 0.45:
        return False, []

    signals = []

    ppda_draw = shap_draw_dict.get('delta_avg_ppda', 0)
    ppda_loss = shap_loss_dict.get('delta_avg_ppda', 0)
    if ppda_draw > 0.04 or ppda_loss > 0.04:
        signals.append('ppda_neutralised')

    sp_draw = shap_draw_dict.get('delta_avg_set_piece_shot_pct', 0)
    sp_loss = shap_loss_dict.get('delta_avg_set_piece_shot_pct', 0)
    if sp_draw > 0.03 or sp_loss > 0.03:
        signals.append('set_piece_threat')

    am_draw = shap_draw_dict.get('archetype_matchup_id', 0)
    am_loss = shap_loss_dict.get('archetype_matchup_id', 0)
    if am_draw >= 0 or am_loss >= 0:
        signals.append('archetype_disadvantage')

    if not signals:
        return False, []
    return True, signals


def _upset_explanation(pred, signals):
    parts = []
    arch_key = f'{pred["archetype_a"]} vs {pred["archetype_b"]}'
    desc     = ARCHETYPE_TACTICAL_DESC.get(arch_key, 'historically variable')

    if 'ppda_neutralised' in signals:
        parts.append(
            f'{pred["team_b_name"]}\'s defensive shape neutralises '
            f'{pred["team_a_name"]}\'s pressing advantage (PPDA SHAP toward upset)'
        )
    if 'set_piece_threat' in signals:
        parts.append(
            f'{pred["team_b_name"]} has a set-piece dependency edge '
            f'— dead-ball moments can flip the game'
        )
    if 'archetype_disadvantage' in signals:
        parts.append(
            f'{arch_key} matchup is {desc} — structural disadvantage for favourite'
        )

    verdict = (
        f'"{pred["archetype_a"]} pressed into stalemate"'
        if pred['archetype_a'] == 'High Press' else
        f'"{pred["archetype_b"]} counter-attack threat is real"'
        if pred['archetype_b'] == 'Counter-Attack' else
        '"Deep defensive block creates upset conditions"'
    )
    return '. '.join(parts) + '. ' + verdict


def run_upset_detector(explainer, model_calib):
    conn = _get_conn()
    cur  = conn.cursor()

    # Build archetype centroid vectors (averaged over all member teams)
    cluster_centroids = {}
    for arch in ARCHETYPES:
        cur.execute(
            "SELECT style_vector FROM team_style_profiles WHERE archetype_name = %s",
            (arch,)
        )
        all_svs = [np.array(r[0]) for r in cur.fetchall()]
        cluster_centroids[arch] = np.mean(all_svs, axis=0)

    rows          = []
    proxy_teams   = []
    db_count      = 0
    proxy_count   = 0
    groups_seen   = {}

    print('\n' + '='*62)
    print(' TactiQ — WC2026 UPSET WATCHLIST (Official Draw)')
    print(' 72 group stage fixtures across 12 groups')
    print('='*62)

    for team_a_name, team_b_name, group in WC2026_GROUP_FIXTURES:
        team_a = _load_team_from_db(team_a_name, cur, cluster_centroids)
        team_b = _load_team_from_db(team_b_name, cur, cluster_centroids)

        if team_a['is_proxy']:
            proxy_count += 1
            if team_a_name not in proxy_teams:
                proxy_teams.append(team_a_name)
        else:
            db_count += 1

        if team_b['is_proxy']:
            proxy_count += 1
            if team_b_name not in proxy_teams:
                proxy_teams.append(team_b_name)
        else:
            db_count += 1

        x_vec    = _build_feature_vector(team_a, team_b)
        X_single = x_vec.reshape(1, -1)

        calib_proba = model_calib.predict_proba(X_single)[0]
        p_win, p_draw, p_loss = float(calib_proba[0]), float(calib_proba[1]), float(calib_proba[2])
        pred_idx = int(np.argmax(calib_proba))
        pred_cls = LABEL_NAMES[pred_idx]

        sv_single  = explainer(X_single)
        shap_win   = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 0].tolist()))
        shap_draw  = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 1].tolist()))
        shap_loss  = dict(zip(FEATURE_NAMES, sv_single.values[0, :, 2].tolist()))

        top_feat_win  = max(shap_win,  key=lambda k: abs(shap_win[k]))
        top_feat_draw = max(shap_draw, key=lambda k: abs(shap_draw[k]))

        pred_info = {
            'team_a_name':     team_a_name,
            'team_b_name':     team_b_name,
            'archetype_a':     team_a['archetype_name'],
            'archetype_b':     team_b['archetype_name'],
            'p_win':           p_win,
            'p_draw':          p_draw,
            'p_loss':          p_loss,
            'predicted_class': pred_cls,
        }

        candidate, signals = is_upset_candidate(pred_info, shap_win, shap_draw, shap_loss)
        explanation = _upset_explanation(pred_info, signals) if candidate else ''

        rows.append({
            'group':              group,
            'team_a':             team_a_name,
            'team_b':             team_b_name,
            'archetype_a':        team_a['archetype_name'],
            'archetype_b':        team_b['archetype_name'],
            'p_win_a':            round(p_win,   3),
            'p_draw':             round(p_draw,  3),
            'p_loss_a':           round(p_loss,  3),
            'is_upset_candidate': candidate,
            'top_upset_feature':  top_feat_draw if candidate else top_feat_win,
            'upset_explanation':  explanation,
            'upset_rank':         None,
            'p_not_fav_win':      round(p_draw + p_loss, 3),
            # DB fields
            'team_a_id':          team_a.get('team_id'),
            'team_b_id':          team_b.get('team_id'),
            'shap_win':           shap_win,
            'shap_draw':          shap_draw,
            'shap_loss':          shap_loss,
            'top_feature_win':    top_feat_win,
            'top_feature_draw':   top_feat_draw,
            'predicted_class':    pred_cls,
        })

        groups_seen.setdefault(group, []).append(candidate)

    cur.close()
    conn.close()

    # Print coverage stats
    total_team_slots = len(WC2026_GROUP_FIXTURES) * 2
    print(f'\n Fixtures with DB vectors  : {db_count} / {total_team_slots}')
    print(f' Fixtures using proxy      : {proxy_count} / {total_team_slots}')
    print(f' Proxy teams               : {", ".join(sorted(proxy_teams))}')

    # Rank upset candidates
    upset_rows = [r for r in rows if r['is_upset_candidate']]
    upset_rows.sort(key=lambda x: x['p_not_fav_win'], reverse=True)
    for rank, r in enumerate(upset_rows, 1):
        r['upset_rank'] = rank

    if not upset_rows:
        print('\nNo upset candidates detected under current thresholds.')
    else:
        for r in upset_rows:
            print(f'\n⚠  UPSET CANDIDATE #{r["upset_rank"]}')
            print(f'   Group {r["group"]}  |  {r["team_b"]} ({r["archetype_b"]}) vs {r["team_a"]} ({r["archetype_a"]})')
            print(f'   Model      : {r["team_a"]} WIN {r["p_win_a"]*100:.0f}% '
                  f'| Draw {r["p_draw"]*100:.0f}% | {r["team_b"]} WIN {r["p_loss_a"]*100:.0f}%')
            top_f = r['top_upset_feature']
            top_v = r['shap_draw'].get(top_f, r['shap_loss'].get(top_f, 0))
            print(f'   Upset signal : {top_f} (SHAP {top_v:+.3f})')
            print(f'   Verdict    : {r["upset_explanation"]}')

    groups_no_upset = [g for g, cands in groups_seen.items() if not any(cands)]
    highest = max(upset_rows, key=lambda x: x['p_not_fav_win']) if upset_rows else None

    print('\nSUMMARY')
    print(f'  Matchups analysed    : {len(rows)}')
    print(f'  Upset candidates     : {len(upset_rows)}')
    print(f'  Groups with 0 upsets : {", ".join(sorted(groups_no_upset)) or "none"}')
    if highest:
        print(f'  Highest upset risk   : Group {highest["group"]}: '
              f'{highest["team_a"]} vs {highest["team_b"]} '
              f'— p(not-fav-win) = {highest["p_not_fav_win"]:.2f}')
    print('=' * 62)

    # Save CSV (without internal DB fields)
    PROCESSED_DIR.mkdir(exist_ok=True)
    _skip = {'shap_win', 'shap_draw', 'shap_loss', 'team_a_id', 'team_b_id',
             'top_feature_win', 'top_feature_draw', 'predicted_class'}
    csv_rows = [{k: v for k, v in r.items() if k not in _skip} for r in rows]
    pd.DataFrame(csv_rows).to_csv(PROCESSED_DIR / 'upset_watchlist.csv', index=False)
    print('Saved → data/processed/upset_watchlist.csv')

    return rows
