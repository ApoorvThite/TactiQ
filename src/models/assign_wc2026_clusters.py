"""Phase 6 Step 3 — Assign clusters to all 48 WC2026 teams and upsert into DB.

For teams already in DB: verify cluster assignment is current.
For new teams: scale → PCA → KMeans.predict() → assign UMAP coords as
cluster mean; insert new row into team_style_profiles.
NEVER calls scaler.fit_transform() or kmeans.fit() — only .transform() / .predict().
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
MODELS_DIR    = ROOT / 'models'
PROCESSED_DIR = ROOT / 'data' / 'processed'

CLUSTERING_FEATURES = [
    'avg_possession_pct', 'avg_ppda', 'avg_pressure_success_rate',
    'avg_xg_created_p90', 'avg_xg_ratio', 'avg_progressive_carry_pct',
    'avg_pass_completion_pct', 'avg_passes_final_third_p90',
    'avg_pass_completion_under_pressure_pct', 'avg_set_piece_shot_pct',
]

ARCHETYPE_NAMES = {0: 'High Press', 1: 'Possession Control', 2: 'Counter-Attack', 3: 'Deep Block'}

# Name aliases for DB lookup
NAME_ALIASES = {
    'Türkiye':              'Turkey',
    'Czechia':              'Czech Republic',
    'DR Congo':             'Congo DR',
    "Côte d'Ivoire":       'Ivory Coast',
    'USA':                  'United States',
    'Korea Republic':       'South Korea',
    'Bosnia-Herzegovina':   'Bosnia and Herzegovina',
}


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def _load_models():
    with open(MODELS_DIR / 'robust_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / 'pca_85.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open(MODELS_DIR / 'kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    umap_reducer = None
    try:
        with open(MODELS_DIR / 'umap_reducer.pkl', 'rb') as f:
            umap_reducer = pickle.load(f)
    except Exception:
        pass
    return scaler, pca, kmeans, umap_reducer


def _compute_cluster_umap_means(cur):
    """Compute mean UMAP coordinates per cluster from existing DB members."""
    cur.execute(
        "SELECT cluster_id, umap_x, umap_y FROM team_style_profiles "
        "WHERE cluster_id IS NOT NULL AND umap_x IS NOT NULL AND umap_y IS NOT NULL"
    )
    from collections import defaultdict
    coords = defaultdict(list)
    for cluster_id, ux, uy in cur.fetchall():
        coords[cluster_id].append((ux, uy))
    means = {}
    for cid, pts in coords.items():
        means[cid] = (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))
    return means


def _compute_archetype_style_centroids(cur):
    """
    Compute mean style vector per archetype from existing DB members.
    Used to give proxy teams a style vector that lies at the archetype centre
    in the ORIGINAL 10-feature space, so KMeans.predict() lands in the right cluster.
    """
    cur.execute(
        "SELECT archetype_name, style_vector FROM team_style_profiles "
        "WHERE archetype_name IS NOT NULL AND style_vector IS NOT NULL"
    )
    from collections import defaultdict
    vectors = defaultdict(list)
    for arch, sv in cur.fetchall():
        vectors[arch].append(np.array(sv))
    return {arch: np.mean(svs, axis=0).tolist() for arch, svs in vectors.items()}


def assign_cluster(style_vector, scaler, pca, kmeans):
    """
    Assign cluster to a 10-dim style vector using fitted models.
    Returns (cluster_id, archetype_name).
    """
    sv = np.array(style_vector).reshape(1, -1)
    sv_scaled = scaler.transform(sv)
    sv_pca    = pca.transform(sv_scaled)
    cluster_id = int(kmeans.predict(sv_pca)[0])
    return cluster_id, ARCHETYPE_NAMES[cluster_id]


def get_umap_coords(sv, scaler, pca, umap_reducer, cluster_id, cluster_umap_means):
    """
    Compute UMAP coordinates for a new team.
    Tries umap_reducer.transform(); falls back to cluster mean coordinates.
    """
    if umap_reducer is not None:
        try:
            sv_scaled = scaler.transform(np.array(sv).reshape(1, -1))
            sv_pca    = pca.transform(sv_scaled)
            umap_coords = umap_reducer.transform(sv_pca)
            return float(umap_coords[0, 0]), float(umap_coords[0, 1])
        except Exception as e:
            print(f'    [WARN] UMAP.transform() failed ({e}) — using cluster mean')

    if cluster_id in cluster_umap_means:
        ux, uy = cluster_umap_means[cluster_id]
        # Add small jitter so the new point doesn't stack exactly on the mean
        ux += np.random.uniform(-0.3, 0.3)
        uy += np.random.uniform(-0.3, 0.3)
        return float(ux), float(uy)

    return 0.0, 0.0


def _get_or_create_team_id(canonical_name, cur, conn):
    """
    Return team_id for a team name.
    If the team is not in the `teams` table, insert a synthetic row
    with an ID starting at 90000 to avoid collisions with StatsBomb IDs.
    """
    cur.execute("SELECT team_id FROM teams WHERE LOWER(team_name) = LOWER(%s)",
                (canonical_name,))
    row = cur.fetchone()
    if row:
        return row[0]

    # Allocate a synthetic ID (max existing ≥ 90000 to avoid StatsBomb collisions)
    cur.execute("SELECT COALESCE(MAX(team_id), 89999) FROM teams WHERE team_id >= 90000")
    max_id = cur.fetchone()[0]
    new_id = max_id + 1

    cur.execute(
        "INSERT INTO teams (team_id, team_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (new_id, canonical_name)
    )
    conn.commit()
    return new_id


def process_teams(scraped_results, in_db_entries):
    """
    Upsert cluster assignments for all 48 WC2026 teams.

    scraped_results: dict team_name → {style_vector, archetype_name, is_proxy}
                     (only for teams that needed scraping)
    in_db_entries:   list of dicts from audit step (already in DB)

    Returns list of all 48 team profile dicts.
    """
    scaler, pca, kmeans, umap_reducer = _load_models()

    conn = _get_conn()
    cur  = conn.cursor()

    cluster_umap_means       = _compute_cluster_umap_means(cur)
    archetype_style_centroids = _compute_archetype_style_centroids(cur)

    all_profiles = []
    inserted     = 0
    skipped      = 0

    in_db_names = {e['team_name'].lower(): e for e in in_db_entries}

    all_teams = set(in_db_names.keys()) | set(k.lower() for k in scraped_results.keys())

    print('\n' + '-'*62)
    print(' Assigning clusters to all 48 WC2026 teams')
    print('-'*62)

    for team_lower in sorted(all_teams):
        # Resolve canonical name
        canonical = None
        for e in in_db_entries:
            if e['team_name'].lower() == team_lower:
                canonical = e['team_name']
                break
        if canonical is None:
            for k in scraped_results:
                if k.lower() == team_lower:
                    canonical = k
                    break
        if canonical is None:
            canonical = team_lower.title()

        in_db_info  = in_db_names.get(team_lower)
        scrape_info = scraped_results.get(canonical) or scraped_results.get(canonical.lower())

        if in_db_info and in_db_info.get('status') in ('IN_DB', 'IN_DB_FUZZY'):
            cur.execute(
                "SELECT team_id, cluster_id, archetype_name, style_vector, matches_played, umap_x, umap_y "
                "FROM team_style_profiles WHERE team_id = %s",
                (in_db_info['team_id'],)
            )
            row = cur.fetchone()
            if row:
                profile = {
                    'team_id':        row[0],
                    'team_name':      canonical,
                    'cluster_id':     row[1],
                    'archetype_name': row[2],
                    'style_vector':   row[3],
                    'matches_played': row[4],
                    'umap_x':         row[5],
                    'umap_y':         row[6],
                    'is_proxy':       False,
                    'source':         'statsbomb',
                }
                all_profiles.append(profile)
                skipped += 1
                print(f'  ✓  {canonical:<30} [existing] → {row[2]}')
                continue

        # New team — insert into teams + team_style_profiles
        if scrape_info and not scrape_info['is_proxy']:
            sv     = scrape_info['style_vector']
            arch   = scrape_info['archetype_name']
            is_prx = False
        else:
            # Use proxy: load archetype centroid from DB (real cluster centre)
            from src.ingestion.scrape_fbref_national import PROXY_ARCHETYPES
            arch   = PROXY_ARCHETYPES.get(canonical, 'Counter-Attack')
            sv     = archetype_style_centroids.get(arch,
                         archetype_style_centroids.get('Counter-Attack', [0.0] * 10))
            is_prx = True

        sv_list = sv if isinstance(sv, list) else sv.tolist()
        cluster_id, cluster_arch = assign_cluster(sv_list, scaler, pca, kmeans)
        ux, uy = get_umap_coords(sv_list, scaler, pca, umap_reducer, cluster_id, cluster_umap_means)

        team_id = _get_or_create_team_id(canonical, cur, conn)

        cur.execute(
            """INSERT INTO team_style_profiles
               (team_id, team_name, cluster_id, archetype_name, style_vector,
                matches_played, umap_x, umap_y)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (team_id) DO UPDATE SET
                 cluster_id     = EXCLUDED.cluster_id,
                 archetype_name = EXCLUDED.archetype_name,
                 style_vector   = EXCLUDED.style_vector,
                 umap_x         = EXCLUDED.umap_x,
                 umap_y         = EXCLUDED.umap_y""",
            (team_id, canonical, cluster_id, cluster_arch,
             json.dumps(sv_list),
             -1 if is_prx else 3,
             ux, uy)
        )
        conn.commit()

        proxy_flag = ' [PROXY]' if is_prx else ' [SCRAPED]'
        print(f'  +  {canonical:<30}{proxy_flag} → {cluster_arch}  (cluster {cluster_id})')

        profile = {
            'team_id':        team_id,
            'team_name':      canonical,
            'cluster_id':     cluster_id,
            'archetype_name': cluster_arch,
            'style_vector':   sv_list,
            'matches_played': -1 if is_prx else 3,
            'umap_x':         ux,
            'umap_y':         uy,
            'is_proxy':       is_prx,
            'source':         'proxy' if is_prx else 'fbref',
        }
        all_profiles.append(profile)
        inserted += 1

    cur.close()
    conn.close()

    print(f'\n  Existing (no change) : {skipped}')
    print(f'  Inserted / updated   : {inserted}')
    print(f'  Total profiles ready : {len(all_profiles)}')

    return all_profiles


if __name__ == '__main__':
    from src.models.audit_wc2026_teams import audit_teams
    in_db, missing, _ = audit_teams()
    profiles = process_teams({}, in_db)
    print(f'\nDone: {len(profiles)} team profiles ready')
