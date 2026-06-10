"""Phase 3 — Style DNA Profiling: PCA → UMAP → K-means → Archetypes."""

import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler
import umap

warnings.filterwarnings('ignore')
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models'
FIGURES_DIR = ROOT / 'docs' / 'figures'
PROCESSED_DIR = ROOT / 'data' / 'processed'

CLUSTERING_FEATURES = [
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

ARCHETYPE_DEFINITIONS = {
    'High Press':          'Low PPDA, high pressures, moderate possession',
    'Possession Control':  'High possession, high pass completion, low PPDA',
    'Counter-Attack':      'Low possession, high progressive carry %, high xG ratio',
    'Deep Block':          'Low pressures, high set-piece %, low xG created',
    'Balanced':            'Mid-range across all features — no dominant tactical trait',
}

CLUSTER_COLORS = ['#e63946', '#2a9d8f', '#f4a261', '#457b9d', '#8338ec']


def _get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tactiq'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


class StyleProfiler:

    def __init__(self):
        self.df = None
        self.X_scaled = None
        self.X_pca = None
        self.embedding = None
        self.cluster_labels = None
        self.archetype_map = {}      # cluster_id → archetype name
        self.scaler = None
        self.pca_85 = None
        self.reducer = None
        self.kmeans = None
        self.chosen_k = None
        self.similarity_matrix = None
        self.k_metrics = {}

    # ------------------------------------------------------------------ #
    #  Step 1 — Load & preprocess                                          #
    # ------------------------------------------------------------------ #
    def load_and_preprocess(self):
        print('\n' + '='*60)
        print(' Step 1 — Feature Selection & Preprocessing')
        print('='*60)

        conn = _get_conn()
        self.df = pd.read_sql(
            "SELECT * FROM team_style_profiles WHERE matches_played >= 3 ORDER BY team_name",
            conn,
        )
        conn.close()

        print(f'Teams loaded from DB      : {len(self.df)}')

        X = self.df[CLUSTERING_FEATURES].copy()

        # Sanitise PPDA sentinels
        X['avg_ppda'] = X['avg_ppda'].where(X['avg_ppda'] < 999, np.nan)
        X['avg_ppda'] = X['avg_ppda'].fillna(X['avg_ppda'].median())

        # Impute xg_ratio NaNs
        X['avg_xg_ratio'] = X['avg_xg_ratio'].fillna(X['avg_xg_ratio'].median())

        self.scaler = RobustScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        print(f'Feature matrix shape      : {self.X_scaled.shape}')
        print(f'Any nulls remaining       : {np.isnan(self.X_scaled).any()}')
        means = self.X_scaled.mean(axis=0)
        stds  = self.X_scaled.std(axis=0)
        print(f'Feature means (scaled)    : {np.round(means, 3).tolist()}')
        print(f'Feature stds  (scaled)    : {np.round(stds,  3).tolist()}')

        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODELS_DIR / 'robust_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print('Saved → models/robust_scaler.pkl')

    # ------------------------------------------------------------------ #
    #  Step 2 — PCA                                                        #
    # ------------------------------------------------------------------ #
    def run_pca(self):
        print('\n' + '='*60)
        print(' Step 2 — PCA Analysis')
        print('='*60)

        pca_full = PCA(n_components=10, random_state=42)
        pca_full.fit(self.X_scaled)

        ev   = pca_full.explained_variance_ratio_
        cumv = np.cumsum(ev)

        print('\nPCA — Variance Explained')
        for i, (v, c) in enumerate(zip(ev, cumv)):
            print(f'  PC{i+1:<2}: {v*100:5.1f}%  (cumulative: {c*100:5.1f}%)')

        n80 = int(np.argmax(cumv >= 0.80)) + 1
        n85 = int(np.argmax(cumv >= 0.85)) + 1
        n90 = int(np.argmax(cumv >= 0.90)) + 1
        print(f'\nComponents needed to explain 80% variance: {n80}')
        print(f'Components needed to explain 85% variance: {n85}')
        print(f'Components needed to explain 90% variance: {n90}')

        # Loadings
        print('\nPC1 top loadings (what does PC1 capture?):')
        pc1 = pca_full.components_[0]
        for feat, load in sorted(zip(CLUSTERING_FEATURES, pc1), key=lambda x: abs(x[1]), reverse=True):
            print(f'  {feat:<45}: {load:+.3f}')

        print('\nPC2 top loadings (what does PC2 capture?):')
        pc2 = pca_full.components_[1]
        for feat, load in sorted(zip(CLUSTERING_FEATURES, pc2), key=lambda x: abs(x[1]), reverse=True):
            print(f'  {feat:<45}: {load:+.3f}')

        # Store full PCA components for variance plot
        self._pca_full_ev   = ev
        self._pca_full_cumv = cumv
        self._n_components_85 = n85

        # Fit 85% PCA
        self.pca_85 = PCA(n_components=n85, random_state=42)
        self.X_pca  = self.pca_85.fit_transform(self.X_scaled)

        print(f'\nPCA (85% variance) n_components = {n85}')
        print(f'X_pca shape: {self.X_pca.shape}')

        with open(MODELS_DIR / 'pca_85.pkl', 'wb') as f:
            pickle.dump(self.pca_85, f)
        print('Saved → models/pca_85.pkl')

    # ------------------------------------------------------------------ #
    #  Step 3 — UMAP                                                       #
    # ------------------------------------------------------------------ #
    def run_umap(self):
        print('\n' + '='*60)
        print(' Step 3 — UMAP 2D Projection')
        print('='*60)

        self.reducer = umap.UMAP(
            n_components=2,
            n_neighbors=8,
            min_dist=0.3,
            metric='euclidean',
            random_state=42,
        )
        self.embedding = self.reducer.fit_transform(self.X_pca)

        self.df['umap_x'] = self.embedding[:, 0]
        self.df['umap_y'] = self.embedding[:, 1]

        print(f'UMAP embedding shape  : {self.embedding.shape}')
        print(f'UMAP x range          : {self.embedding[:,0].min():.2f} to {self.embedding[:,0].max():.2f}')
        print(f'UMAP y range          : {self.embedding[:,1].min():.2f} to {self.embedding[:,1].max():.2f}')

        with open(MODELS_DIR / 'umap_reducer.pkl', 'wb') as f:
            pickle.dump(self.reducer, f)
        print('Saved → models/umap_reducer.pkl')

    # ------------------------------------------------------------------ #
    #  Step 4 — Find optimal k                                             #
    # ------------------------------------------------------------------ #
    def find_optimal_k(self):
        print('\n' + '='*60)
        print(' Step 4 — K-means: Finding Optimal k')
        print('='*60)

        print(f'\n{"k":<4} {"Inertia":>10}  {"Silhouette":>12}  {"Davies-Bouldin":>16}')
        print('-' * 48)

        best_k    = 4
        best_sil  = -1
        metrics   = {}

        for k in range(2, 10):
            km  = KMeans(n_clusters=k, random_state=42, n_init=50)
            lbl = km.fit_predict(self.X_pca)
            inr = km.inertia_
            sil = silhouette_score(self.X_pca, lbl)
            db  = davies_bouldin_score(self.X_pca, lbl)
            metrics[k] = {'inertia': inr, 'silhouette': sil, 'davies_bouldin': db, 'labels': lbl}

            note = ''
            if sil > best_sil and 4 <= k <= 6:
                best_sil = sil
                best_k   = k
                note = ' ← leading'
            print(f'{k:<4} {inr:>10.2f}  {sil:>12.3f}  {db:>16.3f}{note}')

        # Final pick: highest silhouette among k=4–6
        self.k_metrics = metrics
        self.chosen_k  = best_k
        print(f'\nRecommended k (best silhouette in 4–6): {self.chosen_k}')
        print(f'Justification: k={self.chosen_k} yields silhouette={metrics[self.chosen_k]["silhouette"]:.3f} '
              f'with tactically meaningful cluster count.')

    # ------------------------------------------------------------------ #
    #  Step 5 — Final K-means & archetype assignment                       #
    # ------------------------------------------------------------------ #
    def fit_kmeans_and_assign_archetypes(self):
        print('\n' + '='*60)
        print(' Step 5 — Final K-means & Archetype Assignment')
        print('='*60)

        self.kmeans = KMeans(n_clusters=self.chosen_k, random_state=42, n_init=50)
        self.cluster_labels = self.kmeans.fit_predict(self.X_pca)
        self.df['cluster_id'] = self.cluster_labels

        with open(MODELS_DIR / 'kmeans.pkl', 'wb') as f:
            pickle.dump(self.kmeans, f)
        print('Saved → models/kmeans.pkl')

        # Back-transform centroids to original feature space
        centroids_pca    = self.kmeans.cluster_centers_
        centroids_scaled = self.pca_85.inverse_transform(centroids_pca)
        centroids_orig   = self.scaler.inverse_transform(centroids_scaled)
        centroid_df      = pd.DataFrame(centroids_orig, columns=CLUSTERING_FEATURES)

        print('\nCluster Centroids (original feature scale)')
        header = f"{'Feature':<45}"
        for c in range(self.chosen_k):
            header += f' | Cluster {c}'
        print(header)
        print('-' * (45 + 12 * self.chosen_k))
        for feat in CLUSTERING_FEATURES:
            row = f'{feat:<45}'
            for c in range(self.chosen_k):
                row += f' | {centroid_df.loc[c, feat]:>8.3f}'
            print(row)

        # Assign archetype names from centroid profiles
        self.archetype_map = self._assign_archetype_names(centroid_df)

        print('\nCluster assignments:')
        for cid, name in sorted(self.archetype_map.items()):
            count = (self.cluster_labels == cid).sum()
            print(f'  Cluster {cid} → {name} ({count} teams)')

        self.df['archetype_name'] = self.df['cluster_id'].map(self.archetype_map)

        print('\nTeams per archetype:')
        for name in sorted(self.archetype_map.values()):
            teams = self.df[self.df['archetype_name'] == name]['team_name'].tolist()
            print(f'  {name:<22}: {", ".join(sorted(teams))}')

        self._centroid_df = centroid_df

    def _assign_archetype_names(self, centroid_df):
        """Inspect centroids and assign canonical archetype names."""
        k   = self.chosen_k
        list(ARCHETYPE_DEFINITIONS.keys())[:k]

        # Score each cluster on key axes
        scores = {}
        for c in range(k):
            row = centroid_df.iloc[c]
            scores[c] = {
                'ppda':        row['avg_ppda'],                     # low = high press
                'possession':  row['avg_possession_pct'],
                'xg_ratio':    row['avg_xg_ratio'],
                'prog_carry':  row['avg_progressive_carry_pct'],
                'set_piece':   row['avg_set_piece_shot_pct'],
                'xg_created':  row['avg_xg_created_p90'],
            }

        assigned = {}
        used     = set()

        # High Press: lowest PPDA (most aggressive press)
        c_hp = min(range(k), key=lambda c: scores[c]['ppda'])
        assigned[c_hp] = 'High Press'
        used.add(c_hp)

        remaining = [c for c in range(k) if c not in used]

        # Possession Control: highest possession among remaining
        c_pc = max(remaining, key=lambda c: scores[c]['possession'])
        assigned[c_pc] = 'Possession Control'
        used.add(c_pc)

        remaining = [c for c in range(k) if c not in used]

        if len(remaining) >= 1:
            # Counter-Attack: lowest possession + highest prog_carry among remaining
            c_ca = max(remaining, key=lambda c: scores[c]['prog_carry'] - scores[c]['possession'])
            assigned[c_ca] = 'Counter-Attack'
            used.add(c_ca)
            remaining = [c for c in range(k) if c not in used]

        if len(remaining) >= 1:
            # Deep Block: highest set-piece %, lowest xg_created
            c_db = max(remaining, key=lambda c: scores[c]['set_piece'] - scores[c]['xg_created'])
            assigned[c_db] = 'Deep Block'
            used.add(c_db)
            remaining = [c for c in range(k) if c not in used]

        for c in remaining:
            assigned[c] = 'Balanced'

        return assigned

    # ------------------------------------------------------------------ #
    #  Step 6 — Cosine similarity matrix                                   #
    # ------------------------------------------------------------------ #
    def compute_similarity_matrix(self):
        print('\n' + '='*60)
        print(' Step 6 — Cosine Similarity Matrix')
        print('='*60)

        teams = self.df['team_name'].tolist()
        sim   = cosine_similarity(self.X_scaled)
        self.similarity_matrix = pd.DataFrame(sim, index=teams, columns=teams)

        PROCESSED_DIR.mkdir(exist_ok=True)
        self.similarity_matrix.to_csv(PROCESSED_DIR / 'team_similarity_matrix.csv')
        print('Saved → data/processed/team_similarity_matrix.csv')

        # Pairs analysis (exclude self-similarity diagonal)
        n = len(teams)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((teams[i], teams[j], sim[i, j],
                               self.df.iloc[i]['archetype_name'],
                               self.df.iloc[j]['archetype_name']))

        pairs.sort(key=lambda x: x[2], reverse=True)

        print('\nTop 5 most similar team pairs (highest cosine similarity):')
        for rank, (t1, t2, s, a1, a2) in enumerate(pairs[:5], 1):
            print(f'  {rank}. {t1} vs {t2} : {s:.3f} — both {a1}' if a1 == a2
                  else f'  {rank}. {t1} vs {t2} : {s:.3f} — {a1} vs {a2}')

        print('\nTop 5 most different team pairs (lowest cosine similarity):')
        for rank, (t1, t2, s, a1, a2) in enumerate(pairs[-5:][::-1], 1):
            print(f'  {rank}. {t1} vs {t2} : {s:.3f} — {a1} vs {a2}')

        avg_sims = {t: (sim[i].sum() - 1) / (n - 1) for i, t in enumerate(teams)}
        unique_team = min(avg_sims, key=avg_sims.get)
        print('\nMost stylistically unique team (lowest avg similarity to all others):')
        print(f'  {unique_team} : avg similarity = {avg_sims[unique_team]:.3f}')

        self._avg_sims    = avg_sims
        self._pairs       = pairs

    # ------------------------------------------------------------------ #
    #  Step 7 — Save to PostgreSQL                                         #
    # ------------------------------------------------------------------ #
    def save_to_db(self):
        print('\n' + '='*60)
        print(' Step 7 — Saving to PostgreSQL')
        print('='*60)

        conn = _get_conn()
        cur  = conn.cursor()

        # Run schema DDL
        ddl_path = ROOT / 'db' / 'schema' / '003_cluster_tables.sql'
        cur.execute(open(ddl_path).read())
        conn.commit()
        print('Schema applied (003_cluster_tables.sql)')

        # Insert style_clusters
        cur.execute('DELETE FROM style_clusters')
        for cid, name in self.archetype_map.items():
            row = self._centroid_df.iloc[cid]
            teams_in_cluster = (self.df['cluster_id'] == cid).sum()
            cur.execute(
                """INSERT INTO style_clusters
                   (cluster_id, archetype_name, archetype_desc, avg_ppda, avg_possession, avg_xg_ratio, team_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (int(cid), name, ARCHETYPE_DEFINITIONS.get(name, ''),
                 float(row['avg_ppda']), float(row['avg_possession_pct']),
                 float(row['avg_xg_ratio']), int(teams_in_cluster))
            )
        conn.commit()
        print(f'Inserted {self.chosen_k} rows into style_clusters')

        # Update team_style_profiles
        updated = 0
        for _, row in self.df.iterrows():
            svec = json.dumps([float(v) for v in self.X_scaled[self.df.index.get_loc(row.name)]])
            cur.execute(
                """UPDATE team_style_profiles
                   SET cluster_id = %s, archetype_name = %s,
                       umap_x = %s, umap_y = %s, style_vector = %s
                   WHERE team_id = %s""",
                (int(row['cluster_id']), row['archetype_name'],
                 float(row['umap_x']),   float(row['umap_y']),
                 svec, int(row['team_id']))
            )
            updated += 1

        conn.commit()
        cur.close()
        conn.close()
        print(f'Updated {updated} rows in team_style_profiles')

    # ------------------------------------------------------------------ #
    #  Step 8 — Visualizations                                             #
    # ------------------------------------------------------------------ #
    def generate_figures(self):
        print('\n' + '='*60)
        print(' Step 8 — Generating Visualizations')
        print('='*60)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.style.use('dark_background')

        self._fig1_umap_archetypes()
        self._fig2_radar_archetypes()
        self._fig3_silhouette()
        self._fig4_pca_variance()
        self._fig5_similarity_heatmap()

    def _fig1_umap_archetypes(self):
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)

        archetypes = sorted(self.archetype_map.values())
        color_map  = {name: CLUSTER_COLORS[i] for i, name in enumerate(sorted(set(archetypes)))}

        for _, row in self.df.iterrows():
            c = color_map[row['archetype_name']]
            ax.scatter(row['umap_x'], row['umap_y'], color=c, s=80, zorder=3, alpha=0.9)
            ax.annotate(row['team_name'],
                        (row['umap_x'], row['umap_y']),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=6.5, color='white', alpha=0.85, zorder=4)

        # Convex hulls per cluster
        for cid, name in self.archetype_map.items():
            pts = self.df[self.df['cluster_id'] == cid][['umap_x', 'umap_y']].values
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            alpha=0.12, color=color_map[name], zorder=1)
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color_map[name], linewidth=1.2, alpha=0.6, zorder=2)
                except Exception:
                    pass

        handles = [mpatches.Patch(color=color_map[n], label=f'{n} ({(self.df["archetype_name"]==n).sum()})')
                   for n in sorted(color_map)]
        ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.4)
        ax.set_title('TactiQ — Tactical Style Space (UMAP Projection)', fontsize=14, pad=14)
        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        fig.tight_layout()
        path = FIGURES_DIR / 'fig1_umap_archetypes.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved → {path}')

    def _fig2_radar_archetypes(self):
        radar_features = [
            'avg_possession_pct',
            'avg_ppda',            # will invert
            'avg_xg_ratio',
            'avg_progressive_carry_pct',
            'avg_pass_completion_pct',
            'avg_passes_final_third_p90',
            'avg_pass_completion_under_pressure_pct',
            'avg_set_piece_shot_pct',
        ]
        labels = ['Possession', 'Press\n(inv. PPDA)', 'xG Ratio',
                  'Prog. Carry', 'Pass Cmplt.', 'Final 3rd\nPasses',
                  'Composure\nUnder Press', 'Set Piece\n%']

        X_radar = self.df[radar_features].copy().values.astype(float)
        # Invert PPDA axis (index 1) so higher = more aggressive
        ppda_idx = 1
        X_radar[:, ppda_idx] = 1 / (X_radar[:, ppda_idx] + 1e-9)

        col_min = X_radar.min(axis=0)
        col_max = X_radar.max(axis=0)
        X_norm  = (X_radar - col_min) / (col_max - col_min + 1e-9)

        n_axes = len(radar_features)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]

        ncols = min(self.chosen_k, 3)
        nrows = (self.chosen_k + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                  dpi=150, subplot_kw=dict(polar=True))
        if self.chosen_k == 1:
            axes = [[axes]]
        axes_flat = np.array(axes).flatten()

        archetypes_sorted = sorted(self.archetype_map.items())
        color_map = {name: CLUSTER_COLORS[i] for i, name in enumerate(sorted(set(self.archetype_map.values())))}

        for idx, (cid, name) in enumerate(archetypes_sorted):
            ax = axes_flat[idx]
            mask  = self.df['cluster_id'] == cid
            centroid = X_norm[mask].mean(axis=0).tolist() + [X_norm[mask].mean(axis=0)[0]]

            ax.plot(angles, centroid, color=color_map[name], linewidth=2)
            ax.fill(angles, centroid, color=color_map[name], alpha=0.35)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=7.5, color='white')
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=6, color='grey')
            ax.set_ylim(0, 1)
            ', '.join(sorted(self.df[mask]['team_name'].tolist()))
            ax.set_title(f'{name}\n({mask.sum()} teams)', size=9, pad=14, color=color_map[name])
            ax.tick_params(colors='white')

        for idx in range(self.chosen_k, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle('TactiQ — Archetype Radar Profiles', fontsize=13, y=1.01)
        fig.tight_layout()
        path = FIGURES_DIR / 'fig2_radar_archetypes.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved → {path}')

    def _fig3_silhouette(self):
        ks   = sorted(self.k_metrics.keys())
        inrs = [self.k_metrics[k]['inertia']    for k in ks]
        sils = [self.k_metrics[k]['silhouette'] for k in ks]

        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
        ax2 = ax1.twinx()

        ax1.plot(ks, inrs, 'o-', color='#f4a261', linewidth=2, markersize=6, label='Inertia')
        ax2.plot(ks, sils, 's--', color='#2a9d8f', linewidth=2, markersize=6, label='Silhouette')

        ax1.axvline(self.chosen_k, color='white', linestyle=':', linewidth=1.5,
                    label=f'Chosen k={self.chosen_k}')

        ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax1.set_ylabel('Inertia', fontsize=11, color='#f4a261')
        ax2.set_ylabel('Silhouette Score', fontsize=11, color='#2a9d8f')
        ax1.tick_params(axis='y', labelcolor='#f4a261')
        ax2.tick_params(axis='y', labelcolor='#2a9d8f')
        ax1.set_xticks(ks)

        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='upper right', fontsize=9, framealpha=0.4)
        ax1.set_title('K-means Cluster Selection — Elbow & Silhouette', fontsize=13, pad=12)
        fig.tight_layout()
        path = FIGURES_DIR / 'fig3_silhouette.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved → {path}')

    def _fig4_pca_variance(self):
        ev   = self._pca_full_ev
        cumv = self._pca_full_cumv
        comp = list(range(1, len(ev) + 1))

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
        ax2 = ax1.twinx()

        ax1.bar(comp, ev * 100, color='#457b9d', alpha=0.8, label='Individual variance')
        ax2.plot(comp, cumv * 100, 'o-', color='#e63946', linewidth=2, markersize=6, label='Cumulative')

        for thresh, label in [(80, '80%'), (90, '90%'), (95, '95%')]:
            ax2.axhline(thresh, color='grey', linestyle='--', linewidth=1, alpha=0.6)
            ax2.text(10.1, thresh, label, va='center', fontsize=8, color='grey')

        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Individual Variance Explained (%)', fontsize=11, color='#457b9d')
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11, color='#e63946')
        ax1.tick_params(axis='y', labelcolor='#457b9d')
        ax2.tick_params(axis='y', labelcolor='#e63946')
        ax1.set_xticks(comp)
        ax2.set_ylim(0, 105)

        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='center right', fontsize=9, framealpha=0.4)
        ax1.set_title('PCA — Variance Explained by Component', fontsize=13, pad=12)
        fig.tight_layout()
        path = FIGURES_DIR / 'fig4_pca_variance.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved → {path}')

    def _fig5_similarity_heatmap(self):
        # Order teams by cluster
        ordered_df = self.df.sort_values(['cluster_id', 'team_name'])
        ordered_teams = ordered_df['team_name'].tolist()
        sim_ordered = self.similarity_matrix.loc[ordered_teams, ordered_teams].values

        fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
        im = ax.imshow(sim_ordered, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Cosine Similarity', fraction=0.03)

        ax.set_xticks(range(len(ordered_teams)))
        ax.set_yticks(range(len(ordered_teams)))
        ax.set_xticklabels(ordered_teams, rotation=90, fontsize=5.5)
        ax.set_yticklabels(ordered_teams, fontsize=5.5)

        # Draw white cluster boundaries
        boundaries = []
        prev_cid = ordered_df.iloc[0]['cluster_id']
        count = 0
        for _, row in ordered_df.iterrows():
            if row['cluster_id'] != prev_cid:
                boundaries.append(count - 0.5)
                prev_cid = row['cluster_id']
            count += 1

        for b in boundaries:
            ax.axhline(b, color='white', linewidth=1.5)
            ax.axvline(b, color='white', linewidth=1.5)

        ax.set_title('TactiQ — Team Style Similarity Matrix', fontsize=13, pad=12)
        fig.tight_layout()
        path = FIGURES_DIR / 'fig5_similarity_heatmap.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved → {path}')

    # ------------------------------------------------------------------ #
    #  Final summary                                                        #
    # ------------------------------------------------------------------ #
    def print_summary(self):
        print('\n')
        print('=' * 60)
        print(' TACTIQ — Phase 3 Style DNA Profiling Complete')
        print('=' * 60)

        sil = self.k_metrics[self.chosen_k]['silhouette']
        db  = self.k_metrics[self.chosen_k]['davies_bouldin']

        print('\nCLUSTERING RESULTS')
        print(f'  Teams clustered          : {len(self.df)}')
        print(f'  Optimal k chosen         : {self.chosen_k}')
        print(f'  Silhouette score         : {sil:.3f}')
        print(f'  Davies-Bouldin score     : {db:.3f}')

        print('\nARCHETYPES')
        for cid, name in sorted(self.archetype_map.items()):
            mask  = self.df['cluster_id'] == cid
            row   = self._centroid_df.iloc[cid]
            count = mask.sum()
            print(f'  {name:<22}: {count} teams  '
                  f'(centroid PPDA: {row["avg_ppda"]:.2f}, '
                  f'Poss: {row["avg_possession_pct"]:.1f}%, '
                  f'xGR: {row["avg_xg_ratio"]:.2f})')

        pairs = self._pairs
        print('\nSIMILARITY MATRIX')
        t1, t2, s, *_ = pairs[0]
        print(f'  Most similar pair        : {t1} vs {t2} ({s:.3f})')
        t1, t2, s, *_ = pairs[-1]
        print(f'  Most different pair      : {t1} vs {t2} ({s:.3f})')
        ut = min(self._avg_sims, key=self._avg_sims.get)
        print(f'  Most unique team         : {ut} (avg sim: {self._avg_sims[ut]:.3f})')

        print('\nSAVED ARTIFACTS')
        artifacts = [
            'models/robust_scaler.pkl',
            'models/pca_85.pkl',
            'models/umap_reducer.pkl',
            'models/kmeans.pkl',
            'data/processed/team_similarity_matrix.csv',
            'docs/figures/fig1_umap_archetypes.png',
            'docs/figures/fig2_radar_archetypes.png',
            'docs/figures/fig3_silhouette.png',
            'docs/figures/fig4_pca_variance.png',
            'docs/figures/fig5_similarity_heatmap.png',
        ]
        for a in artifacts:
            exists = (ROOT / a).exists()
            print(f'  {a:<48} {"✓" if exists else "✗"}')

        print('\nDATABASE UPDATES')
        print(f'  style_clusters rows      : {self.chosen_k}')
        print(f'  team_style_profiles rows with cluster_id : {len(self.df)} / {len(self.df)}')

        print('\n' + '=' * 60)
        print(f' Phase 3 complete. {len(self.df)} teams fingerprinted.')
        print(' Ready for Phase 4: Matchup Model Training.')
        print('=' * 60)
