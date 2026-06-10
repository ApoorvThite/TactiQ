# Phase 3 — Style DNA Profiling

**Date completed:** 2026-06-09  
**Phase status:** Complete

---

## Summary

Phase 3 transforms the 54-team `team_style_profiles` table into tactical archetype clusters — the core "Style DNA" identity of TactiQ. The pipeline runs PCA → UMAP → K-means over 10 selected tactical features, assigns human-readable archetype names from centroid inspection, stores a 54×54 cosine-similarity matrix, and saves all model artifacts for use in Phase 4.

All 54 teams were clustered into **4 archetypes** with a silhouette score of **0.353** (above the 0.25 acceptability threshold).

---

## Clustering Configuration

- **Features used (10):**
  - `avg_possession_pct` — tempo / ball control
  - `avg_ppda` — pressing intensity (lower = more aggressive)
  - `avg_pressure_success_rate` — press effectiveness
  - `avg_xg_created_p90` — attacking output
  - `avg_xg_ratio` — attack vs defence balance
  - `avg_progressive_carry_pct` — directness / transition speed
  - `avg_pass_completion_pct` — technical quality
  - `avg_passes_final_third_p90` — attacking ambition
  - `avg_pass_completion_under_pressure_pct` — composure under press
  - `avg_set_piece_shot_pct` — set piece dependency

- **Scaler:** RobustScaler — PPDA (1.4–2.9) and xG ratio (0.4–7.0) have very different absolute ranges and outliers; RobustScaler's median/IQR scaling prevents high-variance features from dominating clustering distance calculations.

- **PCA components used for UMAP input:** 3 (explains 85.2% of variance)

- **UMAP parameters:** n_neighbors=8, min_dist=0.3, metric='euclidean', random_state=42

- **K-means:** k=4, silhouette=0.353, Davies-Bouldin=0.883, n_init=50

---

## Archetype Definitions

### High Press (5 teams)
**Teams:** Argentina, Brazil, Ecuador, Germany, Spain

**Centroid feature values:**
- PPDA: 1.64 (lowest of all clusters — most aggressive press)
- Possession: 61.2%
- xG Ratio: 5.36 (dominant attacking superiority)

**Tactical interpretation:** Elite possession-based teams with relentless pressing. These teams dominate the ball, apply sustained pressure, and convert that dominance into clear-cut chances. All 5 are traditional powerhouses expected deep at WC2026.

---

### Possession Control (21 teams)
**Teams:** Austria, Belgium, Canada, Colombia, Croatia, Denmark, England, France, Italy, Mexico, Netherlands, Peru, Portugal, Saudi Arabia, Serbia, Switzerland, Tunisia, Turkey, Ukraine, United States, Uruguay

**Centroid feature values:**
- PPDA: 1.84
- Possession: 51.6%
- xG Ratio: 1.67

**Tactical interpretation:** The largest cluster — technically competent teams that control the ball and create through structured build-up. They press moderately and generate decent xG. This is the "good team without elite dominance" bucket, spanning several European heavyweights and organised Americas sides.

---

### Counter-Attack (9 teams)
**Teams:** Czech Republic, Iceland, Iran, Nigeria, Romania, Russia, Senegal, Slovenia, Sweden

**Centroid feature values:**
- PPDA: 1.94
- Possession: 38.1% (lowest of all clusters)
- Progressive carry %: 13.99% (highest)

**Tactical interpretation:** Low-possession teams that rely on fast transitions and progressive carrying. They cede territory and hit on the break. The high progressive carry % and low possession are the defining signature.

---

### Deep Block (19 teams)
**Teams:** Albania, Australia, Cameroon, Costa Rica, Egypt, Finland, Georgia, Ghana, Hungary, Japan, Morocco, North Macedonia, Panama, Poland, Qatar, Scotland, Slovakia, South Korea, Wales

**Centroid feature values:**
- PPDA: 2.22 (highest — least pressing)
- Possession: 41.2%
- xG Ratio: 0.73 (lowest — defensive balance)

**Tactical interpretation:** Defensively-oriented teams that sit deep, concede possession, and limit chances. Low xG ratio reflects defensive concessions exceeding offensive output. Set pieces are a higher proportion of their attacking threat (4.3% vs cluster average).

---

## PCA Findings

**PC1 (57.9% variance) — Overall tactical quality / attacking dominance:**  
Top loadings: `avg_xg_ratio` (+0.689), `avg_xg_created_p90` (+0.314), `avg_passes_final_third_p90` (+0.308), `avg_possession_pct` (+0.305). PC1 is essentially a "team quality" axis — high positive values = dominant attacking sides with ball control.

**PC2 (15.4% variance) — Technical composure vs set-piece / counter dependency:**  
Top loadings: `avg_pass_completion_pct` (+0.512), `avg_pass_completion_under_pressure_pct` (+0.512), `avg_xg_ratio` (-0.402), `avg_set_piece_shot_pct` (-0.357). PC2 separates technically clean passers from teams relying on set pieces and counterattacking xG.

**PC3 (11.8% variance, cumulative 85.2%) was used as the UMAP input threshold.**  
The first 3 components capture the most meaningful tactical variance while filtering noise from the remaining 7 components.

---

## Similarity Findings

- **Most similar pair:** Germany vs Spain (0.988) — both High Press, nearly identical tactical profiles across all 10 features
- **Most different pair:** Austria vs Georgia (-0.907) — Possession Control vs Deep Block; stylistic opposites
- **Most stylistically unique team:** Saudi Arabia (avg similarity: -0.103) — sits within Possession Control but with an unusual combination of low PPDA and low xG ratio relative to possession, making them a tactical outlier

---

## Visualizations

**fig1_umap_archetypes.png** — Teams in 2D UMAP space coloured by archetype. High Press (Argentina, Spain, Germany, Brazil, Ecuador) clusters tightly in one corner, while Deep Block teams spread broadly. Convex hull boundaries make cluster separation visible.

**fig2_radar_archetypes.png** — One radar per archetype on 8 normalised axes. High Press clearly leads on xG Ratio and Possession; Deep Block sits near the centre on most axes except PPDA (where they score low = least pressing).

**fig3_silhouette.png** — Elbow curve shows steepest descent at k=2, but silhouette plateau from k=4–6 confirms 4 is the best balance of cluster quality and tactical interpretability.

**fig4_pca_variance.png** — PC1 alone explains 57.9%; the first 3 components cross 85%. The remaining 7 components contribute noise, justifying PCA denoising before UMAP.

**fig5_similarity_heatmap.png** — 54×54 heatmap ordered by cluster. Clear block structure visible: within-cluster pairs show green (high similarity), cross-archetype pairs (especially High Press vs Deep Block) show red (negative cosine similarity).

---

## Script Output

```
============================================================
 Step 1 — Feature Selection & Preprocessing
============================================================
Teams loaded from DB      : 54
Feature matrix shape      : (54, 10)
Any nulls remaining       : False
Feature means (scaled)    : [0.008, 0.009, -0.055, 0.211, 0.586, -0.085, 0.027, 0.022, 0.027, 0.021]
Feature stds  (scaled)    : [0.77, 0.893, 0.637, 0.839, 1.683, 0.718, 0.802, 0.804, 0.802, 0.77]
Saved → models/robust_scaler.pkl

PCA — Variance Explained
  PC1 :  57.9%  (cumulative:  57.9%)
  PC2 :  15.4%  (cumulative:  73.3%)
  PC3 :  11.8%  (cumulative:  85.2%)
  PC4 :   6.4%  (cumulative:  91.6%)
  PC5 :   4.1%  (cumulative:  95.7%)
  ...

UMAP embedding shape  : (54, 2)
UMAP x range          : -4.66 to 4.72
UMAP y range          : -0.83 to 2.46

k   Inertia     Silhouette   Davies-Bouldin
2   206.14      0.487        0.826
3   142.94      0.342        0.955
4   107.19      0.353        0.883  ← chosen
5    80.75      0.338        0.862
6    66.91      0.346        0.816
...

Cluster 0 → Deep Block (19 teams)
Cluster 1 → High Press (5 teams)
Cluster 2 → Possession Control (21 teams)
Cluster 3 → Counter-Attack (9 teams)

Most similar pair  : Germany vs Spain (0.988)
Most different pair: Austria vs Georgia (-0.907)
Most unique team   : Saudi Arabia (avg sim: -0.103)

============================================================
 Phase 3 complete. 54 teams fingerprinted.
 Ready for Phase 4: Matchup Model Training.
============================================================
```

---

## Decisions & Rationale

**Why RobustScaler not StandardScaler:**  
`avg_xg_ratio` ranges from 0.36–7.03 (Brazil extreme outlier), while `avg_ppda` ranges 1.41–2.94. StandardScaler normalises by mean/std, so Brazil's xG ratio outlier inflates the standard deviation and compresses all other teams toward zero on that axis. RobustScaler uses median and IQR (interquartile range), making it resistant to such outliers. Each feature contributes proportionally to clustering distance.

**Why PCA before UMAP:**  
With n=54 (small dataset) and 10 correlated features, UMAP on raw features can overfit local structure and produce unstable embeddings. Running PCA first decorrelates the features, removes noise from lower components (PCs 4–10 contribute only 14.8% variance), and gives UMAP a cleaner 3D input space. The result is a more stable, reproducible manifold.

**Why n_neighbors=8:**  
UMAP's `n_neighbors` controls the balance between local and global structure. For n=54, the rule of thumb is sqrt(n) ≈ 7–8. Using 15 (the default) would over-smooth the small dataset; 8 preserves within-cluster structure while still capturing inter-cluster relationships.

**How archetype names were assigned:**  
Names were assigned after inspecting `centroids_original` — not before. The algorithm scores each cluster on PPDA (press intensity), possession, xG ratio, progressive carry %, and set-piece %. The cluster with the lowest PPDA gets "High Press"; highest possession among the rest gets "Possession Control"; highest progressive carry % relative to possession gets "Counter-Attack"; the remainder (highest set-piece %, lowest xG) gets "Deep Block".

---

## Phase 4 Preview

The cosine similarity matrix (`team_similarity_matrix.csv`) and cluster assignments feed Phase 4's XGBoost matchup model. Rather than using similarity scores directly, Phase 4 computes the element-wise **difference** between two teams' scaled 10-feature style vectors. This delta vector (shape: 10,) becomes the primary feature input, capturing not just "how similar" but "in which tactical dimensions do these teams diverge" — which is the signal that predicts match outcomes.
