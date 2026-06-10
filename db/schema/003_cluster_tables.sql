-- TactiQ Phase 3 — Cluster tables and team_style_profiles extensions

CREATE TABLE IF NOT EXISTS style_clusters (
    cluster_id      INTEGER PRIMARY KEY,
    archetype_name  TEXT NOT NULL,
    archetype_desc  TEXT,
    avg_ppda        FLOAT,
    avg_possession  FLOAT,
    avg_xg_ratio    FLOAT,
    team_count      INTEGER
);

ALTER TABLE team_style_profiles
    ADD COLUMN IF NOT EXISTS cluster_id     INTEGER REFERENCES style_clusters(cluster_id),
    ADD COLUMN IF NOT EXISTS archetype_name TEXT,
    ADD COLUMN IF NOT EXISTS umap_x         FLOAT,
    ADD COLUMN IF NOT EXISTS umap_y         FLOAT,
    ADD COLUMN IF NOT EXISTS style_vector   JSONB;
