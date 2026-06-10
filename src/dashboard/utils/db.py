"""Cached database queries for the TactiQ dashboard."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).resolve().parents[3] / '.env')


@st.cache_resource
def get_engine():
    # Streamlit Community Cloud: read from st.secrets
    # Local dev: read from .env
    try:
        cfg = st.secrets["postgres"]
        url = (
            f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['dbname']}"
        )
    except (KeyError, FileNotFoundError):
        url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME', 'tactiq')}"
        )
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=0)


@st.cache_data(ttl=300)
def load_team_profiles() -> pd.DataFrame:
    """
    All WC2026 teams with style features.
    For proxy teams (matches_played < 0), features are extracted from
    the style_vector JSONB column via COALESCE.
    """
    query = """
        SELECT
            tsp.team_id,
            tsp.team_name,
            tsp.archetype_name,
            tsp.cluster_id,
            tsp.umap_x,
            tsp.umap_y,
            tsp.matches_played,
            tsp.win_rate,
            -- Use individual columns when available; fall back to style_vector
            COALESCE(tsp.avg_possession_pct,
                (tsp.style_vector->>0)::float) AS avg_possession_pct,
            COALESCE(tsp.avg_ppda,
                (tsp.style_vector->>1)::float) AS avg_ppda,
            COALESCE(tsp.avg_pressure_success_rate,
                (tsp.style_vector->>2)::float) AS avg_pressure_success_rate,
            COALESCE(tsp.avg_xg_created_p90,
                (tsp.style_vector->>3)::float) AS avg_xg_created_p90,
            COALESCE(tsp.avg_xg_ratio,
                (tsp.style_vector->>4)::float) AS avg_xg_ratio,
            COALESCE(tsp.avg_progressive_carry_pct,
                (tsp.style_vector->>5)::float) AS avg_progressive_carry_pct,
            COALESCE(tsp.avg_pass_completion_pct,
                (tsp.style_vector->>6)::float) AS avg_pass_completion_pct,
            COALESCE(tsp.avg_passes_final_third_p90,
                (tsp.style_vector->>7)::float) AS avg_passes_final_third_p90,
            COALESCE(tsp.avg_pass_completion_under_pressure_pct,
                (tsp.style_vector->>8)::float) AS avg_pass_completion_under_pressure_pct,
            COALESCE(tsp.avg_set_piece_shot_pct,
                (tsp.style_vector->>9)::float) AS avg_set_piece_shot_pct
        FROM team_style_profiles tsp
        WHERE tsp.team_name IN (
            SELECT DISTINCT team_a_name FROM wc2026_group_predictions
            UNION
            SELECT DISTINCT team_b_name FROM wc2026_group_predictions
        )
        ORDER BY tsp.team_name
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn)

    # Derived columns
    df['is_proxy']       = df['matches_played'] < 0
    df['press_intensity'] = 1.0 / df['avg_ppda'].clip(lower=0.5)
    return df


@st.cache_data(ttl=300)
def load_group_predictions() -> pd.DataFrame:
    query = """
        SELECT
            id, group_label, team_a_name, team_b_name,
            team_a_archetype, team_b_archetype,
            team_a_is_proxy, team_b_is_proxy,
            predicted_class, p_win, p_draw, p_loss,
            top_shap_feature_win, top_shap_feature_draw,
            is_upset_candidate, upset_explanation
        FROM wc2026_group_predictions
        ORDER BY group_label, id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=300)
def load_qualification_probs() -> pd.DataFrame:
    query = """
        SELECT
            team_name, group_label, archetype_name, is_proxy,
            p_first, p_second, p_third, p_fourth,
            p_qualify_direct, p_best_third, p_qualify_r32,
            avg_sim_points, avg_sim_gd
        FROM wc2026_qualification_probs
        ORDER BY group_label, p_qualify_r32 DESC
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=300)
def load_shap_values() -> pd.DataFrame:
    query = """
        SELECT
            team_a_name, team_b_name, predicted_class,
            p_win, p_draw, p_loss,
            shap_values_win, shap_values_draw, shap_values_loss,
            top_feature_win, top_feature_draw,
            is_upset_candidate, upset_explanation
        FROM matchup_shap_values
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=600)
def load_similarity_matrix() -> pd.DataFrame | None:
    """Load the 54×54 cosine similarity matrix from Phase 3."""
    csv = Path(__file__).resolve().parents[3] / 'data' / 'processed' / 'team_similarity_matrix.csv'
    if not csv.exists():
        return None
    return pd.read_csv(csv, index_col=0)


@st.cache_data(ttl=600)
def load_upset_watchlist() -> pd.DataFrame:
    """Load the upset watchlist from wc2026_group_predictions."""
    query = """
        SELECT
            group_label, team_a_name AS favourite, team_b_name AS underdog,
            team_a_archetype AS arch_favourite, team_b_archetype AS arch_underdog,
            predicted_class, p_win, p_draw, p_loss,
            (p_draw + p_loss) AS p_not_fav_win,
            top_shap_feature_draw AS top_upset_signal,
            upset_explanation, is_upset_candidate
        FROM wc2026_group_predictions
        WHERE is_upset_candidate = TRUE
        ORDER BY (p_draw + p_loss) DESC
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)
