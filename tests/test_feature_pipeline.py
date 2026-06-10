"""Unit tests for the feature vector construction pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.upset_detector import (
    FEATURE_NAMES,
    _build_feature_vector,
)

STYLE_DIM = 10  # number of style features per team

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_team(archetype='High Press', matches_played=20, is_proxy=False):
    rng = np.random.default_rng(42)
    return {
        'team_name':      'Test FC',
        'archetype_name': archetype,
        'style_vector':   rng.uniform(0.0, 1.0, STYLE_DIM).tolist(),
        'matches_played': matches_played,
        'is_proxy':       is_proxy,
    }


# ── Feature vector shape and content ──────────────────────────────────────────

class TestBuildFeatureVector:
    def test_output_length_equals_feature_names(self):
        team_a = _make_team('High Press')
        team_b = _make_team('Counter-Attack')
        vec = _build_feature_vector(team_a, team_b)
        assert len(vec) == len(FEATURE_NAMES)

    def test_delta_is_a_minus_b(self):
        sv_a = np.ones(STYLE_DIM) * 0.8
        sv_b = np.ones(STYLE_DIM) * 0.3
        team_a = {**_make_team(), 'style_vector': sv_a.tolist()}
        team_b = {**_make_team(), 'style_vector': sv_b.tolist()}
        vec = _build_feature_vector(team_a, team_b)
        expected_delta = sv_a - sv_b
        np.testing.assert_allclose(vec[:STYLE_DIM], expected_delta, atol=1e-9)

    def test_is_home_flag(self):
        team_a = _make_team()
        team_b = _make_team()
        vec_home = _build_feature_vector(team_a, team_b, is_home=True)
        vec_away = _build_feature_vector(team_a, team_b, is_home=False)
        home_idx = FEATURE_NAMES.index('is_home')
        assert vec_home[home_idx] == 1.0
        assert vec_away[home_idx] == 0.0

    def test_archetype_matchup_id_range(self):
        archetypes = ['High Press', 'Possession Control', 'Counter-Attack', 'Deep Block']
        idx = FEATURE_NAMES.index('archetype_matchup_id')
        for arch_a in archetypes:
            for arch_b in archetypes:
                vec = _build_feature_vector(_make_team(arch_a), _make_team(arch_b))
                assert 0 <= vec[idx] <= 15, f"matchup_id out of range for {arch_a} vs {arch_b}"

    def test_competition_weight_world_cup(self):
        vec = _build_feature_vector(_make_team(), _make_team(), competition='FIFA World Cup')
        cw_idx = FEATURE_NAMES.index('competition_weight')
        assert vec[cw_idx] == 1.0

    def test_delta_matches_played(self):
        team_a = {**_make_team(), 'matches_played': 30}
        team_b = {**_make_team(), 'matches_played': 10}
        vec = _build_feature_vector(team_a, team_b)
        dmp_idx = FEATURE_NAMES.index('delta_matches_played')
        assert vec[dmp_idx] == pytest.approx(20.0)

    def test_no_nan_or_inf(self):
        team_a = _make_team('Possession Control', matches_played=15)
        team_b = _make_team('Deep Block', matches_played=8, is_proxy=True)
        vec = _build_feature_vector(team_a, team_b)
        assert np.all(np.isfinite(vec)), "Feature vector contains NaN or Inf"
