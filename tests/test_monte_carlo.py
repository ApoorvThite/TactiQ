"""Unit tests for Monte Carlo probability normalization and group-stage logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.simulate_group_stage import _sample_result


# ── _sample_result ────────────────────────────────────────────────────────────

class TestSampleResult:
    def test_returns_tuple_of_correct_length(self):
        rng = np.random.default_rng(0)
        result = _sample_result(0.5, 0.25, 0.25, rng)
        assert len(result) == 4  # pts_a, pts_b, xg_a, xg_b

    def test_win_gives_three_points_to_a(self):
        # Force win by setting p_win=1
        rng = np.random.default_rng(0)
        pts_a, pts_b, _, _ = _sample_result(1.0, 0.0, 0.0, rng)
        assert pts_a == 3
        assert pts_b == 0

    def test_loss_gives_three_points_to_b(self):
        rng = np.random.default_rng(0)
        pts_a, pts_b, _, _ = _sample_result(0.0, 0.0, 1.0, rng)
        assert pts_a == 0
        assert pts_b == 3

    def test_draw_gives_one_point_each(self):
        rng = np.random.default_rng(0)
        pts_a, pts_b, xg_a, xg_b = _sample_result(0.0, 1.0, 0.0, rng)
        assert pts_a == 1
        assert pts_b == 1
        assert xg_a == xg_b  # draws use same Poisson draw for both

    def test_xg_values_are_non_negative_integers(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            _, _, xg_a, xg_b = _sample_result(0.45, 0.25, 0.30, rng)
            assert isinstance(xg_a, int) and xg_a >= 0
            assert isinstance(xg_b, int) and xg_b >= 0

    def test_normalizes_drifted_probabilities(self):
        # Simulate CSV round-trip drift: probs don't sum to exactly 1
        rng = np.random.default_rng(7)
        # These sum to 1.0001 — should not raise ValueError
        try:
            _sample_result(0.4501, 0.2500, 0.3001, rng)
        except ValueError:
            pytest.fail("_sample_result raised ValueError on drifted probabilities")

    def test_large_sample_win_frequency(self):
        # With p_win=0.6, wins should dominate over many trials
        rng = np.random.default_rng(99)
        wins = sum(
            1 for _ in range(1000)
            if _sample_result(0.6, 0.2, 0.2, rng)[0] == 3
        )
        assert wins > 450, f"Expected >450 wins, got {wins}"

    def test_total_points_per_match(self):
        # Each match distributes exactly 3 points (win/loss) or 2 points (draw)
        rng = np.random.default_rng(5)
        for _ in range(200):
            pts_a, pts_b, _, _ = _sample_result(0.4, 0.3, 0.3, rng)
            assert pts_a + pts_b in (2, 3)
