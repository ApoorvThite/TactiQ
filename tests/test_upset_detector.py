"""Unit tests for the upset detector signal logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.upset_detector import is_upset_candidate, _upset_explanation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pred(p_win=0.55, p_draw=0.25, p_loss=0.20,
          archetype_a='High Press', archetype_b='Deep Block',
          team_a='Spain', team_b='Jordan'):
    return {
        'p_win': p_win, 'p_draw': p_draw, 'p_loss': p_loss,
        'archetype_a': archetype_a, 'archetype_b': archetype_b,
        'team_a_name': team_a, 'team_b_name': team_b,
    }


def _shap_neutral():
    # archetype_matchup_id uses >= 0 as signal threshold, so use -0.1 for "no signal"
    return {
        'delta_avg_ppda': 0.0,
        'delta_avg_set_piece_shot_pct': 0.0,
        'archetype_matchup_id': -0.1,
        'delta_avg_possession_pct': 0.0,
        'delta_avg_pressure_success_rate': 0.0,
    }


def _shap_with(key, value):
    s = _shap_neutral()
    s[key] = value
    return s


# ── Gate conditions ────────────────────────────────────────────────────────────

class TestUpsetCandidateGates:
    def test_no_upset_when_p_win_too_low(self):
        pred = _pred(p_win=0.35, p_draw=0.35, p_loss=0.30)
        result, signals = is_upset_candidate(
            pred, _shap_with('delta_avg_ppda', 0.10), _shap_neutral(), _shap_neutral()
        )
        assert result is False
        assert signals == []

    def test_no_upset_when_underdog_is_high_press(self):
        pred = _pred(p_win=0.55, archetype_b='High Press')
        result, signals = is_upset_candidate(
            pred, _shap_with('delta_avg_ppda', 0.10), _shap_neutral(), _shap_neutral()
        )
        assert result is False

    def test_no_upset_when_favourite_too_dominant(self):
        # p_draw + p_loss = 0.40 < 0.45
        pred = _pred(p_win=0.60, p_draw=0.25, p_loss=0.15)
        result, signals = is_upset_candidate(
            pred, _shap_with('delta_avg_ppda', 0.10), _shap_neutral(), _shap_neutral()
        )
        assert result is False

    def test_no_upset_when_no_shap_signals_triggered(self):
        pred = _pred(p_win=0.55, p_draw=0.25, p_loss=0.20)
        result, signals = is_upset_candidate(
            pred, _shap_neutral(), _shap_neutral(), _shap_neutral()
        )
        assert result is False
        assert signals == []


# ── Signal detection ──────────────────────────────────────────────────────────

class TestUpsetSignals:
    def test_ppda_neutralised_triggers_from_draw(self):
        pred = _pred()
        result, signals = is_upset_candidate(
            pred,
            _shap_neutral(),
            _shap_with('delta_avg_ppda', 0.05),  # draw SHAP > 0.04
            _shap_neutral(),
        )
        assert result is True
        assert 'ppda_neutralised' in signals

    def test_ppda_neutralised_triggers_from_loss(self):
        pred = _pred()
        result, signals = is_upset_candidate(
            pred,
            _shap_neutral(),
            _shap_neutral(),
            _shap_with('delta_avg_ppda', 0.05),  # loss SHAP > 0.04
        )
        assert result is True
        assert 'ppda_neutralised' in signals

    def test_set_piece_threat_triggers(self):
        pred = _pred()
        result, signals = is_upset_candidate(
            pred,
            _shap_neutral(),
            _shap_with('delta_avg_set_piece_shot_pct', 0.04),  # > 0.03
            _shap_neutral(),
        )
        assert result is True
        assert 'set_piece_threat' in signals

    def test_archetype_disadvantage_triggers(self):
        # archetype_matchup_id >= 0 is always true for non-negative SHAP
        pred = _pred()
        result, signals = is_upset_candidate(
            pred,
            _shap_neutral(),
            _shap_with('archetype_matchup_id', 0.01),
            _shap_neutral(),
        )
        assert result is True
        assert 'archetype_disadvantage' in signals

    def test_multiple_signals_returned(self):
        pred = _pred()
        shap_draw = _shap_neutral()
        shap_draw['delta_avg_ppda'] = 0.06
        shap_draw['delta_avg_set_piece_shot_pct'] = 0.05
        shap_draw['archetype_matchup_id'] = 0.02
        result, signals = is_upset_candidate(
            pred, _shap_neutral(), shap_draw, _shap_neutral()
        )
        assert result is True
        assert len(signals) == 3

    def test_boundary_ppda_exactly_at_threshold_not_triggered(self):
        pred = _pred()
        result, _ = is_upset_candidate(
            pred,
            _shap_neutral(),
            _shap_with('delta_avg_ppda', 0.04),  # exactly 0.04, not > 0.04
            _shap_neutral(),
        )
        assert result is False


# ── Explanation text ──────────────────────────────────────────────────────────

class TestUpsetExplanation:
    def test_explanation_is_non_empty_string(self):
        pred = _pred()
        text = _upset_explanation(pred, ['ppda_neutralised'])
        assert isinstance(text, str)
        assert len(text) > 10

    def test_explanation_mentions_team_names(self):
        pred = _pred(team_a='Germany', team_b='Algeria')
        text = _upset_explanation(pred, ['ppda_neutralised'])
        assert 'Germany' in text or 'Algeria' in text

    def test_explanation_for_all_signals(self):
        pred = _pred()
        text = _upset_explanation(pred, ['ppda_neutralised', 'set_piece_threat', 'archetype_disadvantage'])
        assert len(text) > 50
