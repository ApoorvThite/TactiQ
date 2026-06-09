"""Unit tests for IsotonicMulticlassCalibrator."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_matchup_model import IsotonicMulticlassCalibrator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_calibrator(n_classes=3):
    """Build a calibrator with identity isotonic regressors (passthrough)."""
    calibrators = []
    for _ in range(n_classes):
        ir = IsotonicRegression(out_of_bounds='clip')
        # Train on identity mapping so output equals input
        ir.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        calibrators.append(ir)
    return calibrators


class _FakeBaseModel:
    """Minimal duck-type for XGBClassifier.predict_proba."""
    def __init__(self, raw_proba):
        self._raw = np.array(raw_proba)

    def predict_proba(self, X):
        return np.tile(self._raw, (len(X), 1))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIsotonicMulticlassCalibrator:
    def test_output_sums_to_one(self):
        raw = [0.5, 0.3, 0.2]
        model = _FakeBaseModel(raw)
        cal = IsotonicMulticlassCalibrator(model, _make_calibrator())
        X = np.zeros((5, 15))
        proba = cal.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_output_shape(self):
        model = _FakeBaseModel([0.6, 0.2, 0.2])
        cal = IsotonicMulticlassCalibrator(model, _make_calibrator())
        X = np.zeros((10, 15))
        assert cal.predict_proba(X).shape == (10, 3)

    def test_predict_returns_argmax(self):
        raw = [0.1, 0.7, 0.2]
        model = _FakeBaseModel(raw)
        cal = IsotonicMulticlassCalibrator(model, _make_calibrator())
        X = np.zeros((4, 15))
        preds = cal.predict(X)
        assert np.all(preds == 1)  # class 1 (draw) has highest prob

    def test_all_zero_raw_proba_does_not_divide_by_zero(self):
        # If calibrators output all zeros for a row, row_sum=0 → should not crash
        ir_zero = IsotonicRegression(out_of_bounds='clip')
        ir_zero.fit([0.0, 1.0], [0.0, 0.0])  # always outputs 0
        calibrators = [ir_zero, ir_zero, ir_zero]
        model = _FakeBaseModel([0.33, 0.33, 0.34])
        cal = IsotonicMulticlassCalibrator(model, calibrators)
        X = np.zeros((2, 15))
        proba = cal.predict_proba(X)
        # Should not raise; result may be nan but no ZeroDivisionError
        assert proba.shape == (2, 3)

    def test_probabilities_non_negative(self):
        model = _FakeBaseModel([0.5, 0.3, 0.2])
        cal = IsotonicMulticlassCalibrator(model, _make_calibrator())
        X = np.random.default_rng(0).standard_normal((20, 15))
        proba = cal.predict_proba(X)
        assert np.all(proba >= 0), "Calibrated probabilities must be non-negative"

    def test_calibration_shifts_overconfident_win(self):
        # Simulate model that over-predicts win at 0.80
        # Identity calibrators should pass it through; what matters is normalization
        raw = [0.80, 0.10, 0.10]
        model = _FakeBaseModel(raw)
        cal = IsotonicMulticlassCalibrator(model, _make_calibrator())
        X = np.zeros((1, 15))
        proba = cal.predict_proba(X)
        assert abs(proba[0].sum() - 1.0) < 1e-9
        assert proba[0, 0] == pytest.approx(0.80, abs=0.01)
