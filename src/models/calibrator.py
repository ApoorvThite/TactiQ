"""Standalone calibrator — kept separate so pickle can load it without training deps."""

import numpy as np


class IsotonicMulticlassCalibrator:
    """Wraps an XGBClassifier with per-class isotonic regression calibration."""

    def __init__(self, base_model, calibrators):
        self.base_model  = base_model
        self.calibrators = calibrators

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        cal = np.column_stack([
            self.calibrators[i].predict(raw[:, i]) for i in range(raw.shape[1])
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.where(row_sums == 0, 1, row_sums)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
