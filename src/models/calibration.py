"""Probability calibration: Platt (sigmoid) scaling and isotonic regression.

Why calibration matters here: targeting decisions use expected value = p x value - cost. A model
that ranks users perfectly but over-predicts probabilities by 2x doubles the perceived value of
every user, so thresholds, budgets and ROI estimates are all wrong even though AUC is unchanged.
Class-weighted training is the standard example: it barely changes the ranking but inflates the
probabilities by orders of magnitude.

Leakage contract: calibrators are fit on the VALIDATION split only (``fit_calibrator(method, p_val,
y_val)`` takes no other data) and are then applied, frozen, to test scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils import load_config

CALIBRATION_METHODS: tuple[str, ...] = ("sigmoid", "isotonic")
_LOGIT_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Fitted monotone map from raw scores to calibrated probabilities."""

    def __init__(self, method: str, floor: float):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
        self.method = method
        self.floor = floor

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> Calibrator:
        y = np.asarray(y_val)
        if self.method == "sigmoid":
            # Platt scaling: logistic regression on the logit of the score (2 parameters), ~no penalty.
            self.model_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            self.model_.fit(_logit(p_val).reshape(-1, 1), y)
        else:
            self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model_.fit(np.asarray(p_val, dtype=np.float64), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Calibrated probabilities as float32, clipped to [floor, 1 - floor] (isotonic can output 0)."""
        if self.method == "sigmoid":
            out = self.model_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        else:
            out = self.model_.predict(np.asarray(p, dtype=np.float64))
        return np.clip(out, self.floor, 1 - self.floor).astype(np.float32)


def fit_calibrator(method: str, p_val: np.ndarray, y_val: np.ndarray) -> Calibrator:
    """Fit a calibrator on validation scores/labels. It never sees test data by construction."""
    return Calibrator(method, load_config()["predictive"]["calibration_prob_floor"]).fit(p_val, y_val)


def select_calibration_method(p_val: np.ndarray, y_val: np.ndarray, methods: Sequence[str] = CALIBRATION_METHODS,
                              folds: int = 2, seed: int = 0) -> tuple[str, dict[str, float]]:
    """Pick the method with the lowest cross-fitted log loss *within validation*.

    Fitting and scoring a calibrator on the same rows favours the more flexible method (isotonic),
    so each method is fit on ``folds - 1`` parts of validation and scored on the held-out part.
    Returns the best method and the mean held-out log loss per method.
    """
    p_val, y_val = np.asarray(p_val), np.asarray(y_val)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(y_val))
    scores: dict[str, float] = {}
    for m in methods:
        losses = []
        for k in range(folds):
            tr, te = fold != k, fold == k
            q = fit_calibrator(m, p_val[tr], y_val[tr]).predict(p_val[te]).astype(np.float64)
            y = y_val[te]
            losses.append(float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))))
        scores[m] = float(np.mean(losses))
    return min(scores, key=scores.get), scores
