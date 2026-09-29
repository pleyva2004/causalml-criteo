"""Tests for calibrators, the logistic baseline, tree helpers and interpretation utilities."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import RAW_FEATURES
from src.models import calibration
from src.models.baseline import LogisticBaseline
from src.models.calibration import fit_calibrator, select_calibration_method
from src.models.evaluation import expected_calibration_error, row_log_loss
from src.models.interpret import raw_feature_of
from src.models.tree_models import fit_lgbm_early_stopping, make_rf


def _miscalibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True rate p^2 but the model reports p: over-predicts by a growing factor."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 30, n)
    return (rng.random(n) < p**2 * 3).astype(int), p


@pytest.mark.parametrize("method", calibration.CALIBRATION_METHODS)
def test_calibrator_fit_on_val_improves_test_and_uses_no_test_data(method):
    y_val, p_val = _miscalibrated(300_000, 0)
    y_te, p_te = _miscalibrated(300_000, 1)
    cal = fit_calibrator(method, p_val, y_val)
    q = cal.predict(p_te)
    assert row_log_loss(y_te, q).mean() < row_log_loss(y_te, p_te).mean()
    assert expected_calibration_error(y_te, q) < expected_calibration_error(y_te, p_te)
    assert q.min() >= cal.floor and q.max() <= 1 - cal.floor
    # contract: fitting takes validation scores and labels only; predict is a pure map of scores
    assert list(inspect.signature(fit_calibrator).parameters) == ["method", "p_val", "y_val"]
    assert np.array_equal(cal.predict(p_te), q)


def test_calibrator_is_monotone_and_method_selection_returns_known_method():
    y, p = _miscalibrated(100_000, 2)
    cal = fit_calibrator("isotonic", p, y)
    grid = np.linspace(0, 0.5, 200)
    assert np.all(np.diff(cal.predict(grid)) >= 0)
    best, scores = select_calibration_method(p, y, folds=2, seed=0)
    assert best in calibration.CALIBRATION_METHODS and set(scores) == set(calibration.CALIBRATION_METHODS)
    with pytest.raises(ValueError):
        fit_calibrator("platt2", p, y)


def _raw_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = {}
    for i in range(12):
        X[f"f{i}"] = rng.normal(size=n) if i in (0, 2, 7, 10) else rng.integers(0, 5, size=n).astype(float) * 1.7
    df = pd.DataFrame(X)
    logit = -6 + 1.5 * df["f0"]
    df["conversion"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_logistic_baseline_learns_signal_and_probabilities_are_valid():
    df = _raw_frame(60_000, 0)
    m = LogisticBaseline(C=1.0, max_iter=200).fit(df, df["conversion"].to_numpy())
    p = m.predict_proba1(df.iloc[:1000])
    assert p.dtype == np.float32 and ((p > 0) & (p < 1)).all()
    j = m.cols_.index("f0")
    assert m.model_.coef_[0, j] > 0.5


def test_random_forest_and_lgbm_early_stopping_run():
    df = _raw_frame(20_000, 1)
    X, y = df[RAW_FEATURES].iloc[:15_000], df["conversion"].to_numpy()[:15_000]
    Xv, yv = df[RAW_FEATURES].iloc[15_000:], df["conversion"].to_numpy()[15_000:]
    rf = make_rf(20, 50).fit(X, y)
    assert rf.predict_proba(Xv).shape == (5_000, 2)
    lg = fit_lgbm_early_stopping(X, y, Xv, yv, {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 50})
    assert lg.best_iteration_ is not None and 1 <= lg.best_iteration_ <= lg.n_estimators


def test_raw_feature_mapping():
    assert raw_feature_of("f3") == "f3" and raw_feature_of("f11_logfreq") == "f11" and raw_feature_of("f8_is_2") == "f8"
    with pytest.raises(ValueError):
        raw_feature_of("visit")
