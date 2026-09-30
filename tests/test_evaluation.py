"""Tests for metrics, thresholds and the Poisson bootstrap (synthetic data, no real dataset)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.models.evaluation import (
    _RankState,
    bootstrap_ci,
    bootstrap_rank_metrics,
    classification_metrics,
    expected_calibration_error,
    paired_diff,
    reliability_table,
    select_thresholds,
    threshold_metrics,
    wilson_interval,
)


def _synthetic(n: int = 60_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(0.5, 60, n), 1e-6, 1 - 1e-6)
    return (rng.random(n) < p).astype(int), p


def test_metrics_match_sklearn():
    y, p = _synthetic()
    m = classification_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["log_loss"] == pytest.approx(log_loss(y, p), rel=1e-9)
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["base_rate"] == pytest.approx(y.mean())
    assert m["pr_auc_lift_over_base"] == pytest.approx(m["pr_auc"] / y.mean())


def test_ece_of_calibrated_predictor_is_near_zero_and_miscalibration_is_detected():
    y, p = _synthetic(n=400_000, seed=1)
    assert expected_calibration_error(y, p) < 5e-4
    assert expected_calibration_error(y, np.clip(2 * p, 0, 1)) > 5 * expected_calibration_error(y, p)


def test_reliability_table_partitions_rows_and_wilson_bounds():
    y, p = _synthetic(n=20_000, seed=2)
    t = reliability_table(y, p, 10)
    assert t["n"].sum() == len(y) and t["positives"].sum() == y.sum()
    assert (t["obs_lo"] <= t["obs_rate"]).all() and (t["obs_rate"] <= t["obs_hi"]).all()
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.06


def test_threshold_selection_and_metrics():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(50), np.zeros(950)].astype(int)
    p = np.r_[rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.7, 950)]
    thr = select_thresholds(y, p, top_frac=0.05)
    assert np.quantile(p, 0.95) == pytest.approx(thr["top_k"])
    assert (p >= thr["top_k"]).mean() == pytest.approx(0.05, abs=0.002)
    # the F1-optimal threshold must beat (or tie) any other grid threshold on the same data
    best = threshold_metrics(y, p, thr["max_f1"])["f1"]
    assert best >= max(threshold_metrics(y, p, t)["f1"] for t in np.linspace(0.05, 0.95, 19)) - 1e-12
    tm = threshold_metrics(y, p, 0.5)
    assert tm["tp"] + tm["fn"] == 50 and tm["fp"] + tm["tn"] == 950
    assert tm["precision"] == pytest.approx(tm["tp"] / (tm["tp"] + tm["fp"]))


def test_weighted_rank_state_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(4)
    p = np.round(rng.random(5000), 2)                      # many ties
    y = (rng.random(5000) < 0.1 + 0.3 * p).astype(int)
    w = rng.poisson(1.0, 5000).astype(float)
    auc, ap = _RankState(y, p).auc_ap(w)
    assert auc == pytest.approx(roc_auc_score(y, p, sample_weight=w))
    assert ap == pytest.approx(average_precision_score(y, p, sample_weight=w))


def test_bootstrap_ci_contains_point_and_shrinks_with_n():
    y, p = _synthetic(n=30_000, seed=5)
    point, lo, hi = bootstrap_ci("roc_auc", y, p, reps=100, seed=0)
    assert lo <= point <= hi
    point2, lo2, hi2 = bootstrap_ci("pr_auc", y, p, reps=100, seed=0)
    assert lo2 <= point2 <= hi2
    y2, p2 = _synthetic(n=120_000, seed=5)
    _, lo3, hi3 = bootstrap_ci("roc_auc", y2, p2, reps=100, seed=0)
    assert (hi3 - lo3) < (hi - lo)
    with pytest.raises(ValueError):
        bootstrap_ci("accuracy", y, p)


def test_paired_bootstrap_detects_a_better_model():
    y, p = _synthetic(n=40_000, seed=6)
    rng = np.random.default_rng(7)
    noisy = np.clip(p * np.exp(rng.normal(0, 1.5, len(p))), 1e-6, 1 - 1e-6)
    d = bootstrap_rank_metrics(y, {"good": p, "bad": noisy}, reps=60, seed=0)
    r = paired_diff(d["good"]["roc_auc"], d["bad"]["roc_auc"], roc_auc_score(y, p), roc_auc_score(y, noisy))
    assert r["diff"] > 0 and r["lo"] > 0 and r["p_value_two_sided"] < 0.05
