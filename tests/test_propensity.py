"""Tests for src.causal.propensity (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.causal.propensity import (
    clip_propensity,
    crossfit_predict,
    fit_lgbm,
    kish_ess,
    lgbm_propensity,
    logistic_propensity,
    make_folds,
    propensity_diagnostics,
)

SMALL_LGBM = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 100}


def test_make_folds_is_balanced_stratified_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 4, 10_001)
    f = make_folds(strata, 5, seed=1)
    assert np.array_equal(f, make_folds(strata, 5, seed=1))
    assert not np.array_equal(f, make_folds(strata, 5, seed=2))
    sizes = np.bincount(f)
    assert sizes.max() - sizes.min() <= 1
    for s in range(4):
        per = np.bincount(f[strata == s], minlength=5)
        assert per.max() - per.min() <= 1
    with pytest.raises(ValueError):
        make_folds(strata, 1, seed=0)


def test_crossfit_predict_never_predicts_a_row_with_its_own_model() -> None:
    folds = make_folds(np.zeros(1000, dtype=int), 4, seed=0)
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def fit_predict(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        seen.append((tr, te))
        return np.full(len(te), float(len(seen)))

    out = crossfit_predict(fit_predict, folds)
    assert len(seen) == 4
    for tr, te in seen:
        assert np.intersect1d(tr, te).size == 0 and len(tr) + len(te) == 1000
    assert set(np.unique(out)) == {1.0, 2.0, 3.0, 4.0}


def test_lgbm_propensity_is_flat_under_randomization_and_early_stopping_keeps_it_small() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30_000, 5)).astype(np.float32)
    t = rng.random(30_000) < 0.85
    folds = make_folds(t.astype(int), 3, seed=0)
    info: dict = {}
    e = lgbm_propensity(X, t, folds, seed=0, overrides=SMALL_LGBM, early_stopping_rounds=20, info=info)
    e_overfit = lgbm_propensity(X, t, folds, seed=0, overrides=SMALL_LGBM)
    assert max(info["trees"]) < 100                           # nothing to learn: stops early
    assert abs(e.mean() - 0.85) < 0.01
    assert e.std() < 0.5 * e_overfit.std()                    # the fixed tree budget fits noise


def test_lgbm_and_logistic_propensity_recover_a_known_propensity() -> None:
    rng = np.random.default_rng(1)
    n = 40_000
    X = rng.normal(size=(n, 3))
    e_true = 1 / (1 + np.exp(-(1.2 + 0.8 * X[:, 0] - 0.5 * X[:, 1])))
    t = rng.random(n) < e_true
    folds = make_folds(t.astype(int), 3, seed=0)
    e_lgbm = lgbm_propensity(X.astype(np.float32), t, folds, 0, SMALL_LGBM, early_stopping_rounds=20)
    e_lr = logistic_propensity(lambda idx: X[idx], t, folds, seed=0, fit_rows=20_000, predict_chunk=5000)
    assert np.corrcoef(e_lr, e_true)[0, 1] > 0.99
    assert np.corrcoef(e_lgbm, e_true)[0, 1] > 0.9
    assert np.abs(e_lr - e_true).mean() < 0.02


def test_fit_lgbm_without_early_stopping_uses_the_full_budget() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 0.5).astype(int)
    m = fit_lgbm("classifier", X, y, seed=0, overrides={"n_estimators": 25})
    assert m.n_estimators_ == 25


def test_clip_and_kish_ess() -> None:
    assert clip_propensity(np.array([0.0, 0.5, 1.0]), (0.01, 0.99)).tolist() == [0.01, 0.5, 0.99]
    with pytest.raises(ValueError):
        clip_propensity(np.array([0.5]), (0.6, 0.4))
    assert kish_ess(np.ones(100)) == pytest.approx(100)
    assert kish_ess(np.r_[1000.0, np.ones(99)]) < 2


def test_propensity_diagnostics_under_a_constant_propensity() -> None:
    rng = np.random.default_rng(3)
    t = rng.random(10_000) < 0.85
    d = propensity_diagnostics(np.full(10_000, 0.85), t, (0.01, 0.99), design_p=0.85)
    assert d["by_arm"]["control"]["ipw_ess_ratio"] == pytest.approx(1.0)
    assert d["auc_treatment_vs_control"] == pytest.approx(0.5)
    assert d["mean_abs_deviation_from_design"] == 0.0
    assert d["share_below_clip"] == 0.0 and d["share_above_clip"] == 0.0
