"""Tests for the CATE learners on a synthetic randomized experiment with a known tau(x)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.causal import learners as L
from src.causal.causal_forest import fit_causal_forest, forest_ate, predict_cate
from src.models.tree_models import make_lgbm

P_TREAT = 0.85


def _data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """tau(x) = 0.1 * 1[x0 > 0]; the baseline depends on x1 (prognostic, not effect-modifying)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)).astype(np.float32), columns=["x0", "x1", "x2", "x3"])
    t = (rng.random(n) < P_TREAT).astype(np.int64)
    tau = 0.1 * (X["x0"].to_numpy() > 0)
    mu0 = 0.1 + 0.1 * (X["x1"].to_numpy() > 0)
    y = (rng.random(n) < mu0 + tau * t).astype(np.int64)
    return X, t, y, tau


def _outcome() -> object:
    return make_lgbm("classifier", n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=100,
                     n_jobs=2)


def _effect() -> object:
    return make_lgbm("regressor", n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=200,
                     n_jobs=2)


@pytest.fixture(scope="module")
def fitted() -> dict[str, np.ndarray]:
    X, t, y, _ = _data(40_000, seed=0)
    Xv, tv, yv, _ = _data(10_000, seed=1)
    Xte, _, _, tau_te = _data(10_000, seed=2)
    e = np.full(len(t), t.mean())
    val = L.ValSet(Xv, tv, yv, np.full(len(tv), t.mean()))
    tl = L.TLearner(_outcome, early_stopping_rounds=20)
    learners = {
        "s_learner": L.SLearner(_outcome, early_stopping_rounds=20),
        "t_learner": tl,
        "x_learner": L.XLearner(_outcome, _effect, 20, t_learner=tl),
        "dr_learner": L.DRLearner(_outcome, _effect, 20, n_folds=2, seed=0),
        "class_transformation": L.TransformedOutcomeLearner(_outcome, _effect, 20),
    }
    preds = {}
    for name, learner in learners.items():
        learner.fit(X, t, y, e, val)
        preds[name] = learner.predict(Xte, np.full(len(Xte), t.mean()))
    return {"preds": preds, "tau": tau_te}


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "dr_learner"])
def test_learners_recover_effect_ordering_and_size(fitted: dict, name: str) -> None:
    tau_hat, tau = fitted["preds"][name], fitted["tau"]
    assert roc_auc_score(tau > 0, tau_hat) > 0.8                          # ranks x0 > 0 above x0 <= 0
    assert stats.spearmanr(tau_hat, tau).statistic > 0.5
    gap = tau_hat[tau > 0].mean() - tau_hat[tau == 0].mean()
    assert gap == pytest.approx(0.1, abs=0.04)                            # recovers the effect size


@pytest.mark.parametrize("name", ["s_learner", "class_transformation"])
def test_weaker_learners_still_get_the_sign(fitted: dict, name: str) -> None:
    tau_hat, tau = fitted["preds"][name], fitted["tau"]
    assert tau_hat[tau > 0].mean() > tau_hat[tau == 0].mean()
    assert roc_auc_score(tau > 0, tau_hat) > 0.6


def test_pseudo_outcomes_are_unbiased_for_the_ate() -> None:
    X, t, y, tau = _data(200_000, seed=3)
    e = np.full(len(t), P_TREAT)
    mu0 = 0.1 + 0.1 * (X["x1"].to_numpy() > 0)
    true_ate = tau.mean()
    # transformed outcome: E[Z | X] = tau(X)
    z = L.TransformedOutcomeLearner.transform(t, y, e)
    assert z.mean() == pytest.approx(true_ate, abs=4 * z.std() / np.sqrt(len(z)))
    # AIPW pseudo-outcome with a *wrong* outcome model is still unbiased when e is correct
    phi = L.DRLearner(_outcome).pseudo_outcome(t, y, e, np.zeros(len(t)), np.zeros(len(t)))
    assert phi.mean() == pytest.approx(true_ate, abs=4 * phi.std() / np.sqrt(len(phi)))
    phi_good = L.DRLearner(_outcome).pseudo_outcome(t, y, e, mu0, mu0 + tau)
    assert phi_good.std() < phi.std()                                    # good nuisances reduce variance


def test_x_learner_weights_follow_the_propensity() -> None:
    X, t, y, _ = _data(20_000, seed=4)
    xl = L.XLearner(_outcome, _effect).fit(X, t, y)
    tau0, tau1 = xl.tau0_.predict(X[:50]), xl.tau1_.predict(X[:50])
    np.testing.assert_allclose(xl.predict(X[:50], np.ones(50)), tau0, rtol=1e-6)   # g = 1 -> tau_0 only
    np.testing.assert_allclose(xl.predict(X[:50], np.zeros(50)), tau1, rtol=1e-6)  # g = 0 -> tau_1 only
    np.testing.assert_allclose(xl.predict(X[:50]), t.mean() * tau0 + (1 - t.mean()) * tau1, rtol=1e-6)


def test_selection_mask_is_deterministic_order_free_and_balanced() -> None:
    ids = np.arange(100_000, dtype=np.int64) * 7 + 3
    m = L.selection_mask(ids, seed=42, share=0.5)
    assert abs(m.mean() - 0.5) < 0.01
    perm = np.random.default_rng(0).permutation(len(ids))
    np.testing.assert_array_equal(L.selection_mask(ids[perm], 42, 0.5), m[perm])
    assert not np.array_equal(L.selection_mask(ids, 43, 0.5), m)


def test_importance_aggregation_and_segments() -> None:
    agg = L.aggregate_to_raw({"f3": 1.0, "f3_logfreq": 1.0, "f0": 2.0})
    assert agg["f3"] == pytest.approx(0.5) and agg["f0"] == pytest.approx(0.5) and agg.sum() == pytest.approx(1)
    vals = pd.Series(np.r_[np.zeros(600), np.ones(300), np.arange(100) + 2.0])
    lab = L.segment_labels(vals, "f1", n_bins=3)                           # categorical: 2 top levels + other
    assert set(lab.unique()) == {"#1 (60%)", "#2 (30%)", "other"}
    cont = L.segment_labels(pd.Series(np.arange(1000.0)), "f0", n_bins=5)  # continuous: quantile bins
    assert cont.nunique() == 5


def test_causal_forest_recovers_heterogeneity() -> None:
    X, t, y, _ = _data(30_000, seed=5)
    Xte, _, _, tau_te = _data(5_000, seed=6)
    small = {"n_estimators": 40, "min_samples_leaf": 100, "cv": 2,
             "nuisance_model": {"n_estimators": 50, "num_leaves": 7, "min_child_samples": 100}}
    cf = fit_causal_forest(X, t, y, seed=0, params=small)
    tau_hat = predict_cate(cf, Xte, chunk_rows=2_000)
    assert roc_auc_score(tau_te > 0, tau_hat) > 0.8
    ate = forest_ate(cf)
    assert ate["ci_low"] - 0.02 < 0.05 < ate["ci_high"] + 0.02            # true ATE = 0.1 * P(x0 > 0) = 0.05
