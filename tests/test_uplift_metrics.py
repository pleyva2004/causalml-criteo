"""Tests for the uplift evaluation metrics (synthetic data with known answers; sklift as reference)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.metrics import auc

from src.causal import uplift_metrics as um

warnings.filterwarnings("ignore", category=FutureWarning)   # sklift uses a deprecated sklearn helper
sk = pytest.importorskip("sklift.metrics")


def _rct(n: int = 20_000, seed: int = 0, p: float = 0.85) -> dict[str, np.ndarray]:
    """Randomized experiment with a known heterogeneous effect tau(x) = 0.1 * 1[x0 > 0]."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    t = (rng.random(n) < p).astype(np.int64)
    tau = 0.1 * (x0 > 0)
    y = (rng.random(n) < 0.1 + tau * t).astype(np.int64)
    return {"x0": x0, "t": t, "y": y, "tau": tau, "rng": rng}


def test_curves_match_sklift_without_ties() -> None:
    d = _rct()
    s = d["x0"] + d["rng"].normal(scale=0.5, size=len(d["x0"]))          # continuous => no ties
    xq, yq = sk.qini_curve(d["y"], s, d["t"])
    xm, ym = um.qini_curve(d["y"], d["t"], s)
    np.testing.assert_allclose(ym, np.interp(xm, xq, yq), atol=1e-9)
    xu, yu = sk.uplift_curve(d["y"], s, d["t"])
    xm, ym = um.uplift_curve(d["y"], d["t"], s)
    np.testing.assert_allclose(ym, np.interp(xm, xu, yu), atol=1e-9)
    for k in (0.05, 0.1, 0.3):
        assert um.uplift_at_k(d["y"], d["t"], s, k) == pytest.approx(
            sk.uplift_at_k(d["y"], s, d["t"], strategy="overall", k=k), abs=1e-12)


def test_areas_match_sklift_and_normaliser_difference_is_documented() -> None:
    d = _rct(seed=1)
    y, t = d["y"], d["t"]
    s = d["x0"] + d["rng"].normal(scale=0.5, size=len(y))
    m = um.uplift_metrics(y, t, s, grid_points=None)
    n, n_t = len(y), t.sum()
    # unnormalised areas: identical to sklearn.auc on sklift's curves (after the documented scaling)
    xq, yq = sk.qini_curve(y, s, t)
    area_sk = auc(xq, yq) - auc([0, n], [0, yq[-1]])
    assert m["qini_coefficient"] * n * n_t == pytest.approx(area_sk, rel=1e-9)
    xu, yu = sk.uplift_curve(y, s, t)
    assert m["auuc"] * n * n == pytest.approx(auc(xu, yu), rel=1e-9)
    # normalised Qini: same numerator, normaliser differs only by sklift's chord across the perfect
    # curve's final tie block (all control responders) -> tiny relative difference
    assert m["qini_normalized"] == pytest.approx(sk.qini_auc_score(y, s, t), rel=2e-3)
    # the perfect curves agree exactly wherever sklift evaluates them (tie-block ends)
    xp, yp = sk.perfect_qini_curve(y, t)
    _, ym = um.qini_curve(y, t, y * (2 * t - 1))
    np.testing.assert_allclose(ym[xp.astype(int)], yp, atol=1e-9)
    xp, yp = sk.perfect_uplift_curve(y, t)
    ev = um.UpliftEvaluator(y, t, grid_points=None)
    ev.add_perfect()
    up = um.curves_from_counts(ev.counts(um.PERFECT_UPLIFT))["uplift"]
    np.testing.assert_allclose(up[xp.astype(int)], yp, atol=1e-9)


def test_curve_endpoints_equal_overall_incremental_outcomes() -> None:
    d = _rct(seed=2)
    y, t = d["y"], d["t"]
    s = d["rng"].random(len(y))
    dim = y[t == 1].mean() - y[t == 0].mean()
    x, u = um.uplift_curve(y, t, s, grid_points=100)
    assert u[0] == 0.0 and x[-1] == len(y)
    assert u[-1] == pytest.approx(dim * len(y), rel=1e-12)
    _, q = um.qini_curve(y, t, s, grid_points=100)
    assert q[-1] == pytest.approx(y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1 - t).sum(), rel=1e-12)
    assert um.uplift_metrics(y, t, s, grid_points=100)["ate"] == pytest.approx(dim, rel=1e-12)


def test_oracle_beats_random_and_random_is_near_zero() -> None:
    d = _rct(n=40_000, seed=3)
    y, t = d["y"], d["t"]
    oracle = um.uplift_metrics(y, t, d["tau"] + 1e-6 * d["x0"], grid_points=1000)["qini_coefficient"]
    randoms = [um.uplift_metrics(y, t, np.random.default_rng(s).random(len(y)), grid_points=1000)["qini_coefficient"]
               for s in range(10)]
    # tau = 0.1 on half the population: q(f) = 0.1 f up to f = 0.5, then 0.05 -> area over random = 0.0125
    assert oracle == pytest.approx(0.0125, abs=0.004)
    assert abs(np.mean(randoms)) < 0.1 * oracle
    assert max(randoms) < oracle


def test_ties_are_resolved_in_expectation_and_order_invariant() -> None:
    d = _rct(seed=4)
    y, t = d["y"], d["t"]
    # a constant score cannot rank anyone: the Qini curve is exactly the random line
    m = um.uplift_metrics(y, t, np.zeros(len(y)), grid_points=None)
    assert m["qini_coefficient"] == pytest.approx(0.0, abs=1e-15)
    # heavily tied score: metrics do not depend on the row order
    s = np.round(d["x0"])
    perm = d["rng"].permutation(len(y))
    a = um.uplift_metrics(y, t, s, budgets=[0.1], grid_points=None)
    b = um.uplift_metrics(y[perm], t[perm], s[perm], budgets=[0.1], grid_points=None)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-15)


def test_tie_interpolation_equals_mean_over_random_tie_breaks() -> None:
    rng = np.random.default_rng(5)
    n = 60
    y = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    s = rng.integers(0, 4, n).astype(float)                # 4 big tie blocks
    ev = um.UpliftEvaluator(y, t, grid_points=None)
    ev.add("s", s)
    got = ev.counts("s")
    mc = np.zeros_like(got)
    reps = 4000
    for _ in range(reps):
        order = np.lexsort((rng.random(n), -s))
        key = 2 * t[order] + y[order]
        onehot = np.eye(4)[key]
        mc += np.vstack([np.zeros(4), np.cumsum(onehot, axis=0)])
    np.testing.assert_allclose(got, mc / reps, atol=0.25)   # MC error ~ 0.5/sqrt(4000) per unit


def test_bootstrap_weights_and_paired_differences() -> None:
    d = _rct(n=5_000, seed=6)
    y, t = d["y"], d["t"]
    s = d["x0"] + d["rng"].normal(size=len(y))
    ev = um.UpliftEvaluator(y, t, grid_points=100)
    ev.add("s", s)
    np.testing.assert_allclose(ev.counts("s", np.ones(len(y))), ev.counts("s"))
    # weighted counts equal a naive weighted recount (weights apply to the original rows)
    w = d["rng"].integers(0, 3, len(y)).astype(float)
    order = np.argsort(-s, kind="stable")
    k = 37 * len(y) // 100
    top = order[:k]
    naive = [np.sum(w[top] * ((t[top] == tt) & (y[top] == yy))) for tt in (0, 1) for yy in (0, 1)]
    np.testing.assert_allclose(ev.counts("s", w)[37], naive)
    out = um.evaluate_scorings(y, t, {"a": s, "b": s.copy()}, budgets=[0.1], n_boot=20, seed=0,
                               grid_points=100, pairs=[("a", "b")])
    diff = out["differences"]
    assert np.allclose(diff[["estimate", "ci_low", "ci_high"]].to_numpy(), 0.0)


def test_constant_propensity_weights_leave_results_unchanged() -> None:
    d = _rct(n=5_000, seed=7)
    y, t = d["y"], d["t"]
    s = d["x0"]
    a = um.evaluate_scorings(y, t, {"s": s}, budgets=[0.1], n_boot=0, seed=0, grid_points=100)["metrics"]
    # e = observed treated share -> stabilised weights are exactly 1 -> identical results
    w1 = um.stabilized_ipw_weights(t, np.full(len(y), t.mean()))
    np.testing.assert_allclose(w1, 1.0)
    b = um.evaluate_scorings(y, t, {"s": s}, budgets=[0.1], n_boot=0, seed=0, grid_points=100,
                             sample_weight=w1)["metrics"]
    np.testing.assert_allclose(a["estimate"].to_numpy(), b["estimate"].to_numpy(), rtol=1e-12)
    # any constant e -> constant weight per arm -> Hajek arm means (and hence lift/Qini) are unchanged
    w2 = um.stabilized_ipw_weights(t, np.full(len(y), 0.7))
    c = um.evaluate_scorings(y, t, {"s": s}, budgets=[0.1], n_boot=0, seed=0, grid_points=100,
                             sample_weight=w2)["metrics"]
    for k in ("ate", "qini_coefficient", "uplift_at_10", "anyway_share_at_10"):
        assert c.set_index("metric").at[k, "estimate"] == pytest.approx(a.set_index("metric").at[k, "estimate"],
                                                                      rel=1e-10)
    plain = um.diff_in_means(y, t)
    hajek = um.diff_in_means(y, t, weights=np.ones(len(y)))
    assert hajek["estimate"] == pytest.approx(plain["estimate"])
    assert hajek["se"] == pytest.approx(plain["se"], rel=1e-3)


def test_blp_recovers_calibration_slope_and_matches_statsmodels() -> None:
    d = _rct(n=60_000, seed=8, p=0.5)
    y, t, tau = d["y"], d["t"], d["tau"]
    baseline = np.full(len(y), 0.1) + 0.01 * d["rng"].normal(size=len(y))
    res = um.blp_heterogeneity_test(y, t, tau, baseline, 0.5)
    assert res["beta2"] == pytest.approx(1.0, abs=4 * res["beta2_se"])
    assert res["p_value_one_sided"] < 1e-6
    rnd = um.blp_heterogeneity_test(y, t, d["rng"].random(len(y)), baseline, 0.5)
    assert abs(rnd["beta2_z"]) < 4
    # same coefficients and HC1 SEs as statsmodels OLS (constant propensity -> constant weights)
    tt = t - 0.5
    X = np.column_stack([np.ones(len(y)), baseline - baseline.mean(), tt, tt * (tau - tau.mean())])
    fit = sm.OLS(y.astype(float), X).fit(cov_type="HC1")
    assert res["beta2"] == pytest.approx(fit.params[3], rel=1e-8)
    assert res["beta2_se"] == pytest.approx(fit.bse[3], rel=1e-8)


def test_uplift_by_group_ranks_and_sizes() -> None:
    d = _rct(n=20_000, seed=9)
    tab = um.uplift_by_group(d["y"], d["t"], d["tau"] + 1e-3 * d["x0"], n_groups=10, seed=0)
    assert tab["n"].max() - tab["n"].min() <= 1
    assert tab["mean_predicted"].is_monotonic_decreasing
    top, bottom = tab.iloc[:5]["observed"].mean(), tab.iloc[5:]["observed"].mean()
    assert top - bottom == pytest.approx(0.1, abs=0.03)


def test_budget_tags_and_grid_validation() -> None:
    tags = [um.budget_tag(k) for k in (0.001, 0.004, 0.015, 0.025, 0.05, 0.1)]
    assert len(set(tags)) == len(tags) and um.budget_tag(0.05) == "5"
    with pytest.raises(ValueError):
        um.make_grid(100, [0.0, 0.25, 0.5])                 # must end at f = 1
    with pytest.raises(ValueError):
        um.UpliftEvaluator(np.array([0, 1]), np.array([1, 1]))   # no control units
