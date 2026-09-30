"""Tests for src.causal.ate: every estimator is checked on synthetic data with a known answer."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from src.causal.ate import (
    aipw,
    array_chunks,
    auc_delong,
    auc_permutation_null,
    binomial_bootstrap,
    bootstrap_effects,
    confounded_selection_probs,
    crossfit_outcome_models,
    difference_in_means,
    g_computation,
    independent_columns,
    ipw_hajek,
    ipw_horvitz_thompson,
    joint_lrt,
    lin_regression_adjustment,
    mde_two_proportions,
    paired_gap,
    power_two_proportions,
    sample_ratio_test,
    two_proportion_inference,
    wald_iv,
)
from src.causal.propensity import make_folds

SMALL_LGBM = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 50}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _randomized_binary(n: int, seed: int, p: float = 0.85) -> tuple[np.ndarray, ...]:
    """Randomized experiment with a binary outcome, heterogeneous effect and a known ATE."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.random(n) < p
    p0 = _sigmoid(-2.0 + 1.2 * X[:, 0] + 0.6 * X[:, 1])
    p1 = np.clip(p0 + 0.04 + 0.03 * (X[:, 2] > 0), 0, 1)
    y = np.where(t, rng.random(n) < p1, rng.random(n) < p0).astype(float)
    return X, t, y, float((p1 - p0).mean())


# ---------------------------------------------------------------- two-proportion inference

def test_two_proportion_inference_matches_statsmodels() -> None:
    x_t, n_t, x_c, n_c = 530, 10_000, 70, 2_000
    res = two_proportion_inference(x_t, n_t, x_c, n_c)
    z, p = proportions_ztest([x_t, x_c], [n_t, n_c])          # pooled SE by default
    assert res["z_test"]["statistic"] == pytest.approx(z, rel=1e-10)
    assert res["z_test"]["p_value"] == pytest.approx(p, rel=1e-8)
    assert 10 ** res["z_test"]["log10_p_value"] == pytest.approx(p, rel=1e-8)
    assert res["robustness_tests"]["pearson_chi2"]["equals_z_squared"]
    wald = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="wald")
    assert res["absolute_effect"]["ci95_wald"] == pytest.approx(list(wald), rel=1e-9)
    newc = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="newcomb")
    assert res["absolute_effect"]["ci95_newcombe"] == pytest.approx(list(newc), rel=1e-6)
    ratio = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="log", compare="ratio")
    assert res["relative_lift"]["ci95_delta_log_rr"] == pytest.approx([ratio[0] - 1, ratio[1] - 1], rel=1e-6)


def test_relative_lift_undefined_when_control_has_no_events() -> None:
    res = two_proportion_inference(40, 1000, 0, 200)             # exposure-like manipulation check
    assert res["relative_lift"]["estimate"] is None
    assert res["absolute_effect"]["estimate"] == pytest.approx(0.04)


def test_wald_and_log_rr_intervals_have_nominal_coverage() -> None:
    rng = np.random.default_rng(3)
    n_t, n_c, p_t, p_c = 20_000, 4_000, 0.05, 0.04
    sims = 3000
    x_t = rng.binomial(n_t, p_t, sims)
    x_c = rng.binomial(n_c, p_c, sims)
    cov_d = cov_r = 0
    for a, b in zip(x_t, x_c):
        r = two_proportion_inference(int(a), n_t, int(b), n_c)
        lo, hi = r["absolute_effect"]["ci95_wald"]
        cov_d += lo <= p_t - p_c <= hi
        lo, hi = r["relative_lift"]["ci95_delta_log_rr"]
        cov_r += lo <= p_t / p_c - 1 <= hi
    assert 0.935 < cov_d / sims < 0.965
    assert 0.935 < cov_r / sims < 0.965


# ---------------------------------------------------------------- exact binomial bootstrap

def test_binomial_bootstrap_matches_brute_force_row_bootstrap() -> None:
    """The arm-stratified row bootstrap and the two-binomial shortcut have the same distribution."""
    rng = np.random.default_rng(0)
    n_t, n_c = 800, 300
    y_t = (rng.random(n_t) < 0.12).astype(int)
    y_c = (rng.random(n_c) < 0.07).astype(int)
    x_t, x_c = int(y_t.sum()), int(y_c.sum())
    B = 4000
    brute = np.random.default_rng(1)
    row_t = y_t[brute.integers(0, n_t, size=(B, n_t))].sum(axis=1)       # resample rows with replacement
    row_c = y_c[brute.integers(0, n_c, size=(B, n_c))].sum(axis=1)
    # (1) the resampled positive count is exactly Binomial(n_t, x_t/n_t): chi-square goodness of fit
    k = np.arange(n_t + 1)
    pmf = stats.binom.pmf(k, n_t, x_t / n_t)
    lo, hi = stats.binom.ppf([0.005, 0.995], n_t, x_t / n_t).astype(int)
    edges = np.r_[0, np.arange(lo, hi + 1), n_t + 1]
    obs = np.histogram(row_t, bins=edges)[0]
    exp = np.array([pmf[a:b].sum() for a, b in pairwise(edges)]) * B
    keep = exp >= 5
    chi2 = ((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum()
    assert stats.chi2.sf(chi2, keep.sum() - 1) > 1e-3
    # (2) the effect distributions agree (two independent Monte Carlo samples of the same law)
    p_t, p_c = binomial_bootstrap(x_t, n_t, x_c, n_c, B, np.random.default_rng(2))
    d_row, d_bin = row_t / n_t - row_c / n_c, p_t - p_c
    assert stats.ks_2samp(d_row, d_bin).pvalue > 1e-3
    assert d_bin.std() == pytest.approx(d_row.std(), rel=0.06)
    for q in (0.025, 0.975):
        assert np.quantile(d_bin, q) == pytest.approx(np.quantile(d_row, q), abs=0.25 * d_row.std())
    # (3) and both match the analytic Wald SE
    se = np.sqrt((x_t / n_t) * (1 - x_t / n_t) / n_t + (x_c / n_c) * (1 - x_c / n_c) / n_c)
    assert d_bin.std() == pytest.approx(se, rel=0.06)


def test_bootstrap_effects_reports_percentile_cis() -> None:
    out, draws = bootstrap_effects(500, 10_000, 60, 2_000, 1000, np.random.default_rng(0))
    lo, hi = out["absolute"]["ci95_percentile"]
    assert lo < 0.05 - 0.03 < hi
    assert "relative" in out and len(draws["relative"]) == 1000


# ---------------------------------------------------------------- power

def test_mde_has_nominal_power_by_simulation() -> None:
    p_c, n_t, n_c = 0.05, 8500, 1500
    mde = mde_two_proportions(p_c, n_t, n_c, alpha=0.05, power=0.8)
    assert power_two_proportions(p_c, mde, n_t, n_c) == pytest.approx(0.8, abs=1e-6)
    assert power_two_proportions(p_c, 0.0, n_t, n_c) == pytest.approx(0.05, abs=1e-9)
    rng = np.random.default_rng(0)
    sims = 20_000
    x_t = rng.binomial(n_t, p_c + mde, sims)
    x_c = rng.binomial(n_c, p_c, sims)
    pool = (x_t + x_c) / (n_t + n_c)
    z = (x_t / n_t - x_c / n_c) / np.sqrt(pool * (1 - pool) * (1 / n_t + 1 / n_c))
    assert np.mean(np.abs(z) > stats.norm.ppf(0.975)) == pytest.approx(0.8, abs=0.015)


def test_sample_ratio_test() -> None:
    assert sample_ratio_test(8500, 1500, 0.85)["p_value"] == pytest.approx(1.0)
    assert sample_ratio_test(8700, 1300, 0.85)["p_value"] < 1e-6


# ---------------------------------------------------------------- randomization checks

def test_auc_delong_matches_sklearn_and_bootstrap() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.random(n) < 0.3
    s = np.round(0.5 * y + rng.normal(size=n), 1)               # rounding creates ties
    res = auc_delong(y, s)
    assert res["auc"] == pytest.approx(roc_auc_score(y, s), rel=1e-12)
    boot = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        boot.append(roc_auc_score(y[i], s[i]))
    assert res["se_delong"] == pytest.approx(np.std(boot), rel=0.15)


def test_auc_null_moments_match_permutations_with_ties() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.random(n) < 0.85
    s = rng.integers(0, 6, n).astype(float)                      # heavy ties, independent of y
    res = auc_delong(y, s)
    null = auc_permutation_null(y, s, reps=2000, rng=np.random.default_rng(2))
    assert null.mean() == pytest.approx(0.5, abs=4 * res["se_null"] / np.sqrt(2000))
    assert null.std() == pytest.approx(res["se_null"], rel=0.08)


def test_independent_columns_drops_constant_duplicate_and_full_dummy_sets() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    level = rng.integers(0, 3, 500)
    dummies = np.column_stack([level == k for k in range(3)]).astype(float)   # sums to the intercept
    X = np.column_stack([a, b, 2 * a - b, np.ones(500), dummies])
    keep = independent_columns(X)
    assert len(keep) == 2 + 2                                   # a, b and two of the three dummies


def test_joint_lrt_detects_dependence_only_when_present() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 5))
    t_rand = rng.random(20_000) < 0.85
    t_dep = rng.random(20_000) < _sigmoid(1.7 + 0.15 * X[:, 0])
    assert joint_lrt(X, t_rand)["p_value"] > 0.01
    res = joint_lrt(X, t_dep)
    assert res["p_value"] < 1e-6 and res["df"] == 5


# ---------------------------------------------------------------- estimators

def test_lin_chunked_matches_statsmodels_interacted_ols() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 4))
    t = rng.random(n) < 0.7
    Y = np.column_stack([1 + X @ [1, -0.5, 0.2, 0] + t * (0.5 + 0.3 * X[:, 0]) + rng.normal(size=n),
                         (rng.random(n) < _sigmoid(-1 + X[:, 1] + 0.4 * t)).astype(float)])
    res = lin_regression_adjustment(array_chunks(X, t, Y, chunk_rows=700), return_phi=True)
    Xc = X - X.mean(0)
    D = np.column_stack([np.ones(n), t, Xc, t[:, None] * Xc])
    for j in range(2):
        ref = {c: sm.OLS(Y[:, j], D).fit(cov_type=c) for c in ("HC0", "HC1", "HC2")}
        assert res[j]["estimate"] == pytest.approx(ref["HC2"].params[1], rel=1e-9)
        for c in ("HC0", "HC1", "HC2"):
            assert res[j][f"se_{c.lower()}"] == pytest.approx(ref[c].bse[1], rel=1e-8)
        # the influence function reproduces the estimate, and its variance is HC0 plus the x_bar term
        assert res[j]["phi"].mean() == pytest.approx(res[j]["estimate"], rel=1e-8)
        assert res[j]["phi"].std() / np.sqrt(n) >= res[j]["se_hc0"] * 0.999


def test_estimators_recover_the_ate_in_a_randomized_experiment() -> None:
    X, t, y, ate = _randomized_binary(40_000, seed=1)
    folds = make_folds(t * 2 + y, 3, seed=0)
    om = crossfit_outcome_models(X.astype(np.float32), t, y, folds, seed=0, overrides=SMALL_LGBM)
    e = np.full(len(t), 0.85)
    results = {
        "dim": difference_in_means(y, t),
        "lin": lin_regression_adjustment(array_chunks(X, t, y, 10_000))[0],
        "ht": ipw_horvitz_thompson(y, t, e),
        "hajek": ipw_hajek(y, t, e),
        "aipw": aipw(y, t, e, om["mu0"], om["mu1"]),
    }
    for name, r in results.items():
        assert abs(r["estimate"] - ate) < 4 * r["se"], name
    rel = results["aipw"]["relative_lift"]                    # E[Y(1)]/E[Y(0)] - 1 from the AIPW means
    p0 = _sigmoid(-2.0 + 1.2 * X[:, 0] + 0.6 * X[:, 1])       # same DGP as _randomized_binary
    p1 = np.clip(p0 + 0.04 + 0.03 * (X[:, 2] > 0), 0, 1)
    assert abs(rel["estimate"] - (p1.mean() / p0.mean() - 1)) < 4 * rel["se"]
    # covariate adjustment tightens the interval when X predicts Y
    assert results["lin"]["se"] < results["dim"]["se"]
    assert results["aipw"]["se"] < results["dim"]["se"]
    g = g_computation(om["s_mu0"], om["s_mu1"], folds)
    assert g["se"] is None and len(g["fold_estimates"]) == 3


def test_hajek_with_constant_propensity_is_exactly_the_difference_in_means() -> None:
    _, t, y, _ = _randomized_binary(5000, seed=2)
    h = ipw_hajek(y, t, np.full(len(t), 0.85))
    d = difference_in_means(y, t)
    assert h["estimate"] == pytest.approx(d["estimate"], abs=1e-15)
    assert h["se"] == pytest.approx(d["se"], rel=1e-3)


def test_ipw_and_aipw_remove_confounding_that_biases_the_difference_in_means() -> None:
    rng = np.random.default_rng(4)
    n = 60_000
    x = rng.normal(size=n)
    e = _sigmoid(1.0 + 1.2 * x)
    t = rng.random(n) < e
    y = 1.0 * t + 2.0 * x + rng.normal(size=n)                  # true ATE = 1
    dim = difference_in_means(y, t)
    assert abs(dim["estimate"] - 1.0) > 10 * dim["se"]
    for est in (ipw_horvitz_thompson(y, t, e), ipw_hajek(y, t, e)):
        assert abs(est["estimate"] - 1.0) < 4 * est["se"]
    folds = make_folds(t.astype(int), 3, seed=0)
    om = crossfit_outcome_models(x[:, None], t, y, folds, seed=0, overrides=SMALL_LGBM, s_learner=False)
    r = aipw(y, t, e, om["mu0"], om["mu1"])
    assert abs(r["estimate"] - 1.0) < 4 * r["se"]


def test_paired_gap_uses_the_correlation_between_estimators() -> None:
    X, t, y, _ = _randomized_binary(20_000, seed=5)
    d = difference_in_means(y, t, return_phi=True)
    lin = lin_regression_adjustment(array_chunks(X, t, y, 5000), return_phi=True)[0]
    g = paired_gap(lin["phi"], d["phi"], lin["estimate"], d["estimate"])
    assert g["se_paired"] < 0.8 * np.hypot(lin["se"], d["se"])
    assert abs(g["z"]) < 4                                      # both unbiased under randomization
    same = paired_gap(d["phi"], d["phi"], d["estimate"], d["estimate"])
    assert same["gap"] == 0 and same["se_paired"] == 0


def test_wald_iv_recovers_the_complier_effect_under_self_selected_exposure() -> None:
    rng = np.random.default_rng(6)
    n = 300_000
    u = rng.normal(size=n)                                      # unobserved activity
    t = rng.random(n) < 0.85
    complier = (u + rng.normal(size=n)) > 1.3                   # active users get exposed
    d = (t & complier).astype(int)
    p0 = 0.05 + 0.3 * stats.norm.cdf(u)
    y = (rng.random(n) < p0 + 0.10 * d).astype(int)             # effect of exposure = +0.10
    iv = wald_iv(y, t, d, reps=1000, rng=np.random.default_rng(0))
    assert iv["one_sided_noncompliance"]
    assert abs(iv["late"] - 0.10) < 4 * iv["se_delta"]
    lo, hi = iv["bootstrap"]["ci95_percentile"]
    assert lo < 0.10 < hi
    assert iv["bootstrap"]["se"] == pytest.approx(iv["se_delta"], rel=0.15)
    assert iv["complier_mean_untreated"] == pytest.approx(p0[complier].mean(), abs=4 * iv["se_delta"])
    naive = y[t & (d == 1)].mean() - y[t & (d == 0)].mean()
    assert naive > 0.10 + 5 * iv["se_delta"]                    # self-selection inflates the naive contrast


def test_confounded_selection_keeps_covariates_and_has_the_stated_propensity() -> None:
    rng = np.random.default_rng(7)
    n = 400_000
    z = rng.random(n)
    t = rng.random(n) < 0.85
    keep, e_true = confounded_selection_probs(z, t, 0.85, (1.0, 0.1), 0.5)
    marginal = np.where(t, keep, 0.0).mean() + np.where(~t, keep, 0.0).mean()
    sel = rng.random(n) < keep
    # the overall keep rate is flat in z, so the selected sample has the original covariate distribution
    bins = np.digitize(z, [0.25, 0.5, 0.75])
    rates = [sel[bins == b].mean() for b in range(4)]
    assert max(rates) - min(rates) < 0.01 and marginal == pytest.approx(0.575, abs=0.005)
    # the treated share within z-bins matches the stated propensity
    for b in range(4):
        m = sel & (bins == b)
        assert t[m].mean() == pytest.approx(e_true[m].mean(), abs=0.01)
    y = (rng.random(n) < 0.05 + 0.2 * z + 0.02 * t).astype(float)   # ATE = 0.02
    naive = difference_in_means(y[sel], t[sel])
    fixed = ipw_hajek(y[sel], t[sel], e_true[sel])
    assert naive["estimate"] - 0.02 > 10 * naive["se"]
    assert abs(fixed["estimate"] - 0.02) < 4 * fixed["se"]
    with pytest.raises(ValueError, match="outside"):
        confounded_selection_probs(z, t, 0.85, (1.0, 0.1), 0.99)
