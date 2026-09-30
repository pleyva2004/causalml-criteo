"""Average treatment effects of ad eligibility on the Criteo randomized experiment.

Two pipeline stages live here:

* ``run_stats`` (stage ``stats``, Phase 3) -- classical inference on the randomized assignment: rates,
  absolute and relative effects with analytic and exact-bootstrap CIs, two-proportion tests, power /
  minimum detectable effect, and randomization checks (sample ratio, covariate balance, a classifier
  two-sample test and a joint likelihood-ratio test).
* ``run_causal_ate`` (stage ``causal``, Phases 8-9) -- a comparison of average-effect estimators
  (difference in means, Lin regression adjustment, g-computation, IPW, AIPW), the effect of *exposure*
  via randomized assignment as an instrument, and a confounded-subsample demonstration of why
  propensity scores matter in observational data.

Identification. ``treatment`` is randomized *assignment* (ad eligibility), so T is independent of the
potential outcomes (Y(0), Y(1)) and of the pre-treatment features X. Then E[Y | T=t] = E[Y(t)], and
E[Y|T=1] - E[Y|T=0] is unbiased for the average effect of assignment (the intention-to-treat effect,
ITT) with no outcome model at all. Every estimator below targets that same number; they differ only in
variance and in what they would need if randomization were not available. Numbers are properties of
this benchmark sample (the release is non-uniformly sub-sampled), not real-world incrementality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.causal.propensity import (
    clip_propensity,
    fit_lgbm,
    lgbm_propensity,
    logistic_propensity,
    make_folds,
    propensity_diagnostics,
)
from src.data.load import FEATURES, SPLIT_CODES, load_criteo
from src.features.build_features import RAW_FEATURES, FeatureBuilder
from src.models.tree_models import make_lgbm
from src.utils import get_logger, load_config, save_json, save_table, timer

log = get_logger(__name__)

ChunkFactory = Callable[[], Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]]


# ----------------------------------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------------------------------

def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _normal_ci(est: float, se: float, alpha: float) -> list[float]:
    z = _z(alpha)
    return [float(est - z * se), float(est + z * se)]


def _log10_two_sided_p(z: float) -> float:
    """log10 of the two-sided normal p-value, computed in log space so it never underflows to 0."""
    return float((np.log(2.0) + stats.norm.logsf(abs(z))) / np.log(10.0))


def _p_text(log10_p: float) -> str:
    return f"p = {10 ** log10_p:.3g}" if log10_p > -300 else f"p < 1e-300 (log10 p = {log10_p:.0f})"


def _percentile_ci(draws: np.ndarray, alpha: float) -> list[float]:
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def _subsample_idx(n: int, rows: int | None, rng: np.random.Generator) -> np.ndarray:
    """Sorted uniform subsample of ``rows`` indices (all indices when ``rows`` is None or >= n)."""
    if rows is None or rows >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=rows, replace=False))


# ----------------------------------------------------------------------------------------------------
# Phase 3: two-proportion inference
# ----------------------------------------------------------------------------------------------------

def wilson_interval(x: int, n: int, alpha: float = 0.05) -> list[float]:
    """Wilson score interval for a binomial proportion (well behaved near 0, unlike the Wald interval)."""
    z = _z(alpha)
    p = x / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(center - half), float(center + half)]


def newcombe_interval(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> list[float]:
    """Newcombe (1998, method 10) hybrid score interval for p_T - p_C, built from two Wilson intervals.

    A robustness column for the Wald interval: it has better coverage when a rate is near 0 or 1.
    """
    p_t, p_c = x_t / n_t, x_c / n_c
    l_t, u_t = wilson_interval(x_t, n_t, alpha)
    l_c, u_c = wilson_interval(x_c, n_c, alpha)
    d = p_t - p_c
    return [float(d - np.sqrt((p_t - l_t) ** 2 + (u_c - p_c) ** 2)),
            float(d + np.sqrt((u_t - p_t) ** 2 + (p_c - l_c) ** 2))]


def two_proportion_inference(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict[str, Any]:
    """Effect of assignment on a binary outcome from the four counts of a 2x2 table.

    * absolute effect p_T - p_C with the unpooled Wald CI (and Newcombe's score interval as a check);
    * relative lift p_T / p_C - 1 with a delta-method CI on the log risk ratio,
      Var(log RR) ~= (1 - p_T) / x_T + (1 - p_C) / x_C, exponentiated back (so it is asymmetric);
    * two-proportion z-test with the pooled SE (the SE under H0: p_T = p_C), two-sided;
    * robustness tests of the same null: Pearson chi-square (= z^2 exactly, without continuity
      correction), likelihood-ratio G-test, and Fisher's exact test (conditional on the margins).
    """
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
    out: dict[str, Any] = {
        "rates": {"treatment": p_t, "control": p_c,
                  "treatment_ci95_wilson": wilson_interval(x_t, n_t, alpha),
                  "control_ci95_wilson": wilson_interval(x_c, n_c, alpha),
                  "counts": {"x_t": int(x_t), "n_t": int(n_t), "x_c": int(x_c), "n_c": int(n_c)}},
        "absolute_effect": {"estimate": diff, "se_unpooled": se, "ci95_wald": _normal_ci(diff, se, alpha),
                            "ci95_newcombe": newcombe_interval(x_t, n_t, x_c, n_c, alpha)},
    }
    if x_t > 0 and x_c > 0:
        log_rr = float(np.log(p_t / p_c))
        se_log = float(np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c))
        lo, hi = _normal_ci(log_rr, se_log, alpha)
        out["relative_lift"] = {"estimate": float(np.exp(log_rr) - 1), "log_rr": log_rr, "se_log_rr": se_log,
                                "ci95_delta_log_rr": [float(np.exp(lo) - 1), float(np.exp(hi) - 1)]}
    else:
        out["relative_lift"] = {"estimate": None, "note": "undefined: an arm has zero events"}

    p_pool = (x_t + x_c) / (n_t + n_c)
    se0 = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c)))
    z = diff / se0
    log10_p = _log10_two_sided_p(z)
    out["z_test"] = {"statistic": float(z), "se_pooled": se0, "p_value": float(2 * stats.norm.sf(abs(z))),
                     "log10_p_value": log10_p}
    table = np.array([[x_t, n_t - x_t], [x_c, n_c - x_c]])
    chi2 = stats.chi2_contingency(table, correction=False)
    g = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    fisher = stats.fisher_exact(table)
    out["robustness_tests"] = {
        "pearson_chi2": {"statistic": float(chi2.statistic), "p_value": float(chi2.pvalue),
                         "equals_z_squared": bool(np.isclose(chi2.statistic, z**2, rtol=1e-6))},
        "g_test": {"statistic": float(g.statistic), "p_value": float(g.pvalue)},
        "fisher_exact": {"odds_ratio": float(fisher.statistic), "p_value": float(fisher.pvalue)},
    }
    return out


def binomial_bootstrap(x_t: int, n_t: int, x_c: int, n_c: int, reps: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact shortcut for the arm-stratified nonparametric bootstrap of two binomial means.

    The stratified bootstrap redraws n_T rows with replacement from the treated arm and n_C rows from the
    control arm. Each redrawn treated row is positive with probability x_T / n_T independently of the
    others (uniform draw among n_T rows, x_T of them positive), so the resampled positive count is
    *exactly* Binomial(n_T, p_T_hat), independent of the control arm's Binomial(n_C, p_C_hat). Any
    statistic that depends on the resample only through the two counts -- the difference, the ratio --
    therefore has the same bootstrap distribution under both procedures. Drawing two binomials per
    replicate replaces resampling 14M rows. (``tests/test_ate.py`` checks this against a brute-force
    row bootstrap.) Returns the resampled rates (p_T*, p_C*).
    """
    return rng.binomial(n_t, x_t / n_t, size=reps) / n_t, rng.binomial(n_c, x_c / n_c, size=reps) / n_c


def bootstrap_effects(x_t: int, n_t: int, x_c: int, n_c: int, reps: int, rng: np.random.Generator,
                      alpha: float = 0.05) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Percentile bootstrap CIs for the absolute and relative effect (exact binomial shortcut)."""
    p_t, p_c = binomial_bootstrap(x_t, n_t, x_c, n_c, reps, rng)
    diff = p_t - p_c
    out: dict[str, Any] = {"reps": reps, "method": "stratified bootstrap via exact binomial counts, percentile CI",
                           "absolute": {"ci95_percentile": _percentile_ci(diff, alpha),
                                        "se": float(diff.std(ddof=1)), "mean": float(diff.mean())}}
    draws = {"absolute": diff}
    if x_c > 0 and x_t > 0 and (p_c > 0).all():
        rel = p_t / p_c - 1.0
        out["relative"] = {"ci95_percentile": _percentile_ci(rel, alpha), "se": float(rel.std(ddof=1)),
                           "mean": float(rel.mean())}
        draws["relative"] = rel
    return out, draws


def power_two_proportions(p_c: float, delta: float, n_t: int, n_c: int, alpha: float = 0.05) -> float:
    """Power of the two-sided pooled two-proportion z-test when p_T = p_C + delta (normal approximation).

    Reject when |D| > z_{1-alpha/2} * SE0 with SE0 the pooled (null) SE; under the alternative
    D ~ N(delta, SE1^2) with the unpooled SE1. Both rejection tails are counted.
    """
    p_t = p_c + delta
    p_bar = (n_t * p_t + n_c * p_c) / (n_t + n_c)
    se0 = np.sqrt(p_bar * (1 - p_bar) * (1 / n_t + 1 / n_c))
    se1 = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = _z(alpha)
    return float(stats.norm.cdf((delta - z * se0) / se1) + stats.norm.cdf((-delta - z * se0) / se1))


def mde_two_proportions(p_c: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable (absolute, positive) effect: the delta at which ``power_two_proportions`` = power."""
    f = lambda d: power_two_proportions(p_c, d, n_t, n_c, alpha) - power
    hi = 1.0 - p_c - 1e-12
    if f(hi) < 0:
        return float("nan")
    return float(brentq(f, 1e-12, hi, xtol=1e-14, rtol=1e-10))


# ----------------------------------------------------------------------------------------------------
# Phase 3: randomization checks
# ----------------------------------------------------------------------------------------------------

def sample_ratio_test(n_t: int, n_c: int, expected: float) -> dict[str, Any]:
    """Sample-ratio-mismatch check: chi-square goodness of fit of the arm counts to the design ratio."""
    n = n_t + n_c
    res = stats.chisquare([n_t, n_c], f_exp=[n * expected, n * (1 - expected)])
    return {"observed_treated_share": n_t / n, "expected": expected,
            "observed_ci95_wilson": wilson_interval(n_t, n), "chi2": float(res.statistic),
            "p_value": float(res.pvalue), "n_t": int(n_t), "n_c": int(n_c)}


def standardized_mean_differences(X: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """SMD per column, (mean_T - mean_C) / sqrt((var_T + var_C) / 2), with its SE under exact balance.

    Under randomization every SMD is noise of size ~sqrt(1/n_T + 1/n_C), so at 14M rows the usual 0.1 rule
    of thumb is far too loose; the noise band is the informative reference.
    """
    t = np.asarray(t).astype(bool)
    n1, n0 = int(t.sum()), int((~t).sum())
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=np.float64)
        a, b = v[t], v[~t]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        rows.append({"feature": c, "mean_treatment": a.mean(), "mean_control": b.mean(), "smd": smd,
                     "se_null": np.sqrt(1 / n1 + 1 / n0)})
    out = pd.DataFrame(rows)
    out["z"] = out["smd"] / out["se_null"]
    return out


def auc_delong(y: np.ndarray, score: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """ROC-AUC with DeLong's SE/CI (midranks handle ties) and its tie-corrected permutation-null moments.

    DeLong: V10_i = share of negatives scored below positive i (ties count 1/2), V01_j = share of
    positives scored above negative j; Var(AUC) = Var(V10)/n1 + Var(V01)/n0. Both are computed from
    ranks in O(n log n). Under H0 (labels exchangeable) AUC has mean 1/2 and the Mann-Whitney variance
    with the tie correction; the one-sided p-value tests AUC > 1/2 (the classifier found signal).
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    n = n1 + n0
    r_all = stats.rankdata(s)
    auc = float((r_all[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    v10 = (r_all[y] - stats.rankdata(s[y])) / n0
    v01 = 1.0 - (r_all[~y] - stats.rankdata(s[~y])) / n1
    se = float(np.sqrt(v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0))
    _, counts = np.unique(s, return_counts=True)
    counts = counts.astype(np.float64)                      # cubes of ~1e6 overflow int64
    var_u = n1 * n0 / 12.0 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1)))
    se_null = float(np.sqrt(max(var_u, 0.0)) / (n1 * n0))
    z_null = (auc - 0.5) / se_null if se_null > 0 else 0.0
    return {"auc": auc, "se_delong": se, "ci95_delong": _normal_ci(auc, se, alpha), "se_null": se_null,
            "z_null": float(z_null), "p_value_one_sided_analytic": float(stats.norm.sf(z_null)),
            "n_pos": n1, "n_neg": n0, "distinct_scores": len(counts)}


def auc_permutation_null(y: np.ndarray, score: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation distribution of the AUC: relabel which rows are 'negative' at random, keep the ranks.

    Ranks are computed once; each replicate only sums the ranks of a random subset, so 1,000
    permutations of a 2.8M-row test split take seconds.
    """
    y = np.asarray(y).astype(bool)
    r = stats.rankdata(np.asarray(score, dtype=np.float64))
    n, n1 = len(r), int(y.sum())
    n0 = n - n1
    total = r.sum()
    null = np.empty(reps)
    for b in range(reps):
        neg = rng.choice(n, size=n0, replace=False)
        null[b] = (total - r[neg].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return null


def logloss_improvement(y: np.ndarray, p_model: np.ndarray, p_ref: float | np.ndarray,
                        alpha: float = 0.05) -> dict[str, Any]:
    """Paired comparison of held-out log-loss: mean(loss_ref - loss_model) with a CLT CI (positive = model better)."""
    y = np.asarray(y, dtype=np.float64)
    eps = 1e-15
    pm = np.clip(p_model, eps, 1 - eps)
    pr = np.clip(np.broadcast_to(p_ref, y.shape), eps, 1 - eps)
    loss_m = -(y * np.log(pm) + (1 - y) * np.log1p(-pm))
    loss_r = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    d = loss_r - loss_m
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"logloss_model": float(loss_m.mean()), "logloss_reference": float(loss_r.mean()),
            "improvement": float(d.mean()), "se": se, "ci95": _normal_ci(float(d.mean()), se, alpha),
            "z": float(d.mean() / se) if se > 0 else 0.0}


def independent_columns(X: np.ndarray, rel_tol: float = 1e-7) -> np.ndarray:
    """Indices of a maximal set of linearly independent columns *given an intercept* (pivoted QR).

    Columns are centered first, so constant columns and columns collinear with the intercept (e.g. a
    full set of one-hot dummies) are dropped too. Needed for unpenalized fits (Lin OLS, statsmodels Logit).
    """
    A = np.asarray(X, dtype=np.float64)
    A = A - A.mean(axis=0)
    _, R, piv = linalg.qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    if d.size == 0 or d[0] == 0:
        return np.array([], dtype=int)
    rank = int((d > rel_tol * d[0]).sum())
    return np.sort(piv[:rank])


def joint_lrt(X: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    """Likelihood-ratio test of H0: all slopes are 0 in logit P(T=1|X) = a + X b (statsmodels Logit).

    Under randomization T is independent of X, so LR = 2 (ll_full - ll_null) ~ chi2(k). Also counts the
    individually 'significant' coefficients, which should be ~5% of k by chance alone.
    """
    import statsmodels.api as sm

    res = sm.Logit(np.asarray(t, dtype=np.float64), sm.add_constant(np.asarray(X, dtype=np.float64),
                                                                     has_constant="add")).fit(disp=0, maxiter=100)
    k = int(res.df_model)
    n_sig = int((res.pvalues[1:] < 0.05).sum())
    return {"llr": float(res.llr), "df": k, "p_value": float(res.llr_pvalue), "n": int(res.nobs),
            "converged": bool(res.mle_retvals.get("converged", False)),
            "pseudo_r2_mcfadden": float(res.prsquared),
            "coefficients_p_below_0.05": n_sig, "expected_by_chance": 0.05 * k,
            "binomial_p_value_for_that_count": float(stats.binomtest(n_sig, k, 0.05, alternative="greater").pvalue)}


# ----------------------------------------------------------------------------------------------------
# Phases 8-9: ATE estimators
# ----------------------------------------------------------------------------------------------------

def difference_in_means(y: np.ndarray, t: np.ndarray, alpha: float = 0.05, return_phi: bool = False
                        ) -> dict[str, Any]:
    """Difference in means with the Neyman variance s1^2/n1 + s0^2/n0.

    Treatment: binary randomized assignment. Outcome model: none. Estimand: ATE of assignment (ITT).
    Assumptions: T independent of (Y(0), Y(1)) (randomization), SUTVA (no interference between users,
    a single version of treatment). Unbiased for the ATE; the Neyman variance is exact for the
    super-population ATE under iid sampling (conservative for the finite-sample ATE).
    Limitations: ignores covariates, so it is not the most precise unbiased estimator; invalid
    without randomization. ``return_phi`` adds the per-row influence function
    phi_i = T (Y - ybar1) / p - (1-T) (Y - ybar0) / (1-p), p = n1/n (for paired comparisons).
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    est = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    out = {"estimate": est, "se": se, "ci95": _normal_ci(est, se, alpha), "n": len(y),
           "n_t": len(y1), "n_c": len(y0), "mean_t": float(y1.mean()), "mean_c": float(y0.mean())}
    if return_phi:
        p = len(y1) / len(y)
        out["phi"] = np.where(t, (y - y1.mean()) / p, -(y - y0.mean()) / (1 - p))
    return out


def paired_gap(phi_a: np.ndarray, phi_b: np.ndarray, est_a: float, est_b: float,
               alpha: float = 0.05) -> dict[str, Any]:
    """Difference between two estimators computed on the *same* rows, with the SE of the difference from
    their influence functions: SE = sd(phi_a - phi_b) / sqrt(n). The two estimates are highly correlated,
    so sqrt(se_a^2 + se_b^2) would grossly overstate the uncertainty of the gap."""
    d = np.asarray(phi_a) - np.asarray(phi_b)
    gap = float(est_a - est_b)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    z = gap / se if se > 0 else 0.0
    return {"gap": gap, "se_paired": se, "ci95": _normal_ci(gap, se, alpha), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def array_chunks(X: np.ndarray, t: np.ndarray, Y: np.ndarray, chunk_rows: int) -> ChunkFactory:
    """Chunk factory over in-memory arrays for ``lin_regression_adjustment``."""
    Y = Y.reshape(len(Y), -1)

    def gen() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for i in range(0, len(t), chunk_rows):
            yield X[i:i + chunk_rows], t[i:i + chunk_rows], Y[i:i + chunk_rows]

    return gen


def lin_regression_adjustment(chunks: ChunkFactory, alpha: float = 0.05, return_phi: bool = False
                              ) -> list[dict[str, Any]]:
    """Lin (2013) regression adjustment: OLS of Y on T, centered X and T x centered X, with HC SEs.

    Treatment: binary assignment. Outcome model: linear in the design X, fully interacted with T
    (a separate linear fit per arm). Estimand: ATE of assignment -- the coefficient on T when X is
    centered at the full-sample mean. Assumptions: randomization and SUTVA only; the linear model need
    *not* be correct (Lin 2013: under randomization the interacted estimator is consistent and never less
    precise asymptotically than the difference in means). SEs: Eicker-Huber-White HC0/HC1/HC2 sandwich.
    Limitations: linear adjustment captures only the linear part of E[Y|X]; with a rare binary outcome
    the variance reduction is bounded by the (small) R^2 of Y on X.

    Implementation: the interacted design [1, T, Xc, T*Xc] spans the same space as the block design
    [(1-T)Z, T*Z] with Z = [1, X]; the latter is block diagonal by arm, so OLS reduces to one fit per arm
    and the hat matrix (hence HC2 leverages) is the per-arm hat matrix. With beta_k the per-arm
    coefficients and c = [1, x_bar], the ATE is c'(beta_1 - beta_0) and its sandwich variance is
    sum_k sum_{i in arm k} w_i (z_i' G_k^{-1} c)^2 with w_i = e_i^2 (HC0) or e_i^2 / (1 - h_ii) (HC2).
    Everything is accumulated over row chunks (two passes), so the 14M x ~180 design never materializes.
    ``chunks()`` must return a fresh iterator of (X, t, Y) with Y of shape (n, m); returns one dict per
    outcome column. ``return_phi`` adds the per-row influence function
    phi_i = ATE + s_i n (z_i' G_k^{-1} c) e_i + (x_i - x_bar)'(beta_1 - beta_0) with s_i = +1 (treated) / -1;
    the first term reproduces HC0, the second accounts for estimating x_bar (population rather than
    conditional-on-X variance). Used only for paired comparisons with other estimators.
    """
    G = [None, None]
    B = [None, None]
    ysum = [None, None]
    yss = [None, None]
    n_arm = [0, 0]
    colsum = None
    n = 0
    for X, t, Y in chunks():
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(len(X), -1)
        t = np.asarray(t).astype(bool)
        Z = np.hstack([np.ones((len(X), 1)), X])
        colsum = X.sum(axis=0) if colsum is None else colsum + X.sum(axis=0)
        n += len(X)
        for k, mask in ((1, t), (0, ~t)):
            Zk, Yk = Z[mask], Y[mask]
            g, b = Zk.T @ Zk, Zk.T @ Yk
            G[k] = g if G[k] is None else G[k] + g
            B[k] = b if B[k] is None else B[k] + b
            ysum[k] = Yk.sum(0) if ysum[k] is None else ysum[k] + Yk.sum(0)
            yss[k] = (Yk**2).sum(0) if yss[k] is None else yss[k] + (Yk**2).sum(0)
            n_arm[k] += int(mask.sum())
    if min(n_arm) == 0:
        raise ValueError("lin_regression_adjustment: both arms need rows")
    p = G[0].shape[0]
    c = np.r_[1.0, colsum / n]
    Ginv = [np.linalg.pinv(G[k], hermitian=True) for k in (0, 1)]
    beta = [Ginv[k] @ B[k] for k in (0, 1)]
    a = [Ginv[k] @ c for k in (0, 1)]
    ate = c @ (beta[1] - beta[0])
    m = len(ate)

    s_hc0 = np.zeros(m)
    s_hc2 = np.zeros(m)
    sse = [np.zeros(m), np.zeros(m)]
    max_lev = 0.0
    phi_parts: list[np.ndarray] = []
    for X, t, Y in chunks():
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(len(X), -1)
        t = np.asarray(t).astype(bool)
        Z = np.hstack([np.ones((len(X), 1)), X])
        if return_phi:
            ph = Z @ (beta[1] - beta[0]) - ate            # (x_i - x_bar)'(beta_1 - beta_0)
        for k, mask in ((1, t), (0, ~t)):
            Zk = Z[mask]
            E = Y[mask] - Zk @ beta[k]
            u = Zk @ a[k]
            u2 = u**2
            h = np.einsum("ij,ij->i", Zk @ Ginv[k], Zk)
            max_lev = max(max_lev, float(h.max(initial=0.0)))
            E2 = E**2
            if return_phi:
                ph[mask] += (1.0 if k == 1 else -1.0) * n * u[:, None] * E
            s_hc0 += (E2 * u2[:, None]).sum(0)
            s_hc2 += (E2 * (u2 / (1.0 - h))[:, None]).sum(0)
            sse[k] += E2.sum(0)
        if return_phi:
            phi_parts.append(ph)
    n_params = 2 * p
    phi_all = np.vstack(phi_parts) if return_phi else None
    out = []
    for j in range(m):
        se_hc2 = float(np.sqrt(s_hc2[j]))
        r2 = {name: float(1 - sse[k][j] / (yss[k][j] - ysum[k][j] ** 2 / n_arm[k]))
              for name, k in (("treatment", 1), ("control", 0))}
        out.append({"estimate": float(ate[j]), "se": se_hc2, "ci95": _normal_ci(float(ate[j]), se_hc2, alpha),
                    "se_hc0": float(np.sqrt(s_hc0[j])), "se_hc1": float(np.sqrt(s_hc0[j] * n / (n - n_params))),
                    "se_hc2": se_hc2, "n": n, "n_params": n_params, "max_leverage": max_lev,
                    "r2_by_arm": r2})
        if return_phi:
            out[-1]["phi"] = phi_all[:, j] + float(ate[j])
    return out


def ipw_horvitz_thompson(y: np.ndarray, t: np.ndarray, e: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Horvitz-Thompson IPW: mean(T Y / e) - mean((1-T) Y / (1-e)).

    Treatment: binary assignment; weights from the propensity e(x). Outcome model: none. Estimand: ATE.
    Assumptions: conditional ignorability T indep. (Y(0),Y(1)) | X, positivity 0 < e(x) < 1, e correctly
    specified, SUTVA. SE from the influence function treating e as fixed; with an *estimated* e in an RCT
    this is conservative (estimating e acts like covariate adjustment -- Hirano, Imbens & Ridder 2003).
    Limitations: not invariant to shifting Y by a constant; high variance when weights are extreme.
    With the known design propensity it differs from the difference in means only because the realized
    treated share n_T/n is not exactly 0.85.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    psi = t * y / e - (1 - t) * y / (1 - e)
    est = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {"estimate": est, "se": se, "ci95": _normal_ci(est, se, alpha), "n": len(y)}


def ipw_hajek(y: np.ndarray, t: np.ndarray, e: np.ndarray, alpha: float = 0.05, return_phi: bool = False
              ) -> dict[str, Any]:
    """Hajek (normalized) IPW: weighted mean of Y in each arm with weights 1/e and 1/(1-e).

    Same assumptions as Horvitz-Thompson; normalizing the weights makes it location invariant and
    usually more stable. SE from the linearized influence function
    phi_i = w1_i (Y_i - m1) / mean(w1) - w0_i (Y_i - m0) / mean(w0), with e treated as fixed.
    With a constant propensity (the design value 0.85) the weights are constant within arm, so the
    Hajek estimator *is* the difference in means, exactly.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    w1 = t / e
    w0 = (1 - t) / (1 - e)
    m1 = float((w1 * y).sum() / w1.sum())
    m0 = float((w0 * y).sum() / w0.sum())
    phi = w1 * (y - m1) / w1.mean() - w0 * (y - m0) / w0.mean()
    est = m1 - m0
    se = float(phi.std(ddof=1) / np.sqrt(len(phi)))
    out = {"estimate": est, "se": se, "ci95": _normal_ci(est, se, alpha), "n": len(y), "mean_t": m1, "mean_c": m0}
    if return_phi:
        out["phi"] = phi + est
    return out


def aipw(y: np.ndarray, t: np.ndarray, e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray,
         alpha: float = 0.05, return_phi: bool = False) -> dict[str, Any]:
    """Augmented IPW (doubly robust) with the efficient influence function.

    phi_i = mu1(x) - mu0(x) + T (Y - mu1(x)) / e(x) - (1-T) (Y - mu0(x)) / (1 - e(x)); ATE = mean(phi),
    SE = sd(phi)/sqrt(n). Treatment: binary assignment. Outcome model: per-arm regressions mu_t(x) (here
    cross-fitted LightGBM). Estimand: ATE. Assumptions: ignorability given X, positivity, SUTVA, and
    *either* e or mu correct (double robustness); for the SE, cross-fitting and product-rate
    convergence ||e_hat - e|| * ||mu_hat - mu|| = o(n^-1/2). With the known RCT propensity it is unbiased
    whatever mu is. Limitations: SE is asymptotic; extreme weights inflate variance. Also reports the
    AIPW potential-outcome means E[Y(1)], E[Y(0)] and the relative lift E[Y(1)]/E[Y(0)] - 1 with a
    delta-method SE from their influence functions.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    phi1 = mu1 + t * (y - mu1) / e                       # IF of E[Y(1)]
    phi0 = mu0 + (1 - t) * (y - mu0) / (1 - e)           # IF of E[Y(0)]
    phi = phi1 - phi0
    est = float(phi.mean())
    se = float(phi.std(ddof=1) / np.sqrt(len(phi)))
    out = {"estimate": est, "se": se, "ci95": _normal_ci(est, se, alpha), "n": len(y)}
    m1, m0 = float(phi1.mean()), float(phi0.mean())
    if m0 > 0:
        rel = m1 / m0 - 1.0
        se_rel = float(((phi1 - (m1 / m0) * phi0) / m0).std(ddof=1) / np.sqrt(len(phi)))
        out["relative_lift"] = {"estimate": rel, "se": se_rel, "ci95": _normal_ci(rel, se_rel, alpha),
                                "mean_y1": m1, "mean_y0": m0}
    if return_phi:
        out["phi"] = phi
    return out


def g_computation(s_mu0: np.ndarray, s_mu1: np.ndarray, folds: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """g-computation (standardization) with an S-learner: mean over units of mu(x, 1) - mu(x, 0).

    Treatment: binary assignment, entered as one feature of a single outcome model. Outcome model: one
    LightGBM classifier on (X, T), cross-fitted (each row predicted by a model that did not see it).
    Estimand: ATE. Assumptions: ignorability given X, positivity, SUTVA, and a *correct* outcome model --
    unlike AIPW there is no residual correction, so any bias of mu (e.g. regularization shrinking the
    small T effect toward 0) passes straight into the estimate. SE: none. A plug-in ML estimator has no
    valid analytic SE (that is what AIPW's residual term buys), and a bootstrap would refit the model
    thousands of times. The fold-to-fold spread sd(fold estimates)/sqrt(K) is reported only as a
    descriptive number: folds share most of their training data, so it badly understates uncertainty
    and is deliberately *not* turned into a CI.
    """
    tau = np.asarray(s_mu1) - np.asarray(s_mu0)
    ks = np.unique(folds)
    fold_est = np.array([tau[folds == k].mean() for k in ks])
    est = float(tau.mean())
    return {"estimate": est, "se": None, "ci95": [None, None], "n": len(tau),
            "fold_estimates": fold_est.tolist(),
            "fold_spread_se_descriptive": float(fold_est.std(ddof=1) / np.sqrt(len(ks))),
            "se_note": "no valid SE; fold spread is descriptive only (understates uncertainty)"}


def crossfit_outcome_models(X: np.ndarray, t: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int,
                            overrides: dict[str, Any] | None = None, s_learner: bool = True,
                            early_stopping_rounds: int | None = None, inner_val_frac: float = 0.1,
                            constant_mu0: float | None = None) -> dict[str, Any]:
    """Out-of-fold outcome predictions for one outcome.

    * per-arm models (T-learner style) mu_1(x) = E[Y | X, T=1], mu_0(x) = E[Y | X, T=0] -- used by AIPW;
    * optionally one S-learner mu(x, t) with T as a feature, evaluated at t=1 and t=0 -- g-computation.
    Binary outcomes use a LightGBM classifier (probabilities), others a regressor. With
    ``early_stopping_rounds`` each model picks its tree count on an inner split of its training rows.
    The number of trees per model and fold is returned under ``"trees"``. ``constant_mu0`` skips the control
    model when E[Y | X, T=0] is known structurally (e.g. exposure: control users can never be exposed).
    """
    t = np.asarray(t).astype(bool)
    y = np.asarray(y)
    binary = bool(np.isin(np.unique(y), [0, 1]).all())
    kind = "classifier" if binary else "regressor"

    trees: dict[str, list[int]] = {}

    def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte_list: list[np.ndarray], name: str) -> list[np.ndarray]:
        model = fit_lgbm(kind, Xtr, ytr, seed, overrides, early_stopping_rounds, inner_val_frac)
        trees.setdefault(name, []).append(int(model.best_iteration_ or model.n_estimators))
        return [model.predict_proba(Xte)[:, 1] if binary else model.predict(Xte) for Xte in Xte_list]

    out: dict[str, Any] = {k: np.full(len(y), np.nan)
                           for k in (["mu0", "mu1"] + (["s_mu0", "s_mu1"] if s_learner else []))}
    for k in np.unique(folds):
        te = np.flatnonzero(folds == k)
        tr = np.flatnonzero(folds != k)
        Xte = X[te]
        for arm, key in ((True, "mu1"), (False, "mu0")):
            if key == "mu0" and constant_mu0 is not None:
                out["mu0"][te] = constant_mu0
                continue
            idx = tr[t[tr] == arm]
            out[key][te] = fit_predict(X[idx], y[idx], [Xte], key)[0]
        if s_learner:
            Xs = np.hstack([X[tr], t[tr, None].astype(X.dtype)])
            ones = np.ones((len(te), 1), dtype=X.dtype)
            p1, p0 = fit_predict(Xs, y[tr], [np.hstack([Xte, ones]), np.hstack([Xte, 0 * ones])], "s_learner")
            out["s_mu1"][te], out["s_mu0"][te] = p1, p0
    out["trees"] = trees
    return out


# ----------------------------------------------------------------------------------------------------
# Phase 9: exposure vs assignment (instrumental variables)
# ----------------------------------------------------------------------------------------------------

def mean_difference(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """mean(a) - mean(b) with the Welch (unpooled) SE. A descriptive contrast -- causal only if the groups
    are exchangeable."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    est = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return {"estimate": est, "se": se, "ci95": _normal_ci(est, se, alpha), "n": len(a) + len(b),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def wald_iv(y: np.ndarray, t: np.ndarray, d: np.ndarray, alpha: float = 0.05, reps: int = 2000,
            rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Wald / IV estimator of the effect of exposure D on Y, with randomized assignment T as instrument.

    LATE = ITT_Y / ITT_D = (E[Y|T=1] - E[Y|T=0]) / (E[D|T=1] - E[D|T=0]).
    Assumptions: (1) T randomized; (2) exclusion -- T affects Y only through D (being eligible for ads
    without seeing one does nothing); (3) monotonicity -- no one is exposed *because* they were assigned
    to control (holds trivially here: control users cannot be exposed); (4) relevance -- ITT_D != 0;
    SUTVA. Estimand: the average effect of exposure on compliers (users who are exposed iff assigned).
    With one-sided noncompliance (P(D=1|T=0) = 0) compliers are exactly the exposed treated users, so
    LATE = the effect of treatment on the treated (exposed). Complier means: E[Y(1)|C] = E[Y|T=1,D=1] and
    E[Y(0)|C] = E[Y(1)|C] - LATE, which gives a relative lift among the exposed.

    CIs: delta method, Var(a/b) ~= (Var a - 2 (a/b) Cov(a,b) + (a/b)^2 Var b) / b^2 with the arm-wise
    covariance of Y and D; and an exact stratified bootstrap -- for binary Y and D each arm's resample is
    fully described by its (D, Y) cell counts, which are Multinomial(n_arm, observed cell shares)
    (the multinomial analogue of ``binomial_bootstrap``).
    """
    y = np.asarray(y).astype(np.int64)
    d = np.asarray(d).astype(np.int64)
    t = np.asarray(t).astype(bool)
    if not (np.isin(y, [0, 1]).all() and np.isin(d, [0, 1]).all()):
        raise ValueError("wald_iv expects binary y and d (the bootstrap uses cell counts)")
    n1, n0 = int(t.sum()), int((~t).sum())
    y1, y0, d1, d0 = y[t], y[~t], d[t], d[~t]
    itt_y = y1.mean() - y0.mean()
    itt_d = d1.mean() - d0.mean()
    if itt_d == 0:
        raise ValueError("wald_iv: no first stage (assignment does not change exposure)")
    late = itt_y / itt_d
    var_y = y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0
    var_d = d1.var(ddof=1) / n1 + d0.var(ddof=1) / n0
    cov = np.cov(y1, d1)[0, 1] / n1 + (np.cov(y0, d0)[0, 1] / n0 if d0.std() > 0 else 0.0)
    se = float(np.sqrt((var_y - 2 * late * cov + late**2 * var_d) / itt_d**2))
    one_sided = int(d0.sum()) == 0
    out: dict[str, Any] = {
        "itt_y": float(itt_y), "itt_d_first_stage": float(itt_d), "late": float(late), "se_delta": se,
        "ci95_delta": _normal_ci(float(late), se, alpha), "one_sided_noncompliance": one_sided,
        "first_stage_f": float(itt_d**2 / var_d), "n": len(y),
    }
    exposed = t & (d == 1)
    if one_sided and exposed.any():
        y1c = float(y[exposed].mean())
        y0c = y1c - float(late)
        out["complier_mean_treated"] = y1c
        out["complier_mean_untreated"] = y0c
        out["complier_relative_lift"] = float(late / y0c) if y0c > 0 else None
    if rng is not None and reps > 0:
        cells_t = np.bincount(2 * d1 + y1, minlength=4) / n1
        cells_c = np.bincount(2 * d0 + y0, minlength=4) / n0
        ct = rng.multinomial(n1, cells_t, size=reps)
        cc = rng.multinomial(n0, cells_c, size=reps)
        yb1 = (ct[:, 1] + ct[:, 3]) / n1
        db1 = (ct[:, 2] + ct[:, 3]) / n1
        yb0 = (cc[:, 1] + cc[:, 3]) / n0
        db0 = (cc[:, 2] + cc[:, 3]) / n0
        late_b = (yb1 - yb0) / (db1 - db0)
        out["bootstrap"] = {"reps": reps, "method": "exact stratified bootstrap via multinomial (D,Y) cell counts",
                            "ci95_percentile": _percentile_ci(late_b, alpha), "se": float(late_b.std(ddof=1))}
        if one_sided and "complier_mean_untreated" in out:
            y1c_b = ct[:, 3] / np.maximum(ct[:, 2] + ct[:, 3], 1)
            y0c_b = y1c_b - late_b
            rel_b = late_b / y0c_b
            out["bootstrap"]["complier_mean_untreated_ci95"] = _percentile_ci(y0c_b, alpha)
            out["bootstrap"]["complier_relative_lift_ci95"] = _percentile_ci(rel_b, alpha)
    return out


# ----------------------------------------------------------------------------------------------------
# Phase 9: observational-data demonstration on a confounded subsample of the RCT
# ----------------------------------------------------------------------------------------------------

def confounded_selection_probs(z: np.ndarray, t: np.ndarray, p_treat: float, control_keep: tuple[float, float],
                               treated_keep_low: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep-probabilities that confound T with a covariate score z in [0, 1], and the implied propensity.

    Controls are kept with probability q0(z) = c_lo + (c_hi - c_lo) z (decreasing in z when c_hi < c_lo),
    treated users with q1(z) chosen so that p q1(z) + (1-p) q0(z) = K is *constant*. Consequences:
    (i) P(S=1 | X) = K does not depend on X, so the selected sample has the same covariate distribution
    as the RCT and its ATE equals the RCT ATE -- the benchmark is well defined; (ii) selection depends on
    (X, T) only, not on Y, so T is ignorable given X in the selected sample, with known propensity
    e(z) = p q1(z) / K. Treated users are over-represented among high-z (high-baseline) users.
    Returns (keep probability per row, true propensity per row).
    """
    c_lo, c_hi = control_keep
    q0 = c_lo + (c_hi - c_lo) * np.asarray(z, dtype=np.float64)
    K = p_treat * treated_keep_low + (1 - p_treat) * c_lo
    q1 = (K - (1 - p_treat) * q0) / p_treat
    if (q0.min() < 0) or (q0.max() > 1) or (q1.min() < 0) or (q1.max() > 1):
        raise ValueError("confounded_selection_probs: keep probabilities outside [0, 1]; adjust obs_demo config")
    keep = np.where(np.asarray(t).astype(bool), q1, q0)
    return keep, p_treat * q1 / K


def weighted_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """(Weighted) standardized mean difference of one covariate between arms."""
    t = np.asarray(t).astype(bool)
    w = np.ones(len(x)) if w is None else np.asarray(w)
    m1 = np.average(x[t], weights=w[t])
    m0 = np.average(x[~t], weights=w[~t])
    v1 = np.average((x[t] - m1) ** 2, weights=w[t])
    v0 = np.average((x[~t] - m0) ** 2, weights=w[~t])
    return float((m1 - m0) / np.sqrt((v1 + v0) / 2))


def ecdf_score(x: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF in (0, 1); ties (the features' point masses) share a value."""
    return (stats.rankdata(x) - 0.5) / len(x)


# ----------------------------------------------------------------------------------------------------
# Estimator catalogue (written into causal.json next to every number)
# ----------------------------------------------------------------------------------------------------

_ESTIMAND = "ATE of assignment (ad eligibility) = ITT, in this benchmark sample"
_RCT = "T randomized independently of (Y(0), Y(1)) and X; SUTVA (no interference, one version of treatment)"
_OBS = ("conditional ignorability T indep. (Y(0), Y(1)) | X (implied by randomization here); positivity "
        "0 < e(x) < 1; SUTVA")

ESTIMATORS: dict[str, dict[str, str]] = {
    "dim": {"label": "Difference in means", "treatment": "binary assignment T", "outcome_model": "none",
            "propensity": "not used", "estimand": _ESTIMAND, "assumptions": _RCT,
            "se_method": "Neyman sqrt(s1^2/n1 + s0^2/n0)",
            "limitations": "ignores covariates (not the most precise unbiased estimator); invalid without randomization"},
    "lin": {"label": "Lin regression adjustment", "treatment": "binary assignment T, interacted with centered X",
            "outcome_model": "linear in the linear design matrix, separate slopes per arm (OLS)",
            "propensity": "not used", "estimand": _ESTIMAND,
            "assumptions": _RCT + "; the linear model need not be correct (Lin 2013)",
            "se_method": "HC2 sandwich (HC0/HC1 also reported)",
            "limitations": "only the linear part of E[Y|X] reduces variance; small R^2 for rare binary outcomes"},
    "g_comp_s_lgbm": {"label": "g-computation (S-learner LightGBM)", "treatment": "T as a feature of one model",
                      "outcome_model": "LightGBM classifier on (X, T), cross-fitted", "propensity": "not used",
                      "estimand": _ESTIMAND,
                      "assumptions": _OBS + "; outcome model correctly specified (no residual correction)",
                      "se_method": "none (plug-in ML estimator; fold spread reported as descriptive only)",
                      "limitations": "regularization can shrink a small T effect toward 0 (plug-in bias)"},
    "ipw_ht_design": {"label": "IPW Horvitz-Thompson, design e = 0.85", "treatment": "binary assignment T",
                      "outcome_model": "none", "propensity": "known design value 0.85", "estimand": _ESTIMAND,
                      "assumptions": _RCT, "se_method": "influence function, e fixed",
                      "limitations": "differs from DiM only through n_T/n - 0.85; not location invariant"},
    "ipw_hajek_design": {"label": "IPW Hajek, design e = 0.85", "treatment": "binary assignment T",
                         "outcome_model": "none", "propensity": "known design value 0.85", "estimand": _ESTIMAND,
                         "assumptions": _RCT, "se_method": "linearized influence function, e fixed",
                         "limitations": "identical to the difference in means (constant weights within arm)"},
    "ipw_ht_logistic": {"label": "IPW Horvitz-Thompson, logistic e", "treatment": "binary assignment T",
                        "outcome_model": "none", "propensity": "logistic regression, cross-fitted, clipped",
                        "estimand": _ESTIMAND, "assumptions": _OBS + "; propensity model correct",
                        "se_method": "influence function, e treated as fixed (conservative)",
                        "limitations": "high variance with extreme weights; not location invariant"},
    "ipw_hajek_logistic": {"label": "IPW Hajek, logistic e", "treatment": "binary assignment T",
                           "outcome_model": "none", "propensity": "logistic regression, cross-fitted, clipped",
                           "estimand": _ESTIMAND, "assumptions": _OBS + "; propensity model correct",
                           "se_method": "linearized influence function, e treated as fixed (conservative)",
                           "limitations": "biased if the propensity model is misspecified"},
    "ipw_ht_lgbm": {"label": "IPW Horvitz-Thompson, LightGBM e", "treatment": "binary assignment T",
                    "outcome_model": "none", "propensity": "LightGBM, cross-fitted, clipped", "estimand": _ESTIMAND,
                    "assumptions": _OBS + "; propensity model consistent",
                    "se_method": "influence function, e treated as fixed (conservative)",
                    "limitations": "high variance with extreme weights; not location invariant"},
    "ipw_hajek_lgbm": {"label": "IPW Hajek, LightGBM e", "treatment": "binary assignment T",
                       "outcome_model": "none", "propensity": "LightGBM, cross-fitted, clipped", "estimand": _ESTIMAND,
                       "assumptions": _OBS + "; propensity model consistent",
                       "se_method": "linearized influence function, e treated as fixed (conservative)",
                       "limitations": "biased if the propensity model is inconsistent"},
    "ipw_hajek_lgbm_overfit": {"label": "IPW Hajek, LightGBM e without early stopping",
                               "treatment": "binary assignment T", "outcome_model": "none",
                               "propensity": "LightGBM with a fixed tree budget, no early stopping (sensitivity)",
                               "estimand": _ESTIMAND, "assumptions": _OBS + "; propensity model consistent",
                               "se_method": "linearized influence function, e treated as fixed",
                               "limitations": ("sensitivity check: noise fitted into e(x) is amplified by "
                                               "1/(1-e) in the small control arm")},
    "aipw_design": {"label": "AIPW, LightGBM mu + design e", "treatment": "binary assignment T",
                    "outcome_model": "per-arm LightGBM classifiers, cross-fitted",
                    "propensity": "known design value 0.85", "estimand": _ESTIMAND,
                    "assumptions": _RCT + "; unbiased for any outcome model only if 0.85 is the true e(x) -- "
                                   "the balance checks show it is not exactly",
                    "se_method": "efficient influence function",
                    "limitations": "gain over DiM limited by how much of Var(Y) the outcome models explain"},
    "aipw_lgbm": {"label": "AIPW, LightGBM mu + LightGBM e", "treatment": "binary assignment T",
                  "outcome_model": "per-arm LightGBM classifiers, cross-fitted",
                  "propensity": "LightGBM, cross-fitted, clipped", "estimand": _ESTIMAND,
                  "assumptions": _OBS + "; either e or mu consistent (double robustness); cross-fitting",
                  "se_method": "efficient influence function",
                  "limitations": "asymptotic SE; extreme weights inflate variance"},
    "dim_ml_sample": {"label": "Difference in means (ML subsample)", "treatment": "binary assignment T",
                      "outcome_model": "none", "propensity": "not used", "estimand": _ESTIMAND,
                      "assumptions": _RCT, "se_method": "Neyman",
                      "limitations": "reference for the ML estimators when they run on a subsample"},
}

IDENTIFICATION = (
    "By design, treatment is randomized assignment (ad eligibility), so T is independent of the potential outcomes "
    "(Y(0), Y(1)) and of X. Then E[Y|T=1] = E[Y(1)|T=1] = E[Y(1)] and E[Y|T=0] = E[Y(0)], and the difference in "
    "means E[Y|T=1] - E[Y|T=0] is an unbiased estimate of the ATE of assignment (the intention-to-treat effect) "
    "with no model of Y. SUTVA (no interference between users, one version of treatment) is also required. If "
    "randomization holds, covariates are not needed for identification and regression, IPW and AIPW only change "
    "the variance; if T depends on X in the sample (see the randomization checks), the difference in means can be "
    "biased and adjustment for X (under ignorability given X) becomes necessary rather than optional."
)
BALANCE_CAN_SHOW = (
    "Balance checks can detect a failed or leaky randomization: a treated share different from the design, "
    "features that predict assignment (as in v1 of this dataset, where the test id leaked into features), or "
    "systematic mean differences in observed features."
)
BALANCE_CANNOT_SHOW = (
    "They cannot prove independence from unobserved variables (randomization guarantees that; data cannot), "
    "cannot test SUTVA or the exclusion restriction, and a non-significant test has finite power: it only "
    "rules out imbalance the classifier could learn from this many rows. Passing them is necessary, not sufficient."
)
DATA_CAVEAT = ("Rates and effects are properties of this benchmark sample: the release is non-uniformly "
               "sub-sampled, so they are not real-world incrementality.")


def _interpretation(outcome: str, inf: dict[str, Any], alpha: float) -> str:
    r, a, z = inf["rates"], inf["absolute_effect"], inf["z_test"]
    lo, hi = a["ci95_wald"]
    level = round((1 - alpha) * 100)
    rel = inf["relative_lift"]
    rel_txt = (f", a {rel['estimate']:.1%} relative lift" if rel.get("estimate") is not None
               else " (relative lift undefined: the control rate is 0)")
    verdict = "rejects" if z["log10_p_value"] < np.log10(alpha) else "does not reject"
    return (f"In this benchmark sample the {outcome} rate was {r['treatment']:.4%} in the ad-eligible (treatment) "
            f"arm vs {r['control']:.4%} in control, a difference of {a['estimate'] * 100:+.4f} percentage points "
            f"({level}% CI {lo * 100:.4f} to {hi * 100:.4f}){rel_txt}; the z-test {verdict} H0: p_T = p_C "
            f"(z = {z['statistic']:.1f}, {_p_text(z['log10_p_value'])}). Read causally this is the effect of "
            f"assignment, which relies on randomization (see randomization.finding).")


def _power_block(p_c: float, obs: float, n_t: int, n_c: int, alpha: float, power: float) -> dict[str, Any]:
    mde = mde_two_proportions(p_c, n_t, n_c, alpha, power)
    return {"n_t": n_t, "n_c": n_c, "mde_absolute": mde,
            "mde_relative": mde / p_c if p_c > 0 else None,
            "observed_effect_over_mde": obs / mde if mde and np.isfinite(mde) else None,
            "power_at_observed_effect": power_two_proportions(p_c, obs, n_t, n_c, alpha)}


# ----------------------------------------------------------------------------------------------------
# Stage `stats`
# ----------------------------------------------------------------------------------------------------

def _balance_finding(rand: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Plain-language verdict of the randomization checks, generated from the numbers (either way)."""
    lg = rand["c2st"]["models"]["lightgbm"]
    lr = rand["c2st"]["models"]["logistic_regression"]
    lrt = rand["c2st"]["joint_lrt"]
    smd = rand["smd"]
    detected = lg["ci95_delong"][0] > 0.5 or lr["ci95_delong"][0] > 0.5 or lrt["p_value"] < alpha
    q = lg["score_quantiles"]
    numbers = (f"held-out LightGBM AUC {lg['auc']:.4f} (DeLong 95% CI {lg['ci95_delong'][0]:.4f}-"
               f"{lg['ci95_delong'][1]:.4f}, permutation p = {lg['p_value_permutation_one_sided']:.3g}); logistic AUC "
               f"{lr['auc']:.4f} ({lr['ci95_delong'][0]:.4f}-{lr['ci95_delong'][1]:.4f}); joint LR test "
               f"chi2({lrt['df']}) = {lrt['llr']:.1f}, p = {lrt['p_value']:.3g}; max |SMD| = {smd['max_abs_smd']:.4f} "
               f"= {smd['max_abs_z']:.1f} null SDs; LightGBM P(T=1|X) on the test split spans {q['q01']:.3f}-"
               f"{q['q99']:.3f} (1st-99th percentile) around the design 0.85")
    if detected:
        text = (f"Small but statistically detectable dependence of treatment on the features: {numbers}. The "
                f"paper's C2ST (p = 0.137) did not detect it, plausibly for lack of power at its sample size. "
                f"Assignment was randomized within each advertiser test, but the pooled benchmark does not behave "
                f"like a completely randomized experiment: it behaves like a conditionally randomized (stratified) "
                f"one, in which identification requires conditioning on X. Possible mechanisms (hypotheses, not "
                f"documented): per-test mixing (tests with different user populations and baseline rates pooled "
                f"after re-sampling to a common ratio) and non-uniform, label-dependent negative sampling. The "
                f"propensity spread is small, but stage `causal` shows the imbalance is prognostic, so the "
                f"covariate-adjusted estimates (Lin, AIPW with an estimated propensity) are primary and the "
                f"difference in means is the reference.")
    else:
        text = (f"No detectable dependence of treatment on the features: {numbers}. Consistent with randomization "
                f"(and with the paper's C2ST, p = 0.137); the difference in means is the primary estimate.")
    return {"imbalance_detected": bool(detected), "text": text}


def _randomization_classifiers(mode: str, cfg: dict[str, Any], rng: np.random.Generator
                               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """C2ST (LightGBM and logistic regression predicting T from X, scored on the test split) and the joint LRT."""
    scfg = cfg["stats"]
    seed = cfg["seed"]
    alpha = scfg["alpha"]
    cols = [*FEATURES, "treatment"]
    train = load_criteo(mode, split="train", columns=cols)
    fb_tree = FeatureBuilder(kind="tree").fit(train[RAW_FEATURES])
    fb_lin = FeatureBuilder(kind="linear").fit(train[RAW_FEATURES])
    tr = train.iloc[rng.permutation(len(train))[:scfg["c2st_train_rows"]]]
    del train
    val = load_criteo(mode, split="val", columns=cols)
    va = val.iloc[_subsample_idx(len(val), scfg["c2st_val_rows"], rng)]
    del val
    test = load_criteo(mode, split="test", columns=cols)
    t_tr = tr["treatment"].to_numpy()
    t_te = test["treatment"].to_numpy()
    p_const = float(t_tr.mean())

    lgbm = make_lgbm("classifier", seed=seed)
    lgbm.fit(fb_tree.transform(tr[RAW_FEATURES]), t_tr,
             eval_X=(fb_tree.transform(va[RAW_FEATURES]),), eval_y=(va["treatment"].to_numpy(),),
             callbacks=[lgb.early_stopping(scfg["c2st_early_stopping_rounds"], verbose=False)])
    s_lgbm = lgbm.predict_proba(fb_tree.transform(test[RAW_FEATURES]))[:, 1]

    X_lin_tr = fb_lin.transform(tr[RAW_FEATURES]).to_numpy(np.float64)
    logit = LogisticRegression(C=scfg["c2st_logistic_C"], max_iter=scfg["logistic_max_iter"])
    logit.fit(X_lin_tr, t_tr)
    chunk = 500_000
    s_lr = np.concatenate([logit.predict_proba(
        fb_lin.transform(test[RAW_FEATURES].iloc[i:i + chunk]).to_numpy(np.float64))[:, 1]
                           for i in range(0, len(test), chunk)])

    out: dict[str, Any] = {"train_rows": len(tr), "val_rows_early_stopping": len(va), "test_rows": len(test),
                           "constant_predictor": p_const, "paper_c2st_p_value": 0.137, "models": {}}
    plot: dict[str, Any] = {"roc": {}, "null": {}}
    for name, s in (("lightgbm", s_lgbm), ("logistic_regression", s_lr)):
        auc = auc_delong(t_te, s, alpha)
        null = auc_permutation_null(t_te, s, scfg["c2st_permutations"], rng)
        p_perm = float((1 + (null >= auc["auc"]).sum()) / (len(null) + 1))
        fpr, tpr, _ = roc_curve(t_te, s)
        grid = np.linspace(0, 1, scfg["roc_points"])
        plot["roc"][name] = (grid, np.interp(grid, fpr, tpr), auc)
        plot["null"][name] = (null, auc["auc"], p_perm)
        out["models"][name] = {**auc, "p_value_permutation_one_sided": p_perm,
                               "permutations": len(null), "null_auc_sd": float(null.std(ddof=1)),
                               "logloss_vs_constant": logloss_improvement(t_te, s, p_const, alpha),
                               "score_sd": float(s.std()), "score_range": [float(s.min()), float(s.max())],
                               "score_quantiles": {f"q{int(q * 100):02d}": float(v) for q, v in
                                                   zip((0.01, 0.05, 0.5, 0.95, 0.99),
                                                       np.quantile(s, (0.01, 0.05, 0.5, 0.95, 0.99)))}}
    out["models"]["lightgbm"]["best_iteration"] = int(lgbm.best_iteration_ or lgbm.n_estimators)
    out["models"]["logistic_regression"]["n_features"] = int(X_lin_tr.shape[1])

    # Joint likelihood-ratio test on an unpenalized logit with linearly independent columns.
    keep = independent_columns(X_lin_tr[:scfg["qr_rank_rows"]])
    n_lrt = min(scfg["lrt_rows"], len(tr))
    lrt = joint_lrt(X_lin_tr[:n_lrt][:, keep], t_tr[:n_lrt])
    lrt["columns_dropped_collinear"] = int(X_lin_tr.shape[1] - len(keep))
    out["joint_lrt"] = lrt
    return out, plot


def run_stats(mode: str = "dev") -> dict[str, Any]:
    """Stage ``stats`` (Phase 3): classical inference on the randomized assignment. See module docstring."""
    from src.visualization.causal_plots import plot_stats_balance, plot_stats_effects

    cfg = load_config()
    scfg = cfg["stats"]
    alpha, power = scfg["alpha"], scfg["power"]
    rng = np.random.default_rng(cfg["seed"])
    outcomes = scfg["outcomes"]
    reps = cfg["causal"]["bootstrap_reps"]

    df = load_criteo(mode, columns=["treatment", "split", *outcomes])
    t = df["treatment"].to_numpy().astype(bool)
    split = df["split"].to_numpy()
    is_test = split == SPLIT_CODES["test"]
    n_t, n_c = int(t.sum()), int((~t).sum())
    n_t_test, n_c_test = int((t & is_test).sum()), int((~t & is_test).sum())

    res: dict[str, Any] = {"mode": mode, "n_rows": len(df), "n_treatment": n_t, "n_control": n_c, "alpha": alpha,
                           "identification": IDENTIFICATION, "caveat": DATA_CAVEAT, "outcomes": {}}
    draws_all: dict[str, dict[str, np.ndarray]] = {}
    table = []
    for o in outcomes:
        y = df[o].to_numpy()
        x_t, x_c = int(y[t].sum()), int(y[~t].sum())
        inf = two_proportion_inference(x_t, n_t, x_c, n_c, alpha)
        boot, draws = bootstrap_effects(x_t, n_t, x_c, n_c, reps, rng, alpha)
        draws_all[o] = draws
        diff, p_c = inf["absolute_effect"]["estimate"], inf["rates"]["control"]
        z = inf["z_test"]
        entry = {
            **inf,
            "hypothesis_test": {
                "null": f"H0: p_T = p_C (assignment does not change the {o} rate)",
                "alternative": "H1: p_T != p_C (two-sided)",
                "test": "two-proportion z-test with pooled SE",
                "statistic": z["statistic"], "p_value": z["p_value"], "log10_p_value": z["log10_p_value"],
                "ci95": inf["absolute_effect"]["ci95_wald"],
                "interpretation": _interpretation(o, inf, alpha),
            },
            "bootstrap": boot,
            "power": {"alpha": alpha, "target_power": power,
                      "full_data": _power_block(p_c, diff, n_t, n_c, alpha, power),
                      "test_split": _power_block(p_c, diff, n_t_test, n_c_test, alpha, power)},
        }
        if o == "exposure":
            entry["note"] = ("Manipulation check: control users cannot be exposed by construction, so the "
                             "exposure difference is the share of treated users actually shown an ad.")
        res["outcomes"][o] = entry
        rel = inf["relative_lift"]
        table.append({
            "outcome": o, "rate_treatment": inf["rates"]["treatment"], "rate_control": p_c,
            "absolute_effect": diff, "wald_lo": inf["absolute_effect"]["ci95_wald"][0],
            "wald_hi": inf["absolute_effect"]["ci95_wald"][1],
            "newcombe_lo": inf["absolute_effect"]["ci95_newcombe"][0],
            "newcombe_hi": inf["absolute_effect"]["ci95_newcombe"][1],
            "boot_lo": boot["absolute"]["ci95_percentile"][0], "boot_hi": boot["absolute"]["ci95_percentile"][1],
            "relative_lift": rel.get("estimate"),
            "rel_delta_lo": rel.get("ci95_delta_log_rr", [None, None])[0],
            "rel_delta_hi": rel.get("ci95_delta_log_rr", [None, None])[1],
            "rel_boot_lo": boot.get("relative", {}).get("ci95_percentile", [None, None])[0],
            "rel_boot_hi": boot.get("relative", {}).get("ci95_percentile", [None, None])[1],
            "z": z["statistic"], "log10_p": z["log10_p_value"],
            "mde_full": entry["power"]["full_data"]["mde_absolute"],
            "mde_test_split": entry["power"]["test_split"]["mde_absolute"],
        })
    save_table(pd.DataFrame(table), "stats_effects")

    rand: dict[str, Any] = {"identification": IDENTIFICATION, "what_balance_checks_can_show": BALANCE_CAN_SHOW,
                            "what_they_cannot_show": BALANCE_CANNOT_SHOW}
    srm = sample_ratio_test(n_t, n_c, scfg["design_treatment_ratio"])
    srm["note"] = ("The v2 release re-sampled every incrementality test to the same treatment share, so a share "
                   "this close to the design is expected by construction; the check guards against processing bugs.")
    rand["sample_ratio"] = srm
    rand["treated_share_by_split"] = {s: float(t[split == c].mean()) for s, c in SPLIT_CODES.items()}
    del df

    with timer("stats: covariate SMDs (all rows)", log):
        feats = load_criteo(mode, columns=[*FEATURES, "treatment", "split"])
        fb_tree = FeatureBuilder(kind="tree").fit(feats.loc[feats["split"] == SPLIT_CODES["train"], RAW_FEATURES])
        smd = standardized_mean_differences(fb_tree.transform(feats[RAW_FEATURES]), feats["treatment"].to_numpy())
        del feats
    save_table(smd, "stats_balance_smd")
    rand["smd"] = {"max_abs_smd": float(smd["smd"].abs().max()), "max_abs_z": float(smd["z"].abs().max()),
                   "n_features": len(smd), "noise_sd": float(smd["se_null"].iloc[0]),
                   "n_abs_z_above_1.96": int((smd["z"].abs() > 1.96).sum()),
                   "chi2_sum_z2": float((smd["z"] ** 2).sum()),
                   "note": ("sum of z^2 is only indicative: the 20 columns (raw values and log-frequency encodings "
                            "of the same features) are correlated, so it is not exactly chi2(20)")}
    with timer("stats: classifier two-sample test + LRT", log):
        c2st, c2st_plot = _randomization_classifiers(mode, cfg, rng)
    rand["c2st"] = c2st
    rand["finding"] = _balance_finding(rand, alpha)
    res["randomization"] = rand

    figs = [plot_stats_effects(res["outcomes"], draws_all, [o for o in outcomes if o != "exposure"], alpha),
            plot_stats_balance(c2st_plot, smd, res["randomization"])]
    res["figures"] = figs
    save_json(res, "stats")
    lg = c2st["models"]["lightgbm"]
    head = {o: {"ate": res["outcomes"][o]["absolute_effect"]["estimate"],
                "ci95": res["outcomes"][o]["absolute_effect"]["ci95_wald"],
                "relative_lift": res["outcomes"][o]["relative_lift"].get("estimate")} for o in outcomes}
    head["c2st_lgbm_auc"] = lg["auc"]
    head["c2st_lgbm_p_perm"] = lg["p_value_permutation_one_sided"]
    head["lrt_p"] = c2st["joint_lrt"]["p_value"]
    return head


# ----------------------------------------------------------------------------------------------------
# Stage `causal`
# ----------------------------------------------------------------------------------------------------

def _implication(primary: dict[str, Any], gaps: dict[str, Any], est: dict[str, Any],
                 outcomes: list[str]) -> dict[str, Any]:
    """What the DiM-vs-adjusted comparison implies, generated from the numbers (either way)."""
    per: dict[str, Any] = {}
    parts = []
    for o in outcomes:
        d, a, g = est[o]["dim"], est[o]["aipw_lgbm"], gaps[o]["aipw_lgbm"]
        pb = est[o]["prognostic_balance"]["mu0"]
        per[o] = {"dim": d["estimate"], "aipw": a["estimate"], "dim_minus_aipw": -g["gap"],
                  "dim_minus_aipw_z": -g["z"], "aipw_below_dim_share": -g["gap"] / d["estimate"],
                  "dim_over_aipw_ratio": d["estimate"] / a["estimate"], "prognostic_mu0_gap": pb["estimate"]}
        parts.append(f"{o}: AIPW {a['estimate'] * 100:.4f} pp (95% CI {a['ci95'][0] * 100:.4f}-{a['ci95'][1] * 100:.4f}) "
                     f"vs difference in means {d['estimate'] * 100:.4f} pp; DiM - AIPW = {-g['gap'] * 100:.4f} pp "
                     f"(paired SE {g['se_paired'] * 100:.4f}, z = {-g['z']:.1f}), i.e. the adjusted effect is "
                     f"{-g['gap'] / d['estimate']:.0%} {'smaller' if g['gap'] < 0 else 'larger'} than DiM; the mean "
                     f"predicted untreated outcome mu0(X) differs by {pb['estimate'] * 100:+.4f} pp (SE "
                     f"{pb['se'] * 100:.4f}) between the treated and control arms")
    if primary["imbalance_detected"]:
        direction = ("treated users have higher predicted baseline outcomes, so the difference in means overstates "
                     "the effect of assignment" if per[outcomes[0]]["prognostic_mu0_gap"] > 0 else
                     "treated users have lower predicted baseline outcomes, so the difference in means understates "
                     "the effect of assignment")
        head = ("Treatment was randomized within each advertiser test, but in the pooled benchmark T is weakly "
                f"predictable from X and the imbalance is prognostic ({direction}). The data behave like a "
                "conditionally randomized (stratified) experiment: identification needs T indep. (Y(0), Y(1)) | X. ")
        tail = (" Covariate-adjusted estimates are primary. They remove bias only along the observed X. Possible "
                "mechanisms (hypotheses, not documented): per-test mixing and non-uniform, label-dependent negative "
                "sampling.")
    else:
        head, tail = "No detectable imbalance; the difference in means is primary. ", ""
    return {"text": head + "; ".join(parts) + "." + tail, "by_outcome": per}


def _choose_confounder(df: pd.DataFrame, is_train: np.ndarray, candidates: list[str], outcome: str) -> dict[str, Any]:
    """Most outcome-predictive ordered feature on the training split (|AUC - 0.5|), used to build the
    confounded sample. Chosen on train so the choice does not depend on the rows the demo evaluates on."""
    y = df.loc[is_train, outcome].to_numpy()
    aucs = {c: float(roc_auc_score(y, df.loc[is_train, c].to_numpy())) for c in candidates}
    best = max(aucs, key=lambda c: abs(aucs[c] - 0.5))
    return {"feature": best, "orientation": 1 if aucs[best] >= 0.5 else -1, "outcome": outcome,
            "auc_by_candidate": aucs}


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a / b


def _row(est_id: str, outcome: str, r: dict[str, Any], sample: str) -> dict[str, Any]:
    return {"estimator": est_id, "label": ESTIMATORS[est_id]["label"], "outcome": outcome,
            "estimate": r["estimate"], "se": r["se"], "ci_low": r["ci95"][0], "ci_high": r["ci95"][1],
            "n": r["n"], "sample": sample, "se_method": ESTIMATORS[est_id]["se_method"]}


def run_observational_demo(df: pd.DataFrame, t: np.ndarray, Y: dict[str, np.ndarray], outcomes: list[str],
                           benchmarks: dict[str, dict[str, dict[str, Any]]], fb_tree: FeatureBuilder,
                           lin_design: Callable[[np.ndarray], np.ndarray], is_train: np.ndarray,
                           cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Phase 9 demo: confound the RCT by selective dropping, then compare naive vs adjusted estimators.

    Per seed: draw ``base_rows`` users uniformly, keep each with a probability that depends on treatment and
    on the ECDF score z of a strongly outcome-predictive feature (see ``confounded_selection_probs``), so
    treated users are over-represented among high-baseline users. The selected sample has the RCT's
    covariate distribution, so its ATE is the full-RCT ATE, which is the benchmark. Estimators: naive DiM,
    Lin regression, Hajek IPW with the true (oracle) and a cross-fitted LightGBM propensity, and
    cross-fitted AIPW. Reports bias, RMSE and CI coverage across seeds against two full-data benchmarks:
    the difference in means ("dim") and AIPW with the estimated propensity ("aipw"). Two, because the full
    data itself shows a small T-X dependence: the 'known selection' propensity p q1 / K assumes P(T=1|X) =
    0.85 before selection, so it corrects only the constructed confounding (target ~ the DiM benchmark),
    while estimators that adjust for X also absorb the natural imbalance (target ~ the AIPW benchmark).
    """
    ccfg = cfg["causal_ate"]
    ocfg = ccfg["obs_demo"]
    alpha = cfg["stats"]["alpha"]
    design_p = cfg["stats"]["design_treatment_ratio"]
    overrides = ccfg["lgbm_overrides"]
    es_rounds, inner_val = ccfg["early_stopping_rounds"], ccfg["inner_val_frac"]
    conf = _choose_confounder(df, is_train, ocfg["confounder_candidates"], outcomes[0])
    feat = df[conf["feature"]].to_numpy()
    rows: list[dict[str, Any]] = []
    balance: list[dict[str, Any]] = []
    for seed in ocfg["seeds"]:
        rng = np.random.default_rng(seed)
        base = _subsample_idx(len(df), ocfg["base_rows"], rng)
        z = ecdf_score(feat[base])
        z = z if conf["orientation"] > 0 else 1.0 - z
        keep_p, e_true = confounded_selection_probs(z, t[base], design_p, tuple(ocfg["control_keep"]),
                                                    ocfg["treated_keep_low"])
        sel = rng.random(len(base)) < keep_p
        idx = base[sel]
        ts = t[idx]
        e_oracle = e_true[sel]
        X = fb_tree.transform(df[RAW_FEATURES].iloc[idx]).to_numpy(np.float32)
        strata = ts.astype(np.int64)
        for o in outcomes:
            strata = strata * 2 + Y[o][idx].astype(np.int64)
        folds = make_folds(strata, ocfg["folds"], seed)
        e_hat = clip_propensity(lgbm_propensity(X, ts, folds, seed, overrides, es_rounds, inner_val),
                                ccfg["propensity_clip"])
        zs = z[sel]
        w = np.where(ts, 1 / e_hat, 1 / (1 - e_hat))
        balance.append({"seed": seed, "n_selected": len(idx), "treated_share": float(ts.mean()),
                        "smd_confounder_raw": weighted_smd(zs, ts),
                        "smd_confounder_ipw_lgbm": weighted_smd(zs, ts, w),
                        "corr_e_hat_e_true": float(np.corrcoef(e_hat, e_oracle)[0, 1]),
                        "e_true_range": [float(e_oracle.min()), float(e_oracle.max())]})
        Ymat = np.column_stack([Y[o][idx] for o in outcomes])
        lin = lin_regression_adjustment(array_chunks(lin_design(idx), ts, Ymat, ccfg["lin_chunk_rows"]), alpha)
        for j, o in enumerate(outcomes):
            y = Y[o][idx]
            om = crossfit_outcome_models(X, ts, y, folds, seed, overrides, s_learner=False,
                                         early_stopping_rounds=es_rounds, inner_val_frac=inner_val)
            ests = {"naive_dim": difference_in_means(y, ts, alpha), "lin": lin[j],
                    "ipw_hajek_known_selection": ipw_hajek(y, ts, e_oracle, alpha),
                    "ipw_hajek_lgbm": ipw_hajek(y, ts, e_hat, alpha),
                    "aipw_lgbm": aipw(y, ts, e_hat, om["mu0"], om["mu1"], alpha)}
            for name, r in ests.items():
                row = {"seed": seed, "outcome": o, "estimator": name, "estimate": r["estimate"],
                       "se": r["se"], "ci_low": r["ci95"][0], "ci_high": r["ci95"][1], "n": len(idx)}
                for bname, bm in benchmarks.items():
                    b = bm[o]["estimate"]
                    row[f"benchmark_{bname}"] = b
                    row[f"covers_{bname}"] = bool(r["ci95"][0] <= b <= r["ci95"][1])
                rows.append(row)
        log.info("obs demo seed %d: %d selected rows, treated share %.3f", seed, len(idx), ts.mean())
    reps = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for (o, name), g in reps.groupby(["outcome", "estimator"], sort=False):
        entry: dict[str, Any] = {"mean_estimate": float(g["estimate"].mean()),
                                 "sd_estimate": float(g["estimate"].std(ddof=1)), "mean_se": float(g["se"].mean()),
                                 "sd_over_mean_se": float(g["estimate"].std(ddof=1) / g["se"].mean()),
                                 "replicates": len(g)}
        for bname in benchmarks:
            err = g["estimate"] - g[f"benchmark_{bname}"]
            entry[f"vs_{bname}"] = {
                "bias": float(err.mean()), "bias_se": float(err.std(ddof=1) / np.sqrt(len(g))),
                "rmse": float(np.sqrt((err**2).mean())), "bias_over_mean_se": float(err.mean() / g["se"].mean()),
                "coverage": float(g[f"covers_{bname}"].mean()), "covered": int(g[f"covers_{bname}"].sum())}
        summary.setdefault(o, {})[name] = entry
    out = {
        "design": ("Uniform draw of base_rows users per seed; controls kept with probability decreasing in the "
                   "confounder score z, treated users with the complementary probability so the overall keep "
                   "rate is constant in z. Selection depends on (X, T) only: T is ignorable given X in the "
                   "selected sample, its covariate distribution matches the RCT, and the RCT ATE is the target."),
        "confounder": conf, "config": ocfg,
        "benchmarks": {b: {o: {"estimate": bm[o]["estimate"], "ci95": bm[o]["ci95"]} for o in outcomes}
                       for b, bm in benchmarks.items()},
        "benchmark_note": ("Benchmarks are full-data estimates with their own sampling error, positively correlated "
                           "with each replicate (same users), so coverage is approximate."),
        "balance": balance, "summary": summary,
    }
    return out, reps


def run_causal_ate(mode: str = "dev") -> dict[str, Any]:
    """Stage ``causal`` (Phases 8-9): compare ATE estimators, estimate propensities, exposure IV, obs demo."""
    from src.visualization.causal_plots import (
        plot_causal_forest,
        plot_exposure_iv,
        plot_obs_demo,
        plot_propensity,
    )

    cfg = load_config()
    ccfg = cfg["causal_ate"]
    alpha = cfg["stats"]["alpha"]
    design_p = cfg["stats"]["design_treatment_ratio"]
    seed = cfg["seed"]
    outcomes = cfg["causal"]["outcomes"]
    K = cfg["causal"]["crossfit_folds"]
    overrides = ccfg["lgbm_overrides"]
    es_rounds, inner_val = ccfg["early_stopping_rounds"], ccfg["inner_val_frac"]
    clip = ccfg["propensity_clip"]
    rng = np.random.default_rng(seed)
    timings: dict[str, float] = {}

    df = load_criteo(mode, columns=[*FEATURES, "treatment", "exposure", "split", *outcomes])
    n = len(df)
    t = df["treatment"].to_numpy().astype(bool)
    d = df["exposure"].to_numpy()
    Y = {o: df[o].to_numpy().astype(np.float64) for o in outcomes}
    is_train = df["split"].to_numpy() == SPLIT_CODES["train"]
    fb_tree = FeatureBuilder(kind="tree").fit(df.loc[is_train, RAW_FEATURES])
    fb_lin = FeatureBuilder(kind="linear").fit(df.loc[is_train, RAW_FEATURES])
    train_idx = np.flatnonzero(is_train)
    qr_idx = train_idx[_subsample_idx(len(train_idx), cfg["stats"]["qr_rank_rows"], rng)]
    keep_cols = independent_columns(fb_lin.transform(df[RAW_FEATURES].iloc[qr_idx]).to_numpy(np.float64))
    lin_names = [fb_lin.feature_names_out_[i] for i in keep_cols]
    raw = df[RAW_FEATURES]

    def lin_design(idx: np.ndarray) -> np.ndarray:
        return fb_lin.transform(raw.iloc[idx])[lin_names].to_numpy(np.float32)

    est: dict[str, dict[str, Any]] = {o: {} for o in outcomes}
    rows: list[dict[str, Any]] = []
    full_label = f"all rows (n={n:,})"

    # 1. difference in means, and IPW with the known design propensity (identity checks)
    phis: dict[str, dict[str, np.ndarray]] = {o: {} for o in outcomes}
    for o in outcomes:
        est[o]["dim"] = difference_in_means(Y[o], t, alpha, return_phi=True)
        phis[o]["dim"] = est[o]["dim"].pop("phi")
        est[o]["ipw_ht_design"] = ipw_horvitz_thompson(Y[o], t, np.full(n, design_p), alpha)
        est[o]["ipw_hajek_design"] = ipw_hajek(Y[o], t, np.full(n, design_p), alpha)
    identity = {o: {"hajek_design_minus_dim": est[o]["ipw_hajek_design"]["estimate"] - est[o]["dim"]["estimate"],
                    "ht_design_minus_dim": est[o]["ipw_ht_design"]["estimate"] - est[o]["dim"]["estimate"]}
                for o in outcomes}
    identity["treated_share_minus_design"] = float(t.mean() - design_p)
    identity["note"] = ("Hajek IPW with a constant propensity equals the difference in means exactly (constant "
                        "weights cancel within arm); Horvitz-Thompson differs only because the realized treated "
                        "share is not exactly the design value.")

    # 2. Lin regression adjustment on all rows (chunked, two passes)
    chunk = ccfg["lin_chunk_rows"]

    def lin_chunks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for i in range(0, n, chunk):
            idx = np.arange(i, min(i + chunk, n))
            yield lin_design(idx), t[idx], np.column_stack([Y[o][idx] for o in outcomes])

    with timer("causal: Lin regression adjustment", log) as tm:
        lin = lin_regression_adjustment(lin_chunks, alpha, return_phi=True)
    timings["lin"] = tm["seconds"]
    for j, o in enumerate(outcomes):
        phis[o]["lin"] = lin[j].pop("phi")
        est[o]["lin"] = {**lin[j], "n_covariates": len(lin_names)}

    # 3. cross-fitted propensity scores and outcome models
    ml_idx = _subsample_idx(n, ccfg["ml_rows"], rng)
    ml_label = full_label if len(ml_idx) == n else f"uniform subsample (n={len(ml_idx):,})"
    t_ml = t[ml_idx]
    strata = t_ml.astype(np.int64)
    for o in outcomes:
        strata = strata * 2 + Y[o][ml_idx].astype(np.int64)
    folds = make_folds(strata, K, seed)
    X_tree = fb_tree.transform(raw.iloc[ml_idx]).to_numpy(np.float32)
    trees: dict[str, Any] = {}
    with timer("causal: LightGBM propensity (cross-fit)", log) as tm:
        info: dict[str, Any] = {}
        e_lgbm = lgbm_propensity(X_tree, t_ml, folds, seed, overrides, es_rounds, inner_val, info=info)
        trees["propensity"] = info["trees"]
    timings["propensity_lgbm"] = tm["seconds"]
    # Sensitivity: the same booster with a fixed tree budget and no early stopping (overfits T noise).
    of_cfg = ccfg.get("propensity_overfit_check")
    e_overfit = None
    if of_cfg:
        with timer("causal: LightGBM propensity without early stopping (sensitivity)", log) as tm:
            e_overfit = lgbm_propensity(X_tree, t_ml, folds, seed, {**overrides, **of_cfg})
        timings["propensity_lgbm_overfit"] = tm["seconds"]
    with timer("causal: logistic propensity (cross-fit)", log) as tm:
        e_lr = logistic_propensity(lambda i: lin_design(ml_idx[i]), t_ml, folds, seed, C=ccfg["logistic_C"],
                                   fit_rows=ccfg["logistic_fit_rows"])
    timings["propensity_logistic"] = tm["seconds"]
    prop = {"design_propensity": design_p, "clip_bounds": clip, "sample": ml_label, "folds": K,
            "lightgbm": propensity_diagnostics(e_lgbm, t_ml, clip, design_p),
            "logistic": propensity_diagnostics(e_lr, t_ml, clip, design_p)}
    if e_overfit is not None:
        prop["lightgbm_no_early_stopping"] = propensity_diagnostics(e_overfit, t_ml, clip, design_p)
        prop["lightgbm_no_early_stopping"]["config"] = {**overrides, **of_cfg}
    for key, e in (("lightgbm", e_lgbm), ("logistic", e_lr), ("lightgbm_no_early_stopping", e_overfit)):
        if e is not None:
            prop[key]["auc_delong"] = auc_delong(t_ml, e, alpha)
    prop["lightgbm"]["trees_per_fold"] = trees["propensity"]
    imbalance = prop["lightgbm"]["auc_delong"]["ci95_delong"][0] > 0.5
    primary = {
        "estimator": "aipw_lgbm" if imbalance else "dim", "imbalance_detected": bool(imbalance),
        "propensity_auc_lightgbm": prop["lightgbm"]["auc_delong"]["auc"],
        "propensity_auc_ci95": prop["lightgbm"]["auc_delong"]["ci95_delong"],
        "rule": ("Decision rule (fixed after the dev-mode balance checks flagged imbalance, before the full run): "
                 "if the cross-fitted LightGBM propensity separates the arms (DeLong 95% CI of its AUC excludes "
                 "0.5), AIPW with the estimated propensity is the primary ATE -- it removes bias from the observed "
                 "covariate imbalance under ignorability given X -- and the difference in means is the reference. "
                 "Otherwise the difference in means is primary."),
    }
    e_lgbm_c, e_lr_c = clip_propensity(e_lgbm, clip), clip_propensity(e_lr, clip)
    e_design = np.full(len(ml_idx), design_p)

    # Covariate-adjusted first stage for the IV: AIPW effect of assignment on exposure (mu0 = 0 structurally).
    d_ml = d[ml_idx].astype(np.float64)
    with timer("causal: exposure model (cross-fit)", log) as tm:
        om_d = crossfit_outcome_models(X_tree, t_ml, d_ml, folds, seed, overrides, s_learner=False,
                                       early_stopping_rounds=es_rounds, inner_val_frac=inner_val, constant_mu0=0.0)
    timings["exposure_model"] = tm["seconds"]
    trees["exposure"] = om_d["trees"]
    first_stage_aipw = aipw(d_ml, t_ml, e_lgbm_c, om_d["mu0"], om_d["mu1"], alpha, return_phi=True)
    phi_d = first_stage_aipw.pop("phi")
    del om_d
    late_adj: dict[str, Any] = {}

    for o in outcomes:
        y = Y[o][ml_idx]
        if len(ml_idx) < n:
            est[o]["dim_ml_sample"] = difference_in_means(y, t_ml, alpha, return_phi=True)
            phis[o]["dim_ml_sample"] = est[o]["dim_ml_sample"].pop("phi")
        est[o]["ipw_ht_logistic"] = ipw_horvitz_thompson(y, t_ml, e_lr_c, alpha)
        est[o]["ipw_hajek_logistic"] = ipw_hajek(y, t_ml, e_lr_c, alpha)
        est[o]["ipw_ht_lgbm"] = ipw_horvitz_thompson(y, t_ml, e_lgbm_c, alpha)
        est[o]["ipw_hajek_lgbm"] = ipw_hajek(y, t_ml, e_lgbm_c, alpha, return_phi=True)
        phis[o]["ipw_hajek_lgbm"] = est[o]["ipw_hajek_lgbm"].pop("phi")
        if e_overfit is not None:
            est[o]["ipw_hajek_lgbm_overfit"] = ipw_hajek(y, t_ml, clip_propensity(e_overfit, clip), alpha,
                                                         return_phi=True)
            phis[o]["ipw_hajek_lgbm_overfit"] = est[o]["ipw_hajek_lgbm_overfit"].pop("phi")
        with timer(f"causal: outcome models for {o} (cross-fit)", log) as tm:
            om = crossfit_outcome_models(X_tree, t_ml, y, folds, seed, overrides, s_learner=True,
                                         early_stopping_rounds=es_rounds, inner_val_frac=inner_val)
        timings[f"outcome_models_{o}"] = tm["seconds"]
        trees[f"outcome_{o}"] = om["trees"]
        est[o]["aipw_lgbm"] = aipw(y, t_ml, e_lgbm_c, om["mu0"], om["mu1"], alpha, return_phi=True)
        phis[o]["aipw_lgbm"] = est[o]["aipw_lgbm"].pop("phi")
        est[o]["aipw_design"] = aipw(y, t_ml, e_design, om["mu0"], om["mu1"], alpha, return_phi=True)
        phis[o]["aipw_design"] = est[o]["aipw_design"].pop("phi")
        tau_y, tau_d = est[o]["aipw_lgbm"]["estimate"], first_stage_aipw["estimate"]
        late = tau_y / tau_d
        phi_late = (phis[o]["aipw_lgbm"] - late * phi_d) / tau_d
        se_late = float(phi_late.std(ddof=1) / np.sqrt(len(phi_late)))
        late_adj[o] = {"late": late, "se": se_late, "ci95": _normal_ci(late, se_late, alpha),
                       "itt_y_aipw": tau_y, "itt_d_aipw": first_stage_aipw,
                       "method": ("ratio of cross-fitted AIPW effects of assignment on the outcome and on exposure; "
                                  "SE from the influence function of the ratio (delta method on paired IFs)")}
        est[o]["g_comp_s_lgbm"] = g_computation(om["s_mu0"], om["s_mu1"], folds, alpha)
        est[o]["outcome_model_fit"] = {
            "auc_mu_observed_arm": float(roc_auc_score(y, np.where(t_ml, om["mu1"], om["mu0"]))),
            "mean_cate_t_learner": float((om["mu1"] - om["mu0"]).mean()),
            "mean_cate_s_learner": float((om["s_mu1"] - om["s_mu0"]).mean()),
            "share_s_learner_cate_exactly_zero": float(((om["s_mu1"] - om["s_mu0"]) == 0).mean()),
            "trees_per_fold": om["trees"],
        }
        # Prognostic-score balance: if the arms differ in their predicted untreated outcome mu0(X), the
        # difference in means inherits that difference as bias (DiM - adjusted ATE ~ this gap).
        est[o]["prognostic_balance"] = {
            "mu0": mean_difference(om["mu0"][t_ml], om["mu0"][~t_ml], alpha),
            "mu1": mean_difference(om["mu1"][t_ml], om["mu1"][~t_ml], alpha),
            "note": ("mean cross-fitted predicted outcome, treated minus control arm; SEs treat predictions as "
                     "fixed. Under randomization both should be ~0."),
        }
        del om
    del X_tree

    ref = {o: est[o]["dim"] for o in outcomes}
    gaps: dict[str, Any] = {"note": (
        "Gap = estimator - difference in means on the same rows; SE of the gap from the paired influence functions "
        "(the estimates are strongly correlated, so their separate SEs would overstate the gap's uncertainty). "
        "Only for estimators whose influence function accounts for nuisance estimation: Lin (OLS sandwich) and "
        "cross-fitted AIPW (Neyman-orthogonal). IPW with an estimated propensity is listed as a plain difference: "
        "its gap to DiM comes entirely from e_hat, which its IF treats as fixed, so a paired SE would be invalid.")}
    for o in outcomes:
        dim_key = "dim" if len(ml_idx) == n else "dim_ml_sample"
        gaps[o] = {}
        for k in ("ipw_hajek_lgbm", "ipw_hajek_lgbm_overfit"):
            if k in est[o]:
                gaps[o][k] = {"gap": est[o][k]["estimate"] - est[o][dim_key]["estimate"], "se_paired": None,
                              "gap_over_se_dim": (est[o][k]["estimate"] - est[o][dim_key]["estimate"])
                              / est[o]["dim"]["se"]}
        for k in ("lin", "aipw_lgbm", "aipw_design"):
            if k == "lin":
                g = paired_gap(phis[o]["lin"], phis[o]["dim"], est[o]["lin"]["estimate"], est[o]["dim"]["estimate"], alpha)
            else:
                g = paired_gap(phis[o][k], phis[o][dim_key], est[o][k]["estimate"], est[o][dim_key]["estimate"], alpha)
            g["gap_over_se_dim"] = g["gap"] / est[o]["dim"]["se"]
            g["gap_relative_to_dim"] = g["gap"] / est[o]["dim"]["estimate"]
            gaps[o][k] = g
        for k, r in est[o].items():
            if k in ESTIMATORS:
                r["se_ratio_vs_dim"] = _ratio(r["se"], ref[o]["se"])
                rows.append(_row(k, o, r, full_label if r["n"] == n else ml_label))
    del phis
    primary["estimates"] = {o: est[o][primary["estimator"]] for o in outcomes}
    implication = _implication(primary, gaps, est, outcomes)

    # 7. exposure vs assignment
    with timer("causal: exposure / IV", log):
        exposure: dict[str, Any] = {
            "assumptions": {
                "random_assignment": "T randomized (instrument independent of potential outcomes and exposures)",
                "exclusion_restriction": "assignment affects visit/conversion only through ad exposure",
                "monotonicity": "no defiers; trivially true because control users cannot be exposed",
                "relevance": "first stage P(E=1|T=1) - P(E=1|T=0) != 0",
                "sutva": "no interference between users"},
            "why_naive_is_biased": (
                "Exposure is not randomized: among eligible users, those who are online more (and so more likely "
                "to visit or buy anyway) are more likely to be shown an ad. Exposed-vs-control compares these "
                "high-baseline users with an average user; exposed-vs-unexposed-treated compares them with the "
                "low-activity users who were never reached. Both mix the ad effect with baseline differences "
                "(E[Y(0)|exposed] > E[Y(0)]); the IV uses only the randomized contrast and rescales it by the "
                "first stage."),
        }
        exp_t, unexp_t = t & (d == 1), t & (d == 0)
        for o in outcomes:
            y = Y[o]
            iv = wald_iv(y, t, d, alpha, cfg["causal"]["bootstrap_reps"], rng)
            naive_c = mean_difference(y[exp_t], y[~t], alpha)
            naive_u = mean_difference(y[exp_t], y[unexp_t], alpha)
            exposure[o] = {"itt": est[o]["dim"], "iv_wald": iv, "iv_wald_covariate_adjusted": late_adj[o],
                           "itt_aipw": est[o]["aipw_lgbm"],
                           "naive_exposed_vs_control": naive_c,
                           "naive_exposed_vs_unexposed_treated": naive_u,
                           "selection_bias_exposed_vs_control": naive_c["estimate"] - iv["late"],
                           "selection_bias_exposed_vs_unexposed": naive_u["estimate"] - iv["late"],
                           "rates": {"exposed_treated": float(y[exp_t].mean()),
                                     "unexposed_treated": float(y[unexp_t].mean()),
                                     "control": float(y[~t].mean())}}

    # 8. observational-data demonstration
    with timer("causal: observational demo", log) as tm:
        benchmarks = {"dim": ref, "aipw": {o: est[o]["aipw_lgbm"] for o in outcomes}}
        obs, obs_reps = run_observational_demo(df, t, Y, outcomes, benchmarks, fb_tree, lin_design, is_train, cfg)
    timings["observational_demo"] = tm["seconds"]
    save_table(obs_reps, "causal_obs_demo_replicates")

    table = pd.DataFrame(rows)
    save_table(table, "causal_ate_estimates")
    figs = [plot_causal_forest(table, outcomes, primary["estimator"]),
            plot_propensity(e_lgbm, e_lr, t_ml, prop, e_overfit),
            plot_obs_demo(obs_reps, obs, outcomes), plot_exposure_iv(exposure, outcomes)]
    res = {
        "mode": mode, "n_rows": n, "alpha": alpha, "identification": IDENTIFICATION, "caveat": DATA_CAVEAT,
        "note_on_sample": ("RCT estimators use every row: this is estimation of one population quantity, not "
                           "prediction, so there is no train/test split to respect. Cross-fitting plays the "
                           "role of the held-out split for the ML nuisance models."),
        "ml_sample": ml_label, "crossfit_folds": K, "lgbm_overrides": overrides,
        "nuisance_tuning": {"early_stopping_rounds": es_rounds, "inner_val_frac": inner_val, "trees": trees,
                            "note": ("tree counts chosen by early stopping on a random inner split of each "
                                     "training fold; the held-out fold is never used for fitting or tuning")},
        "linear_design_columns": len(lin_names),
        "primary": primary, "implication": implication, "gaps_vs_dim": gaps,
        "estimators": ESTIMATORS, "estimates": est, "identity_checks": identity, "propensity": prop,
        "exposure": exposure, "observational_demo": obs, "timings_seconds": timings, "figures": figs,
    }
    save_json(res, "causal")
    head: dict[str, Any] = {}
    for o in outcomes:
        head[o] = {k: {"est": est[o][k]["estimate"], "se": est[o][k]["se"]}
                   for k in ("dim", "lin", "aipw_lgbm", "g_comp_s_lgbm")}
        head[o]["late"] = exposure[o]["iv_wald"]["late"]
        head[o]["late_adjusted"] = exposure[o]["iv_wald_covariate_adjusted"]["late"]
        head[o]["aipw_minus_dim_z"] = gaps[o]["aipw_lgbm"]["z"]
    head["primary"] = primary["estimator"]
    head["propensity_auc"] = primary["propensity_auc_lightgbm"]
    return head
