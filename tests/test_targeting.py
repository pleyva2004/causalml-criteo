"""Targeting tests on synthetic randomized data with a known heterogeneous effect."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.targeting import (
    budget_mask,
    dim_value,
    eiv_mask,
    evaluate_masks,
    ht_value,
    ipw_value,
    rank_of,
    tiebreak_key,
    value_metrics,
)

P = 0.85


def make_data(n: int, seed: int):
    """Binary outcome, tau(x) = 0.02 + 0.08*x1 with x1 in {0,1} (30% high responders)."""
    rng = np.random.default_rng(seed)
    x1 = (rng.random(n) < 0.3).astype(int)
    base = 0.03 + 0.02 * x1
    tau = 0.02 + 0.08 * x1
    t = (rng.random(n) < P).astype(int)
    y = (rng.random(n) < base + tau * t).astype(int)
    return x1, tau, t, y


def test_ipw_unbiased_for_known_policy_value():
    """Policy 'treat x1==1': true value = n * E[1{x1=1} tau] = n * 0.3 * 0.10."""
    n, est = 100_000, []
    for seed in range(20):
        x1, _, t, y = make_data(n, seed)
        est.append(ht_value(x1 == 1, t, y, P) / n)
    truth = 0.3 * 0.10
    se = np.std(est, ddof=1) / np.sqrt(len(est))
    assert abs(np.mean(est) - truth) < 4 * se + 1e-4


def make_confounded(n: int, seed: int):
    """Propensity depends on x1 (0.6 vs 0.9) and so does the baseline rate: constant-p DiM is biased for
    'treat all' (true value per user 0.02 + 0.08 * 0.3 = 0.044)."""
    rng = np.random.default_rng(seed)
    x1 = (rng.random(n) < 0.3).astype(int)
    e = np.where(x1 == 1, 0.6, 0.9)
    t = (rng.random(n) < e).astype(int)
    y = (rng.random(n) < 0.03 + 0.20 * x1 + (0.02 + 0.08 * x1) * t).astype(int)
    return x1, e, t, y


def test_ipw_recovers_value_where_constant_p_dim_is_biased():
    n = 400_000
    x1, e, t, y = make_confounded(n, 0)
    allm = np.ones(n, dtype=bool)
    truth = 0.044 * n
    assert ipw_value(allm, t, y, e) == pytest.approx(truth, rel=0.05)
    assert ht_value(allm, t, y, e) == pytest.approx(truth, rel=0.06)
    assert abs(dim_value(allm, t, y) - truth) > 0.25 * truth          # confounded DiM is off by a lot


def test_hajek_and_ht_agree_with_known_propensity():
    n = 400_000
    x1, e, t, y = make_confounded(n, 1)
    assert ipw_value(x1 == 1, t, y, e) == pytest.approx(ht_value(x1 == 1, t, y, e), rel=0.05)


def test_budget_masks_exact_and_deterministic_with_ties():
    n = 1000
    score = np.round(np.random.default_rng(0).random(n), 1)          # heavy ties
    tie = tiebreak_key(n, 3)
    for frac in (0.05, 0.1, 0.3, 1.0):
        m1, m2 = budget_mask(score, frac, tie), budget_mask(score, frac, tie)
        assert m1.sum() == round(frac * n)
        assert (m1 == m2).all()
    m = budget_mask(score, 0.2, tie)
    assert score[m].min() >= score[~m].max()                          # top scores are selected


def test_eiv_policy_never_treats_negative_eiv():
    rng = np.random.default_rng(0)
    tau = rng.normal(0.001, 0.003, 5000)
    value, cost = 50.0, 0.05                                          # cutoff tau > 0.001
    tie = tiebreak_key(len(tau), 0)
    for frac in (0.1, 0.5, 1.0):
        mask = eiv_mask(tau, frac, value, cost, tie)
        assert (tau[mask] * value - cost > 0).all()
        assert mask.sum() <= round(frac * len(tau))
    assert eiv_mask(tau, 1.0, value, 1e9, tie).sum() == 0


def test_treat_none_zero_and_paired_self_difference_zero():
    x1, _, t, y = make_data(20_000, 2)
    none = np.zeros(len(t), dtype=bool)
    assert ht_value(none, t, y, P) == 0.0
    ev = evaluate_masks({"a": x1 == 1, "b": x1 == 1, "none": none}, t, y, P, 200, np.random.default_rng(0))
    assert (ev.boot_inc["a"] - ev.boot_inc["b"] == 0).all()
    assert ev.inc["none"] == 0.0 and (ev.boot_inc["none"] == 0).all()


def test_random_k_is_k_times_treat_all():
    n = 300_000
    _, _, t, y = make_data(n, 4)
    p = t.mean()
    z_all = ht_value(np.ones(n, bool), t, y, p)
    for k in (0.1, 0.3):
        vals = []
        for s in range(10):
            mask = np.random.default_rng(s).random(n) < k
            vals.append(ht_value(mask, t, y, p))
        assert np.mean(vals) == pytest.approx(k * z_all, rel=0.15)


def test_bootstrap_matches_row_bootstrap_and_covers_truth():
    """Poisson cell bootstrap SE should match a direct row bootstrap of the Hajek-IPW estimator."""
    n = 30_000
    x1, e, t, y = make_confounded(n, 5)
    mask = x1 == 1
    ev = evaluate_masks({"m": mask}, t, y, e, 400, np.random.default_rng(1), prop_bins=5)
    rng = np.random.default_rng(2)
    direct = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        direct.append(ipw_value(mask[i], t[i], y[i], e[i]))
    assert np.std(ev.boot_inc["m"]) == pytest.approx(np.std(direct), rel=0.25)
    truth = n * 0.3 * 0.10
    lo, hi = np.percentile(ev.boot_inc["m"], [0.5, 99.5])
    assert lo < truth < hi


def test_targeting_beats_random_when_tau_known():
    n = 200_000
    x1, tau, t, y = make_data(n, 6)
    tie = tiebreak_key(n, 0)
    rank = rank_of(tau, tie)
    masks = {"uplift": rank < int(0.2 * n), "random": np.random.default_rng(0).random(n) < 0.2}
    ev = evaluate_masks(masks, t, y, P, 300, np.random.default_rng(3))
    d = ev.boot_inc["uplift"] - ev.boot_inc["random"]
    assert np.percentile(d, 2.5) > 0
    m = value_metrics(ev.inc["uplift"], ev.n_treated["uplift"], n, 50.0, 0.05)
    assert m["inc_per_treated"] > 0.05
