"""Unit tests for the EDA computations on small synthetic frames with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import FEATURES
from src.data.preprocess import (
    balance_table,
    duplicate_counts,
    exposure_funnel,
    rates_by_group,
    spearman_matrix,
    standardized_mean_difference,
    structural_violations,
    wilson_ci,
)


def _frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for f in ("f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"):
        df[f] = rng.integers(0, 5, size=n).astype(float)
    df["treatment"] = rng.integers(0, 2, size=n).astype("int8")
    df["exposure"] = ((df.treatment == 1) & (rng.random(n) < 0.3)).astype("int8")
    df["visit"] = (rng.random(n) < 0.05 + 0.3 * df.exposure).astype("int8")
    df["conversion"] = ((df.visit == 1) & (rng.random(n) < 0.1)).astype("int8")
    return df


def test_smd_identical_groups_is_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert standardized_mean_difference(x, x.copy()) == 0.0


def test_smd_known_value_and_constant():
    a, b = np.array([1.0, 3.0]), np.array([0.0, 2.0])   # means differ by 1, both var = 2
    assert standardized_mean_difference(a, b) == pytest.approx(1 / np.sqrt(2))
    assert standardized_mean_difference(np.ones(5), np.ones(5)) == 0.0


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(5, 10)            # textbook: 5/10 -> (0.2366, 0.7634)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo0, hi0 = wilson_ci(0, 100)         # zero successes: lower bound 0, upper > 0
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.05
    assert np.isnan(wilson_ci(0, 0)[0])


def test_rates_by_group_counts():
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 6, "visit": [1, 0, 0, 0, 1, 1, 0, 0, 0, 0]})
    r = rates_by_group(df, "g", ["visit"]).set_index("g")
    assert r.loc["a", "n"] == 4 and r.loc["a", "visit_rate"] == 0.25
    assert r.loc["b", "visit_count"] == 2
    assert (r.visit_ci_low < r.visit_rate).all() and (r.visit_ci_high > r.visit_rate).all()


def test_exposure_funnel_counts():
    df = pd.DataFrame({
        "treatment": [0, 0, 1, 1, 1, 1], "exposure": [0, 0, 1, 1, 0, 0],
        "visit": [0, 1, 1, 0, 1, 0], "conversion": [0, 0, 1, 0, 0, 0],
    })
    f = exposure_funnel(df).set_index("group")
    assert list(f.n) == [2, 4, 2, 2]
    assert f.loc["control", "exposure_count"] == 0
    assert f.loc["treated", "exposure_rate"] == 0.5
    assert f.loc["treated_exposed", "conversion_count"] == 1
    assert f.loc["treated_unexposed", "visit_count"] == 1


def test_duplicate_counting():
    df = _frame(100)
    dup = pd.concat([df, df.iloc[:7]], ignore_index=True)
    c = duplicate_counts(dup)
    assert c["duplicate_rows_features_and_labels"] == 7
    assert c["duplicate_rows_features_only"] >= 7
    # same features, different label -> counted as a feature-only duplicate only
    alt = df.iloc[:1].copy()
    alt["conversion"] = 1 - alt["conversion"]
    c2 = duplicate_counts(pd.concat([df, alt], ignore_index=True))
    assert c2["duplicate_rows_features_and_labels"] == 0 and c2["duplicate_rows_features_only"] == 1


def test_structural_violations_detected():
    df = _frame(500)
    assert structural_violations(df) == {"control_exposed": 0, "conversion_without_visit": 0}
    df.loc[df.index[df.treatment == 0][0], "exposure"] = 1
    assert structural_violations(df)["control_exposed"] == 1


def test_balance_table_randomized_frame_is_balanced():
    df = _frame(6000)
    cont, cat = ["f0", "f2", "f7", "f10"], ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
    bal = balance_table(df, cont, cat, min_level_count=50)
    assert list(bal.feature) == FEATURES
    assert (bal.abs_smd < 0.15).all()
    assert bal.loc[bal.type == "continuous", "ks_stat"].notna().all()
    assert bal.loc[bal.type == "categorical", "chi2_pvalue"].notna().all()


def test_balance_detects_shift():
    df = _frame(4000)
    df.loc[df.treatment == 1, "f0"] += 1.0
    bal = balance_table(df, ["f0", "f2", "f7", "f10"], ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"], min_level_count=50)
    assert bal.set_index("feature").loc["f0", "abs_smd"] > 0.8


def test_spearman_matrix_shape_and_symmetry():
    df = _frame(1000)
    m = spearman_matrix(df, ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"])
    assert m.shape == (16, 16)
    assert np.allclose(m.to_numpy(), m.to_numpy().T, equal_nan=True)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)


def test_mode_coupling_detects_shared_default():
    from src.data.preprocess import mode_coupling
    df = _frame(1000)
    default = np.random.default_rng(1).random(1000) < 0.8
    df.loc[default, "f0"] = 7.0
    df.loc[default, "f1"] = 3.0
    df.loc[~default, "f1"] = 1.0
    m = mode_coupling(df).set_index(["feature_a", "feature_b"])
    assert m.loc[("f0", "f1"), "jaccard"] == pytest.approx(1.0)
