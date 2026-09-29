"""Tests for the data loader's pure functions and the feature builder (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import SPLIT_CODES, assign_splits, stratified_subsample, validate_raw
from src.features.build_features import CATEGORICAL, CONTINUOUS, RAW_FEATURES, FeatureBuilder, build_matrices


def _synthetic(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in CONTINUOUS})
    for j, c in enumerate(CATEGORICAL):
        levels = rng.normal(size=5 + j)                     # projected category values
        df[c] = rng.choice(levels, size=n, p=np.r_[0.5, np.full(4 + j, 0.5 / (4 + j))])
    df = df[RAW_FEATURES]
    df["treatment"] = (rng.random(n) < 0.85).astype(np.int8)
    df["visit"] = (rng.random(n) < 0.05).astype(np.int8)
    df["conversion"] = (df["visit"] & (rng.random(n) < 0.1)).astype(np.int8)
    df["exposure"] = (df["treatment"] & (rng.random(n) < 0.04)).astype(np.int8)
    return df


def test_validate_raw_counts_structural_violations() -> None:
    df = _synthetic()
    assert validate_raw(df, expected_rows=len(df))["control_exposed"] == 0
    df.loc[df.index[df.treatment == 0][0], "exposure"] = 1
    df.loc[df.index[df.visit == 0][0], "conversion"] = 1
    checks = validate_raw(df, expected_rows=None)
    assert checks["control_exposed"] == 1 and checks["conversion_without_visit"] == 1


def test_validate_raw_rejects_bad_schema() -> None:
    df = _synthetic(1000)
    with pytest.raises(ValueError, match="Expected"):
        validate_raw(df, expected_rows=999)
    df.loc[0, "visit"] = 2
    with pytest.raises(ValueError, match="non-binary"):
        validate_raw(df, expected_rows=None)


def test_assign_splits_is_stratified_and_deterministic() -> None:
    df = _synthetic(50_000)
    fr = {"train": 0.6, "val": 0.2, "test": 0.2}
    a, b = assign_splits(df, fr, seed=1), assign_splits(df, fr, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, assign_splits(df, fr, seed=2))
    shares = pd.Series(a).value_counts(normalize=True)
    assert shares[SPLIT_CODES["train"]] == pytest.approx(0.6, abs=1e-3)
    # the treatment ratio and outcome rates are preserved in every split
    for code in SPLIT_CODES.values():
        part = df[a == code]
        assert part.treatment.mean() == pytest.approx(df.treatment.mean(), abs=2e-3)
        assert part.visit.mean() == pytest.approx(df.visit.mean(), abs=2e-3)


def test_assign_splits_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        assign_splits(_synthetic(100), {"train": 0.7, "val": 0.2, "test": 0.2}, seed=0)


def test_stratified_subsample_fraction() -> None:
    df = _synthetic(40_000)
    df["split"] = assign_splits(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    sub = stratified_subsample(df, 0.1, seed=0)
    assert len(sub) == pytest.approx(4_000, rel=0.01)
    assert sub.treatment.mean() == pytest.approx(df.treatment.mean(), abs=3e-3)


def test_feature_builder_tree_design() -> None:
    df = _synthetic()
    fb, (X,) = build_matrices(df, kind="tree")
    assert list(X.columns[:12]) == RAW_FEATURES
    assert X.shape[1] == 12 + len(CATEGORICAL)
    assert X.dtypes.eq(np.float32).all() and not X.isna().any().any()


def test_feature_builder_linear_design_standardized_and_label_free() -> None:
    train, test = _synthetic(seed=0), _synthetic(seed=1)
    fb = FeatureBuilder(kind="linear", min_category_count=10, max_onehot_levels=3).fit(train)
    Xtr = fb.transform(train)
    num_cols = CONTINUOUS + [f"{c}_logfreq" for c in CATEGORICAL]
    assert np.allclose(Xtr[num_cols].mean(), 0, atol=1e-4)
    # labels never influence the fit: permuting outcomes leaves the transform unchanged
    shuffled = train.assign(conversion=np.random.default_rng(3).permutation(train.conversion.to_numpy()))
    Xs = FeatureBuilder(kind="linear", min_category_count=10, max_onehot_levels=3).fit(shuffled).transform(test)
    assert Xs.equals(fb.transform(test))
    onehots = [c for c in Xtr.columns if "_is_" in c]
    assert len(onehots) == 3 * len(CATEGORICAL)


def test_feature_builder_unseen_levels_map_to_zero_frequency() -> None:
    train = _synthetic()
    fb = FeatureBuilder(kind="tree").fit(train)
    new = train.head(5).copy()
    new[CATEGORICAL[0]] = 12345.678                           # level never seen in training
    assert (fb.transform(new)[f"{CATEGORICAL[0]}_logfreq"] == 0).all()


def test_feature_builder_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        FeatureBuilder().transform(_synthetic(10))
