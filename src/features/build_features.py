"""Feature construction shared by every model.

The 12 Criteo features are anonymized: 4 are continuous and 8 are hashed categoricals whose values
are random projections of the original tokens (so their numeric *order* carries no meaning). Two
design matrices are produced from the same fitted state:

* ``kind="tree"``  -- raw values plus a log-frequency encoding of each categorical. Trees can split
  on the raw projected values; the frequency encoding gives them an ordered, meaningful axis.
* ``kind="linear"`` -- clipped + standardized continuous features, standardized log-frequency
  encodings, and one-hot indicators for the most frequent levels of each categorical.

All statistics (frequencies, clip bounds, means/SDs, retained levels) are learned from the data
passed to ``fit`` -- always the training split -- and no label is used, so there is no target leakage.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import load_config

_FEAT_CFG = load_config()["features"]
CONTINUOUS: list[str] = list(_FEAT_CFG["continuous"])
CATEGORICAL: list[str] = list(_FEAT_CFG["categorical"])
RAW_FEATURES: list[str] = [f"f{i}" for i in range(12)]

Kind = Literal["tree", "linear"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Label-free feature transformer (sklearn API). Fit on the training split only.

    Args:
        kind: ``"tree"`` or ``"linear"`` design matrix (see module docstring).
        min_category_count: levels rarer than this in the fit data are pooled (no one-hot column).
        max_onehot_levels: at most this many one-hot columns per categorical (linear design only).
        clip_quantiles: continuous features are clipped to these fit-data quantiles (linear only).
    """

    def __init__(self, kind: Kind = "tree", min_category_count: int | None = None,
                 max_onehot_levels: int | None = None, clip_quantiles: tuple[float, float] = (0.001, 0.999)):
        self.kind = kind
        self.min_category_count = min_category_count
        self.max_onehot_levels = max_onehot_levels
        self.clip_quantiles = clip_quantiles

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureBuilder:
        if self.kind not in ("tree", "linear"):
            raise ValueError(f"kind must be 'tree' or 'linear', got {self.kind!r}")
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"FeatureBuilder.fit: missing columns {sorted(missing)}")
        min_count = self.min_category_count if self.min_category_count is not None else _FEAT_CFG["min_category_count"]
        max_levels = self.max_onehot_levels if self.max_onehot_levels is not None else _FEAT_CFG["max_onehot_levels"]
        n = len(X)
        self.n_fit_rows_ = n
        self.freq_maps_: dict[str, pd.Series] = {}
        self.onehot_levels_: dict[str, np.ndarray] = {}
        for c in CATEGORICAL:
            counts = X[c].value_counts()
            self.freq_maps_[c] = np.log1p(counts)            # log count; unseen levels map to 0
            kept = counts[counts >= min_count].index.to_numpy()[:max_levels]
            self.onehot_levels_[c] = kept
        lo, hi = self.clip_quantiles
        self.clip_ = {c: (float(X[c].quantile(lo)), float(X[c].quantile(hi))) for c in CONTINUOUS}
        # Standardization statistics for the linear design, computed on the clipped / encoded fit data.
        enc = self._encode_numeric(X)
        self.mean_ = enc.mean()
        self.std_ = enc.std().replace(0.0, 1.0)
        self.feature_names_out_ = list(self.transform(X.iloc[:1]).columns)
        return self

    def _logfreq(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {f"{c}_logfreq": X[c].map(self.freq_maps_[c]).fillna(0.0).to_numpy() for c in CATEGORICAL},
            index=X.index,
        )

    def _encode_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        cont = pd.DataFrame({c: X[c].clip(*self.clip_[c]) for c in CONTINUOUS}, index=X.index)
        return pd.concat([cont, self._logfreq(X)], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "freq_maps_"):
            raise RuntimeError("FeatureBuilder must be fit before transform")
        if self.kind == "tree":
            out = pd.concat([X[RAW_FEATURES], self._logfreq(X)], axis=1)
            return out.astype(np.float32)
        num = (self._encode_numeric(X) - self.mean_) / self.std_
        dummies = {}
        for c in CATEGORICAL:
            vals = X[c].to_numpy()
            for j, level in enumerate(self.onehot_levels_[c]):
                dummies[f"{c}_is_{j}"] = (vals == level)
        out = pd.concat([num, pd.DataFrame(dummies, index=X.index)], axis=1)
        return out.astype(np.float32)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_)


def build_matrices(train: pd.DataFrame, *others: pd.DataFrame, kind: Kind = "tree"
                   ) -> tuple[FeatureBuilder, list[pd.DataFrame]]:
    """Fit a ``FeatureBuilder`` on ``train`` and transform ``train`` plus any other frames."""
    fb = FeatureBuilder(kind=kind).fit(train[RAW_FEATURES])
    return fb, [fb.transform(df[RAW_FEATURES]) for df in (train, *others)]


def select_features(fb: FeatureBuilder, drop_constant: bool = True, X_ref: pd.DataFrame | None = None) -> list[str]:
    """Feature selection used before modeling: drop columns that are constant on the reference data.

    With the linear design, rare one-hot levels can be all-zero in small samples; they carry no
    information and make coefficients unidentifiable.
    """
    names = list(fb.feature_names_out_)
    if not drop_constant or X_ref is None:
        return names
    std = X_ref[names].std()
    return [c for c in names if std[c] > 0]
