"""Logistic regression baseline for P(conversion | X): the interpretable reference model.

L2-regularised logistic regression on the *linear* design matrix (clipped + standardized continuous
features, standardized log-frequency encodings, one-hot indicators of frequent levels). No class
weights, so the output is a probability on the natural scale. Treatment is NOT a feature: this is
the "who converts" model, not a treatment-effect model.

The feature builder is fit on the rows the model is trained on (label-free), so wrapping both in one
object lets cross-validation refit the design inside every fold without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.build_features import RAW_FEATURES, FeatureBuilder, select_features
from src.utils import get_logger, load_config

log = get_logger(__name__)


class LogisticBaseline:
    """FeatureBuilder(linear) + L2 logistic regression.

    Args:
        C: inverse regularisation strength (sklearn convention: the loss is *summed* over rows, so
            with millions of rows even C=0.01 is a very weak penalty).
        max_iter: L-BFGS iterations.
    """

    def __init__(self, C: float = 1.0, max_iter: int | None = None):
        self.C = C
        self.max_iter = max_iter if max_iter is not None else load_config()["predictive"]["lr_max_iter"]

    def fit(self, df_raw: pd.DataFrame, y: np.ndarray) -> "LogisticBaseline":
        self.fb_ = FeatureBuilder(kind="linear").fit(df_raw[RAW_FEATURES])
        X = self.fb_.transform(df_raw[RAW_FEATURES])
        # Columns constant on the training rows (rare one-hot levels in small samples) are dropped.
        self.cols_ = select_features(self.fb_, X_ref=X)
        self.model_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter)
        self.model_.fit(X[self.cols_].to_numpy(), np.asarray(y))
        if self.model_.n_iter_[0] >= self.max_iter:
            log.warning("LogisticBaseline (C=%g) hit max_iter=%d without converging", self.C, self.max_iter)
        return self

    def design(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Linear design matrix restricted to the columns the model uses."""
        return self.fb_.transform(df_raw[RAW_FEATURES])[self.cols_]

    def predict_proba1(self, df_raw: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
        """P(conversion = 1) as float32, computed in chunks to bound memory."""
        out = np.empty(len(df_raw), dtype=np.float32)
        for s in range(0, len(df_raw), chunk):
            X = self.design(df_raw.iloc[s:s + chunk]).to_numpy()
            out[s:s + chunk] = self.model_.predict_proba(X)[:, 1]
        return out
