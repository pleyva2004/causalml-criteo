"""Causal forest (EconML ``CausalForestDML``) for the CATE of a randomized binary treatment.

Method (Athey, Tibshirani & Wager 2019, "Generalized Random Forests"; the DML variant of
Chernozhukov et al. 2018): outcomes and treatment are first residualized with cross-fitted nuisance
models, ``Y - m(X)`` and ``T - e(X)``; the forest then splits to maximise heterogeneity of the local
residual-on-residual slope and estimates ``tau(x)`` from the units that share leaves with ``x``.

Assumptions: (i) unconfoundedness given X -- true by design here (randomized assignment), and the
propensity model only has to absorb the small chance imbalance seen in the EDA; (ii) overlap --
every user had an 85% chance of treatment; (iii) SUTVA. ``honest=True`` uses disjoint halves of
each tree's subsample for choosing splits and for estimating leaf effects, which is what makes the
forest's confidence intervals valid.

Scale: GRF's split search is far slower than LightGBM histograms, so the forest is trained on
``causal.causal_forest_train_rows`` training rows (see config) and predicts every val/test row in
chunks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from scipy import stats

from src.models.tree_models import make_lgbm
from src.utils import get_logger, load_config

log = get_logger(__name__)


def make_causal_forest(seed: int, params: dict[str, Any] | None = None) -> CausalForestDML:
    """``CausalForestDML`` with LightGBM nuisances (outcome classifier, propensity classifier)."""
    cfg = load_config()
    p = {**cfg["cate"]["causal_forest"], **(params or {})}
    nuisance = p.pop("nuisance_model")
    n_jobs = cfg["compute"]["n_jobs"]
    return CausalForestDML(
        model_y=make_lgbm("classifier", seed=seed, **nuisance),
        model_t=make_lgbm("classifier", seed=seed, **nuisance),
        discrete_outcome=True, discrete_treatment=True,
        cv=p["cv"], n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"],
        max_samples=p["max_samples"], honest=True, inference=True, drate=True,
        n_jobs=n_jobs, random_state=seed,
    )


def fit_causal_forest(X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int,
                      params: dict[str, Any] | None = None) -> CausalForestDML:
    """Fit on the given rows (the caller subsamples; see module docstring)."""
    model = make_causal_forest(seed, params)
    model.fit(np.asarray(y), np.asarray(t), X=X.to_numpy(dtype=np.float64))
    return model


def predict_cate(model: CausalForestDML, X: pd.DataFrame, chunk_rows: int) -> np.ndarray:
    """Point CATE predictions in chunks (bounded memory on millions of rows)."""
    arr = X.to_numpy(dtype=np.float64)
    out = [np.asarray(model.effect(arr[i:i + chunk_rows])).ravel() for i in range(0, len(arr), chunk_rows)]
    return np.concatenate(out).astype(np.float32)


def forest_ate(model: CausalForestDML, alpha: float = 0.05) -> dict[str, float]:
    """Doubly robust ATE on the forest's training rows (``drate=True``) with a normal CI.

    A sanity check: it should agree with the difference in means up to sampling error.
    """
    ate = float(np.asarray(model.ate_).ravel()[0])
    se = float(np.asarray(model.ate_stderr_).ravel()[0])
    z = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": ate, "se": se, "ci_low": ate - z * se, "ci_high": ate + z * se}


def forest_importances(model: CausalForestDML, feature_names: list[str]) -> pd.Series:
    """Split-based heterogeneity importance (how much each feature's splits increased effect
    heterogeneity), normalised to sum to 1."""
    imp = np.asarray(model.feature_importances_, dtype=np.float64).ravel()
    s = pd.Series(imp, index=feature_names)
    return s / s.sum() if s.sum() > 0 else s
