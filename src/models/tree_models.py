"""Tree-ensemble model factories (LightGBM, random forest).

``make_lgbm`` is shared by the predictive stage and by the causal meta-learners so that every
outcome model in the project uses the same, documented base learner.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import load_config

# Conservative defaults for 12 features and millions of rows; tuned values (if any) are passed in.
LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def make_lgbm(kind: str = "classifier", seed: int | None = None, **overrides: Any) -> lgb.LGBMModel:
    """LightGBM classifier or regressor with project defaults, seed and thread count from config."""
    cfg = load_config()
    params = {**LGBM_DEFAULTS, "random_state": cfg["seed"] if seed is None else seed,
              "n_jobs": cfg["compute"]["n_jobs"], **overrides}
    if kind == "classifier":
        return lgb.LGBMClassifier(**params)
    if kind == "regressor":
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"kind must be 'classifier' or 'regressor', got {kind!r}")


def make_rf(n_estimators: int, min_samples_leaf: int, seed: int | None = None, **overrides: Any) -> RandomForestClassifier:
    """Random forest for rare positives, parameterised from ``predictive.rf`` in the config.

    Large ``min_samples_leaf`` keeps leaf frequencies from being all-zero or driven by one or two
    positives (the base rate is ~0.3%); ``max_samples`` bounds the per-tree bootstrap so forests on
    millions of rows stay tractable. Class weights are left at None so probabilities stay on the
    natural scale.
    """
    cfg = load_config()
    rf = cfg["predictive"]["rf"]
    params: dict[str, Any] = {
        "n_estimators": n_estimators, "min_samples_leaf": min_samples_leaf,
        "max_features": rf["max_features"], "max_samples": rf["max_samples"], "bootstrap": True,
        "random_state": cfg["seed"] if seed is None else seed, "n_jobs": cfg["compute"]["n_jobs"],
    }
    return RandomForestClassifier(**{**params, **overrides})


def fit_lgbm_early_stopping(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                            params: dict[str, Any], scale_pos_weight: float | None = None,
                            seed: int | None = None) -> lgb.LGBMClassifier:
    """Fit LightGBM with early stopping on validation log loss (never on the test split).

    ``params`` are ``make_lgbm`` overrides (num_leaves, learning_rate, ...); the tree budget is
    ``predictive.lgbm_max_trees``. Validation log loss is a proper scoring rule, so the stopping
    point favours calibrated probabilities rather than a particular ranking metric.
    """
    pcfg = load_config()["predictive"]
    over = {"n_estimators": pcfg["lgbm_max_trees"], **params}
    if scale_pos_weight is not None:
        over["scale_pos_weight"] = scale_pos_weight
    model = make_lgbm("classifier", seed=seed, **over)
    model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(pcfg["lgbm_early_stopping_rounds"], verbose=False)])
    return model


def fit_lgbm_fixed(X_train: pd.DataFrame, y_train: np.ndarray, params: dict[str, Any],
                   scale_pos_weight: float | None = None, seed: int | None = None) -> lgb.LGBMClassifier:
    """Fit LightGBM with a fixed number of trees (``params['n_estimators']``), no validation data."""
    over = dict(params)
    if scale_pos_weight is not None:
        over["scale_pos_weight"] = scale_pos_weight
    return make_lgbm("classifier", seed=seed, **over).fit(X_train, y_train)
