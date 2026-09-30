"""Propensity scores e(x) = P(T=1 | X=x): cross-fitted estimation and overlap diagnostics.

In the Criteo experiment the propensity is *known by design*: every user was assigned to treatment
with probability 0.85, independently of X. Estimating it anyway serves two purposes:

* a diagnostic -- an estimated e(x) that is flat at 0.85 is one more sign that randomization held
  (a classifier cannot tell the arms apart);
* the machinery is the same one needed for observational data, where e(x) is unknown and must be
  modeled; ``src.causal.ate`` uses it on a deliberately confounded subsample to show why.

Every estimate here is *cross-fitted*: rows are split into K folds and each row's score comes from a
model that never saw that row, so the weights are not contaminated by in-sample overfitting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.models.tree_models import make_lgbm
from src.utils import get_logger

log = get_logger(__name__)

FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]


def make_folds(strata: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Stratified fold ids (0..n_folds-1): each stratum is shuffled and dealt round-robin to folds.

    Stratifying on treatment x outcome keeps the (rare) treated/control positives evenly spread, so
    every per-arm outcome model sees a similar number of conversions.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(strata), dtype=np.int8)
    offset = 0
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = idx[rng.permutation(len(idx))]
        # continue the round-robin where the previous stratum stopped so fold sizes stay equal
        folds[idx] = (np.arange(len(idx)) + offset) % n_folds
        offset = (offset + len(idx)) % n_folds
    return folds


def crossfit_predict(fit_predict: FitPredict, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions: for each fold k, ``fit_predict(train_idx, test_idx)`` fits on the other
    folds and returns predictions for the rows of fold k."""
    out = np.full(len(folds), np.nan)
    for k in np.unique(folds):
        test_idx = np.flatnonzero(folds == k)
        train_idx = np.flatnonzero(folds != k)
        out[test_idx] = fit_predict(train_idx, test_idx)
    if np.isnan(out).any():
        raise RuntimeError("crossfit_predict: some rows received no out-of-fold prediction")
    return out


def fit_lgbm(kind: str, X: np.ndarray, y: np.ndarray, seed: int, overrides: dict[str, Any] | None = None,
             early_stopping_rounds: int | None = None, inner_val_frac: float = 0.1) -> lgb.LGBMModel:
    """Fit the project LightGBM on (X, y); with ``early_stopping_rounds`` the number of trees is chosen on a
    random ``inner_val_frac`` of *these* rows (the training folds), never on the held-out fold.

    Early stopping matters most for the propensity model: under (near-)randomization there is almost no
    signal in T, so a fixed large tree budget mostly fits noise, and that noise is amplified by 1/(1-e).
    """
    model = make_lgbm(kind, seed=seed, **(overrides or {}))
    y = np.asarray(y)
    if not early_stopping_rounds:
        return model.fit(X, y)
    val = np.random.default_rng(seed).random(len(y)) < inner_val_frac
    return model.fit(X[~val], y[~val], eval_X=(X[val],), eval_y=(y[val],),
                     callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])


def lgbm_propensity(X: np.ndarray, t: np.ndarray, folds: np.ndarray, seed: int,
                    overrides: dict[str, Any] | None = None, early_stopping_rounds: int | None = None,
                    inner_val_frac: float = 0.1, info: dict[str, Any] | None = None) -> np.ndarray:
    """Cross-fitted LightGBM propensity (tree design matrix, project-default booster + ``overrides``).

    If ``info`` is given, the number of trees used per fold is recorded under ``info["trees"]``.
    """
    t = np.asarray(t)
    trees: list[int] = []

    def fit_predict(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        model = fit_lgbm("classifier", X[tr], t[tr], seed, overrides, early_stopping_rounds, inner_val_frac)
        trees.append(int(model.best_iteration_ or model.n_estimators))
        return model.predict_proba(X[te])[:, 1]

    out = crossfit_predict(fit_predict, folds)
    if info is not None:
        info["trees"] = trees
    return out


def logistic_propensity(design: Callable[[np.ndarray], np.ndarray], t: np.ndarray, folds: np.ndarray,
                        seed: int, C: float = 1.0, fit_rows: int | None = None,
                        predict_chunk: int = 500_000, max_iter: int = 500) -> np.ndarray:
    """Cross-fitted logistic-regression propensity on the linear design matrix.

    ``design(idx)`` returns the (standardized) linear design for the given row indices, so the full
    14M-row matrix never has to exist in memory. Each fold's model is fit on at most ``fit_rows``
    randomly chosen training-fold rows (a linear model with ~180 coefficients is saturated long
    before millions of rows) and predicts its held-out fold in chunks.
    """
    t = np.asarray(t)
    rng = np.random.default_rng(seed)

    def fit_predict(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        if fit_rows is not None and len(tr) > fit_rows:
            tr = np.sort(rng.choice(tr, size=fit_rows, replace=False))
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(design(tr), t[tr])
        preds = [model.predict_proba(design(te[i:i + predict_chunk]))[:, 1]
                 for i in range(0, len(te), predict_chunk)]
        return np.concatenate(preds)

    return crossfit_predict(fit_predict, folds)


def clip_propensity(e: np.ndarray, bounds: tuple[float, float] | list[float]) -> np.ndarray:
    """Clip scores to ``[lo, hi]`` so no unit gets an unbounded inverse-probability weight."""
    lo, hi = bounds
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"clip bounds must satisfy 0 < lo < hi < 1, got {bounds}")
    return np.clip(e, lo, hi)


def kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2: the number of equally-weighted units carrying
    the same information as the weighted sample. Equals len(w) when all weights are equal."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def propensity_diagnostics(e: np.ndarray, t: np.ndarray, clip: tuple[float, float] | list[float],
                           design_p: float | None = None) -> dict[str, Any]:
    """Distribution and overlap diagnostics for an estimated propensity.

    Reports per-arm quantiles, the share of units beyond the clip bounds, Kish effective sample size of
    the inverse-probability weights per arm, how well e(x) separates the arms (ROC-AUC; 0.5 = no
    separation, as expected under randomization), and the deviation from the design propensity.
    """
    e = np.asarray(e, dtype=float)
    t = np.asarray(t).astype(bool)
    lo, hi = clip
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    by_arm: dict[str, Any] = {}
    for name, mask in (("treatment", t), ("control", ~t)):
        vals = e[mask]
        w = 1.0 / vals if name == "treatment" else 1.0 / (1.0 - vals)
        ess = kish_ess(w)
        by_arm[name] = {
            "n": int(mask.sum()), "mean": float(vals.mean()), "sd": float(vals.std()),
            "quantiles": {f"q{int(q * 100):02d}": float(v) for q, v in zip(qs, np.quantile(vals, qs))},
            "ipw_ess": ess, "ipw_ess_ratio": ess / mask.sum(),
            "max_weight_share": float(w.max() / w.sum()),
        }
    out: dict[str, Any] = {
        "by_arm": by_arm,
        "min": float(e.min()), "max": float(e.max()),
        "share_below_clip": float((e < lo).mean()), "share_above_clip": float((e > hi).mean()),
        "clip_bounds": [lo, hi],
        "auc_treatment_vs_control": float(roc_auc_score(t, e)),
        "mean_e_minus_mean_t": float(e.mean() - t.mean()),
    }
    if design_p is not None:
        out["design_propensity"] = design_p
        out["mean_abs_deviation_from_design"] = float(np.abs(e - design_p).mean())
        out["share_within_0.02_of_design"] = float((np.abs(e - design_p) <= 0.02).mean())
    return out
