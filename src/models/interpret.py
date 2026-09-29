"""Interpretation of the conversion-prediction models (Phase 7).

Everything here describes ASSOCIATIONS with P(conversion | X). None of it is a causal statement,
and none of it is a treatment-effect statement: a feature that predicts who converts can be
irrelevant to who converts *because of* the ad (and vice versa). The tidy table produced by
``importance_table`` has one row per raw feature (f0..f11) so it can be compared with
treatment-effect importance from the uplift stages.

Encoded columns (log-frequency encodings, one-hot indicators) are aggregated back to the raw
feature they came from via ``raw_feature_of``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score

from src.features.build_features import RAW_FEATURES, FeatureBuilder
from src.models.baseline import LogisticBaseline
from src.models.evaluation import row_log_loss
from src.utils import get_logger

log = get_logger(__name__)

_RAW_RE = re.compile(r"^(f\d+)(?:_.*)?$")


def raw_feature_of(col: str) -> str:
    """Map an encoded column name (``f3``, ``f3_logfreq``, ``f3_is_2``) to its raw feature (``f3``)."""
    m = _RAW_RE.match(col)
    if not m:
        raise ValueError(f"cannot map column {col!r} to a raw feature")
    return m.group(1)


def lr_coefficients(lr: LogisticBaseline, sample: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Standardized logistic-regression coefficients with 95% CIs from an unpenalised statsmodels GLM.

    Fit on ``rows`` training rows (same design and columns as ``lr``). Continuous and log-frequency
    columns are standardized, so a coefficient is the change in log-odds per +1 SD; one-hot columns
    are 0/1 indicators of the j-th most frequent level of a hashed categorical (level order carries
    no meaning beyond frequency). Interpret as associations only. Hashed categoricals are
    anonymized, so the coefficients say *which* anonymous feature matters, not *why*.
    """
    idx = np.random.default_rng(seed).choice(len(sample), size=min(rows, len(sample)), replace=False)
    sub = sample.iloc[idx]
    X = sm.add_constant(lr.design(sub).astype(np.float64), has_constant="add")
    y = sub["conversion"].to_numpy()
    try:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=50)
        ci = res.conf_int()
        table = pd.DataFrame({"term": X.columns, "coef": res.params.to_numpy(), "se": res.bse.to_numpy(),
                              "ci_lo": ci[0].to_numpy(), "ci_hi": ci[1].to_numpy()})
    except Exception as exc:  # singular / non-converging fit: fall back to point estimates, say so
        log.warning("statsmodels GLM failed (%s); using sklearn coefficients without CIs", exc)
        table = pd.DataFrame({"term": ["const", *lr.cols_],
                              "coef": np.r_[lr.model_.intercept_, lr.model_.coef_.ravel()], "se": np.nan,
                              "ci_lo": np.nan, "ci_hi": np.nan})
    table["z"] = table["coef"] / table["se"]
    table["odds_ratio"] = np.exp(table["coef"])
    table["n_rows_fit"] = len(sub)
    table["n_positives_fit"] = int(y.sum())
    table = table[table["term"] != "const"].copy()
    table["raw_feature"] = table["term"].map(raw_feature_of)
    return table.reset_index(drop=True)


def lgbm_gain_by_raw(model) -> pd.DataFrame:
    """LightGBM total-gain importance, summed over encoded columns per raw feature, as a share of the total."""
    gain = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    by_raw = gain.groupby(gain.index.map(raw_feature_of)).sum()
    return (by_raw / by_raw.sum()).rename("gain_share").reindex(RAW_FEATURES).fillna(0.0).rename_axis("raw_feature").reset_index()


def permutation_importance_raw(model, fb: FeatureBuilder, df: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    """Permutation importance of each RAW feature on a held-out subsample (validation, never test).

    The raw column is permuted and the design matrix rebuilt, so a feature's raw and frequency-encoded
    columns are shuffled together (permuting only one would leave a leaked copy). Reports the mean and
    SD over ``repeats`` of the increase in log loss and the drop in PR-AUC relative to the unpermuted
    baseline. Correlated features share importance, and with ~0.3% positives PR-AUC is noisy: read the SDs.
    """
    rng = np.random.default_rng(seed)
    y = df["conversion"].to_numpy()
    base_p = model.predict_proba(fb.transform(df[RAW_FEATURES]))[:, 1]
    base_ll, base_ap = row_log_loss(y, base_p).mean(), average_precision_score(y, base_p)
    rows = []
    for f in RAW_FEATURES:
        d_ll, d_ap = [], []
        for _ in range(repeats):
            shuffled = df[RAW_FEATURES].copy()
            shuffled[f] = shuffled[f].to_numpy()[rng.permutation(len(shuffled))]
            p = model.predict_proba(fb.transform(shuffled))[:, 1]
            d_ll.append(row_log_loss(y, p).mean() - base_ll)
            d_ap.append(base_ap - average_precision_score(y, p))
        rows.append({"raw_feature": f, "perm_logloss_increase_mean": np.mean(d_ll), "perm_logloss_increase_sd": np.std(d_ll, ddof=1),
                     "perm_prauc_drop_mean": np.mean(d_ap), "perm_prauc_drop_sd": np.std(d_ap, ddof=1)})
    return pd.DataFrame(rows)


def shap_summary(model, X: pd.DataFrame) -> dict:
    """TreeSHAP values (log-odds scale) for ``X`` plus mean |SHAP| per encoded column and per raw feature."""
    import shap
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):                   # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                           # (n, features, classes)
        values = values[:, :, 1]
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    by_raw = mean_abs.groupby(mean_abs.index.map(raw_feature_of)).sum().reindex(RAW_FEATURES).fillna(0.0)
    return {"values": values, "columns": list(X.columns), "mean_abs": mean_abs, "mean_abs_by_raw": by_raw,
            "n_rows": len(X), "X": X}


def importance_table(gain: pd.DataFrame, perm: pd.DataFrame, shap_res: dict, lr_tab: pd.DataFrame) -> pd.DataFrame:
    """One row per raw feature: LightGBM gain share, permutation importance, SHAP, logistic coefficients.

    Columns: ``gain_share``; ``perm_*`` (mean/SD, on validation); ``shap_mean_abs`` (sum of mean |SHAP| over
    the feature's encoded columns, log-odds) and ``shap_share``; ``lr_sum_abs_coef`` (sum of |standardized
    coef| over the feature's columns) and ``lr_max_abs_z``. ``*_rank`` columns rank by SHAP, permutation
    log loss and gain (1 = most important). All describe P(conversion | X), not treatment effects.
    """
    out = gain.merge(perm, on="raw_feature")
    s = shap_res["mean_abs_by_raw"].rename("shap_mean_abs")
    out = out.merge(s.rename_axis("raw_feature").reset_index(), on="raw_feature")
    out["shap_share"] = out["shap_mean_abs"] / out["shap_mean_abs"].sum()
    agg = lr_tab.assign(abs_coef=lr_tab["coef"].abs(), abs_z=lr_tab["z"].abs()).groupby("raw_feature").agg(
        lr_sum_abs_coef=("abs_coef", "sum"), lr_max_abs_z=("abs_z", "max")).reset_index()
    out = out.merge(agg, on="raw_feature", how="left")
    out["shap_rank"] = out["shap_mean_abs"].rank(ascending=False).astype(int)
    out["perm_logloss_rank"] = out["perm_logloss_increase_mean"].rank(ascending=False).astype(int)
    out["gain_rank"] = out["gain_share"].rank(ascending=False).astype(int)
    out["n_shap_rows"] = shap_res["n_rows"]
    return out
