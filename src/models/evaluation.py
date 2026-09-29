"""Predictive evaluation of P(conversion | X): metrics, bootstrap CIs, cross-validation, and the ``predict`` stage.

What this stage answers -- and what it does not
-----------------------------------------------
It answers "who converts?" (prediction, an association). It does NOT answer "who converts *because
of* the ad?" (a treatment effect): treatment is not a feature, and a high P(conversion | X) says
nothing about the *difference* between P(conversion | X, treated) and P(conversion | X, control).
Feature importance here therefore need not match treatment-effect importance (compared elsewhere).

Why not accuracy, and why PR-AUC
--------------------------------
Only 0.29% of users convert, so "never convert" is 99.71% accurate. ROC-AUC is insensitive to the
huge negative class (its false-positive rate is divided by ~2.8M negatives), so it can look strong
while the flagged list is mostly false positives. PR-AUC (average precision) works with precision,
whose denominator is the flagged list; its no-skill baseline is the prevalence (~0.0029), not 0.5.
Log loss and Brier score are proper scoring rules and measure probability quality (calibration).

Evaluation protocol: features, hyperparameters (stratified K-fold on a train subsample, features
refit inside every fold), early stopping, thresholds and calibrators use TRAIN/VALIDATION only; the
test split is touched once, at the end, for reporting.

Bootstrap: test-set CIs use the Poisson(1) bootstrap (each row gets an independent Poisson(1)
multiplicity), which is asymptotically equivalent to the multinomial bootstrap and lets ranking
metrics be computed from one pre-sorted order per model, vectorised, with the *same* replicate
weights for every model -- so model differences are paired.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.load import FEATURES, load_criteo, stratified_subsample
from src.features.build_features import RAW_FEATURES, FeatureBuilder
from src.models.baseline import LogisticBaseline
from src.utils import (get_logger, load_config, repo_path, save_json, save_table, timer)

log = get_logger(__name__)

_LOSS_EPS = 1e-15
_Z95 = 1.959963984540054


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------
def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row negative log-likelihood (natural log), probabilities clipped to [1e-15, 1 - 1e-15]."""
    p = np.clip(np.asarray(p, dtype=np.float64), _LOSS_EPS, 1 - _LOSS_EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def wilson_interval(k: np.ndarray | float, n: np.ndarray | float, z: float = _Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion k/n (accurate for small rates and counts)."""
    k, n = np.asarray(k, dtype=np.float64), np.asarray(n, dtype=np.float64)
    phat = k / np.maximum(n, 1)
    denom = 1 + z**2 / np.maximum(n, 1)
    centre = (phat + z**2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(phat * (1 - phat) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Equal-frequency reliability table: mean predicted vs observed rate per bin, with Wilson CI.

    Equal-width bins are useless here (nearly all predictions are < 1%), so bins hold equal numbers
    of rows.
    """
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    edges = np.linspace(0, len(p), n_bins + 1).astype(np.int64)
    cy, cp = np.concatenate([[0.0], np.cumsum(y[order])]), np.concatenate([[0.0], np.cumsum(p[order])])
    n = np.diff(edges)
    k = cy[edges[1:]] - cy[edges[:-1]]
    lo, hi = wilson_interval(k, n)
    return pd.DataFrame({"mean_pred": (cp[edges[1:]] - cp[edges[:-1]]) / n, "obs_rate": k / n,
                         "obs_lo": lo, "obs_hi": hi, "n": n, "positives": k})


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """ECE with equal-frequency bins: sum_b (n_b / n) |mean(p_b) - mean(y_b)|.

    Absolute (not relative) error, so for a 0.3% base rate values of ~1e-4 are already meaningful;
    compare against the base rate. A perfectly calibrated predictor has ECE -> 0 as n grows.
    """
    t = reliability_table(y, p, n_bins)
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["mean_pred"] - t["obs_rate"])))


def classification_metrics(y: np.ndarray, p: np.ndarray, n_bins: int | None = None) -> dict[str, float]:
    """Threshold-free metrics: ROC-AUC, PR-AUC (with base-rate reference), log loss, Brier, ECE."""
    n_bins = n_bins or load_config()["predictive"]["calibration_bins"]
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    base = float(y.mean())
    pr = float(average_precision_score(y, p))
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": pr, "base_rate": base, "pr_auc_lift_over_base": pr / base,
        "log_loss": float(row_log_loss(y, p).mean()), "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p, n_bins), "mean_pred": float(p.mean()),
        "mean_pred_over_base_rate": float(p.mean() / base),
    }


def loss_ci(y: np.ndarray, p: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Normal-approximation 95% CIs (mean, lo, hi) for log loss and Brier: both are means of per-row losses."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out = {}
    for name, per_row in (("log_loss", row_log_loss(y, p)), ("brier", (p - y) ** 2)):
        m, se = float(per_row.mean()), float(per_row.std(ddof=1) / np.sqrt(len(per_row)))
        out[name] = (m, m - _Z95 * se, m + _Z95 * se)
    return out


def select_thresholds(y_val: np.ndarray, p_val: np.ndarray, top_frac: float) -> dict[str, float]:
    """Operating points chosen on VALIDATION scores only.

    ``max_f1``: the score threshold maximising F1 on the precision-recall curve.
    ``top_k``: the score of the (1 - top_frac) quantile, i.e. flag the top ``top_frac`` of users.
    A row is flagged when ``p >= threshold``.
    """
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return {"max_f1": float(thr[int(np.argmax(f1))]),
            "top_k": float(np.quantile(np.asarray(p_val, dtype=np.float64), 1 - top_frac))}


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts, precision/recall/F1 (+ Wilson CI for precision, recall) when flagging ``p >= threshold``."""
    y = np.asarray(y).astype(bool)
    flag = np.asarray(p) >= threshold
    tp, fp = int(np.sum(flag & y)), int(np.sum(flag & ~y))
    fn, tn = int(np.sum(~flag & y)), int(np.sum(~flag & ~y))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    plo, phi = wilson_interval(tp, tp + fp)
    rlo, rhi = wilson_interval(tp, tp + fn)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            "precision_lo": float(plo), "precision_hi": float(phi), "recall_lo": float(rlo), "recall_hi": float(rhi),
            "flagged_fraction": (tp + fp) / len(y), "lift_over_base": prec / max(y.mean(), 1e-12)}


# ---------------------------------------------------------------------------------------------
# Bootstrap (Poisson, vectorised, paired across models)
# ---------------------------------------------------------------------------------------------
class _RankState:
    """Pre-sorted view of one model's scores for repeated weighted AUC / AP evaluation."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.order = np.argsort(p, kind="stable")
        ps = np.asarray(p)[self.order]
        self.starts = np.flatnonzero(np.concatenate([[True], ps[1:] != ps[:-1]]))
        self.y = np.asarray(y, dtype=np.float64)[self.order]

    def auc_ap(self, w: np.ndarray) -> tuple[float, float]:
        """Weighted ROC-AUC (ties count 1/2) and step-wise average precision (sklearn's definition)."""
        ws = w[self.order]
        pos = np.add.reduceat(ws * self.y, self.starts)          # ascending score groups
        neg = np.add.reduceat(ws * (1.0 - self.y), self.starts)
        tot_p, tot_n = pos.sum(), neg.sum()
        auc = float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (tot_p * tot_n))
        pd_, nd_ = pos[::-1], neg[::-1]                          # descending: highest scores first
        tp, fp = np.cumsum(pd_), np.cumsum(nd_)
        seen = tp + fp                                           # 0 while the leading groups all drew weight 0
        prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        ap = float(((pd_ / tot_p) * prec).sum())
        return auc, ap


def bootstrap_rank_metrics(y: np.ndarray, preds: dict[str, np.ndarray], reps: int, seed: int
                           ) -> dict[str, dict[str, np.ndarray]]:
    """Poisson-bootstrap replicates of ROC-AUC and PR-AUC for several models with shared weights.

    Returns ``{model: {"roc_auc": array(reps), "pr_auc": array(reps)}}``. Sharing the row weights
    across models makes differences between models paired.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    states = {m: _RankState(y, p) for m, p in preds.items()}
    out = {m: {"roc_auc": np.empty(reps), "pr_auc": np.empty(reps)} for m in preds}
    for r in range(reps):
        w = rng.poisson(1.0, size=len(y)).astype(np.float64)
        for m, st in states.items():
            out[m]["roc_auc"][r], out[m]["pr_auc"][r] = st.auc_ap(w)
    return out


def bootstrap_ci(metric: str, y: np.ndarray, p: np.ndarray, reps: int = 200, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point estimate, lo, hi): percentile bootstrap CI of ``roc_auc`` or ``pr_auc`` (point from sklearn)."""
    if metric not in ("roc_auc", "pr_auc"):
        raise ValueError(f"metric must be 'roc_auc' or 'pr_auc', got {metric!r}")
    point = roc_auc_score(y, p) if metric == "roc_auc" else average_precision_score(y, p)
    draws = bootstrap_rank_metrics(y, {"m": np.asarray(p)}, reps, seed)["m"][metric]
    return float(point), *(float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def summarize_draws(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_diff(draws_a: np.ndarray, draws_b: np.ndarray, point_a: float, point_b: float) -> dict[str, float]:
    """Paired bootstrap of metric(A) - metric(B): point estimate, 95% CI, two-sided bootstrap p-value."""
    d = draws_a - draws_b
    lo, hi = summarize_draws(d)
    p = 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))
    return {"diff": float(point_a - point_b), "lo": lo, "hi": hi, "p_value_two_sided": float(min(p, 1.0))}


# ---------------------------------------------------------------------------------------------
# Cross-validation (Phase 5)
# ---------------------------------------------------------------------------------------------
CV_METRICS = ("roc_auc", "pr_auc", "log_loss", "brier")


def _fast_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "log_loss": float(row_log_loss(y, p).mean()), "brier": float(np.mean((p - y) ** 2))}


def stratified_train_sample(train: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Stratified (treatment x visit x conversion) subsample of the training split, at most ``rows`` rows."""
    frac = min(1.0, rows / len(train))
    if frac >= 1.0:
        return train
    return stratified_subsample(train, frac, seed, strata_cols=("treatment", "visit", "conversion"))


def cv_configs() -> list[dict]:
    """The small, explicit search space (<= 8 configs per the brief; here 3 + 2 + 4 = 9 fits per fold)."""
    from itertools import product
    pc = load_config()["predictive"]
    cfgs = [{"model": "logistic_regression", "config": f"C={C:g}", "params": {"C": C}} for C in pc["lr_C_grid"]]
    cfgs += [{"model": "random_forest", "config": f"min_samples_leaf={m}", "params": {"min_samples_leaf": m}}
             for m in pc["rf"]["min_samples_leaf_grid"]]
    for nl, lr in product(pc["lgbm_grid"]["num_leaves"], pc["lgbm_grid"]["learning_rate"]):
        trees = pc["lgbm_cv_trees"][str(lr)]
        cfgs.append({"model": "lightgbm", "config": f"leaves={nl},lr={lr:g}",
                     "params": {"num_leaves": nl, "learning_rate": lr, "n_estimators": trees}})
    return cfgs


def cross_validate(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratified K-fold over every config; identical folds for all models; features refit per fold.

    The FeatureBuilder (frequency encodings, clip bounds, standardization, one-hot levels) is fit on
    each fold's training part only, so held-out rows never influence the design. Returns one row per
    (model, config, fold).
    """
    from src.models.tree_models import fit_lgbm_fixed, make_rf
    pc = load_config()["predictive"]
    y_all = sample["conversion"].to_numpy()
    skf = StratifiedKFold(n_splits=pc["cv_folds"], shuffle=True, random_state=seed)
    rows: list[dict] = []
    for k, (tri, vai) in enumerate(skf.split(sample, y_all)):
        tr, va = sample.iloc[tri], sample.iloc[vai]
        ytr, yva = y_all[tri], y_all[vai]
        fb = FeatureBuilder(kind="tree").fit(tr[RAW_FEATURES])
        Xtr, Xva = fb.transform(tr[RAW_FEATURES]), fb.transform(va[RAW_FEATURES])
        for cfg in cv_configs():
            t0 = time.perf_counter()
            if cfg["model"] == "logistic_regression":
                p = LogisticBaseline(**cfg["params"]).fit(tr, ytr).predict_proba1(va)
            elif cfg["model"] == "random_forest":
                m = make_rf(pc["rf"]["cv_n_estimators"], seed=seed, **cfg["params"]).fit(Xtr, ytr)
                p = m.predict_proba(Xva)[:, 1]
            else:
                p = fit_lgbm_fixed(Xtr, ytr, cfg["params"], seed=seed).predict_proba(Xva)[:, 1]
            secs = time.perf_counter() - t0
            rows.append({"model": cfg["model"], "config": cfg["config"], "fold": k, "fit_predict_seconds": secs,
                         **_fast_metrics(yva, p)})
            log.info("CV fold %d %-20s %-24s pr_auc=%.4f logloss=%.5f (%.0fs)", k, cfg["model"], cfg["config"],
                     rows[-1]["pr_auc"], rows[-1]["log_loss"], secs)
        del Xtr, Xva
    return pd.DataFrame(rows)


def summarize_cv(folds: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    """Mean and SD (across folds) of each metric per config; flags the winner within each model family."""
    higher_better = selection_metric in ("roc_auc", "pr_auc")
    g = folds.groupby(["model", "config"], sort=False)
    out = g[list(CV_METRICS)].mean().add_suffix("_mean").join(g[list(CV_METRICS)].std(ddof=1).add_suffix("_sd"))
    out["fit_predict_seconds_mean"] = g["fit_predict_seconds"].mean()
    out = out.reset_index()
    col = f"{selection_metric}_mean"
    out["selected"] = False
    for m, sub in out.groupby("model"):
        best = sub[col].idxmax() if higher_better else sub[col].idxmin()
        out.loc[best, "selected"] = True
    return out


# ---------------------------------------------------------------------------------------------
# The `predict` stage
# ---------------------------------------------------------------------------------------------
def _predict_chunked(model: object, fb: FeatureBuilder, df: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(df), dtype=np.float32)
    for s in range(0, len(df), chunk):
        X = fb.transform(df.iloc[s:s + chunk][RAW_FEATURES])
        out[s:s + chunk] = model.predict_proba(X)[:, 1]
    return out


MODEL_KEYS = ("logistic_regression", "random_forest", "lightgbm", "lightgbm_weighted")


def run_predictive(mode: str = "dev") -> dict:
    """Phases 4-7: fit, cross-validate, calibrate and interpret P(conversion | X). See module docstring.

    Writes ``predict.json``, tables ``predict_*``, figures ``predict_*`` and
    ``data/processed/predictions_{mode}.parquet`` (validation + test rows).
    """
    from src.models import interpret
    from src.models.calibration import CALIBRATION_METHODS, fit_calibrator, select_calibration_method
    from src.models.tree_models import fit_lgbm_early_stopping, fit_lgbm_fixed, make_rf
    from src.visualization import model_plots as mp

    cfg = load_config()
    pc, seed = cfg["predictive"], cfg["seed"]
    times: dict[str, float] = {}
    cols = ["row_id", *FEATURES, "treatment", "conversion", "visit", "split"]
    train = load_criteo(mode, "train", cols)
    val = load_criteo(mode, "val", cols)
    test = load_criteo(mode, "test", cols)
    ytr, yva, yte = (d["conversion"].to_numpy() for d in (train, val, test))
    base_rate = float(ytr.mean())
    log.info("rows train/val/test = %d/%d/%d; train conversion rate %.4f%%", len(train), len(val), len(test),
             100 * base_rate)

    # ---- Phase 5: cross-validation on a stratified train subsample ---------------------------
    cv_sample = stratified_train_sample(train, pc["cv_sample_rows"], seed)
    with timer("cross-validation", log) as t:
        cv_folds = cross_validate(cv_sample, seed)
    times["cv_seconds"] = t["seconds"]
    cv_summary = summarize_cv(cv_folds, pc["selection_metric"])
    save_table(cv_folds, "predict_cv_folds")
    save_table(cv_summary, "predict_cv_comparison")
    del cv_sample
    cv_cfg = {c["config"]: c for c in cv_configs()}
    selected = {r.model: cv_cfg[r.config] for r in cv_summary[cv_summary["selected"]].itertuples()}
    log.info("selected: %s", {m: c["config"] for m, c in selected.items()})

    # ---- Phase 4: final fits ------------------------------------------------------------------
    scores_val: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}

    with timer("logistic regression", log) as t:
        lr_sample = stratified_train_sample(train, pc["lr_train_rows"], seed + 1)
        lr = LogisticBaseline(**selected["logistic_regression"]["params"]).fit(lr_sample, lr_sample["conversion"].to_numpy())
        scores_val["logistic_regression"] = lr.predict_proba1(val)
        scores_test["logistic_regression"] = lr.predict_proba1(test)
    times["logistic_regression_seconds"] = t["seconds"]

    tree_fb = FeatureBuilder(kind="tree").fit(train[RAW_FEATURES])
    Xva = tree_fb.transform(val[RAW_FEATURES])

    with timer("random forest", log) as t:
        rf_sample = stratified_train_sample(train, pc["rf_train_rows"], seed + 2)
        rf = make_rf(pc["rf"]["n_estimators"], seed=seed, **selected["random_forest"]["params"])
        rf.fit(tree_fb.transform(rf_sample[RAW_FEATURES]), rf_sample["conversion"].to_numpy())
        scores_val["random_forest"] = _predict_chunked(rf, tree_fb, val)
        scores_test["random_forest"] = _predict_chunked(rf, tree_fb, test)
    times["random_forest_seconds"] = t["seconds"]
    del rf_sample

    Xtr = tree_fb.transform(train[RAW_FEATURES])
    lg_params = {k: v for k, v in selected["lightgbm"]["params"].items() if k != "n_estimators"}
    with timer("lightgbm", log) as t:
        lgbm = fit_lgbm_early_stopping(Xtr, ytr, Xva, yva, lg_params, seed=seed)
        scores_val["lightgbm"] = lgbm.predict_proba(Xva)[:, 1].astype(np.float32)
        scores_test["lightgbm"] = _predict_chunked(lgbm, tree_fb, test)
    times["lightgbm_seconds"] = t["seconds"]
    n_trees = int(lgbm.best_iteration_ or lgbm.n_estimators)
    log.info("lightgbm best iteration: %d trees", n_trees)

    # Class-weighted variant: same trees/params, only the loss weighting changes (controlled comparison).
    spw = (1 - base_rate) / base_rate
    with timer("lightgbm (scale_pos_weight)", log) as t:
        lgbm_w = fit_lgbm_fixed(Xtr, ytr, {**lg_params, "n_estimators": n_trees}, scale_pos_weight=spw, seed=seed)
        scores_val["lightgbm_weighted"] = lgbm_w.predict_proba(Xva)[:, 1].astype(np.float32)
        scores_test["lightgbm_weighted"] = _predict_chunked(lgbm_w, tree_fb, test)
    times["lightgbm_weighted_seconds"] = t["seconds"]

    # ---- Test metrics (once) -----------------------------------------------------------------
    metrics = {m: classification_metrics(yte, scores_test[m]) for m in MODEL_KEYS}
    with timer("bootstrap", log) as t:
        draws = bootstrap_rank_metrics(yte, scores_test, pc["bootstrap_reps"], seed)
    times["bootstrap_seconds"] = t["seconds"]
    for m in MODEL_KEYS:
        for name in ("roc_auc", "pr_auc"):
            metrics[m][f"{name}_ci"] = list(summarize_draws(draws[m][name]))
        for name, (_, lo, hi) in loss_ci(yte, scores_test[m]).items():
            metrics[m][f"{name}_ci"] = [lo, hi]
    paired = {}
    for a, b in (("lightgbm", "logistic_regression"), ("lightgbm", "random_forest"), ("lightgbm_weighted", "lightgbm")):
        paired[f"{a}_minus_{b}"] = {
            name: paired_diff(draws[a][name], draws[b][name], metrics[a][name], metrics[b][name])
            for name in ("roc_auc", "pr_auc")}

    # Thresholds are chosen on validation, applied to test.
    thr_rows, conf = [], {}
    for m in MODEL_KEYS:
        thr = select_thresholds(yva, scores_val[m], pc["top_k_fraction"])
        for rule, t_ in thr.items():
            tm = threshold_metrics(yte, scores_test[m], t_)
            thr_rows.append({"model": m, "rule": rule, **tm})
            conf[(m, rule)] = tm
    thr_df = pd.DataFrame(thr_rows)
    save_table(thr_df, "predict_thresholds")
    main_table = pd.DataFrame([
        {"model": m, **{k: v for k, v in metrics[m].items() if not k.endswith("_ci")},
         **{f"{k}_lo": metrics[m][f"{k}_ci"][0] for k in ("roc_auc", "pr_auc", "log_loss", "brier")},
         **{f"{k}_hi": metrics[m][f"{k}_ci"][1] for k in ("roc_auc", "pr_auc", "log_loss", "brier")}}
        for m in MODEL_KEYS])
    save_table(main_table, "predict_test_metrics")

    # ---- Phase 6: calibration (fit on validation, evaluate on test) --------------------------
    cal_rows, cal_scores, cal_choice = [], {}, {}
    for m in MODEL_KEYS:
        best, cv_ll = select_calibration_method(scores_val[m], yva, CALIBRATION_METHODS,
                                                pc["calibration_selection_folds"], seed)
        cal_choice[m] = {"selected_by_crossfit_val_logloss": best, "crossfit_val_logloss": cv_ll}
        variants = {"uncalibrated": scores_test[m]}
        for meth in CALIBRATION_METHODS:
            variants[meth] = fit_calibrator(meth, scores_val[m], yva).predict(scores_test[m])
        cal_scores[m] = variants
        for meth, p in variants.items():
            met = classification_metrics(yte, p)
            ci = loss_ci(yte, p)
            cal_rows.append({"model": m, "method": meth, "selected": meth == best, **met,
                             "log_loss_lo": ci["log_loss"][1], "log_loss_hi": ci["log_loss"][2],
                             "brier_lo": ci["brier"][1], "brier_hi": ci["brier"][2]})
    cal_df = pd.DataFrame(cal_rows)
    save_table(cal_df, "predict_calibration")
    best_lgbm = cal_choice["lightgbm"]["selected_by_crossfit_val_logloss"]
    p_lgbm_cal_test = cal_scores["lightgbm"][best_lgbm]
    p_lgbm_cal_val = fit_calibrator(best_lgbm, scores_val["lightgbm"], yva).predict(scores_val["lightgbm"])

    # ---- Figures: performance + calibration -------------------------------------------------
    figs = []
    display = ("logistic_regression", "random_forest", "lightgbm")
    figs.append(mp.plot_roc_pr(yte, {m: scores_test[m] for m in display}, metrics, base_rate))
    figs.append(mp.plot_calibration(yte, {m: scores_test[m] for m in display}, pc["calibration_bins"]))
    figs.append(mp.plot_confusion(conf, display))
    figs.append(mp.plot_imbalance_tradeoff(metrics, paired))
    figs.append(mp.plot_calibration_methods(yte, cal_scores, pc["calibration_bins"]))
    figs.append(mp.plot_cv(cv_summary))

    # ---- Phase 7: interpretation --------------------------------------------------------------
    rng = np.random.default_rng(seed)
    with timer("interpretation", log) as t:
        lr_tab = interpret.lr_coefficients(lr, lr_sample, min(pc["lr_interpret_rows"], len(lr_sample)), seed)
        save_table(lr_tab, "predict_lr_coefficients")
        gain = interpret.lgbm_gain_by_raw(lgbm)
        n_perm = min(pc["permutation_rows"], len(val))
        perm_idx = rng.choice(len(val), size=n_perm, replace=False)
        perm = interpret.permutation_importance_raw(lgbm, tree_fb, val.iloc[perm_idx], pc["permutation_repeats"], seed)
        n_shap = min(pc["shap_rows"], len(val))
        shap_idx = rng.choice(len(val), size=n_shap, replace=False)
        shap_res = interpret.shap_summary(lgbm, Xva.iloc[shap_idx])
        imp = interpret.importance_table(gain, perm, shap_res, lr_tab)
        save_table(imp, "predict_feature_importance")
    times["interpretation_seconds"] = t["seconds"]
    figs.append(mp.plot_lr_coefficients(lr_tab, pc["top_features_shown"]))
    figs.append(mp.plot_importance(imp))
    figs.append(mp.plot_shap_summary(shap_res))
    figs.append(mp.plot_shap_dependence(shap_res, imp))

    # ---- Artifacts ------------------------------------------------------------------------------
    pred_df = pd.concat([
        pd.DataFrame({"row_id": d["row_id"].to_numpy(), "split": d["split"].to_numpy(),
                      "treatment": d["treatment"].to_numpy(), "conversion": d["conversion"].to_numpy(),
                      "visit": d["visit"].to_numpy(),
                      "p_logistic_regression": s["logistic_regression"], "p_random_forest": s["random_forest"],
                      "p_lightgbm": s["lightgbm"], "p_lightgbm_calibrated": pc_})
        for d, s, pc_ in ((val, scores_val, p_lgbm_cal_val), (test, scores_test, p_lgbm_cal_test))],
        ignore_index=True)
    pred_path = repo_path(cfg["paths"]["processed_dir"]) / f"predictions_{mode}.parquet"
    pred_df.to_parquet(pred_path, index=False)
    model_dir = repo_path(cfg["paths"]["processed_dir"] + "/models")
    lgbm.booster_.save_model(str(model_dir / f"lightgbm_conversion_{mode}.txt"))

    cv_json = cv_summary.to_dict(orient="records")
    result = {
        "mode": mode,
        "question": "Prediction, not causation: P(conversion | X) says who converts, not who converts because of the ad. "
                    "Treatment is not a feature. Feature importance here is an association and need not match "
                    "treatment-effect importance.",
        "why_pr_auc": "Only ~0.29% of users convert. Accuracy is uninformative (always-no is ~99.7% accurate). ROC-AUC "
                      "divides false positives by millions of negatives and so looks good even when a flagged list is mostly "
                      "false positives; PR-AUC's no-skill baseline is the prevalence (base_rate), so its lift over base rate "
                      "is the honest ranking-quality number.",
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "base_rate_train": base_rate, "base_rate_test": float(yte.mean()),
        "training_rows": {"logistic_regression": len(lr_sample), "random_forest": min(pc["rf_train_rows"], len(train)),
                          "lightgbm": len(train), "cv_sample": min(pc["cv_sample_rows"], len(train))},
        "test_metrics": metrics,
        "paired_bootstrap_differences": paired,
        "bootstrap": {"reps": pc["bootstrap_reps"], "type": "Poisson(1) row weights, percentile 95% CI; "
                      "log-loss/Brier CIs are normal-approximation on per-row losses"},
        "thresholds_test": {f"{m}|{r}": v for (m, r), v in conf.items()},
        "cv": {"folds": pc["cv_folds"], "selection_metric": pc["selection_metric"],
               "selection_rationale": "Log loss is a proper scoring rule and much less noisy than PR-AUC with ~0.3% "
                                      "positives; it also favours usable probabilities. PR-AUC is reported alongside.",
               "summary": cv_json},
        "selected_hyperparameters": {m: c["params"] for m, c in selected.items()},
        "lightgbm_final": {"trees_after_early_stopping": n_trees, "scale_pos_weight_variant": spw},
        "calibration": {"selection": cal_choice, "table_file": "results/tables/predict_calibration.csv",
                        "note": "Calibrators fit on validation, evaluated on test. Why it matters: expected value = "
                                "p x value - cost, so a 2x over-prediction doubles the perceived value. "
                                "mean_pred_over_base_rate ~ 1 means unbiased on average.",
                        "rows": cal_df.to_dict(orient="records")},
        "importance_top_raw_features_by_shap": imp.sort_values("shap_mean_abs", ascending=False)["raw_feature"].head(5).tolist(),
        "timings_seconds": times,
        "figures": figs,
    }
    save_json(result, "predict")
    headline = {m: {"roc_auc": round(metrics[m]["roc_auc"], 4), "pr_auc": round(metrics[m]["pr_auc"], 5),
                    "log_loss": round(metrics[m]["log_loss"], 5)} for m in MODEL_KEYS}
    return {"test": headline, "selected": result["selected_hyperparameters"], "timings": times}
