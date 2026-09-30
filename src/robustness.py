"""Phase 13: error and robustness analysis for the headline uplift results.

The CATE stage reports one fit per learner with bootstrap CIs over the *test rows*. That CI ignores two other
sources of variation, which this stage measures directly:

* **training randomness**: refit the selected uplift learner and the response model with several seeds
  (LightGBM bagging/column sampling, causal-forest subsampling) and track the uplift-vs-response gap;
* **training sample**: fit the same learner on two disjoint halves of the training split and compare the
  resulting tau-hat on test (rank correlation, overlap of the top-k% targeted sets);
* **sample size**: learning curves of the Qini coefficient and incremental outcomes vs training rows.

It also checks how the pooled response model behaves within each randomized arm, and how noisy the rare
conversion outcome is relative to visits. All uplift quantities are propensity-weighted (Hajek), because
treatment is not independent of the features in this benchmark (see the ``stats`` and ``causal`` stages).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.causal.causal_forest import fit_causal_forest, predict_cate
from src.causal.learners import (
    SLearner,
    TLearner,
    ValSet,
    XLearner,
    _fit,
    _proba,
    scores_path,
    selection_mask,
)
from src.causal.uplift_metrics import budget_tag, evaluate_scorings, stabilized_ipw_weights
from src.data.load import FEATURES, load_criteo
from src.features.build_features import build_matrices
from src.models.tree_models import make_lgbm
from src.utils import get_logger, load_config, load_json, repo_path, save_json, save_table

log = get_logger(__name__)

SUPPORTED = ("s_learner", "t_learner", "x_learner", "causal_forest")
Predictor = Callable[[pd.DataFrame, np.ndarray], np.ndarray]


def _stable_half(row_id: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 50/50 partition of training rows by a hash of ``row_id`` (order-independent)."""
    return selection_mask(row_id, seed, 0.5)


def fit_learner(name: str, Xtr: pd.DataFrame, t: np.ndarray, y: np.ndarray, val: ValSet, seed: int) -> Predictor:
    """Fit one uplift learner with the CATE stage's settings and return ``predict(X, e) -> tau_hat``."""
    ccfg = load_config()["cate"]
    rounds = ccfg["early_stopping_rounds"]

    def outcome_model() -> Any:
        return make_lgbm("classifier", seed=seed, **ccfg["outcome_model"])

    def effect_model() -> Any:
        return make_lgbm("regressor", seed=seed, **ccfg["effect_model"])

    if name == "causal_forest":
        rows = load_config()["causal"]["causal_forest_train_rows"]
        if rows is not None and rows < len(t):
            idx = np.sort(np.random.default_rng(seed).choice(len(t), size=rows, replace=False))
            Xtr, t, y = Xtr.iloc[idx], t[idx], y[idx]
        cf = fit_causal_forest(Xtr, t, y, seed)
        return lambda X, e: predict_cate(cf, X, ccfg["predict_chunk_rows"])
    learner = {"s_learner": SLearner(outcome_model, early_stopping_rounds=rounds),
               "t_learner": TLearner(outcome_model, early_stopping_rounds=rounds),
               "x_learner": XLearner(outcome_model, effect_model, rounds)}.get(name)
    if learner is None:
        raise ValueError(f"robustness supports {SUPPORTED}, got {name!r}; set robustness.learners in config")
    learner.fit(Xtr, t, y, None, val)
    return learner.predict


def fit_response(Xtr: pd.DataFrame, y: np.ndarray, val: ValSet, seed: int) -> Predictor:
    """P(Y=1 | X) ignoring treatment: the 'target the likeliest converters' baseline."""
    ccfg = load_config()["cate"]
    m = _fit(make_lgbm("classifier", seed=seed, **ccfg["outcome_model"]), Xtr, y, (val.X, val.y),
             ccfg["early_stopping_rounds"])
    return lambda X, e: _proba(m, X)


def evaluate_pair(y: np.ndarray, t: np.ndarray, weights: np.ndarray, tau: np.ndarray, resp: np.ndarray,
                  budgets: list[float], seed: int, n_boot: int = 0) -> dict[str, float]:
    """Propensity-weighted Qini and incremental outcomes for uplift vs response scoring (point estimates)."""
    ev = evaluate_scorings(y, t, {"uplift": tau, "response": resp}, budgets=budgets, n_boot=n_boot, seed=seed,
                           pairs=[("uplift", "response")], sample_weight=weights)
    m = ev["metrics"].set_index(["scorer", "metric"])["estimate"]
    d = ev["differences"].set_index("metric")["estimate"]
    out = {"qini_uplift": m[("uplift", "qini_normalized")], "qini_response": m[("response", "qini_normalized")]}
    for k in budgets:
        tag = budget_tag(k)
        out[f"inc_uplift_{tag}"] = m[("uplift", f"incremental_at_{tag}")]
        out[f"inc_response_{tag}"] = m[("response", f"incremental_at_{tag}")]
        out[f"diff_{tag}"] = d[f"incremental_at_{tag}"]
    return out


def _top_overlap(a: np.ndarray, b: np.ndarray, k: float) -> float:
    """Jaccard overlap of the top-``k`` fraction of users under two scorings."""
    n = round(k * len(a))
    ta, tb = set(np.argsort(-a, kind="stable")[:n]), set(np.argsort(-b, kind="stable")[:n])
    return len(ta & tb) / len(ta | tb)


def _rank_corr(a: np.ndarray, b: np.ndarray, rows: int, seed: int) -> float:
    idx = np.random.default_rng(seed).choice(len(a), size=min(rows, len(a)), replace=False)
    return float(spearmanr(a[idx], b[idx]).statistic)


def response_by_arm(mode: str) -> pd.DataFrame:
    """How the pooled conversion model behaves inside each randomized arm (test split).

    The model never sees treatment, so within control it should over-predict conversion (control converts
    less), and within the treated arm under-predict slightly. Ranking quality per arm is reported too.
    """
    path = repo_path(load_config()["paths"]["processed_dir"]) / f"predictions_{mode}.parquet"
    p = pd.read_parquet(path)
    p = p[p.split == 2]
    rows = []
    for col in ("p_lightgbm", "p_lightgbm_calibrated", "p_logistic_regression"):
        for arm, g in p.groupby("treatment"):
            y, s = g["conversion"].to_numpy(), g[col].to_numpy()
            rows.append({"model": col.removeprefix("p_"), "arm": "treatment" if arm == 1 else "control", "n": len(g),
                         "observed_rate": y.mean(), "mean_predicted": s.mean(), "predicted_over_observed": s.mean() / y.mean(),
                         "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)})
    return pd.DataFrame(rows)


def rare_event_noise(mode: str) -> pd.DataFrame:
    """Relative precision of the propensity-weighted uplift estimates, visit vs conversion (from the CATE stage)."""
    t = pd.read_csv(repo_path(load_config()["paths"]["tables_dir"].replace(
        "results/", "results/dev/" if mode == "dev" else "results/", 1)) / "cate_uplift_metrics.csv")
    t = t[(t.estimator == "ipw") & t.metric.str.startswith("incremental_at_")]
    t = t.assign(rel_halfwidth=(t.ci_high - t.ci_low) / 2 / t.estimate.abs())
    return t[["outcome", "scorer", "metric", "estimate", "ci_low", "ci_high", "rel_halfwidth"]]


def run_robustness(mode: str = "dev") -> dict[str, Any]:
    """Seed, split-half and sample-size robustness of the uplift-vs-response comparison, per outcome."""
    cfg = load_config()
    rcfg, seed0 = cfg["robustness"], cfg["seed"]
    budgets = list(cfg["uplift"]["budgets"])
    selected = load_json("cate").get("selected_learner", {})
    learners = {o: rcfg.get("learners", {}).get(o) or selected.get(o) for o in cfg["causal"]["outcomes"]}

    cols = ["row_id", "split", *FEATURES, "treatment", "visit", "conversion"]
    train = load_criteo(mode, "train", cols).reset_index(drop=True)
    val = load_criteo(mode, "val", cols).reset_index(drop=True)
    test = load_criteo(mode, "test", cols).reset_index(drop=True)
    sc = pd.read_parquet(scores_path(mode), columns=["row_id", "propensity"])
    e_map = sc.set_index("row_id")["propensity"]
    e_va, e_te = e_map.loc[val.row_id].to_numpy(), e_map.loc[test.row_id].to_numpy()
    _, (Xtr, Xva, Xte) = build_matrices(train, val, test, kind="tree")
    t_tr, t_va, t_te = (d["treatment"].to_numpy(np.int64) for d in (train, val, test))
    es = ~selection_mask(val["row_id"].to_numpy(), seed0, cfg["cate"]["val_selection_share"])
    w_te = stabilized_ipw_weights(t_te, e_te)

    seed_rows, half_rows, curve_rows, stability = [], [], [], {}
    timings: dict[str, float] = {}
    for outcome, lname in learners.items():
        y_tr, y_va, y_te = (d[outcome].to_numpy(np.int64) for d in (train, val, test))
        val_es = ValSet(Xva[es], t_va[es], y_va[es], e_va[es])

        # 1) seeds: full training split, different model seeds
        taus, full_fit_row = {}, None
        for s in rcfg["seeds"]:
            t0 = time.perf_counter()
            tau = fit_learner(lname, Xtr, t_tr, y_tr, val_es, seed0 + s)(Xte, e_te)
            resp = fit_response(Xtr, y_tr, val_es, seed0 + s)(Xte, e_te)
            taus[s] = tau
            row = {"outcome": outcome, "learner": lname, "seed": s,
                   **evaluate_pair(y_te, t_te, w_te, tau, resp, budgets, seed0)}
            seed_rows.append(row)
            if s == 0:
                full_fit_row = {k: v for k, v in row.items() if k not in ("outcome", "learner", "seed")}
            timings[f"{outcome}_seed{s}"] = time.perf_counter() - t0
            log.info("robustness %s seed %d: %s", outcome, s, {k: round(v, 4) for k, v in row.items()
                                                                if isinstance(v, float)})
        pairs = [(a, b) for i, a in enumerate(taus) for b in list(taus)[i + 1:]]
        stability[outcome] = {
            "learner": lname,
            "seed_rank_corr_mean": float(np.mean([_rank_corr(taus[a], taus[b], rcfg["corr_rows"], seed0)
                                                  for a, b in pairs])),
            "seed_top10_jaccard_mean": float(np.mean([_top_overlap(taus[a], taus[b], 0.10) for a, b in pairs])),
        }

        # 2) split-half: disjoint halves of the training split
        half = _stable_half(train["row_id"].to_numpy(), seed0 + 7)
        tau_half = []
        for h, mask in (("A", half), ("B", ~half)):
            tau = fit_learner(lname, Xtr[mask], t_tr[mask], y_tr[mask], val_es, seed0)(Xte, e_te)
            resp = fit_response(Xtr[mask], y_tr[mask], val_es, seed0)(Xte, e_te)
            tau_half.append(tau)
            half_rows.append({"outcome": outcome, "learner": lname, "half": h, "train_rows": int(mask.sum()),
                              **evaluate_pair(y_te, t_te, w_te, tau, resp, budgets, seed0)})
        stability[outcome] |= {
            "split_half_rank_corr": _rank_corr(tau_half[0], tau_half[1], rcfg["corr_rows"], seed0),
            "split_half_top10_jaccard": _top_overlap(tau_half[0], tau_half[1], 0.10),
        }
        rng = np.random.default_rng(seed0)
        sample = rng.choice(len(tau_half[0]), size=min(rcfg["scatter_rows"], len(tau_half[0])), replace=False)
        stability[outcome]["_scatter"] = (tau_half[0][sample], tau_half[1][sample])

        # 3) learning curve over training-set size (stratified by the fixed row hash, nested subsets)
        u = np.random.default_rng(seed0 + 13).permutation(len(train))      # nested subsets: prefix of one permutation
        cap = cfg["causal"]["causal_forest_train_rows"] if lname == "causal_forest" else None
        effective_full = len(train) if cap is None else min(cap, len(train))
        for n in [s for s in rcfg["train_sizes"] if s < effective_full]:
            idx = np.sort(u[:n])
            t0 = time.perf_counter()
            tau = fit_learner(lname, Xtr.iloc[idx], t_tr[idx], y_tr[idx], val_es, seed0)(Xte, e_te)
            resp = fit_response(Xtr.iloc[idx], y_tr[idx], val_es, seed0)(Xte, e_te)
            curve_rows.append({"outcome": outcome, "learner": lname, "train_rows": n,
                               "fit_seconds": time.perf_counter() - t0,
                               **evaluate_pair(y_te, t_te, w_te, tau, resp, budgets, seed0)})
            log.info("robustness %s learning curve n=%d done", outcome, n)
        if full_fit_row is not None:      # the largest size is the seed-0 fit above; reuse it instead of refitting
            curve_rows.append({"outcome": outcome, "learner": lname, "train_rows": effective_full,
                               "fit_seconds": timings.get(f"{outcome}_seed0"), **full_fit_row})

    seeds_df, halves_df, curve_df = pd.DataFrame(seed_rows), pd.DataFrame(half_rows), pd.DataFrame(curve_rows)
    arms_df = response_by_arm(mode)
    noise_df = rare_event_noise(mode)
    for df, name in ((seeds_df, "seeds"), (halves_df, "split_half"), (curve_df, "learning_curve"),
                     (arms_df, "response_by_arm"), (noise_df, "rare_event_noise")):
        save_table(df, f"robustness_{name}")

    from src.visualization.robustness_plots import plot_learning_curve, plot_seed_stability, plot_split_half
    figs = [plot_seed_stability(seeds_df, budgets), plot_learning_curve(curve_df),
            plot_split_half({o: v.pop("_scatter") for o, v in stability.items()}, stability)]

    diff_cols = [f"diff_{budget_tag(k)}" for k in budgets]
    seed_summary = {o: {c: {"mean": float(g[c].mean()), "sd": float(g[c].std(ddof=1)), "min": float(g[c].min()),
                            "max": float(g[c].max()), "all_positive": bool((g[c] > 0).all()),
                            "all_negative": bool((g[c] < 0).all())}
                        for c in ["qini_uplift", "qini_response", *diff_cols]}
                    for o, g in seeds_df.groupby("outcome")}
    out = {
        "mode": mode, "learners": learners, "seeds": rcfg["seeds"], "estimator": "propensity-weighted (Hajek) on test",
        "seed_summary": seed_summary, "stability": stability,
        "split_half": halves_df.to_dict(orient="records"), "learning_curve": curve_df.to_dict(orient="records"),
        "response_by_arm": arms_df.to_dict(orient="records"),
        "rare_event_noise": noise_df.to_dict(orient="records"),
        "timings_seconds": timings, "figures": figs,
        "notes": ("Seed runs vary LightGBM bagging/column sampling (and causal-forest subsampling) with the data "
                  "fixed; split-half runs vary the training data with the seed fixed. Differences are uplift minus "
                  "response targeting in incremental outcomes on the 2.8M-row test split."),
    }
    save_json(out, "robustness")
    return {o: {"diff_10pct_mean": s[f"diff_{budget_tag(0.1)}"]["mean"], "diff_10pct_sd": s[f"diff_{budget_tag(0.1)}"]["sd"],
                **{k: round(v, 3) for k, v in stability[o].items() if isinstance(v, float)}}
            for o, s in seed_summary.items()}
