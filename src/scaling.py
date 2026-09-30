"""Phase 14: scaling from development mode to the full 14M-row dataset.

Every measurement runs in a fresh spawned process so that peak resident memory (``ru_maxrss``) belongs to that
one job, not to whatever ran before it. For each model and training-set size we record training time,
inference time on the test split, peak memory, and held-out quality, which doubles as a learning curve:
how much does more data buy?
"""

from __future__ import annotations

import multiprocessing as mp
import resource
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger, get_run_mode, load_config, save_json, save_table, set_run_mode

log = get_logger(__name__)

MODELS = ("logistic_regression", "lightgbm")


def _peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def _measure(model: str, n_train: int | None, mode: str, seed: int) -> dict[str, Any]:
    """Fit one model on ``n_train`` stratified training rows and evaluate on the full test split.

    Runs inside a child process; imports are local so the parent stays light.
    """
    import psutil
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.data.load import FEATURES, load_criteo, stratified_subsample
    from src.features.build_features import build_matrices
    from src.models.tree_models import make_lgbm

    cols = [*FEATURES, "treatment", "visit", "conversion", "split"]
    t0 = time.perf_counter()
    train = load_criteo(mode, split="train", columns=cols)
    test = load_criteo(mode, split="test", columns=cols)
    load_s = time.perf_counter() - t0
    if n_train is not None and n_train < len(train):
        train = stratified_subsample(train, n_train / len(train), seed)
    kind = "linear" if model == "logistic_regression" else "tree"
    t0 = time.perf_counter()
    _, (X_tr, X_te) = build_matrices(train, test, kind=kind)
    features_s = time.perf_counter() - t0
    y_tr, y_te = train["conversion"].to_numpy(), test["conversion"].to_numpy()
    del train, test
    rss_before_fit = psutil.Process().memory_info().rss / 1e9

    cfg = load_config()
    if model == "logistic_regression":
        est = LogisticRegression(C=cfg["scaling"]["logistic_C"], max_iter=cfg["scaling"]["logistic_max_iter"])
    else:
        est = make_lgbm("classifier", seed=seed, n_estimators=cfg["scaling"]["lgbm_n_estimators"])
    t0 = time.perf_counter()
    est.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = est.predict_proba(X_te)[:, 1]
    predict_s = time.perf_counter() - t0
    return {
        "model": model, "n_train": len(y_tr), "n_features": X_tr.shape[1], "n_test": len(y_te),
        "load_s": load_s, "features_s": features_s, "fit_s": fit_s, "predict_s": predict_s,
        "predict_s_per_1m": predict_s / len(y_te) * 1e6,
        "rss_before_fit_gb": rss_before_fit, "peak_rss_gb": _peak_rss_gb(),
        "test_roc_auc": roc_auc_score(y_te, p), "test_pr_auc": average_precision_score(y_te, p),
    }


def _run_isolated(model: str, n_train: int | None, mode: str, seed: int) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(_measure, (model, n_train, mode, seed))


def run_scaling(mode: str = "dev") -> dict[str, Any]:
    """Measure time, memory and test quality for each model across training-set sizes."""
    set_run_mode(mode)
    cfg = load_config()
    seed = cfg["seed"]
    n_train_total = len(__import__("src.data.load", fromlist=["load_criteo"]).load_criteo(
        mode, split="train", columns=["split"]))
    sizes = [s for s in cfg["robustness"]["train_sizes"] if s < n_train_total] + [None]
    rows = []
    for model in MODELS:
        for n in sizes:
            log.info("scaling: %s with %s training rows", model, "all" if n is None else f"{n:,}")
            rows.append(_run_isolated(model, n, mode, seed))
    table = pd.DataFrame(rows)
    table["n_jobs"] = cfg["compute"]["n_jobs"]
    save_table(table, "scaling_measurements")

    from src.visualization.scaling_plots import plot_scaling
    plot_scaling(table)

    full = table[table.n_train == table.n_train.max()].set_index("model")
    out = {
        "mode": get_run_mode(),
        "machine": _machine_info(),
        "n_jobs": cfg["compute"]["n_jobs"],
        "largest_train": {m: full.loc[m, ["n_train", "fit_s", "predict_s_per_1m", "peak_rss_gb",
                                            "test_pr_auc", "test_roc_auc"]].to_dict() for m in full.index},
        "measurements": table.to_dict(orient="records"),
        "notes": ("Each row is a separate spawned process; peak_rss_gb is that process's peak resident memory "
                  "including data loading. Test metrics are on the full test split, so rows are comparable "
                  "across training sizes (a learning curve)."),
    }
    save_json(out, "scaling")
    return {m: {k: round(float(v), 4) for k, v in d.items()} for m, d in out["largest_train"].items()}


def _machine_info() -> dict[str, Any]:
    import os
    import platform

    import psutil
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}


if __name__ == "__main__":
    set_run_mode("dev")
    log.info(run_scaling("dev"))
    np.random.seed(0)
