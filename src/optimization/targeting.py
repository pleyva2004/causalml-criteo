"""Phase 12: budget-constrained targeting policies and their off-policy evaluation.

Decision problem
----------------
Treating user ``i`` costs ``c`` and, if it causes a conversion, earns ``V``. With a CATE estimate
``tau_hat(x_i)`` the expected incremental value of treating user ``i`` is
``EIV_i = tau_hat(x_i) * V - c``. The conversion value ``V`` and cost ``c`` are ILLUSTRATIVE
assumptions from ``configs/config.yaml`` (the benchmark is sub-sampled non-uniformly, so absolute
rates are not real-world rates); the cost/value sensitivity sweep is the real result.

Policies (each maps scores + a budget fraction to a boolean "treat" mask)
    treat_all, treat_none, random-k%, response-k% (top-k% by P(conversion | X)),
    uplift-k% (top-k% by tau_hat), expected_value-k% (top-k% by EIV, only where EIV > 0, so it may
    treat fewer than k%).

Evaluation (randomized TEST split only; VAL is used only to pick budgets in the sensitivity sweep)
    Assumptions: (1) treatment is unconfounded given X with propensity e(x) = P(T=1 | X). Criteo v2 has a
    constant *nominal* ratio of 0.85, but treatment is measurably predictable from X in this data
    (Lin-adjusted ATEs and a LightGBM propensity give smaller effects than the raw difference in means; the
    treated share drifts within score-ranked groups), so we weight by the estimated per-row propensity
    e_hat (``propensity`` column of the scores file; constant p only if that column is absent); (2) the policy is a function of
    X only (scores come from models fit on train, tie-breaks and the random policy use a
    label-independent RNG), hence independent of T given X; (3) SUTVA / no interference; (4) the
    policy value is the *incremental* outcome of offering treatment to the selected users versus
    treating nobody. Under these, for pi(x) in {0, 1}

        Delta_HT(pi)     = sum_i pi_i [ T_i Y_i / p - (1 - T_i) Y_i / (1 - p) ]      (Horvitz-Thompson)
        Delta_Hajek(pi)  = n_pi * ( mean(Y | T=1, pi=1) - mean(Y | T=0, pi=1) )        (self-normalized)

    with e_hat in place of p, and the Hajek form as the PRIMARY estimator, are consistent for
    n * E[pi(X) * tau(X)] (total incremental conversions on the test set) if e_hat is consistent. The
    unnormalized HT with e_hat and the constant-p difference in means are kept as reference columns
    (``inc_ht``, ``inc_dim``); their gap to the primary estimate is reported in targeting.json.
    Nothing here validates the *ranking* quality of tau_hat; it measures the realized value of acting
    on it. Exposure is not used: the policy is an *offer* of treatment (intention-to-treat).

Bootstrap: resampling rows is equivalent to a multinomial draw over the cells
(policy-membership pattern x T x Y). We resample those cell counts, which makes a 2.8M-row bootstrap
cost microseconds and, because every policy is scored on the SAME resampled rows, gives exact paired
bootstrap differences between policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.load import SPLIT_CODES, load_criteo
from src.utils import (get_logger, load_config, load_json, repo_path, save_json, save_table, timer)

log = get_logger(__name__)

OUTCOME = "conversion"
PROP_BINS = int(load_config()["optimization"]["propensity_bins"])
BUDGETED = ["random", "response", "uplift", "expected_value"]   # policies with a budget knob
ALL_POLICIES = [*BUDGETED, "treat_all"]
LEARNERS = ["s_learner", "t_learner", "x_learner", "dr_learner", "causal_forest", "class_transformation"]

ASSUMPTIONS = [
    "Conversion value V and treatment cost c are illustrative assumptions, not data; the benchmark is "
    "sub-sampled non-uniformly, so absolute conversion rates and incrementality are not real-world values.",
    "Treatment is assumed unconfounded given X. Weights use the estimated propensity e_hat(x) from the "
    "scores file (test range reported in the JSON); the nominal 0.85 ratio is not used because treatment is "
    "predictable from X in this sample. Correctness rests on e_hat being well estimated (overlap holds: "
    "e_hat is bounded away from 0 and 1).",
    "Primary estimator is the normalized (Hajek) IPW; unnormalized HT with e_hat and the constant-p "
    "difference in means are reported for reference.",
    "Bootstrap: Poisson-weight bootstrap over cells (policy pattern x T x Y x e_hat bin); within a cell the "
    "IPW weight is its mean, so CIs ignore within-bin weight variation (small with 40 bins).",
    "Policies depend only on X (scores fit on train; tie-breaks and the random policy use a "
    "label-independent RNG), so pi is independent of T given X.",
    "SUTVA: no interference between users. The value is an intention-to-treat offer effect; exposure "
    "is self-selected and not used.",
    "Policy value = incremental conversions versus treating nobody, estimated by inverse-propensity "
    "(Horvitz-Thompson) weighting; the Hajek form is reported alongside.",
    "Budgets in the sensitivity sweep are chosen on the validation split and evaluated on test; "
    "the expected_value policy uses the fixed rule tau_hat > c/V with no tuning.",
    "EIV thresholding assumes tau_hat is on the probability scale; a miscalibrated tau_hat shifts the "
    "expected_value cutoff but not the ranking used by the uplift policy.",
]


# --------------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------------
def tiebreak_key(n: int, seed: int) -> np.ndarray:
    """Fixed random permutation used to break score ties. Depends on ``seed`` only, never on T or Y."""
    return np.random.default_rng(seed).permutation(n)


def rank_of(score: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """0-based rank by descending score, ties resolved by ``tie`` (deterministic, label-free)."""
    order = np.lexsort((tie, -np.asarray(score, dtype=np.float64)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def n_budget(n: int, frac: float) -> int:
    """Number of users a fractional budget selects (rounded, never above n)."""
    return int(min(n, max(0, round(frac * n))))


def budget_mask(score: np.ndarray, frac: float, tie: np.ndarray) -> np.ndarray:
    """Treat exactly ``round(frac * n)`` users with the highest ``score``."""
    return rank_of(score, tie) < n_budget(len(score), frac)


def eiv(tau: np.ndarray, value: float, cost: float) -> np.ndarray:
    """Expected incremental value of treating each user: tau_hat * V - c."""
    return np.asarray(tau, dtype=np.float64) * value - cost


def eiv_mask(tau: np.ndarray, frac: float, value: float, cost: float, tie: np.ndarray) -> np.ndarray:
    """Top-``frac`` users by EIV, restricted to EIV > 0 (so it can treat fewer than ``frac``)."""
    e = eiv(tau, value, cost)
    return (rank_of(e, tie) < n_budget(len(e), frac)) & (e > 0)


def random_mask(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random subset of exactly ``round(frac * n)`` users."""
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=n_budget(n, frac), replace=False)] = True
    return mask


# --------------------------------------------------------------------------------------------
# Off-policy estimators
# --------------------------------------------------------------------------------------------
def _as_prop(e: Any, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(e, dtype=np.float64), (n,))


def ht_terms(t: np.ndarray, y: np.ndarray, e: Any) -> np.ndarray:
    """Per-user Horvitz-Thompson pseudo-outcome T*Y/e - (1-T)*Y/(1-e); its sum over pi=1 is Delta_HT."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    e = _as_prop(e, len(t))
    return t * y / e - (1.0 - t) * y / (1.0 - e)


def ht_value(mask: np.ndarray, t: np.ndarray, y: np.ndarray, e: Any) -> float:
    """Unnormalized IPW (Horvitz-Thompson) total incremental outcome of policy ``mask``."""
    return float(ht_terms(t, y, e)[mask].sum())


def ipw_value(mask: np.ndarray, t: np.ndarray, y: np.ndarray, e: Any) -> float:
    """PRIMARY estimator: normalized (Hajek) IPW, n_pi * (weighted mean Y | T=1 - weighted mean Y | T=0)
    among pi=1 with weights T/e and (1-T)/(1-e). Lower variance than HT and robust to a treated share
    that drifts within score-ranked groups. With constant ``e`` it reduces to ``dim_value``."""
    t = np.asarray(t, dtype=np.float64)[mask]
    y = np.asarray(y, dtype=np.float64)[mask]
    e = _as_prop(e, len(mask))[mask]
    w1, w0 = t / e, (1.0 - t) / (1.0 - e)
    if w1.sum() <= 0 or w0.sum() <= 0:
        return 0.0
    return float(mask.sum() * ((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()))


def dim_value(mask: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Reference only: n_pi * difference in means (implicitly a constant propensity; biased when
    treatment depends on X)."""
    t = np.asarray(t)[mask]
    y = np.asarray(y, dtype=np.float64)[mask]
    if (t == 1).sum() == 0 or (t == 0).sum() == 0:
        return 0.0
    return float(mask.sum() * (y[t == 1].mean() - y[t == 0].mean()))


@dataclass
class Evaluation:
    """Point estimates (exact, from rows) and bootstrap replicates for policies scored on the same rows.

    ``inc`` is the primary Hajek-IPW estimate; ``ht`` (unnormalized IPW) and ``dim`` (constant-propensity
    difference in means) are secondary / reference columns.
    """

    names: list[str]
    n_total: int
    n_treated: dict[str, float]
    inc: dict[str, float]
    ht: dict[str, float]
    dim: dict[str, float]
    boot_n: dict[str, np.ndarray]
    boot_inc: dict[str, np.ndarray]


def evaluate_masks(masks: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray, e: Any, reps: int,
                   rng: np.random.Generator, prop_bins: int = 40, chunk: int = 250) -> Evaluation:
    """Score several policies on one sample; joint (hence paired) Poisson-weight bootstrap.

    Poisson bootstrap: every row gets an independent Poisson(1) weight. Rows sharing a cell
    (policy-membership pattern x T x Y x propensity bin) are exchangeable for every statistic, so the
    summed weight of a cell is Poisson(n_cell); we draw that directly, in chunks of replicates. Within a
    cell the IPW weight is replaced by the cell mean (``prop_bins`` propensity quantile bins), which makes
    a 2.8M-row bootstrap cheap. Point estimates are computed exactly from rows, not from the bins.
    """
    names = list(masks)
    k, n = len(names), len(t)
    t = np.asarray(t, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    e = _as_prop(e, n)
    n_treated = {nm: float(masks[nm].sum()) for nm in names}
    point = {
        "inc": {nm: ipw_value(masks[nm], t, y, e) for nm in names},
        "ht": {nm: ht_value(masks[nm], t, y, e) for nm in names},
        "dim": {nm: dim_value(masks[nm], t, y) for nm in names},
    }

    w = t / e + (1 - t) / (1 - e)
    if np.ptp(e) == 0 or prop_bins <= 1:
        kb, n_kb = np.zeros(n, dtype=np.int64), 1
    else:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, prop_bins + 1)[1:-1]))
        kb, n_kb = np.searchsorted(edges, e, side="right"), len(edges) + 1
    code = np.zeros(n, dtype=np.int64)
    for j, nm in enumerate(names):
        code |= masks[nm].astype(np.int64) << j
    ncell = 4 * n_kb * 2**k
    cell = ((code * 2 + t) * 2 + y) * n_kb + kb
    counts = np.bincount(cell, minlength=ncell).astype(np.float64)
    wmean = np.bincount(cell, weights=w, minlength=ncell) / np.maximum(counts, 1.0)
    idx = np.arange(ncell)
    c_y, c_t, c_code = (idx // n_kb) % 2, (idx // (2 * n_kb)) % 2, idx // (4 * n_kb)
    sel = np.stack([(c_code >> j) & 1 for j in range(k)], axis=1).astype(np.float64)   # (cells, k)

    boot_n, boot_inc = [], []
    for start in range(0, reps, chunk):
        cnt = rng.poisson(counts, size=(min(chunk, reps - start), ncell)).astype(np.float64)
        sw = cnt * wmean
        n_pi = cnt @ sel
        b1, a1 = (sw * c_t) @ sel, (sw * c_t * c_y) @ sel
        b0, a0 = (sw * (1 - c_t)) @ sel, (sw * (1 - c_t) * c_y) @ sel
        with np.errstate(divide="ignore", invalid="ignore"):
            hj = np.where((b1 > 0) & (b0 > 0), n_pi * (a1 / b1 - a0 / b0), 0.0)
        boot_n.append(n_pi)
        boot_inc.append(hj)
    bn, bi = np.concatenate(boot_n), np.concatenate(boot_inc)
    return Evaluation(
        names=names, n_total=n, n_treated=n_treated, inc=point["inc"], ht=point["ht"], dim=point["dim"],
        boot_n={nm: bn[:, j] for j, nm in enumerate(names)},
        boot_inc={nm: bi[:, j] for j, nm in enumerate(names)},
    )


def value_metrics(inc: Any, n_treated: Any, n_total: int, value: float, cost: float) -> dict[str, np.ndarray]:
    """Business metrics from incremental conversions and users treated (scalars or bootstrap arrays)."""
    inc = np.asarray(inc, dtype=np.float64)
    n_tr = np.asarray(n_treated, dtype=np.float64)
    safe = np.maximum(n_tr, 1e-12)
    net = value * inc - cost * n_tr
    return {
        "inc": inc,
        "inc_per_100k": inc / n_total * 1e5,
        "inc_per_treated": np.where(n_tr > 0, inc / safe, np.nan),
        "net": net,
        "net_per_100k": net / n_total * 1e5,
        "roi": np.where((n_tr > 0) & (cost > 0), net / max(cost, 1e-12) / safe, np.nan),
    }


def _ci(x: np.ndarray, level: float) -> tuple[float, float]:
    if np.all(np.isnan(x)):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(x, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def policy_row(ev: Evaluation, name: str, value: float, cost: float, level: float) -> dict[str, float]:
    """One table row: point estimate and percentile CI for every business metric of policy ``name``."""
    pt = value_metrics(ev.inc[name], ev.n_treated[name], ev.n_total, value, cost)
    bt = value_metrics(ev.boot_inc[name], ev.boot_n[name], ev.n_total, value, cost)
    row: dict[str, float] = {"n_treated": ev.n_treated[name], "frac_treated": ev.n_treated[name] / ev.n_total,
                             "inc_ht": ev.ht[name], "inc_dim": ev.dim[name]}
    for key in ("inc", "inc_per_100k", "inc_per_treated", "net", "net_per_100k", "roi"):
        row[key] = float(pt[key])
        row[f"{key}_lo"], row[f"{key}_hi"] = _ci(bt[key], level)
    return row


def diff_row(ev: Evaluation, a: str, b: str, value: float, cost: float, level: float,
             prefix: str) -> dict[str, float]:
    """Paired bootstrap difference (a - b) of incremental conversions and net value."""
    ma = value_metrics(ev.inc[a], ev.n_treated[a], ev.n_total, value, cost)
    mb = value_metrics(ev.inc[b], ev.n_treated[b], ev.n_total, value, cost)
    ba = value_metrics(ev.boot_inc[a], ev.boot_n[a], ev.n_total, value, cost)
    bb = value_metrics(ev.boot_inc[b], ev.boot_n[b], ev.n_total, value, cost)
    out: dict[str, float] = {}
    for key in ("inc", "net"):
        out[f"{prefix}_{key}"] = float(ma[key] - mb[key])
        out[f"{prefix}_{key}_lo"], out[f"{prefix}_{key}_hi"] = _ci(ba[key] - bb[key], level)
    return out


# --------------------------------------------------------------------------------------------
# Scores: real file, or a clearly-labelled dev stand-in
# --------------------------------------------------------------------------------------------
def scores_file(mode: str) -> Path:
    return repo_path(load_config()["paths"]["processed_dir"]) / f"scores_{mode}.parquet"


def build_standin_scores(mode: str = "dev") -> Path:
    """Dev-only stand-in with the real scores schema: T-learner + response model on the train split.

    Written to ``scores_<mode>_standin.parquet``; never valid for results/ (full mode refuses it).
    """
    from src.features.build_features import build_matrices
    from src.models.tree_models import make_lgbm

    cfg = load_config()
    n_est = cfg["optimization"]["standin_estimators"]
    df = load_criteo(mode)
    train = df[df["split"] == SPLIT_CODES["train"]]
    held = df[df["split"].isin([SPLIT_CODES["val"], SPLIT_CODES["test"]])].reset_index(drop=True)
    _, (x_tr, x_held) = build_matrices(train, held, kind="tree")
    out = held[["row_id", "split", "treatment", "visit", "conversion", "exposure"]].copy()
    for outcome in ("visit", "conversion"):
        y = train[outcome].to_numpy()
        tr = train["treatment"].to_numpy() == 1
        m1 = make_lgbm("classifier", n_estimators=n_est).fit(x_tr[tr], y[tr])
        m0 = make_lgbm("classifier", n_estimators=n_est).fit(x_tr[~tr], y[~tr])
        mr = make_lgbm("classifier", n_estimators=n_est).fit(x_tr, y)
        out[f"cate_t_learner_{outcome}"] = m1.predict_proba(x_held)[:, 1] - m0.predict_proba(x_held)[:, 1]
        out[f"response_{outcome}"] = mr.predict_proba(x_held)[:, 1]
    path = repo_path(cfg["paths"]["processed_dir"]) / f"scores_{mode}_standin.parquet"
    out.to_parquet(path, index=False)
    log.warning("STAND-IN scores written to %s (T-learner on dev train). NOT real CATE results.", path)
    return path


def resolve_scores(mode: str, scores_path: str | Path | None) -> tuple[Path, bool]:
    """Pick the scores file. Returns (path, is_standin). Full mode requires the real file."""
    if scores_path is not None:
        path = Path(scores_path)
        if not path.exists():
            raise FileNotFoundError(f"scores_path {path} does not exist.")
        return path, "standin" in path.name
    real = scores_file(mode)
    if real.exists():
        return real, False
    if mode == "full":
        raise FileNotFoundError(f"{real} not found. Run `uv run causalml cate --mode full` first; "
                                "targeting never falls back to stand-in scores in full mode.")
    standin = real.with_name(f"scores_{mode}_standin.parquet")
    log.warning("!!! %s missing: using STAND-IN scores (%s). Dev numbers are NOT real results !!!", real, standin)
    if not standin.exists():
        build_standin_scores(mode)
    return standin, True


def select_learner(columns: list[str], is_standin: bool) -> str:
    """Learner chosen on validation by the cate stage; config default (logged) if unavailable."""
    cfg = load_config()["optimization"]
    chosen: str | None = None
    if not is_standin:
        try:
            chosen = load_json("cate")["selected_learner"][OUTCOME]
        except (FileNotFoundError, KeyError, TypeError):
            log.warning("cate.json has no selected_learner for %s; falling back to config default %s",
                        OUTCOME, cfg["default_learner"])
    else:
        log.warning("stand-in scores: using the only available learner column")
    for cand in [chosen, cfg["default_learner"], *LEARNERS]:
        if cand and f"cate_{cand}_{OUTCOME}" in columns:
            if cand != chosen:
                log.info("using learner %s", cand)
            return cand
    raise KeyError(f"no cate_<learner>_{OUTCOME} column in scores file (columns: {columns})")


def _load_scores(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path).names
    cate_cols = [c for c in schema if c.startswith("cate_") and c.endswith(f"_{OUTCOME}")]
    cols = ["split", "treatment", OUTCOME, f"response_{OUTCOME}", *cate_cols]
    if "propensity" in schema:
        cols.append("propensity")
    else:
        log.warning("no `propensity` column in %s: falling back to the constant empirical share "
                    "(difference-in-means weights; biased if treatment depends on X)", path.name)
    missing = [c for c in cols if c not in schema]
    if missing:
        raise KeyError(f"{path} lacks columns {missing}")
    df = pq.read_table(path, columns=cols).to_pandas()
    if df["split"].dtype == object or str(df["split"].dtype).startswith("str"):
        df["split"] = df["split"].map(SPLIT_CODES)
    return df


# --------------------------------------------------------------------------------------------
# Curves, sensitivity, main stage
# --------------------------------------------------------------------------------------------
@dataclass
class Split:
    """Arrays for one split plus the label-free rankings every policy needs."""

    t: np.ndarray
    y: np.ndarray
    resp: np.ndarray
    tau: np.ndarray
    p: float
    e: np.ndarray          # per-row propensity (estimated e_hat if available, else constant p)
    tie: np.ndarray
    resp_rank: np.ndarray
    tau_rank: np.ndarray
    rand_rank: np.ndarray

    @property
    def n(self) -> int:
        return len(self.t)


def make_split(df: pd.DataFrame, learner: str, seed: int) -> Split:
    t = df["treatment"].to_numpy().astype(np.int64)
    y = df[OUTCOME].to_numpy().astype(np.int64)
    resp = df[f"response_{OUTCOME}"].to_numpy()
    tau = df[f"cate_{learner}_{OUTCOME}"].to_numpy()
    tie = tiebreak_key(len(t), seed)
    p = float(t.mean())
    if "propensity" in df.columns:
        clip = float(load_config()["optimization"]["propensity_clip"])
        e = np.clip(df["propensity"].to_numpy().astype(np.float64), clip, 1 - clip)
    else:
        e = np.full(len(t), p)
    rand = np.random.default_rng(seed + 1).random(len(t))
    return Split(t, y, resp, tau, p, e, tie, rank_of(resp, tie), rank_of(tau, tie),
                 rank_of(rand, tie))


def masks_at(s: Split, frac: float, ratio: float, value: float) -> dict[str, np.ndarray]:
    """All policies' masks at one budget; ``ratio`` = c/V sets the EIV > 0 cutoff (tau_hat > ratio)."""
    m = n_budget(s.n, frac)
    return {
        "random": s.rand_rank < m,
        "response": s.resp_rank < m,
        "uplift": s.tau_rank < m,
        "expected_value": (s.tau_rank < m) & (s.tau * value - ratio * value > 0),
        "treat_all": np.ones(s.n, dtype=bool),
    }


def policy_curves(s: Split, budgets: np.ndarray, value: float, cost: float, reps: int, level: float,
                  seed: int) -> pd.DataFrame:
    """Every budgeted policy at every budget, with CIs and paired differences vs response and random."""
    rng = np.random.default_rng(seed)
    ratio = cost / value
    rows: list[dict[str, Any]] = []
    for b in budgets:
        ev = evaluate_masks(masks_at(s, b, ratio, value), s.t, s.y, s.e, reps, rng, PROP_BINS)
        for name in ALL_POLICIES:
            row: dict[str, Any] = {"policy": name, "budget": float(b),
                                   **policy_row(ev, name, value, cost, level)}
            if name in BUDGETED:
                row.update(diff_row(ev, name, "response", value, cost, level, "diff_vs_response"))
                row.update(diff_row(ev, name, "random", value, cost, level, "diff_vs_random"))
            if name == "random":
                # sanity: E[random-k%] = k * treat_all (oracle-free check), paired bootstrap
                d = ev.boot_inc["random"] - b * ev.boot_inc["treat_all"]
                row["random_minus_k_treat_all"] = ev.inc["random"] - b * ev.inc["treat_all"]
                row["random_minus_k_treat_all_lo"], row["random_minus_k_treat_all_hi"] = _ci(d, level)
            rows.append(row)
    return pd.DataFrame(rows)


def val_optimal_budgets(val: Split, budgets: np.ndarray, value: float, ratios: list[float]
                        ) -> dict[float, dict[str, float]]:
    """Per c/V ratio, the budget maximizing net value on VALIDATION for random/response/uplift.

    Net value at budget g is V * Delta_IPW - c * n_treated, where Delta_IPW is the Hajek-IPW value of the
    prefix of the policy's ranking (cumulative weighted sums). Random uses its expectation g * Delta_IPW(all), because
    a single random draw's val noise is not information about the policy. Budget 0 (treat nobody)
    is always allowed; ties go to the smaller budget.
    """
    w1, w0 = val.t / val.e, (1 - val.t) / (1 - val.e)
    grid = np.concatenate([[0.0], budgets])
    m = np.array([n_budget(val.n, g) for g in grid])

    def hajek_prefix(order: np.ndarray) -> np.ndarray:
        cs = {k: np.concatenate([[0.0], np.cumsum(v[order])]) for k, v in
              (("b1", w1), ("a1", w1 * val.y), ("b0", w0), ("a0", w0 * val.y))}
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.arange(len(order) + 1) * (cs["a1"] / cs["b1"] - cs["a0"] / cs["b0"])
        return np.nan_to_num(h)

    curves = {}
    for name, rank in (("response", val.resp_rank), ("uplift", val.tau_rank)):
        curves[name] = hajek_prefix(np.argsort(rank))[m]
    curves["random"] = grid * ipw_value(np.ones(val.n, dtype=bool), val.t, val.y, val.e)
    out: dict[float, dict[str, float]] = {}
    for r in ratios:
        out[r] = {}
        for name, inc in curves.items():
            net = value * inc - r * value * m
            out[r][name] = float(grid[int(np.argmax(net))])
    return out


def sensitivity_table(test: Split, opt: dict[float, dict[str, float]], value: float, reps: int, level: float,
                      seed: int) -> pd.DataFrame:
    """Evaluate val-chosen budgets on test for each c/V ratio; show who wins and when to treat everyone."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for r, budgets in opt.items():
        cost = r * value
        masks = {
            "random": test.rand_rank < n_budget(test.n, budgets["random"]),
            "response": test.resp_rank < n_budget(test.n, budgets["response"]),
            "uplift": test.tau_rank < n_budget(test.n, budgets["uplift"]),
            "expected_value": test.tau * value - cost > 0,       # fixed rule, no budget cap, no tuning
            "treat_all": np.ones(test.n, dtype=bool),
        }
        ev = evaluate_masks(masks, test.t, test.y, test.e, reps, rng, PROP_BINS)
        nets = {nm: value * ev.inc[nm] - cost * ev.n_treated[nm] for nm in ALL_POLICIES}
        top = max(nets.values())
        tol = 1e-9 * max(1.0, abs(top))
        tied = [nm for nm in ALL_POLICIES if nets[nm] >= top - tol]          # exact ties, e.g. all at 100%
        priority = ["treat_all", "expected_value", "uplift", "response", "random"]
        best = next(nm for nm in priority if nm in tied)
        for name in ALL_POLICIES:
            row = {"cost_value_ratio": r, "cost": cost, "policy": name,
                   "val_chosen_budget": budgets.get(name, np.nan),
                   **policy_row(ev, name, value, cost, level),
                   **diff_row(ev, name, "treat_all", value, cost, level, "diff_vs_treat_all"),
                   "is_best": name == best, "tied_for_best": name in tied}
            rows.append(row)
        rows.append({"cost_value_ratio": r, "cost": cost, "policy": "treat_none", "val_chosen_budget": 0.0,
                     "n_treated": 0.0, "frac_treated": 0.0, "inc": 0.0, "net": 0.0, "net_per_100k": 0.0,
                     "is_best": False, "tied_for_best": False})
    return pd.DataFrame(rows)


def run_targeting(mode: str = "dev", scores_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate targeting policies on the randomized test split and write results (stage ``targeting``)."""
    from src.visualization.targeting_plots import make_targeting_figures

    cfg = load_config()
    opt_cfg = cfg["optimization"]
    value, cost = float(opt_cfg["conversion_value"]), float(opt_cfg["treatment_cost"])
    ratio0 = cost / value
    reps, level = int(opt_cfg["bootstrap_reps"]), float(opt_cfg["ci_level"])
    seed = cfg["seed"] + int(opt_cfg["tiebreak_seed_offset"])
    headline = [float(b) for b in cfg["uplift"]["budgets"]]
    grid = np.unique(np.round(np.linspace(0, 1, int(cfg["uplift"]["curve_points"]) + 1)[1:], 6))
    budgets = np.unique(np.concatenate([grid, headline, [1.0]]))

    path, is_standin = resolve_scores(mode, scores_path)
    if is_standin and mode == "full":
        raise RuntimeError("Stand-in scores must never be used in full mode (results/).")
    df = _load_scores(path)
    learner = select_learner(list(df.columns), is_standin)
    log.info("scores=%s learner=%s outcome=%s c/V=%.5f", path.name, learner, OUTCOME, ratio0)
    val = make_split(df[df["split"] == SPLIT_CODES["val"]].reset_index(drop=True), learner, seed)
    test = make_split(df[df["split"] == SPLIT_CODES["test"]].reset_index(drop=True), learner, seed + 100)
    del df
    if test.n == 0 or val.n == 0:
        raise ValueError("scores file has no val or test rows")

    with timer("policy curves", log):
        curves = policy_curves(test, budgets, value, cost, reps, level, seed)
    with timer("sensitivity", log):
        opt = val_optimal_budgets(val, budgets, value, [float(r) for r in opt_cfg["cost_value_ratios"]])
        sens = sensitivity_table(test, opt, value, reps, level, seed + 1)

    # Headline table: budgeted policies at the configured budgets, plus treat_all / treat_none.
    hb = curves[curves["budget"].isin(headline) & curves["policy"].isin(BUDGETED)]
    all_row = curves[(curves["policy"] == "treat_all") & (curves["budget"] == 1.0)].copy()
    none_row = pd.DataFrame([{"policy": "treat_none", "budget": 0.0, "n_treated": 0.0, "frac_treated": 0.0,
                              "inc": 0.0, "inc_per_100k": 0.0, "net": 0.0, "net_per_100k": 0.0}])
    table = pd.concat([hb, all_row, none_row], ignore_index=True)
    save_table(table, "targeting_policy_values")
    save_table(sens, "targeting_sensitivity")
    save_table(curves, "targeting_curves")

    # Diagnostics: ATE from the treat-all policy (= blanket break-even c/V), HT vs Hajek agreement, sanity.
    ta = all_row.iloc[0]
    ate, ate_lo, ate_hi = (float(ta[k]) / test.n for k in ("inc", "inc_lo", "inc_hi"))     # same estimator
    sanity = curves[(curves["policy"] == "random") & curves["budget"].isin(headline)][
        ["budget", "inc", "random_minus_k_treat_all", "random_minus_k_treat_all_lo",
         "random_minus_k_treat_all_hi"]]
    sanity = sanity.assign(k_times_treat_all=sanity["budget"] * float(ta["inc"]))
    hc = curves[curves["budget"].isin(headline) | (curves["policy"] == "treat_all")]
    scale = hc["inc"].abs().clip(lower=1)
    est_gap = {"max_relative_gap_HT_vs_primary": float(((hc["inc_ht"] - hc["inc"]).abs() / scale).max()),
               "max_relative_gap_DiM_vs_primary": float(((hc["inc_dim"] - hc["inc"]).abs() / scale).max()),
               "treat_all_primary": float(ta["inc"]), "treat_all_ht": float(ta["inc_ht"]),
               "treat_all_dim": float(ta["inc_dim"]),
               "ate_dim_test": float(ta["inc_dim"]) / test.n, "ate_ht_test": float(ta["inc_ht"]) / test.n}
    best_by_ratio = {
        f"{r:g}": s.loc[s["is_best"], "policy"].iloc[0]
        for r, s in sens[sens["policy"] != "treat_none"].groupby("cost_value_ratio")
        if s["is_best"].any()
    }
    zero_best = [f"{r:g}" for r, s in sens.groupby("cost_value_ratio")
                 if s.loc[s["policy"] != "treat_none", "net"].max() <= 0]
    result = {
        "mode": mode, "scores_file": path.name, "scores_are_standin": is_standin,
        "outcome": OUTCOME, "selected_learner": learner,
        "n_test": test.n, "n_val": val.n, "propensity_test": {"constant_share": test.p, "used": "estimated e_hat" if not np.allclose(test.e, test.p) else "constant",
                            "min": float(test.e.min()), "max": float(test.e.max()),
                            "mean": float(test.e.mean())},
        "primary_estimator": "Hajek-normalized IPW with per-row propensity; HT and constant-p DiM are references",
        "value_per_conversion": value, "cost_per_treatment": cost, "cost_value_ratio_default": ratio0,
        "ate_test": {"estimate": ate, "ci_lo": ate_lo, "ci_hi": ate_hi,
                     "note": "treat_all incremental conversions per user; c/V above this makes blanket "
                             "treatment unprofitable"},
        "bootstrap_reps": reps, "ci_level": level, "headline_budgets": headline,
        "assumptions": ASSUMPTIONS,
        "policy_table": table.to_dict(orient="records"),
        "optimal_budget_by_cost_ratio_val": {f"{r:g}": v for r, v in opt.items()},
        "sensitivity": sens.to_dict(orient="records"),
        "best_policy_by_cost_ratio_test": best_by_ratio,
        "ratios_where_no_policy_has_positive_net_value": zero_best,
        "sanity_random_vs_k_times_treat_all": sanity.to_dict(orient="records"),
        "estimator_comparison": est_gap,
    }
    save_json(result, "targeting")
    make_targeting_figures(curves, sens, ratio0, headline)

    ref = table[table["budget"] == 0.2]
    return {
        "learner": learner, "standin": is_standin, "ate_test": ate, "break_even_cost_ratio": ate,
        "inc_per_100k_at_20pct": {r["policy"]: round(float(r["inc_per_100k"]), 2) for _, r in ref.iterrows()},
        "best_policy_by_cost_ratio": best_by_ratio,
    }
