"""Uplift-model evaluation on a randomized test set: curves, areas, uplift@k, calibration, BLP.

Every metric here is estimated from *held-out randomized data*: a scoring ranks users, and inside
each top-``k`` group the treatment effect is estimated by the treated-vs-control difference in
outcome rates. That is valid because treatment is randomized independently of X (so it is also
independent of any score computed from X); it is an estimate of the effect of the *targeting
policy* "treat the top k", not of any individual effect.

Notation for the top ``p`` users of a ranking (``p`` may be fractional, see ties below):
``n_t(p), n_c(p)`` treated / control counts, ``y_t(p), y_c(p)`` outcome sums in each arm,
``N, N_T`` population and treated totals.

* **lift** ``(p)  = y_t/n_t - y_c/n_c`` -- average effect among the targeted (``uplift@k`` at p = kN).
* **uplift curve** ``U(p) = lift(p) * (n_t + n_c)`` -- estimated incremental outcomes if the top ``p``
  were treated instead of nobody (Gutierrez & Gerardy 2017; sklift ``uplift_curve``; causalml calls
  this the *cumulative gain* curve).
* **Qini curve** ``Q(p) = y_t - y_c * n_t / n_c`` -- Radcliffe (2007): incremental outcomes counted
  on the treated scale (sklift ``qini_curve``).

Normalisation used for the reported areas (so they do not grow with the test-set size): curves
are divided by the population (``u = U / N``) or treated total (``q = Q / N_T``), so both curves
end at the overall difference in means (the ATE) at ``f = 1``; the x-axis is the targeted fraction
``f = p / N``.

* ``auuc``            = int_0^1 u(f) df   (random targeting gives u(1) / 2).
* ``qini_coefficient``= int_0^1 [q(f) - f q(1)] df, the area between the Qini curve and the
  random-targeting line (0 in expectation for a random score; units: outcomes per user).
* ``qini_normalized`` = qini_coefficient / (same area for the *perfect* ranking), the quantity
  sklift's ``qini_auc_score(negative_effect=True)`` estimates.
* ``auuc_normalized`` = (auuc - u(1)/2) / (perfect auuc - u(1)/2), the quantity sklift's
  ``uplift_auc_score`` estimates.

The model-curve areas (numerators) equal sklift's exactly when scores have no ties (tested). The
normalisers differ: the perfect rankings *always* contain blocks of identical (T, Y) rows (e.g. all
control responders last), inside which the true curve is curved; sklift evaluates the perfect
curve only at block ends and joins them with a straight line, whereas here it is evaluated at every
position. The perfect curves agree at sklift's evaluation points; qini_normalized typically differs
from sklift by <0.1% (relative), auuc_normalized by several percent. With binary outcomes the
"perfect" rankings assume *every* treated responder is incremental; they are unattainable here, so
normalised values are tiny and the unnormalised coefficient is the primary metric.

**Ties.** Tree models give many users identical scores (e.g. the S-learner predicts exactly zero
effect for many cells). Ties are resolved in *expectation over random tie-breaking*: within a
block of tied scores the cumulative arm counts and outcome sums are linearly interpolated, which
is exactly their expected value when the block is ordered at random. Results are therefore
deterministic and invariant to row order. The Qini/uplift values are then the curve evaluated at
the *expected* counts (not the expected value of the curve, which is nonlinear in the counts).
(sklift instead evaluates only at tie-block ends and interpolates the *curve* linearly; the two
agree exactly when there are no ties.)

**Bootstrap.** CIs resample users with Poisson(1) weights (the standard large-n approximation to
the multinomial bootstrap). The ranking is held fixed (scores come from models fit on train;
resampling the test set does not refit them), so each replicate is one weighted pass
(``np.bincount``) per scoring over precomputed rank segments; paired differences between two
scorings use the *same* weights in every replicate. The targeted fraction is defined on the
original test ordering; in a replicate the realised fraction differs from ``f`` by O(1/sqrt(N)).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)

# Column order of the cumulative-count arrays (key = 2 * treatment + outcome).
C_Y0, C_Y1, T_Y0, T_Y1 = 0, 1, 2, 3
PERFECT_QINI = "__perfect_qini"
PERFECT_UPLIFT = "__perfect_uplift"


def _as_binary(a: np.ndarray | pd.Series, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    uniq = np.unique(arr)
    if not np.isin(uniq, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1, found values {uniq[:5]}")
    return arr.astype(np.int64)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den with 0 where den == 0 (sklift's convention at the very top of a ranking)."""
    out = np.zeros(np.broadcast_shapes(np.shape(num), np.shape(den)), dtype=np.float64)
    np.divide(num, den, out=out, where=np.asarray(den) != 0)
    return out


def make_grid(n: int, grid_points: int | Sequence[float] | None) -> np.ndarray:
    """Targeted fractions at which curves are evaluated: ``grid_points + 1`` evenly spaced values in
    [0, 1], every integer position ``0..n`` (exact, for small n) when ``grid_points`` is None, or an
    explicit increasing sequence of fractions."""
    if grid_points is None:
        return np.arange(n + 1, dtype=np.float64) / n
    if np.ndim(grid_points) == 1:
        g = np.asarray(grid_points, dtype=np.float64)
        if g[0] != 0.0 or g[-1] != 1.0 or np.any(np.diff(g) <= 0):
            # totals (ATE, normalisations) are read at the last grid point, so it must be f = 1
            raise ValueError("explicit grid must be strictly increasing fractions from 0 to 1 inclusive")
        return g
    return np.linspace(0.0, 1.0, int(grid_points) + 1)


@dataclass
class _Ranking:
    key: np.ndarray        # per row (original order): segment id * 4 + 2 * t + y
    n_segments: int
    lo_idx: np.ndarray     # per grid point: index into the boundary set of the tie block start
    hi_idx: np.ndarray     # ... and of the tie block end
    alpha: np.ndarray      # per grid point: position of p inside its tie block, in [0, 1]


class UpliftEvaluator:
    """Evaluate many scorings of the same randomized evaluation set, with paired bootstrap CIs.

    Args:
        y: binary outcome.
        t: binary randomized treatment assignment.
        grid_points: number of intervals of the targeted-fraction grid (None = every position).
    """

    def __init__(self, y: np.ndarray | pd.Series, t: np.ndarray | pd.Series,
                 grid_points: int | Sequence[float] | None = 1000, sample_weight: np.ndarray | None = None):
        self.y = _as_binary(y, "y")
        self.t = _as_binary(t, "t")
        if len(self.y) != len(self.t):
            raise ValueError(f"y and t lengths differ: {len(self.y)} vs {len(self.t)}")
        self.n = len(self.y)
        if self.t.min() == self.t.max():
            raise ValueError("Evaluation set needs both treated and control units.")
        self.grid = make_grid(self.n, grid_points)
        self.sample_weight = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.shape != (self.n,) or not np.isfinite(sw).all() or (sw < 0).any():
                raise ValueError("sample_weight must be a finite, non-negative vector with one entry per row")
            self.sample_weight = sw
        self._rankings: dict[str, _Ranking] = {}

    # ------------------------------------------------------------------ rankings
    def add(self, name: str, score: np.ndarray | pd.Series) -> None:
        """Register a scoring (higher = target first)."""
        s = np.asarray(score, dtype=np.float64)
        if s.shape != (self.n,):
            raise ValueError(f"score {name!r} has shape {s.shape}, expected ({self.n},)")
        if not np.isfinite(s).all():
            raise ValueError(f"score {name!r} contains NaN/inf values")
        order = np.argsort(-s, kind="stable")
        s_sorted = s[order]
        bounds = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1, self.n]
        p = self.grid * self.n
        p_int = np.round(p)
        p = np.where(np.abs(p - p_int) < 1e-9, p_int, p)   # f * n = 3.0000000000000004 must mean 3 rows
        j = np.clip(np.searchsorted(bounds, p, side="right") - 1, 0, len(bounds) - 2)
        lo, hi = bounds[j], bounds[j + 1]
        alpha = np.where(hi > lo, (p - lo) / np.maximum(hi - lo, 1), 0.0)
        needed = np.unique(np.r_[lo, hi, 0, self.n])
        seg_sorted = np.repeat(np.arange(len(needed) - 1, dtype=np.int64), np.diff(needed))
        key = np.empty(self.n, dtype=np.int64)
        key[order] = seg_sorted * 4
        key += 2 * self.t + self.y
        self._rankings[name] = _Ranking(key=key, n_segments=len(needed) - 1,
                                        lo_idx=np.searchsorted(needed, lo), hi_idx=np.searchsorted(needed, hi),
                                        alpha=alpha)

    def add_perfect(self) -> None:
        """Register the two 'perfect' rankings used for normalisation (sklift definitions).

        Qini: treated responders first, control responders last (``y * (2t - 1)``).
        Uplift curve: sklift's ``perfect_uplift_curve`` score ``2 * (y == t) + (y or t)``.
        """
        self.add(PERFECT_QINI, self.y * (2 * self.t - 1))
        cr = np.sum((self.y == 1) & (self.t == 0))
        tn = np.sum((self.y == 0) & (self.t == 1))
        summand = self.y if cr > tn else self.t
        self.add(PERFECT_UPLIFT, 2 * (self.y == self.t) + summand)

    @property
    def names(self) -> list[str]:
        return list(self._rankings)

    # ------------------------------------------------------------------ counts
    def counts(self, name: str, weights: np.ndarray | None = None) -> np.ndarray:
        """Cumulative [control y=0, control y=1, treated y=0, treated y=1] counts at each grid point.

        Shape ``(len(grid), 4)``; with ``weights`` (per row, original order) the counts are weighted,
        on top of the evaluator's ``sample_weight`` (e.g. stabilised inverse-propensity weights, which
        turn every arm mean into a Hajek estimator).
        """
        r = self._rankings[name]
        if self.sample_weight is not None:
            weights = self.sample_weight if weights is None else weights * self.sample_weight
        seg = np.bincount(r.key, weights=weights, minlength=r.n_segments * 4).reshape(r.n_segments, 4)
        cum = np.vstack([np.zeros((1, 4)), np.cumsum(seg, axis=0)])
        lo, hi = cum[r.lo_idx], cum[r.hi_idx]
        return lo + r.alpha[:, None] * (hi - lo)

    def bootstrap_counts(self, n_reps: int, seed: int, names: Sequence[str] | None = None
                         ) -> dict[str, np.ndarray]:
        """Poisson-bootstrap cumulative counts, shape ``(n_reps, len(grid), 4)`` per scoring.

        All scorings share the same weights in each replicate, so differences are paired.
        """
        names = list(names or self._rankings)
        rng = np.random.default_rng(seed)
        out = {nm: np.empty((n_reps, len(self.grid), 4)) for nm in names}
        for b in range(n_reps):
            w = rng.poisson(1.0, size=self.n).astype(np.float64)
            for nm in names:
                out[nm][b] = self.counts(nm, w)
        return out


# ---------------------------------------------------------------------- metrics from counts
def curves_from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    """Lift, uplift and Qini curves from cumulative counts ``(..., G, 4)`` (see module docstring)."""
    n_c = c[..., C_Y0] + c[..., C_Y1]
    n_t = c[..., T_Y0] + c[..., T_Y1]
    y_c, y_t = c[..., C_Y1], c[..., T_Y1]
    lift = _safe_div(y_t, n_t) - _safe_div(y_c, n_c)
    uplift = lift * (n_t + n_c)
    qini = y_t - y_c * _safe_div(n_t, n_c)
    big_n = (n_t + n_c)[..., -1:]
    big_nt = n_t[..., -1:]
    return {"lift": lift, "uplift": uplift, "qini": qini,
            "uplift_norm": uplift / big_n, "qini_norm": qini / big_nt,
            "treated_rate": _safe_div(y_t, n_t), "control_rate": _safe_div(y_c, n_c),
            "n_targeted": n_t + n_c}


def _area(yv: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.trapezoid(yv, grid, axis=-1)


def _grid_index(grid: np.ndarray, k: float) -> int:
    i = int(np.argmin(np.abs(grid - k)))
    if not np.isclose(grid[i], k, atol=1e-9):
        raise ValueError(f"budget {k} is not on the evaluation grid; use a grid that contains it")
    return i


def budget_tag(k: float) -> str:
    """Metric-name suffix for a budget fraction: 0.05 -> '5', 0.025 -> '2p5' (collision-free)."""
    return f"{k * 100:.6g}".replace(".", "p")


def metrics_from_counts(c: np.ndarray, grid: np.ndarray, budgets: Iterable[float],
                        perfect_qini: np.ndarray | None = None,
                        perfect_uplift: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Scalar metrics (one per leading index) from cumulative counts ``(..., G, 4)``."""
    cv = curves_from_counts(c)
    q, u = cv["qini_norm"], cv["uplift_norm"]
    ate = q[..., -1]
    out: dict[str, np.ndarray] = {
        "ate": ate,
        "qini_coefficient": _area(q, grid) - ate / 2.0,
        "auuc": _area(u, grid),
        "auuc_random": u[..., -1] / 2.0,
    }
    if perfect_qini is not None:
        qp = curves_from_counts(perfect_qini)["qini_norm"]
        out["qini_normalized"] = out["qini_coefficient"] / (_area(qp, grid) - qp[..., -1] / 2.0)
    if perfect_uplift is not None:
        up = curves_from_counts(perfect_uplift)["uplift_norm"]
        rand = up[..., -1] / 2.0
        out["auuc_normalized"] = (out["auuc"] - out["auuc_random"]) / (_area(up, grid) - rand)
    for k in budgets:
        i = _grid_index(grid, k)
        tag = budget_tag(k)
        out[f"uplift_at_{tag}"] = cv["lift"][..., i]
        out[f"incremental_at_{tag}"] = cv["uplift"][..., i]
        out[f"treated_rate_at_{tag}"] = cv["treated_rate"][..., i]
        out[f"control_rate_at_{tag}"] = cv["control_rate"][..., i]
        # share of the treated group's outcomes that the control rate says would have happened anyway
        out[f"anyway_share_at_{tag}"] = _safe_div(cv["control_rate"][..., i], cv["treated_rate"][..., i])
    return out


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile bootstrap interval (NaNs dropped)."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def stabilized_ipw_weights(t: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """``t * pbar / e + (1 - t) * (1 - pbar) / (1 - e)``: inverse-propensity weights rescaled so each
    arm's weights sum (in expectation) to its own size. Arm means become Hajek estimators and the
    weighted arm sizes still add up to the number of targeted users."""
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(propensity, dtype=np.float64)
    pbar = t.mean()
    return t * pbar / e + (1 - t) * (1 - pbar) / (1 - e)


def evaluate_scorings(y: np.ndarray, t: np.ndarray, scores: Mapping[str, np.ndarray], *,
                      budgets: Sequence[float], n_boot: int, seed: int, grid_points: int = 1000,
                      pairs: Sequence[tuple[str, str]] = (), alpha: float = 0.05,
                      sample_weight: np.ndarray | None = None) -> dict[str, object]:
    """Point estimates + bootstrap CIs for every scoring, plus paired differences.

    ``sample_weight`` (optional, e.g. ``stabilized_ipw_weights``) weights every arm mean; the
    bootstrap multiplies it by the Poisson weights, so the weights themselves are held fixed.

    Returns a dict with
      * ``metrics``: tidy frame (scorer, metric, estimate, ci_low, ci_high),
      * ``differences``: tidy frame (scorer_a, scorer_b, metric, estimate of a - b, CI, p_boot),
        where ``p_boot`` is the two-sided bootstrap share of replicates on the other side of 0,
      * ``curves``: frame of point-estimate curves on a ``grid_points``-interval grid,
      * ``boot``: per scorer, per metric bootstrap samples (for downstream reuse).
    """
    ev = UpliftEvaluator(y, t, grid_points=grid_points, sample_weight=sample_weight)
    for nm, s in scores.items():
        ev.add(nm, s)
    ev.add_perfect()
    point_c = {nm: ev.counts(nm) for nm in ev.names}
    boot_c = ev.bootstrap_counts(n_boot, seed) if n_boot > 0 else {}

    def _metrics(cdict: Mapping[str, np.ndarray], nm: str) -> dict[str, np.ndarray]:
        return metrics_from_counts(cdict[nm], ev.grid, budgets, cdict[PERFECT_QINI], cdict[PERFECT_UPLIFT])

    rows, boot_m = [], {}
    for nm in scores:
        pm = _metrics(point_c, nm)
        bm = _metrics(boot_c, nm) if boot_c else {k: np.array([np.nan]) for k in pm}
        boot_m[nm] = bm
        for k, v in pm.items():
            lo, hi = percentile_ci(bm[k], alpha)
            rows.append({"scorer": nm, "metric": k, "estimate": float(v), "ci_low": lo, "ci_high": hi})
    diff_rows = []
    for a, b in pairs:
        pa, pb = _metrics(point_c, a), _metrics(point_c, b)
        for k in pa:
            d = boot_m[a][k] - boot_m[b][k]
            lo, hi = percentile_ci(d, alpha)
            fin = d[np.isfinite(d)]
            p_boot = float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))) if fin.size else float("nan")
            diff_rows.append({"scorer_a": a, "scorer_b": b, "metric": k, "estimate": float(pa[k] - pb[k]),
                              "ci_low": lo, "ci_high": hi, "p_boot": p_boot})
    curve_rows = []
    for nm in scores:
        cv = curves_from_counts(point_c[nm])
        curve_rows.append(pd.DataFrame({"scorer": nm, "fraction": ev.grid, "qini": cv["qini_norm"],
                                        "uplift": cv["uplift_norm"], "lift": cv["lift"],
                                        "incremental": cv["uplift"]}))
    return {"metrics": pd.DataFrame(rows), "differences": pd.DataFrame(diff_rows),
            "curves": pd.concat(curve_rows, ignore_index=True), "boot": boot_m,
            "boot_curves": {nm: curves_from_counts(c)["qini_norm"] for nm, c in boot_c.items()}, "grid": ev.grid}


# ---------------------------------------------------------------------- functional API
def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, grid_points: int | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Radcliffe Qini curve Q(p) (count scale) at targeted counts ``p``; tie-averaged."""
    ev = UpliftEvaluator(y, t, grid_points)
    ev.add("s", score)
    return ev.grid * ev.n, curves_from_counts(ev.counts("s"))["qini"]


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, grid_points: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Uplift (cumulative incremental outcomes) curve U(p) at targeted counts ``p``; tie-averaged."""
    ev = UpliftEvaluator(y, t, grid_points)
    ev.add("s", score)
    return ev.grid * ev.n, curves_from_counts(ev.counts("s"))["uplift"]


def uplift_metrics(y: np.ndarray, t: np.ndarray, score: np.ndarray, budgets: Sequence[float] = (),
                   grid_points: int | None = None) -> dict[str, float]:
    """All scalar metrics for one scoring (no bootstrap)."""
    ev = UpliftEvaluator(y, t, grid_points)
    ev.add("s", score)
    ev.add_perfect()
    m = metrics_from_counts(ev.counts("s"), ev.grid, budgets, ev.counts(PERFECT_QINI), ev.counts(PERFECT_UPLIFT))
    return {k: float(v) for k, v in m.items()}


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float) -> float:
    """Treated-minus-control outcome rate among the top ``k`` fraction (sklift ``strategy='overall'``
    when ``k * n`` is an integer and there are no ties; otherwise tie/fraction-averaged)."""
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}")
    ev = UpliftEvaluator(y, t, grid_points=[0.0, k] if k == 1 else [0.0, k, 1.0])
    ev.add("s", score)
    return float(curves_from_counts(ev.counts("s"))["lift"][1])


# ---------------------------------------------------------------------- effect estimates
def diff_in_means(y: np.ndarray, t: np.ndarray, alpha: float = 0.05,
                  weights: np.ndarray | None = None) -> dict[str, float]:
    """Difference in outcome means (treated - control) with a Neyman (unpooled) normal CI.

    Valid for the average effect in any subgroup defined by X alone, because treatment is
    randomized independently of X. With ``weights`` (inverse-propensity weights) each arm mean is a
    Hajek (self-normalised) estimator and the SE uses the linearised variance
    ``sum w^2 (y - m)^2 / (sum w)^2`` per arm, treating the weights as fixed.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t).astype(bool)
    n1, n0 = int(t.sum()), int((~t).sum())
    if n1 == 0 or n0 == 0:
        return {"estimate": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_treated": n1, "n_control": n0, "treated_rate": float("nan"), "control_rate": float("nan")}
    if weights is None:
        m1, m0 = y[t].mean(), y[~t].mean()
        v1 = y[t].var(ddof=1) if n1 > 1 else 0.0
        v0 = y[~t].var(ddof=1) if n0 > 1 else 0.0
        se = float(np.sqrt(v1 / n1 + v0 / n0))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w1, w0 = w[t], w[~t]
        m1, m0 = np.average(y[t], weights=w1), np.average(y[~t], weights=w0)
        var1 = np.sum(w1 ** 2 * (y[t] - m1) ** 2) / w1.sum() ** 2
        var0 = np.sum(w0 ** 2 * (y[~t] - m0) ** 2) / w0.sum() ** 2
        se = float(np.sqrt(var1 + var0))
    z = stats.norm.ppf(1 - alpha / 2)
    est = float(m1 - m0)
    return {"estimate": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se,
            "n_treated": n1, "n_control": n0, "treated_rate": float(m1), "control_rate": float(m0)}


def rank_groups(score: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """Equal-size groups by descending score (1 = highest). Ties are broken at random (seeded), so a
    tied block that straddles a group boundary is split randomly rather than by row order."""
    s = np.asarray(score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(s)), -s))
    groups = np.empty(len(s), dtype=np.int64)
    groups[order] = np.arange(len(s)) * n_groups // len(s) + 1
    return groups


def uplift_by_group(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_groups: int = 10, seed: int = 0,
                    alpha: float = 0.05, weights: np.ndarray | None = None) -> pd.DataFrame:
    """GATES-style calibration table: mean predicted effect vs observed difference in means per
    score group (group 1 = highest predicted effect). Groups are formed from the score only;
    ``weights`` (inverse-propensity) make each group's arm means Hajek estimators."""
    g = rank_groups(score, n_groups, seed)
    y, t, s = np.asarray(y), np.asarray(t), np.asarray(score, dtype=np.float64)
    rows = []
    for k in range(1, n_groups + 1):
        m = g == k
        d = diff_in_means(y[m], t[m], alpha, None if weights is None else np.asarray(weights)[m])
        rows.append({"group": k, "n": int(m.sum()), "mean_predicted": float(s[m].mean()),
                     "min_predicted": float(s[m].min()), "max_predicted": float(s[m].max()),
                     "observed": d["estimate"], "se": d["se"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                     "treated_rate": d["treated_rate"], "control_rate": d["control_rate"]})
    return pd.DataFrame(rows)


def blp_heterogeneity_test(y: np.ndarray, t: np.ndarray, score: np.ndarray, baseline: np.ndarray,
                           propensity: float | np.ndarray) -> dict[str, float]:
    """Best Linear Predictor of the CATE (Chernozhukov, Demirer, Duflo & Fernandez-Val, 2018/2023).

    Regression on held-out data (score and baseline come from models fit on *other* data):

        Y = a0 + a1 * B(X) + b1 * (T - p) + b2 * (T - p) * (S(X) - mean S) + e,

    weighted by 1 / (p(X) (1 - p(X))), with p the (design or estimated) propensity. Under
    randomization, ``b1`` is the ATE and ``b2 = Cov(tau, S) / Var(S)``: ``b2 = 0`` means the score
    carries no information about effect heterogeneity (or there is none), ``b2 = 1`` means S is
    well calibrated. We test H0: b2 = 0 against b2 > 0 with heteroskedasticity-robust (HC1) SEs.
    ``B(X)`` (a proxy for E[Y | T=0, X]) only reduces variance; any fixed function is valid.
    """
    y = np.asarray(y, dtype=np.float64)
    e = np.broadcast_to(np.asarray(propensity, dtype=np.float64), y.shape)
    tt = np.asarray(t, dtype=np.float64) - e
    w = 1.0 / (e * (1.0 - e))
    s = np.asarray(score, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    sd_s = s.std()
    if sd_s == 0:
        raise ValueError("BLP needs a non-constant score")
    X = np.column_stack([np.ones_like(y), b - b.mean(), tt, tt * (s - s.mean())])
    Xw = X * w[:, None]
    xtx_inv = np.linalg.inv(Xw.T @ X)
    beta = xtx_inv @ (Xw.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    meat = (Xw * resid[:, None] ** 2).T @ Xw
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    z2 = beta[3] / se[3]
    return {"ate_beta1": float(beta[2]), "ate_beta1_se": float(se[2]),
            "beta2": float(beta[3]), "beta2_se": float(se[3]),
            "beta2_ci_low": float(beta[3] - 1.96 * se[3]), "beta2_ci_high": float(beta[3] + 1.96 * se[3]),
            "beta2_z": float(z2), "p_value_one_sided": float(stats.norm.sf(z2)),
            "p_value_two_sided": float(2 * stats.norm.sf(abs(z2))), "score_sd": float(sd_s), "n": int(n)}
