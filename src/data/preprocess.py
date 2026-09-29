"""Phase 1 (data understanding) and Phase 2 (EDA) for the Criteo Uplift v2.1 benchmark.

Variable definitions (Diemert et al. 2021; the release itself documents only the first two lines
loosely, the rest is our reading of the paper):

* ``treatment`` -- randomized *assignment*: 1 = user was eligible to be targeted by the advertiser's
  ads, 0 = held out (control). This is the only randomized variable.
* ``exposure`` -- the user was *actually shown* at least one ad. It is post-randomization and
  self-selected (auction outcomes, how active the user is), so ``treatment == 0`` implies
  ``exposure == 0`` but many treated users are never exposed. Assignment != exposure: comparing
  treated vs control estimates an intention-to-treat (ITT) effect of *eligibility*, not the effect
  of seeing an ad; comparing exposed vs unexposed is confounded.
* ``visit`` / ``conversion`` -- outcomes observed in the 2-week test window (visited the advertiser's
  site / purchased). ``visit == 0`` implies ``conversion == 0``.

Everything here is descriptive. No number in this module is a causal effect except the randomized
treated-vs-control contrast, which is an ITT difference in this benchmark sample (the data is
sub-sampled non-uniformly, so absolute rates are not real-world rates).

Interpretation caveats: the point masses at each feature's modal value are *suspected* default or
missing-value encodings after anonymization -- the release does not document this. The numeric order
of the hashed categoricals (f1, f3, f4, f5, f6, f8, f9, f11) is meaningless, so wherever a categorical
enters a correlation or a distance it is represented by level frequency or by per-level indicators.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.data.load import FEATURES, load_criteo
from src.utils import get_logger, load_config, save_json, save_table, timer

log = get_logger(__name__)

OUTCOMES: list[str] = ["exposure", "visit", "conversion"]
LABEL_COLS: list[str] = ["treatment", "exposure", "visit", "conversion"]

VARIABLE_DEFINITIONS: dict[str, str] = {
    "treatment": "Randomized assignment: 1 = user eligible to be targeted by the advertiser's ads, 0 = held-out control.",
    "exposure": ("1 = the user was actually shown at least one ad. Post-randomization and self-selected "
                 "(depends on auctions and user activity), NOT randomized. Zero for every control user."),
    "visit": "Outcome: user visited the advertiser's website within the 2-week test window.",
    "conversion": "Outcome: user converted (purchased) within the 2-week window; implies visit.",
    "assignment_vs_exposure": ("Treated-vs-control is the randomized contrast (intention-to-treat, effect of eligibility). "
                               "Exposed-vs-unexposed compares self-selected groups and is confounded by user activity."),
}


def _cfg() -> dict[str, Any]:
    return load_config()["eda"]


# --------------------------------------------------------------------------------------- statistics
def wilson_ci(k: np.ndarray | float, n: np.ndarray | float, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (well behaved for tiny rates, unlike Wald).

    Returns ``(low, high)``; where ``n == 0`` both are NaN.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def standardized_mean_difference(x_treated: np.ndarray, x_control: np.ndarray) -> float:
    """(mean_T - mean_C) / sqrt((var_T + var_C) / 2). Zero if both groups are constant.

    The sample SMD is the balance metric used instead of p-values: with n ~ 14M any difference is
    "significant", so magnitude (conventionally < 0.1 is balanced) is what matters.
    """
    xt = np.asarray(x_treated, dtype=float)
    xc = np.asarray(x_control, dtype=float)
    pooled = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2) if len(xt) > 1 and len(xc) > 1 else 0.0
    return 0.0 if pooled == 0 else float((xt.mean() - xc.mean()) / pooled)


def rates_by_group(df: pd.DataFrame, group: str | Sequence[str], outcomes: Sequence[str] = OUTCOMES,
                   alpha: float = 0.05) -> pd.DataFrame:
    """Per-group ``n`` plus, for each outcome, ``<o>_rate``, ``<o>_ci_low``, ``<o>_ci_high`` (Wilson)."""
    g = df.groupby(group, observed=True)
    out = g.size().rename("n").to_frame()
    for o in outcomes:
        k = g[o].sum()
        out[f"{o}_count"] = k
        out[f"{o}_rate"] = k / out["n"]
        lo, hi = wilson_ci(k.to_numpy(), out["n"].to_numpy(), alpha)
        out[f"{o}_ci_low"], out[f"{o}_ci_high"] = lo, hi
    return out.reset_index()


def itt_difference(df: pd.DataFrame, outcome: str, alpha: float = 0.05) -> dict[str, float]:
    """Treated minus control difference in an outcome rate with a Wald CI (randomized contrast; ITT)."""
    t, c = df.loc[df.treatment == 1, outcome], df.loc[df.treatment == 0, outcome]
    pt, pc = t.mean(), c.mean()
    se = np.sqrt(pt * (1 - pt) / len(t) + pc * (1 - pc) / len(c))
    z = stats.norm.ppf(1 - alpha / 2)
    d = pt - pc
    return {"treated_rate": float(pt), "control_rate": float(pc), "difference": float(d),
            "ci_low": float(d - z * se), "ci_high": float(d + z * se), "relative_lift": float(d / pc) if pc else float("nan")}


# --------------------------------------------------------------------------------------- phase 1
def duplicate_counts(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> dict[str, int]:
    """Exact duplicate rows on features+labels and on features only (first occurrence not counted).

    Duplicates are kept downstream: they are distinct users who share default feature vectors.
    """
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    return {
        "duplicate_rows_features_and_labels": int(df.duplicated(subset=[*features, *label_cols]).sum()),
        "duplicate_rows_features_only": int(df.duplicated(subset=list(features)).sum()),
    }


def structural_violations(df: pd.DataFrame) -> dict[str, int]:
    """Rechecks the data-generating constraints: T=0 => E=0 and V=0 => C=0."""
    return {
        "control_exposed": int(((df.treatment == 0) & (df.exposure == 1)).sum()),
        "conversion_without_visit": int(((df.visit == 0) & (df.conversion == 1)).sum()),
    }


def dataset_summary(df: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    """One-row-per-metric summary table (``metric``, ``value``) of size, dtypes, labels and integrity checks."""
    dup = duplicate_counts(df)
    viol = structural_violations(df)
    ctrl, trt = df[df.treatment == 0], df[df.treatment == 1]
    rows: list[tuple[str, Any]] = [
        ("rows", len(df)), ("n_features", len(FEATURES)),
        ("n_continuous_features", len(continuous)), ("n_categorical_features", len(categorical)),
        ("feature_dtypes", ",".join(sorted({str(df[f].dtype) for f in FEATURES}))),
        ("label_dtypes", ",".join(sorted({str(df[c].dtype) for c in LABEL_COLS}))),
        ("missing_values_total", int(df[[*FEATURES, *LABEL_COLS]].isna().sum().sum())),
        *dup.items(), *viol.items(),
        ("treatment_share", df.treatment.mean()),
        ("exposure_rate_overall", df.exposure.mean()), ("visit_rate_overall", df.visit.mean()),
        ("conversion_rate_overall", df.conversion.mean()),
        ("exposure_rate_treated", trt.exposure.mean()), ("exposure_rate_control", ctrl.exposure.mean()),
        ("visit_rate_treated", trt.visit.mean()), ("visit_rate_control", ctrl.visit.mean()),
        ("conversion_rate_treated", trt.conversion.mean()), ("conversion_rate_control", ctrl.conversion.mean()),
        ("conversions", int(df.conversion.sum())), ("conversions_control", int(ctrl.conversion.sum())),
        ("visits", int(df.visit.sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def feature_profile(df: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str],
                    tail_quantile: float = 0.999) -> pd.DataFrame:
    """Per-feature profile: type, cardinality, modal point mass, range, quantiles, moments and tail ratio.

    ``mode_share`` is the share of rows at the most frequent value. Large shares suggest default /
    missing-value encodings after anonymization (an interpretation, not documented). ``tail_ratio`` is
    (max - p99.9) / (p99.9 - p50): how far the extreme tail extends beyond the bulk.
    """
    qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, tail_quantile]
    kinds = {**{f: "continuous" for f in continuous}, **{f: "categorical" for f in categorical}}
    rows = []
    for f in FEATURES:
        x = df[f]
        vc = x.value_counts()
        q = x.quantile(sorted(set(qs)))
        p50, ptail = q.loc[0.5], q.loc[tail_quantile]
        rows.append({
            "feature": f, "type": kinds[f], "n_unique": int(x.nunique()),
            "mode_value": float(vc.index[0]), "mode_share": float(vc.iloc[0] / len(x)),
            "second_mode_share": float(vc.iloc[1] / len(x)) if len(vc) > 1 else 0.0,
            "min": float(x.min()), **{f"q{qq:g}": float(q.loc[qq]) for qq in sorted(set(qs))},
            "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
            "tail_ratio": float((x.max() - ptail) / (ptail - p50)) if ptail > p50 else float("nan"),
            "missing": int(x.isna().sum()),
        })
    return pd.DataFrame(rows)


def mode_coupling(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement of the "is at modal value" indicators across features.

    If two features sit at their modal value for exactly the same rows (agreement ~1), their point
    masses were probably produced by one shared default/missing mechanism (interpretation, not documented).
    Columns: ``feature_a``, ``feature_b``, ``agreement`` (share of rows where both indicators match),
    ``jaccard`` (overlap of the two "at mode" sets).
    """
    at_mode = {f: (df[f] == df[f].mode().iloc[0]).to_numpy() for f in FEATURES}
    rows = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            both, either = (at_mode[a] & at_mode[b]).sum(), (at_mode[a] | at_mode[b]).sum()
            rows.append({"feature_a": a, "feature_b": b, "agreement": float((at_mode[a] == at_mode[b]).mean()),
                         "jaccard": float(both / either) if either else np.nan})
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


def exposure_funnel(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Assigned -> exposed -> visited -> converted, per arm and for treated split by exposure.

    Groups: ``control``, ``treated`` (all), ``treated_exposed``, ``treated_unexposed``. The last two are
    self-selected subgroups of the treated arm and must not be compared as if randomized.
    """
    groups = {
        "control": df[df.treatment == 0],
        "treated": df[df.treatment == 1],
        "treated_exposed": df[(df.treatment == 1) & (df.exposure == 1)],
        "treated_unexposed": df[(df.treatment == 1) & (df.exposure == 0)],
    }
    n_treated = len(groups["treated"])
    rows = []
    for name, g in groups.items():
        n = len(g)
        row: dict[str, Any] = {"group": name, "n": n,
                               "share_of_treated": n / n_treated if name.startswith("treated") else np.nan}
        for o in OUTCOMES:
            k = int(g[o].sum())
            lo, hi = wilson_ci(k, n, alpha)
            row.update({f"{o}_count": k, f"{o}_rate": k / n if n else np.nan,
                        f"{o}_ci_low": float(lo), f"{o}_ci_high": float(hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Counts, positive counts and rates of each binary variable (class-imbalance table)."""
    rows = []
    for c in LABEL_COLS:
        k = int(df[c].sum())
        lo, hi = wilson_ci(k, len(df))
        rows.append({"variable": c, "n": len(df), "positives": k, "negatives": len(df) - k,
                     "positive_rate": k / len(df), "ci_low": float(lo), "ci_high": float(hi),
                     "imbalance_ratio": (len(df) - k) / k if k else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- balance
def _pooled_level_table(x: pd.Series, treatment: pd.Series, min_count: int) -> pd.DataFrame:
    """Level x arm counts with levels rarer than ``min_count`` pooled into one 'other' row (value NaN)."""
    ct = pd.crosstab(x, treatment).reindex(columns=[0, 1], fill_value=0)
    ct.columns = ["control", "treated"]
    rare = (ct.sum(axis=1) < min_count)
    common = ct.loc[~rare]
    if rare.any():
        other = ct.loc[rare].sum().to_frame().T
        other.index = [np.nan]
        common = pd.concat([common, other])
    return common


def balance_table(df: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str],
                  min_level_count: int = 1000) -> pd.DataFrame:
    """Covariate balance between arms for every feature.

    * continuous: SMD of raw values, plus a two-sample KS statistic and p-value.
    * categorical: raw numeric values are random hash projections, so an SMD of raw values is
      meaningless. Instead: chi-square test of the level distribution (levels with fewer than
      ``min_level_count`` rows pooled), Cramer's V, and the per-level indicator SMD; the headline
      ``smd`` is the maximum |indicator SMD| over the non-pooled levels.

    ``mean_treated`` / ``mean_control`` are raw means for all features (informational for categoricals).
    """
    t_mask = (df.treatment == 1).to_numpy()
    rows = []
    for f in FEATURES:
        x = df[f].to_numpy()
        xt, xc = x[t_mask], x[~t_mask]
        row: dict[str, Any] = {"feature": f, "type": "continuous" if f in continuous else "categorical",
                               "mean_treated": float(xt.mean()), "mean_control": float(xc.mean())}
        if f in continuous:
            ks = stats.ks_2samp(xt, xc)
            row.update({"smd": standardized_mean_difference(xt, xc), "smd_basis": "raw value",
                        "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                        "chi2_stat": np.nan, "chi2_pvalue": np.nan, "cramers_v": np.nan, "n_levels_tested": np.nan})
        else:
            ct = _pooled_level_table(df[f], df["treatment"], min_level_count)
            n_t, n_c = ct["treated"].sum(), ct["control"].sum()
            chi2, p, _, _ = stats.chi2_contingency(ct.to_numpy())
            v = float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
            pt, pc = ct["treated"] / n_t, ct["control"] / n_c
            denom = np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2).replace(0, np.nan)
            level_smd = ((pt - pc) / denom).fillna(0.0)
            named = level_smd.loc[level_smd.index.notna()]
            row.update({"smd": float(named.abs().max()) if len(named) else 0.0, "smd_basis": "max |per-level indicator SMD|",
                        "ks_stat": np.nan, "ks_pvalue": np.nan, "chi2_stat": float(chi2), "chi2_pvalue": float(p),
                        "cramers_v": v, "n_levels_tested": int(len(ct))})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out


def interpret_balance(bal: pd.DataFrame, n_rows: int, thresholds: Sequence[float], n_treated: int, n_control: int) -> str:
    """One-sentence reading of the balance table; magnitude (SMD) over significance.

    Compares the worst |SMD| with the sampling-noise scale sqrt(1/n_T + 1/n_C) of a perfectly
    randomized experiment, so the text does not claim "noise" when the gap is far larger.
    """
    worst = bal.loc[bal.abs_smd.idxmax()]
    n_sig = int(((bal.ks_pvalue < 0.05) | (bal.chi2_pvalue < 0.05)).sum())
    noise = float(np.sqrt(1 / n_treated + 1 / n_control))
    verdict = ("within" if worst.abs_smd < thresholds[-1] else "ABOVE")
    return (f"With n={n_rows:,} rows even negligible imbalance gives tiny p-values ({n_sig} of {len(bal)} features have p<0.05), "
            f"so judge balance by effect size. The largest |SMD| is {worst.abs_smd:.4f} ({worst.feature}), {verdict} the conventional "
            f"{thresholds[-1]:g} threshold. A perfectly randomized comparison at this n would show |SMD| of order {noise:.4f} "
            f"(sqrt(1/n_T+1/n_C)); observed imbalances of {worst.abs_smd / noise:.0f}x that scale are therefore small in size but "
            f"not sampling noise. Likely cause (interpretation, not documented): the benchmark pools several advertiser tests that were "
            f"re-sampled to a common treatment ratio, so covariate mix can differ slightly by arm. Downstream causal estimators adjust for covariates.")


# --------------------------------------------------------------------------------------- association
def log_frequency(x: pd.Series) -> pd.Series:
    """log(1 + level count) encoding, used to give hashed categoricals a meaningful order."""
    return np.log1p(x.map(x.value_counts()).astype(float))


def spearman_matrix(df: pd.DataFrame, categorical: Sequence[str], outcomes: Sequence[str] = LABEL_COLS) -> pd.DataFrame:
    """Spearman correlation among features and labels.

    Categoricals enter as log level-frequency (their raw numeric order is arbitrary), with the
    ``(freq)`` suffix in the label. Frequencies are computed on the frame passed in.
    """
    enc = pd.DataFrame({(f"{f} (freq)" if f in categorical else f): (log_frequency(df[f]) if f in categorical else df[f])
                        for f in FEATURES})
    for o in outcomes:
        enc[o] = df[o].to_numpy()
    return enc.corr(method="spearman")


def conversion_by_feature(df: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str],
                          n_bins: int = 10, top_levels: int = 8, alpha: float = 0.05) -> pd.DataFrame:
    """Visit/conversion rates per arm within feature bins: association, not causation.

    Continuous features: the modal point mass as bin 0 ('mode') plus quantile bins of the remaining rows.
    Categoricals: the ``top_levels`` most frequent levels plus 'rest' (label ``L1..Lk`` by frequency rank,
    since the hashed values carry no order).
    """
    parts = []
    for f in FEATURES:
        x = df[f]
        if f in continuous:
            # The modal point mass becomes its own bin (label "mode"); the remaining rows are cut into
            # quantile bins so heavy point masses do not collapse the whole feature into one bin.
            xv = x.to_numpy()
            mode_val = x.mode().iloc[0]
            is_mode = xv == mode_val
            edges = np.unique(np.quantile(xv[~is_mode], np.linspace(0, 1, n_bins + 1))) if (~is_mode).any() else np.array([])
            b = np.zeros(len(xv), dtype=int)
            labels = {0: "mode"}
            if len(edges) >= 2:
                b[~is_mode] = 1 + np.clip(np.searchsorted(edges[1:-1], xv[~is_mode], side="right"), 0, len(edges) - 2)
                labels.update({i + 1: f"{edges[i]:.3g}-{edges[i + 1]:.3g}" for i in range(len(edges) - 1)})
            binned = pd.Series(b, index=df.index)
            order = sorted(labels)
        else:
            top = x.value_counts().index[:top_levels]
            rank = {v: i for i, v in enumerate(top)}
            binned = x.map(rank).fillna(top_levels).astype(int)
            order = list(range(top_levels + 1))
            labels = {i: (f"L{i + 1}" if i < top_levels else "rest") for i in order}
        tmp = pd.DataFrame({"bin": binned.to_numpy(), "treatment": df.treatment.to_numpy(),
                            "visit": df.visit.to_numpy(), "conversion": df.conversion.to_numpy()})
        r = rates_by_group(tmp, ["bin", "treatment"], ["visit", "conversion"], alpha)
        r["feature"] = f
        r["type"] = "continuous" if f in continuous else "categorical"
        r["bin_label"] = r["bin"].map(labels)
        r["arm"] = r["treatment"].map({0: "control", 1: "treatment"})
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def categorical_level_shares(df: pd.DataFrame, categorical: Sequence[str], top_levels: int = 8) -> pd.DataFrame:
    """Share of each arm at the ``top_levels`` most frequent levels of every categorical."""
    rows = []
    n_arm = df.treatment.value_counts()
    for f in categorical:
        top = df[f].value_counts().index[:top_levels]
        ct = pd.crosstab(df[f], df.treatment).reindex(top)
        for rank, (level, r) in enumerate(ct.iterrows(), start=1):
            rows.append({"feature": f, "rank": rank, "level": float(level),
                         "share_control": r.get(0, 0) / n_arm[0], "share_treatment": r.get(1, 0) / n_arm[1]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- orchestration
def run_eda(mode: str = "dev") -> dict[str, Any]:
    """Run Phase 1 + Phase 2 and write tables (``eda_*.csv``), ``eda.json`` and figures (``eda_*.png``).

    Rates, balance and funnel use every row; histograms and Spearman correlations use a seeded uniform
    sample of ``eda.sample_rows`` (documented in config). Returns headline numbers.
    """
    from src.visualization import eda_plots as P  # local import: keeps preprocess importable without matplotlib

    cfg, all_cfg = _cfg(), load_config()
    cont, cat = list(all_cfg["features"]["continuous"]), list(all_cfg["features"]["categorical"])
    with timer("load", log):
        df = load_criteo(mode, columns=[*FEATURES, *LABEL_COLS])
    n = len(df)
    log.info("EDA on %d rows (%s mode)", n, mode)
    alpha = cfg["ci_alpha"]

    # ---- phase 1
    with timer("phase 1", log):
        summary = dataset_summary(df, cont, cat)
        profile = feature_profile(df, cont, cat, cfg["tail_quantile"])
        targets = target_distribution(df)
        funnel = exposure_funnel(df, alpha)
        coupling = mode_coupling(df)
        viol = structural_violations(df)
        dup = duplicate_counts(df)
    save_table(summary, "eda_dataset_summary")
    save_table(profile, "eda_feature_profile")
    save_table(targets, "eda_target_distribution")
    save_table(funnel, "eda_exposure_funnel")
    save_table(coupling, "eda_mode_coupling")

    # ---- phase 2 tables
    with timer("phase 2 tables", log):
        arm = df.assign(arm=df.treatment.map({0: "control", 1: "treatment"}))
        by_arm = rates_by_group(arm, "arm", OUTCOMES, alpha)
        bal = balance_table(df, cont, cat, cfg["min_level_count"])
        by_feat = conversion_by_feature(df, cont, cat, cfg["n_bins"], cfg["top_levels"], alpha)
        shares = categorical_level_shares(df, cat, cfg["top_levels"])
    save_table(by_arm, "eda_rates_by_treatment")
    save_table(bal.drop(columns="abs_smd"), "eda_balance")
    save_table(by_feat, "eda_conversion_by_feature")
    save_table(shares, "eda_categorical_level_shares")

    rng = np.random.default_rng(all_cfg["seed"])
    sample = df.iloc[np.sort(rng.choice(n, size=min(cfg["sample_rows"], n), replace=False))].reset_index(drop=True)
    with timer("spearman", log):
        corr = spearman_matrix(sample, cat)
    save_table(corr, "eda_correlation", index=True)

    # ---- figures
    with timer("figures", log):
        P.plot_target_distribution(targets)
        P.plot_rates_by_arm(by_arm)
        P.plot_exposure_funnel(funnel, by_arm)
        P.plot_continuous_by_arm(sample, cont, profile)
        P.plot_categorical_by_arm(shares, cat)
        P.plot_love(bal, cfg["smd_thresholds"])
        P.plot_correlation(corr, len(sample))
        P.plot_conversion_by_feature(by_feat)

    # ---- json
    worst = bal.loc[bal.abs_smd.idxmax()]
    rates = {o: {"overall": float(df[o].mean()),
                 **{a: float(df.loc[df.treatment == t, o].mean()) for a, t in (("control", 0), ("treatment", 1))}}
             for o in OUTCOMES}
    itt = {o: itt_difference(df, o, alpha) for o in ("visit", "conversion")}
    modal = profile.set_index("feature")
    obj = {
        "mode": mode, "rows": n, "n_features": len(FEATURES),
        "variable_definitions": VARIABLE_DEFINITIONS,
        "feature_types": {"continuous": cont, "categorical": cat},
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "treatment_share": float(df.treatment.mean()), "rates": rates,
        "itt_difference_descriptive": itt,
        "target_distribution": targets.to_dict("records"),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": dup, "structural_violations": viol,
        "suspicious_values": {
            "interpretation": ("Dominant modal point masses are suspected default/missing-value encodings after "
                               "anonymization; this is an interpretation, the release does not document it."),
            "mode_share_by_feature": {f: float(modal.loc[f, "mode_share"]) for f in FEATURES},
            "mode_value_by_feature": {f: float(modal.loc[f, "mode_value"]) for f in FEATURES},
            "mode_coupling_top_pairs": coupling.head(6).to_dict("records"),
            "tail_ratio_continuous": {f: float(modal.loc[f, "tail_ratio"]) for f in cont},
        },
        "exposure_funnel": funnel.to_dict("records"),
        "balance": {"max_abs_smd": float(worst.abs_smd), "max_abs_smd_feature": worst.feature,
                    "interpretation": interpret_balance(bal, n, cfg["smd_thresholds"], int(df.treatment.sum()), int((df.treatment == 0).sum())),
                    "table": bal.drop(columns="abs_smd").to_dict("records")},
        "sample_rows_for_histograms_and_correlations": int(len(sample)),
        "spearman_note": "Hashed categoricals enter as log level-frequency because their numeric order is arbitrary.",
    }
    save_json(obj, "eda")

    return {"rows": n, "n_features": len(FEATURES), "treatment_share": obj["treatment_share"],
            "rates": rates, "itt_visit_diff": itt["visit"]["difference"], "itt_conversion_diff": itt["conversion"]["difference"],
            "max_abs_smd": float(worst.abs_smd), "max_abs_smd_feature": worst.feature,
            "n_duplicate_rows": dup["duplicate_rows_features_and_labels"],
            "n_duplicate_rows_features_only": dup["duplicate_rows_features_only"]}
