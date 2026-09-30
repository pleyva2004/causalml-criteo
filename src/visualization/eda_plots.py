"""EDA figures. Each function answers one stated analytical question (in its title/subtitle).

Entity colors come from ``plots.color_for``. Two neutral additions local to this module: ``exposure``
uses ``SERIES[2]`` (green, unused by the visit/conversion/arm entities) and ``treatment share`` uses
``INK_2``. All rates are descriptive; arm contrasts are intention-to-treat (ITT) differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.visualization.plots import (
    DIVERGING,
    INK,
    INK_2,
    MUTED,
    SERIES,
    color_for,
    new_figure,
    save_fig,
)

EXPOSURE_COLOR = SERIES[2]
ARMS = ("control", "treatment")


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%"


def plot_target_distribution(targets: pd.DataFrame) -> str:
    """Q: how imbalanced are the binary variables? Positive-class share, log scale."""
    colors = {"treatment": INK_2, "exposure": EXPOSURE_COLOR, "visit": color_for("visit"), "conversion": color_for("conversion")}
    fig, ax = new_figure(width=7.0, height=3.2)
    t = targets.iloc[::-1].reset_index(drop=True)
    ax.barh(t.variable, 100 * t.positive_rate, color=[colors[v] for v in t.variable], height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("share of users in the positive class (%, log scale)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for i, r in t.iterrows():
        ax.text(100 * r.positive_rate * 1.08, i, f"{_pct(r.positive_rate)}  ({int(r.positives):,} of {int(r.n):,})",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlim(right=100 * t.positive_rate.max() * 12)
    return save_fig(fig, "eda_target_distribution", "Treatment is 85% of users; every outcome is rare",
                    "How imbalanced are the labels? Conversion has a positive rate of a fraction of a percent, so accuracy is uninformative.")


def plot_rates_by_arm(by_arm: pd.DataFrame) -> str:
    """Q: descriptively, do outcome rates differ between randomized arms? (ITT, with Wilson CIs)."""
    fig, axes = new_figure(ncols=2, width=4.6, height=4.0)
    a = by_arm.set_index("arm")
    for ax, o in zip(axes, ("visit", "conversion")):
        for i, arm in enumerate(ARMS):
            r = a.loc[arm, f"{o}_rate"] * 100
            lo, hi = a.loc[arm, f"{o}_ci_low"] * 100, a.loc[arm, f"{o}_ci_high"] * 100
            ax.bar(i, r, color=color_for(arm), width=0.55, label=arm)
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], color=INK, capsize=4, linewidth=1.2)
            ax.text(i, hi * 1.03, f"{r:.3f}%", ha="center", va="bottom", fontsize=9, color=INK_2)
        pt, pc = a.loc["treatment", f"{o}_rate"], a.loc["control", f"{o}_rate"]
        se = np.sqrt(pt * (1 - pt) / a.loc["treatment", "n"] + pc * (1 - pc) / a.loc["control", "n"])
        d = 100 * (pt - pc)
        ax.set_title(f"{o.capitalize()}: ITT diff {d:+.3f} pp [{d - 196 * se:+.3f}, {d + 196 * se:+.3f}]", fontsize=10)
        ax.set_xticks([0, 1], ["control", "treatment"])
        ax.set_ylim(0, a[f"{o}_ci_high"].max() * 100 * 1.25)
        ax.set_ylabel(f"{o} rate (%)")
        ax.grid(axis="x", visible=False)
    return save_fig(fig, "eda_rates_by_arm", "Descriptive ITT: outcome rate by randomized arm",
                    "Do outcomes differ between assigned arms? Bars are rates with 95% Wilson CIs; the difference (pp) uses a Wald CI. "
                    "Benchmark sample, not real-world rates.")


def plot_exposure_funnel(funnel: pd.DataFrame, by_arm: pd.DataFrame) -> str:
    """Q: why is assignment not exposure? Funnel per arm and outcome rates by exposure subgroup."""
    f = funnel.set_index("group")
    fig, axes = new_figure(ncols=3, width=4.6, height=4.0)
    ax = axes[0]
    stages = ["assigned", "exposed", "visited", "converted"]
    y = np.arange(len(stages))[::-1]
    for j, (grp, arm) in enumerate((("control", "control"), ("treated", "treatment"))):
        vals = [1.0, f.loc[grp, "exposure_rate"], f.loc[grp, "visit_rate"], f.loc[grp, "conversion_rate"]]
        yy = y + (0.19 if j == 0 else -0.19)
        for yv, v in zip(yy, vals):
            if v > 0:
                ax.barh(yv, 100 * v, height=0.34, color=color_for(arm), label=arm if yv == yy[0] else None)
                ax.text(100 * v * 1.15, yv, _pct(v), va="center", fontsize=8, color=INK_2)
            else:
                ax.text(0.03, yv, "0 (by design)", va="center", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1500)
    ax.set_yticks(y, stages)
    ax.set_xlabel("% of assigned users (log scale)")
    ax.set_title("Funnel per arm", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="lower right")
    order = [("control", "control"), ("treated_unexposed", "treated,\nnot exposed"), ("treated_exposed", "treated,\nexposed")]
    for ax, o in zip(axes[1:], ("visit", "conversion")):
        for i, (g, label) in enumerate(order):
            r, lo, hi = f.loc[g, f"{o}_rate"] * 100, f.loc[g, f"{o}_ci_low"] * 100, f.loc[g, f"{o}_ci_high"] * 100
            ax.bar(i, r, color=color_for("control" if g == "control" else "treatment"), alpha=1.0 if g != "treated_unexposed" else 0.5, width=0.55)
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], color=INK, capsize=4, linewidth=1.2)
            ax.text(i, hi * 1.03 + 0.0, f"{r:.3g}%", ha="center", va="bottom", fontsize=9, color=INK_2)
        ax.set_xticks(range(3), [l for _, l in order], fontsize=8)
        ax.set_ylim(0, f[f"{o}_ci_high"].max() * 100 * 1.2)
        ax.set_ylabel(f"{o} rate (%)")
        ax.set_title(f"{o.capitalize()} rate by exposure group", fontsize=10)
        ax.grid(axis="x", visible=False)
    return save_fig(fig, "eda_exposure_funnel", "Assignment is not exposure: only a small share of treated users is shown an ad",
                    "How many assigned users are exposed, and how do exposed/unexposed subgroups differ? Exposure is self-selected, so the "
                    "exposed-vs-unexposed contrast is confounded; unexposed treated users convert below control.")


def plot_continuous_by_arm(sample: pd.DataFrame, continuous: list[str], profile: pd.DataFrame, bins: int = 60) -> str:
    """Q: are continuous features distributed alike in both arms (balance) and how are they shaped?"""
    p = profile.set_index("feature")
    fig, axes = new_figure(ncols=len(continuous), width=3.6, height=3.6)
    for ax, f in zip(np.atleast_1d(axes), continuous):
        lo, hi = p.loc[f, "q0.01"], p.loc[f, "q0.99"]
        edges = np.linspace(lo, hi, bins + 1)
        for arm, t in (("control", 0), ("treatment", 1)):
            x = sample.loc[sample.treatment == t, f].to_numpy()
            h, _ = np.histogram(x[(x >= lo) & (x <= hi)], bins=edges)
            ax.stairs(h / max(h.sum(), 1), edges, color=color_for(arm), label=arm, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(f"{f}: {p.loc[f, 'mode_share'] * 100:.1f}% of rows at {p.loc[f, 'mode_value']:.4g}", fontsize=9)
        ax.set_xlabel(f"{f} (1st-99th percentile)")
        ax.set_ylabel("share of arm per bin (log)")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    return save_fig(fig, "eda_continuous_by_arm", "Continuous features: distribution by arm",
                    "Are the arms balanced on continuous features, and what do the point masses look like? Histograms of a seeded sample, "
                    "clipped to the 1st-99th percentile; panel titles give the modal-value share (suspected default value).")


def plot_categorical_by_arm(shares: pd.DataFrame, categorical: list[str]) -> str:
    """Q: do the arms have the same level mix for the hashed categoricals? Top levels by frequency rank."""
    fig, axes = new_figure(ncols=4, nrows=2, width=3.2, height=2.7)
    for ax, f in zip(axes.ravel(), categorical):
        s = shares[shares.feature == f]
        x = np.arange(len(s))
        ax.bar(x - 0.2, 100 * s.share_control, width=0.4, color=color_for("control"), label="control")
        ax.bar(x + 0.2, 100 * s.share_treatment, width=0.4, color=color_for("treatment"), label="treatment")
        ax.set_xticks(x, [f"L{r}" for r in s["rank"]], fontsize=8)
        ax.set_title(f, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of arm")
    axes[0, 0].legend(loc="upper right")
    return save_fig(fig, "eda_categorical_by_arm", "Categorical features: level mix by arm",
                    "Do control and treatment share the same level mix? Levels L1..L8 are ranked by overall frequency (hashed values have no meaningful order).")


def plot_love(bal: pd.DataFrame, thresholds: list[float]) -> str:
    """Q: how large is the covariate imbalance between arms, feature by feature? |SMD| on a log axis."""
    b = bal.sort_values("abs_smd").reset_index(drop=True)
    fig, ax = new_figure(width=7.0, height=4.8)
    floor = 1e-4
    for kind, marker in (("continuous", "o"), ("categorical", "s")):
        s = b[b.type == kind]
        ax.scatter(np.maximum(s.abs_smd, floor), s.index, marker=marker, s=46, color=SERIES[0],
                   label=f"{kind} ({'raw-value SMD' if kind == 'continuous' else 'max per-level SMD'})", zorder=3)
    for th in thresholds:
        ax.axvline(th, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(th, len(b) - 0.4, f" {th:g}", color=INK_2, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(floor, max(0.3, b.abs_smd.max() * 2))
    ax.set_yticks(b.index, b.feature)
    ax.set_xlabel("|standardized mean difference| (log scale; values below 1e-4 drawn at 1e-4)")
    ax.grid(axis="y", visible=True)
    ax.legend(loc="lower right")
    return save_fig(fig, "eda_balance_love", "Covariate balance: |SMD| by feature",
                    "Is treatment assignment balanced on the features? Dashed lines: 0.01 (tight) and 0.1 (conventional) thresholds; "
                    "p-values are uninformative at n=14M.")


def plot_correlation(corr: pd.DataFrame, n_sample: int) -> str:
    """Q: which features and outcomes are monotonically related? Spearman with symmetric diverging colors."""
    m = corr.to_numpy().copy()
    np.fill_diagonal(m, np.nan)
    vmax = float(np.ceil(np.nanmax(np.abs(m)) * 20) / 20)
    cmap = LinearSegmentedColormap.from_list("div", list(DIVERGING))
    k = len(corr)
    fig, ax = new_figure(width=8.6, height=7.4)
    im = ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(k), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k), corr.index, fontsize=8)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            if i != j:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(m[i, j]) > 0.6 * vmax else INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman rho (diagonal omitted)", color=INK_2)
    return save_fig(fig, "eda_correlation", "Spearman correlation: features and outcomes",
                    f"Which variables move together? Seeded sample of {n_sample:,} rows; hashed categoricals are log level-frequency encoded "
                    "(their raw order is arbitrary). Association only.")


def plot_conversion_by_feature(by_feat: pd.DataFrame) -> str:
    """Q: which features are associated with conversion, and does the pattern differ by arm? (association)"""
    feats = list(dict.fromkeys(by_feat.feature))
    fig, axes = new_figure(ncols=4, nrows=3, width=3.3, height=2.6)
    for ax, f in zip(axes.ravel(), feats):
        s = by_feat[by_feat.feature == f]
        for arm, off in (("control", -0.08), ("treatment", 0.08)):
            a = s[s.arm == arm].sort_values("bin")
            x = a["bin"].to_numpy() + off
            r, lo, hi = 100 * a.conversion_rate.to_numpy(), 100 * a.conversion_ci_low.to_numpy(), 100 * a.conversion_ci_high.to_numpy()
            ax.errorbar(x, r, yerr=[np.maximum(r - lo, 0), np.maximum(hi - r, 0)], fmt="o-", markersize=3.5, linewidth=1.2,
                        capsize=2, color=color_for(arm), label=arm)
        bins = s.drop_duplicates("bin").sort_values("bin")
        kind = s.type.iloc[0]
        if kind == "continuous":
            ax.set_xticks(bins["bin"], ["mode" if i == 0 else str(int(i)) for i in bins["bin"]], fontsize=8)
            ax.set_title(f"{f} (mode, then quantile bins)", fontsize=9)
        else:
            ax.set_xticks(bins["bin"], bins.bin_label.tolist(), fontsize=7)
            ax.set_title(f"{f} (level by frequency)", fontsize=9)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("conversion rate (%)")
    axes[0, 1].legend(loc="upper left", fontsize=8)
    return save_fig(fig, "eda_conversion_by_feature", "Conversion rate by feature bin and arm (association, not causation)",
                    "Which features are associated with conversion? Points are rates with 95% Wilson CIs; continuous features as the modal point mass plus quantile bins "
                    "of the rest, categoricals by top-8 levels plus 'rest'.")
