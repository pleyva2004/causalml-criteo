"""Figures for the ``cate`` stage (CATE learners and uplift evaluation). All inputs come from
``src.causal.learners.analyze``; every figure answers one question, stated in its title.

Effects are shown in percentage points (pp) of the outcome rate; incremental outcomes are counts in
the held-out test split. Uncertainty: 95% bootstrap (curves, budgets) or normal (subgroup
differences in means) intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.visualization.plots import (
    INK,
    INK_2,
    MUTED,
    SEQUENTIAL,
    color_for,
    format_count_axis,
    new_figure,
    reference_line,
    save_fig,
    zero_line,
)

PP = 100.0  # rate -> percentage points
# The response baseline shares SERIES[0] with the S-learner in ENTITY_COLORS; in figures where both appear
# it is drawn in ink (the reference every learner is compared with) so the two never share a color.
RESPONSE_ON_LEARNER_PLOTS = INK


def _learner_color(name: str) -> str:
    return RESPONSE_ON_LEARNER_PLOTS if name == "response" else color_for(name)
LABELS = {"s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner", "dr_learner": "DR-learner",
          "causal_forest": "Causal forest", "class_transformation": "Transformed outcome",
          "response": "Response model P(Y|X)", "random": "Random"}


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _pct_axis(ax, axis: str = "x") -> None:
    from matplotlib.ticker import FuncFormatter
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}%"))


def qini_curves(curves: dict[str, pd.DataFrame], bands: dict[str, dict[str, np.ndarray]], grid_band: np.ndarray,
                outcomes: Sequence[str], n_test: int) -> str:
    fig, axes = new_figure(len(outcomes), 1, width=6.2, height=4.6)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        cv = curves[outcome]
        cv = cv[cv["scorer"] != "random"]
        for nm, g in cv.groupby("scorer", sort=False):
            emph = nm in bands[outcome]
            ax.plot(g["fraction"] * 100, g["qini"] * n_test, color=_learner_color(nm), lw=2.4 if emph else 1.3,
                    alpha=1.0 if emph else 0.8, label=_label(nm), zorder=3 if emph else 2)
        for nm, band in bands[outcome].items():
            ax.fill_between(grid_band * 100, band[0] * n_test, band[1] * n_test, color=_learner_color(nm),
                            alpha=0.13, lw=0, zorder=1)
        end = cv.groupby("scorer")["qini"].last().iloc[0] * n_test
        reference_line(ax, [0, 100], [0, end], label="Random targeting")
        zero_line(ax)
        ax.set_xlim(0, 100)
        _pct_axis(ax, "x")
        ax.set_xlabel("Users targeted (ranked by score)")
        ax.set_ylabel(f"Incremental {outcome}s in test split (Qini)")
        format_count_axis(ax, "y")
        sel = [nm for nm in bands[outcome] if nm != "response"]
        ax.set_title(f"{outcome.capitalize()} (selected on validation: {_label(sel[0]) if sel else 'n/a'})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    return save_fig(fig, "cate_qini_curves",
                    "Which ranking concentrates the users whose visits and conversions the ad causes?",
                    f"Qini curves on the held-out test split (N={n_test:,}), propensity-weighted (Hajek) arm means, scaled to "
                    "the whole population; shaded: 95% bootstrap band for the selected learner and the response "
                    "model; dashed: random targeting")


def uplift_calibration(calib: dict[str, tuple[str, pd.DataFrame]], outcomes: Sequence[str]) -> str:
    fig, axes = new_figure(len(outcomes), 1, width=6.0, height=4.2)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        ln, cal = calib[outcome]
        x = cal["group"].to_numpy()
        ax.errorbar(x, cal["observed"] * PP, yerr=[(cal["observed"] - cal["ci_low"]) * PP,
                                                    (cal["ci_high"] - cal["observed"]) * PP],
                    fmt="o", color=color_for(ln), ecolor=color_for(ln), elinewidth=1.4, capsize=3,
                    label="Observed uplift (propensity-weighted, 95% CI)", zorder=3)
        ax.plot(x, cal["mean_predicted"] * PP, marker="D", markersize=6, lw=1.2, color=INK_2, mfc="none",
                label=f"Mean predicted uplift ({_label(ln)})", zorder=4)
        zero_line(ax)
        ax.set_xticks(x)
        ax.set_xlabel("Decile of predicted uplift (1 = highest)")
        ax.set_ylabel(f"Uplift in {outcome} rate (pp)")
        ax.set_title(f"{outcome.capitalize()}: {_label(ln)} (selected on validation)")
        ax.legend(loc="upper right")
    return save_fig(fig, "cate_uplift_calibration",
                    "Do users with higher predicted uplift really respond more to the ad?",
                    "Held-out test split, deciles of the selected learner's score; GATES-style check of the "
                    "ranking and of the scale of the predictions")


def response_vs_uplift(diffs: dict[str, dict[str, Any]], outcomes: Sequence[str], budgets: Sequence[float]) -> str:
    from src.causal.uplift_metrics import budget_tag
    fig, axes = new_figure(len(outcomes), 2, width=6.0, height=3.6)
    axes = np.atleast_2d(axes).reshape(2, len(outcomes))
    xb = np.array(budgets) * 100
    for j, outcome in enumerate(outcomes):
        d = diffs[outcome]
        sel = d["selected"]
        ax = axes[0, j]
        width = min(np.diff(xb).min() if len(xb) > 1 else 5, 5) * 0.22
        for off, nm, key in ((-width, "response", "response"), (0.0, "uplift", sel), (width, "random", "random")):
            est = np.array([d["ipw"][budget_tag(k)]["incremental"][key]["estimate"] for k in budgets])
            lo = np.array([d["ipw"][budget_tag(k)]["incremental"][key]["ci_low"] for k in budgets])
            hi = np.array([d["ipw"][budget_tag(k)]["incremental"][key]["ci_high"] for k in budgets])
            lab = {"response": "Response model (target the likeliest responders)",
                   "uplift": "Uplift model (learner selected on validation)", "random": "Random targeting"}[nm]
            ax.errorbar(xb + off, est, yerr=[est - lo, hi - est], fmt="o", color=color_for(nm), capsize=3,
                        elinewidth=1.4, label=lab)
        zero_line(ax)
        ax.set_xticks(xb)
        _pct_axis(ax, "x")
        ax.set_ylabel(f"Incremental {outcome}s (test split)")
        format_count_axis(ax, "y")
        ax.set_title(f"{outcome.capitalize()}: incremental {outcome}s by budget (uplift model: {_label(sel)})",
                     fontsize=10)
        ax = axes[1, j]
        for off, kind, mfc in ((-width / 2, "ipw", None), (width / 2, "plain", "none")):
            dd = [d[kind][budget_tag(k)]["incremental"]["difference_uplift_minus_response"] for k in budgets]
            est = np.array([x["estimate"] for x in dd])
            lo, hi = np.array([x["ci_low"] for x in dd]), np.array([x["ci_high"] for x in dd])
            ax.errorbar(xb + off, est, yerr=[est - lo, hi - est], fmt="s", color=INK, mfc=mfc or INK, capsize=3,
                        elinewidth=1.4, label={"ipw": "Propensity-weighted (Hajek, primary)",
                                               "plain": "Unweighted difference in means"}[kind])
        zero_line(ax)
        ax.set_xticks(xb)
        _pct_axis(ax, "x")
        ax.set_xlabel("Share of users targeted")
        ax.set_ylabel("Uplift model − response model")
        ax.set_title(f"{outcome.capitalize()}: uplift − response, paired (95% bootstrap CI)", fontsize=10)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, [x + " (top row)" for x in l1] + [x + " (bottom row)" for x in l2], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    return save_fig(fig, "cate_response_vs_uplift",
                    "Does targeting by predicted uplift beat targeting the likeliest converters?",
                    "Held-out test split; incremental outcomes = (treated − control rate among the targeted, "
                    "propensity-weighted) × number targeted; paired bootstrap over the same resamples")


def sure_things(diffs: dict[str, dict[str, Any]], outcomes: Sequence[str], budgets: Sequence[float],
                overall: dict[str, float]) -> str:
    from src.causal.uplift_metrics import budget_tag
    fig, axes = new_figure(len(outcomes), 1, width=6.0, height=4.0)
    axes = np.atleast_1d(axes)
    xb = np.array(budgets) * 100
    for ax, outcome in zip(axes, outcomes):
        d = diffs[outcome]
        sel = d["selected"]
        for off, nm, key in ((-0.6, "response", "response"), (0.6, "uplift", sel)):
            vals = [d["ipw"][budget_tag(k)]["anyway_share"][key] for k in budgets]
            est = np.array([v["estimate"] for v in vals]) * 100
            lo = np.array([v["ci_low"] for v in vals]) * 100
            hi = np.array([v["ci_high"] for v in vals]) * 100
            lab = {"response": "Top-k by response model", "uplift": f"Top-k by uplift model ({_label(sel)})"}[nm]
            ax.errorbar(xb + off, est, yerr=[est - lo, hi - est], fmt="o", color=color_for(nm), capsize=3,
                        elinewidth=1.4, label=lab)
        ax.axhline(overall[outcome] * 100, color=MUTED, ls="--", lw=1.2, label="Whole population")
        ax.set_xticks(xb)
        _pct_axis(ax, "x")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Share of users targeted")
        ax.set_ylabel("Control rate ÷ treated rate (%)")
        ax.set_title(f"{outcome.capitalize()}")
        ax.legend(loc="lower right", fontsize=8)
    return save_fig(fig, "cate_sure_things",
                    "How much of the targeted users' response would have happened without the ad?",
                    "Held-out test split: control outcome rate as a share of the treated outcome rate inside the "
                    "targeted group (propensity-weighted; 100% = no effect); 95% bootstrap CIs")


def tau_distributions(dist: dict[str, pd.DataFrame], outcomes: Sequence[str], ate: dict[str, float],
                      ate_ipw: dict[str, float] | None = None) -> str:
    fig, axes = new_figure(len(outcomes), 1, width=6.0, height=3.8)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        d = dist[outcome].reset_index(drop=True)
        ys = np.arange(len(d))[::-1]
        for y, (_, r) in zip(ys, d.iterrows()):
            c = color_for(r["learner"])
            ax.plot([r["q01"] * PP, r["q99"] * PP], [y, y], color=c, lw=1.2, solid_capstyle="butt")
            ax.plot([r["q10"] * PP, r["q90"] * PP], [y, y], color=c, lw=5, solid_capstyle="butt")
            ax.plot([r["q50"] * PP], [y], "o", color="white", mec=c, mew=1.6, ms=6, zorder=3)
            ax.plot([r["mean"] * PP], [y], "|", color=INK, ms=12, mew=1.6, zorder=4)
        ax.axvline(ate[outcome] * PP, color=MUTED, ls="--", lw=1.2)
        if ate_ipw is not None:
            ax.axvline(ate_ipw[outcome] * PP, color=MUTED, ls=":", lw=1.4)
        zero_line(ax, axis="x")
        ax.set_yticks(ys)
        ax.set_yticklabels([_label(x) for x in d["learner"]])
        ax.grid(axis="x", color="#e1e0d9")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Predicted uplift in {outcome} rate (pp)")
        ax.set_title(f"{outcome.capitalize()}")
    handles = [Line2D([], [], color=INK_2, lw=5, label="10th–90th percentile"),
               Line2D([], [], color=INK_2, lw=1.2, label="1st–99th percentile"),
               Line2D([], [], marker="o", color="white", mec=INK_2, ls="", label="Median"),
               Line2D([], [], marker="|", color=INK, ls="", ms=12, label="Mean"),
               Line2D([], [], color=MUTED, ls="--", label="ATE, difference in means (test)"),
               Line2D([], [], color=MUTED, ls=":", lw=1.4, label="ATE, propensity-weighted (test)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    return save_fig(fig, "cate_tau_distributions",
                    "How much heterogeneity does each learner predict?",
                    "Distribution of predicted individual effects on the held-out test split")


def rank_correlation(corr: dict[str, pd.DataFrame], outcomes: Sequence[str]) -> str:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7f6f3", SEQUENTIAL[0], SEQUENTIAL[2], SEQUENTIAL[4]])
    fig, axes = new_figure(len(outcomes), 1, width=5.6, height=5.0)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        c = corr[outcome]
        ax.imshow(c.to_numpy(), vmin=0.3, vmax=1, cmap=cmap)
        names = [_label(x).replace(" P(Y|X)", "") for x in c.columns]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                v = c.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v >= 0.9 else INK)
        ax.set_title(f"{outcome.capitalize()}")
    return save_fig(fig, "cate_rank_correlation",
                    "Do the learners agree on who benefits most from the ad?",
                    "Spearman rank correlation of predicted scores on the held-out test split (darker = higher; "
                    "color scale 0.3-1); the response model is included for comparison")


def segment_uplift(seg: dict[str, tuple[str, list[str], pd.DataFrame]], outcomes: Sequence[str]) -> str:
    ncols = max(len(v[1]) for v in seg.values())
    fig, axes = new_figure(ncols, len(outcomes), width=4.6, height=3.6)
    axes = np.atleast_2d(axes).reshape(len(outcomes), ncols)
    for i, outcome in enumerate(outcomes):
        ln, top, s = seg[outcome]
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(top):
                ax.axis("off")
                continue
            f = top[j]
            d = s[s["feature"] == f].copy()
            d = d.sort_values("segment", key=lambda x: x.str.replace("other", "~"))
            x = np.arange(len(d))
            ax.errorbar(x, d["observed"] * PP, yerr=[(d["observed"] - d["ci_low"]) * PP,
                                                     (d["ci_high"] - d["observed"]) * PP],
                        fmt="o", color=color_for(outcome), capsize=3, elinewidth=1.4,
                        label="Observed (propensity-weighted, 95% CI)")
            ax.plot(x, d["mean_predicted"] * PP, "D", color=INK_2, mfc="none", ms=6,
                    label=f"Mean predicted ({_label(ln)})")
            zero_line(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(d["segment"], rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{outcome.capitalize()} × {f}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Uplift in {outcome} rate (pp)")
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=7)
    return save_fig(fig, "cate_segment_uplift",
                    "Does the ad's effect differ across segments of the most effect-relevant features?",
                    "Held-out test split: propensity-weighted treated − control difference per segment (quantile "
                    "bins or most frequent hashed levels) vs the selected learner's mean prediction; heterogeneity, "
                    "not mechanism")


def negative_uplift(neg: dict[str, pd.DataFrame], outcomes: Sequence[str]) -> str:
    """Small multiples (one panel per learner, own y-scale): flagged groups range from 0.1% to 23% of
    users, so their confidence intervals differ by two orders of magnitude."""
    learners = list(neg[outcomes[0]]["learner"])
    fig, axes = new_figure(len(learners), len(outcomes), width=2.25, height=2.7)
    axes = np.atleast_2d(axes).reshape(len(outcomes), len(learners))
    for i, outcome in enumerate(outcomes):
        d = neg[outcome].set_index("learner")
        for j, ln in enumerate(learners):
            ax = axes[i, j]
            r = d.loc[ln]
            c = color_for(ln)
            if pd.notna(r.get("observed")):
                ax.errorbar([0], [r["observed"] * PP], yerr=[[(r["observed"] - r["ci_low"]) * PP],
                                                             [(r["ci_high"] - r["observed"]) * PP]],
                            fmt="o", color=c, capsize=4, elinewidth=1.6, ms=7)
                ax.plot([0.35], [r["mean_predicted"] * PP], "D", color=INK_2, mfc="none", ms=6)
            zero_line(ax)
            ax.set_xlim(-0.5, 0.85)
            ax.set_xticks([])
            share = r["share_predicted_negative"]
            ax.set_title(f"{_label(ln)}\n{share:.1%} flagged", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{outcome.capitalize()} uplift (pp)")
    handles = [Line2D([], [], marker="o", color=INK_2, ls="", label="Observed uplift among users with predicted "
                      "uplift < 0 (propensity-weighted, 95% CI)"),
               Line2D([], [], marker="D", color=INK_2, mfc="none", ls="", label="Mean predicted uplift in that group")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=8)
    return save_fig(fig, "cate_negative_uplift",
                    "Are users with negative predicted uplift really hurt by the ad?",
                    "Held-out test split, one panel per learner (independent y-scales): a real 'sleeping dog' group "
                    "would show an observed uplift below zero")


def effect_importance(imp: dict[str, tuple[str, pd.DataFrame]], outcomes: Sequence[str]) -> str:
    fig, axes = new_figure(len(outcomes), 1, width=5.6, height=4.2)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        sel, df = imp[outcome]
        a = df[(df["method"] == "causal_forest_split_importance")].set_index("feature")["importance"]
        b = df[(df["method"] == "surrogate_tree_shap") & (df["learner"] == sel)].set_index("feature")["importance"]
        order = (a.add(b, fill_value=0)).sort_values().index
        y = np.arange(len(order))
        ax.barh(y + 0.2, a.reindex(order).to_numpy(), height=0.38, color=color_for("causal_forest"),
                label="Causal forest: split importance")
        r2 = df[(df["method"] == "surrogate_tree_shap") & (df["learner"] == sel)]["surrogate_r2"].iloc[0]
        ax.barh(y - 0.2, b.reindex(order).to_numpy(), height=0.38, color=color_for(sel) if sel != "causal_forest"
                else SEQUENTIAL[4], label=f"{_label(sel)}: surrogate SHAP (R² = {r2:.3f})")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.grid(axis="x", color="#e1e0d9")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of total importance")
        ax.set_title(f"{outcome.capitalize()}")
        ax.legend(loc="lower right", fontsize=8)
    return save_fig(fig, "cate_effect_importance",
                    "Which features drive the predicted effect of the ad (not the outcome)?",
                    "Importance for the CATE, aggregated to raw features; describes the models, not causal mechanisms")


def treated_share(imb: pd.DataFrame, outcomes: Sequence[str], design_share: float) -> str:
    fig, axes = new_figure(len(outcomes), 1, width=5.6, height=3.8)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        d = imb[imb["outcome"] == outcome]
        ax.errorbar(d["decile"], d["treated_share"] * 100, yerr=[(d["treated_share"] - d["ci_low"]) * 100,
                                                                 (d["ci_high"] - d["treated_share"]) * 100],
                    fmt="o", color=color_for("treatment"), capsize=3, elinewidth=1.4,
                    label="Treated share in decile (95% CI)")
        ax.axhline(design_share * 100, color=MUTED, ls="--", lw=1.2, label="Overall treated share")
        ax.set_xticks(d["decile"])
        ax.set_xlabel(f"Decile of predicted P({outcome}) from the response model (10 = likeliest)")
        ax.set_ylabel("Treated share (%)")
        ax.set_title(f"{outcome.capitalize()}")
        ax.legend(loc="upper left", fontsize=8)
    return save_fig(fig, "cate_treated_share_by_response_decile",
                    "Is treatment assignment independent of users' baseline propensity in this sample?",
                    "Held-out test split; response model fit on train ignoring treatment. Under X-independent "
                    "randomization every decile would sit on the dashed line")


def make_all(fd: dict[str, Any], outcomes: Sequence[str], budgets: Sequence[float], n_test: int, mode: str
             ) -> list[str]:
    paths = [
        treated_share(fd["imbalance"], outcomes, fd["design_share"]),
        qini_curves(fd["curves"], fd["bands"], fd["grid_band"], outcomes, n_test),
        uplift_calibration(fd["calib"], outcomes),
        response_vs_uplift(fd["diffs"], outcomes, budgets),
        sure_things(fd["diffs"], outcomes, budgets, fd["overall_anyway"]),
        tau_distributions(fd["dist"], outcomes, fd["ate"], fd.get("ate_ipw")),
        rank_correlation(fd["corr"], outcomes),
        segment_uplift(fd["seg"], outcomes),
        negative_uplift(fd["neg"], outcomes),
        effect_importance(fd["imp"], outcomes),
    ]
    return paths
