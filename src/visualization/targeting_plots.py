"""Figures for the targeting stage. Each answers one question; every estimate carries its 95% CI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.visualization.plots import (INK_2, MUTED, color_for, new_figure, reference_line, save_fig,
                                     zero_line)

LABELS = {"random": "Random", "response": "Response (P(convert))", "uplift": "Uplift (tau_hat)",
          "expected_value": "Expected value (tau_hat*V - c > 0)", "treat_all": "Treat everyone",
          "treat_none": "Treat nobody"}
BUDGETED = ["random", "response", "uplift", "expected_value"]


def _band(ax, df: pd.DataFrame, policy: str, key: str) -> None:
    d = df[df["policy"] == policy].sort_values("budget")
    color = color_for(policy)
    ax.fill_between(d["budget"] * 100, d[f"{key}_lo"], d[f"{key}_hi"], color=color, alpha=0.12, linewidth=0)
    ax.plot(d["budget"] * 100, d[key], color=color, label=LABELS[policy],
            linestyle="--" if policy == "random" else "-")


def _curve_figure(curves: pd.DataFrame, key: str, ylabel: str, title: str, subtitle: str, name: str,
                  treat_all_marker: bool = True) -> None:
    fig, ax = new_figure(width=7.2, height=4.4)
    for pol in BUDGETED:
        _band(ax, curves, pol, key)
    if treat_all_marker:
        ta = curves[(curves["policy"] == "treat_all")].iloc[-1]
        ax.errorbar([100], [ta[key]], yerr=[[ta[key] - ta[f"{key}_lo"]], [ta[f"{key}_hi"] - ta[key]]],
                    fmt="o", color=color_for("treat_all"), capsize=3, label=LABELS["treat_all"], clip_on=False)
    if key.startswith("net"):
        zero_line(ax)
    ax.set_xlabel("Budget: % of users targeted")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 101)
    ax.legend(loc="best")
    save_fig(fig, name, title, subtitle)


def make_targeting_figures(curves: pd.DataFrame, sens: pd.DataFrame, ratio0: float,
                           headline: list[float]) -> None:
    """Write the four ``targeting_*`` figures."""
    _curve_figure(
        curves, "inc_per_100k", "Incremental conversions per 100k users",
        "Which ranking recovers more incremental conversions at a given budget?",
        "Randomized test split, IPW estimate vs treating nobody; bands are 95% bootstrap CIs; "
        "random is one seeded draw.",
        "targeting_incremental_vs_budget")
    _curve_figure(
        curves, "net_per_100k", "Net value per 100k users ($)",
        f"Net value vs budget at c/V = {ratio0:g}: does any targeting policy beat treating everyone?",
        "Net = V x incremental conversions - c x users treated (illustrative V, c); 95% bootstrap CIs.",
        "targeting_net_value_vs_budget")
    _sensitivity_figure(sens)
    _bar_figure(curves, [b for b in (0.10, 0.20) if b in headline])


def _sensitivity_figure(sens: pd.DataFrame) -> None:
    ratios = sorted(sens["cost_value_ratio"].unique())
    x = np.arange(len(ratios))
    pols = ["random", "response", "uplift", "expected_value", "treat_all"]
    fig, ax = new_figure(width=7.4, height=4.4)
    width = 0.8 / len(pols)
    for j, pol in enumerate(pols):
        d = sens[sens["policy"] == pol].sort_values("cost_value_ratio")
        pos = x + (j - (len(pols) - 1) / 2) * width
        ax.errorbar(pos, d["net_per_100k"], yerr=[d["net_per_100k"] - d["net_per_100k_lo"],
                                                  d["net_per_100k_hi"] - d["net_per_100k"]],
                    fmt="o", ms=4, capsize=2, color=color_for(pol), label=LABELS[pol])
    zero_line(ax)
    ax.set_xticks(x, [f"{r:g}" for r in ratios])
    ax.set_xlabel("Cost / value ratio c/V (break-even for treating everyone = conversion ATE)")
    ax.set_ylabel("Net value per 100k users ($)")
    ax.legend(loc="best")
    save_fig(fig, "targeting_cost_sensitivity",
             "How does the best policy change with the cost/value ratio?",
             "Budgets chosen on validation, evaluated on test; 95% bootstrap CIs. Policies at zero are "
             "equivalent to treating nobody.")


def _bar_figure(curves: pd.DataFrame, budgets: list[float]) -> None:
    fig, axes = new_figure(ncols=len(budgets), width=5.2, height=4.2, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, budgets):
        d = curves[(curves["budget"] == b) & curves["policy"].isin(BUDGETED)].set_index("policy").loc[BUDGETED]
        pos = np.arange(len(d))
        ax.bar(pos, d["inc_per_100k"], color=[color_for(p) for p in d.index], width=0.62,
               yerr=[d["inc_per_100k"] - d["inc_per_100k_lo"], d["inc_per_100k_hi"] - d["inc_per_100k"]],
               error_kw={"ecolor": INK_2, "capsize": 3, "elinewidth": 1})
        ta = curves[(curves["policy"] == "treat_all") & (curves["budget"] == 1.0)].iloc[0]
        reference_line(ax, [-0.5, len(d) - 0.5], [ta["inc_per_100k"]] * 2)
        ax.text(-0.45, ta["inc_per_100k"], "treat everyone", color=MUTED, fontsize=8, va="bottom", ha="left")
        ax.set_xticks(pos, ["Random", "Response", "Uplift", "Exp. value"])
        ax.set_title(f"{b:.0%} budget")
        ax.set_ylabel("Incremental conversions per 100k users")
        zero_line(ax)
    save_fig(fig, "targeting_policy_bars",
             "Uplift vs response targeting at 10% and 20% budgets",
             "Incremental conversions per 100k users on the test split (all users, not only treated); "
             "95% bootstrap CIs; dashed = treat everyone.")
