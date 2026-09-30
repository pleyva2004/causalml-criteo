"""Figures for the robustness stage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.uplift_metrics import budget_tag
from src.visualization.plots import SEQUENTIAL, color_for, format_count_axis, new_figure, save_fig, zero_line


def plot_seed_stability(seeds: pd.DataFrame, budgets: list[float]) -> str:
    """Uplift-minus-response incremental outcomes at each budget, one dot per training seed."""
    outcomes = list(seeds.outcome.unique())
    fig, axes = new_figure(ncols=len(outcomes), width=4.6, height=3.6)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, outcome in zip(axes, outcomes):
        g = seeds[seeds.outcome == outcome]
        for i, k in enumerate(budgets):
            v = g[f"diff_{budget_tag(k)}"].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=36, color=color_for("uplift"), zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=color_for("treat_all"), lw=1.6, zorder=2)
        zero_line(ax)
        ax.set_xticks(range(len(budgets)), [f"top {k:.0%}" for k in budgets])
        ax.set_title(f"{outcome} ({g.learner.iloc[0].replace('_', '-')} vs response)")
        ax.set_ylabel(f"Incremental {outcome}s, uplift − response")
    return save_fig(fig, "robustness_seed_stability",
                    "Does the uplift-vs-response gap survive retraining with new seeds?",
                    "Test split, propensity-weighted. Dots = seeds (same data, different model randomness); "
                    "bar = mean. Above 0 = uplift targeting wins.")


def plot_learning_curve(curve: pd.DataFrame) -> str:
    """Normalized Qini coefficient vs training rows for the uplift learner and the response model."""
    outcomes = list(curve.outcome.unique())
    fig, axes = new_figure(ncols=len(outcomes), width=4.6, height=3.6)
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        g = curve[curve.outcome == outcome].sort_values("train_rows")
        ax.plot(g.train_rows, g.qini_uplift, marker="o", color=color_for("uplift"),
                label=f"uplift ({g.learner.iloc[0].replace('_', '-')})")
        ax.plot(g.train_rows, g.qini_response, marker="o", color=color_for("response"), label="response model")
        ax.set_xscale("log")
        ax.set_xticks(sorted(g.train_rows.unique()))
        format_count_axis(ax)
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Normalized Qini (test)")
        ax.set_title(outcome)
        ax.legend(loc="lower right")
    return save_fig(fig, "robustness_learning_curve", "How much training data does uplift ranking need?",
                    "Normalized Qini coefficient on the 2.8M-row test split, propensity-weighted; nested training subsets.")


def plot_split_half(scatter: dict[str, tuple[np.ndarray, np.ndarray]], stability: dict[str, dict]) -> str:
    """tau-hat from two models trained on disjoint halves of the training split, on the same test users."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = new_figure(ncols=len(scatter), width=4.2, height=4.0)
    axes = np.atleast_1d(axes)
    for ax, (outcome, (a, b)) in zip(axes, scatter.items()):
        lo, hi = np.quantile(np.r_[a, b], [0.005, 0.995])
        ax.hexbin(a, b, gridsize=45, extent=(lo, hi, lo, hi), bins="log", cmap=cmap, mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="#898781", lw=1.0, ls="--")
        s = stability[outcome]
        ax.set_title(f"{outcome}: Spearman {s['split_half_rank_corr']:.2f}, top-10% Jaccard "
                     f"{s['split_half_top10_jaccard']:.2f}", fontsize=9.5)
        ax.set_xlabel("τ̂ from training half A")
        ax.set_ylabel("τ̂ from training half B")
        ax.grid(False)
    return save_fig(fig, "robustness_split_half", "Do two disjoint training halves agree on who responds?",
                    "Same test users scored by the selected learner trained on each half; dashed = perfect agreement.")
