"""Figures for the scaling stage: cost and benefit of more training data."""

from __future__ import annotations

import pandas as pd

from src.visualization.plots import color_for, format_count_axis, new_figure, save_fig


def plot_scaling(table: pd.DataFrame) -> str:
    """Three panels sharing the x-axis (training rows, log scale): fit time, peak memory, test PR-AUC.

    Answers: what does moving from the development sample to 8.4M training rows cost, and what does it buy?
    """
    fig, axes = new_figure(ncols=3, width=4.2, height=3.4)
    panels = [("fit_s", "Training time (s)"), ("peak_rss_gb", "Peak memory (GB)"), ("test_pr_auc", "Test PR-AUC")]
    for ax, (col, label) in zip(axes, panels):
        for model, g in table.groupby("model"):
            g = g.sort_values("n_train")
            ax.plot(g.n_train, g[col], marker="o", color=color_for(model), label=model.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xticks(sorted(table.n_train.unique()))
        format_count_axis(ax)
        ax.set_xlabel("Training rows")
        ax.set_title(label)
    axes[0].legend(loc="upper left")
    return save_fig(fig, "scaling_learning_curve",
                    title="What does more training data cost, and what does it buy?",
                    subtitle="Conversion model; each point is a fresh process evaluated on the full test split")
