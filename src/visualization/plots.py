"""Shared figure style and helpers. Every figure in results/figures goes through ``save_fig``.

Conventions (applied by ``apply_style``):
* one light surface, recessive hairline gridlines on the value axis only, no top/right spines;
* colors follow the *entity*, never its rank -- use ``color_for(name)`` so "control" or
  "uplift targeting" has the same color in every figure;
* text uses ink colors, never series colors; a single y-axis per panel (no dual axes);
* reference lines (perfect calibration, random targeting) are muted gray and dashed.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.utils import figure_path  # noqa: E402

# Validated categorical order (colorblind-safe on adjacent pairs); never cycle past 8.
SERIES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")      # negative pole, neutral midpoint, positive pole

# Fixed entity -> color map. Add new entities here rather than choosing colors inline.
ENTITY_COLORS: dict[str, str] = {
    # experiment arms
    "control": SERIES[0], "treatment": SERIES[1],
    # predictive models
    "logistic_regression": SERIES[0], "random_forest": SERIES[2], "lightgbm": SERIES[1],
    "uncalibrated": MUTED, "sigmoid": SERIES[0], "isotonic": SERIES[1],
    # CATE / uplift learners
    "s_learner": SERIES[0], "t_learner": SERIES[1], "x_learner": SERIES[2], "dr_learner": SERIES[3],
    "causal_forest": SERIES[4], "class_transformation": SERIES[6],
    # targeting policies
    "random": MUTED, "response": SERIES[0], "uplift": SERIES[1], "expected_value": SERIES[2],
    "treat_all": INK_2, "treat_none": AXIS,
    # outcomes
    "visit": SERIES[0], "conversion": SERIES[1],
}


def color_for(entity: str, fallback_index: int = 0) -> str:
    """Stable color for a named entity (see ``ENTITY_COLORS``)."""
    return ENTITY_COLORS.get(entity, SERIES[fallback_index % len(SERIES)])


def apply_style() -> None:
    """Set matplotlib rcParams for every figure in the project."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def new_figure(ncols: int = 1, nrows: int = 1, width: float = 6.4, height: float = 4.0, **kw) -> tuple[Figure, object]:
    """Styled figure; ``width``/``height`` are per panel in inches."""
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), **kw)
    return fig, axes


def reference_line(ax: Axes, x: Iterable[float], y: Iterable[float], label: str | None = None) -> None:
    """Muted dashed reference (perfect calibration, random targeting, zero effect)."""
    ax.plot(list(x), list(y), color=MUTED, linewidth=1.2, linestyle="--", label=label, zorder=1)


def zero_line(ax: Axes, axis: str = "y") -> None:
    """Solid baseline at zero for effect-size plots."""
    (ax.axhline if axis == "y" else ax.axvline)(0.0, color=AXIS, linewidth=1.0, zorder=1)


def format_count_axis(ax: Axes, axis: str = "x") -> None:
    """Human-readable tick labels for counts (100K, 1M) on linear or log axes."""
    from matplotlib.ticker import FuncFormatter

    def fmt(v: float, _pos: int) -> str:
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}K"
        return f"{v:g}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(fmt))
    target.set_minor_formatter(FuncFormatter(lambda v, p: ""))


def save_fig(fig: Figure, name: str, title: str | None = None, subtitle: str | None = None) -> str:
    """Save to ``results/figures/<name>.png`` and close. Returns the repo-relative path.

    The title band is sized in inches so the title and subtitle never overlap, whatever the figure height.
    """
    height = fig.get_size_inches()[1]
    band = (0.42 if title else 0.0) + (0.30 if subtitle else 0.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - band / height) if band else None)
    if title:
        fig.text(0.01, 1 - 0.10 / height, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - (0.44 if title else 0.10) / height, subtitle, ha="left", va="top", fontsize=9,
                 color=INK_2)
    path = figure_path(name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return f"results/figures/{path.name}"
