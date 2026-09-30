"""Figures for the predictive stage (prefix ``predict_``). Each answers one question, stated in its title.

Colors follow the entity (``plots.color_for``). One entity is added here: ``lightgbm_weighted`` (the
class-weighted LightGBM variant) takes ``SERIES[3]``, which no existing predictive entity uses.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.evaluation import reliability_table, row_log_loss
from src.visualization.plots import (
    INK_2,
    MUTED,
    SEQUENTIAL,
    SERIES,
    color_for,
    new_figure,
    reference_line,
    save_fig,
    zero_line,
)

ENTITY_EXTRA = {"lightgbm_weighted": SERIES[3]}
LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM",
          "lightgbm_weighted": "LightGBM (scale_pos_weight)"}
_CMAP = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)


def _color(entity: str) -> str:
    return ENTITY_EXTRA.get(entity) or color_for(entity)


def _thin(x: np.ndarray, y: np.ndarray, n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return x[idx], y[idx]


def plot_roc_pr(y: np.ndarray, scores: dict[str, np.ndarray], metrics: dict, base_rate: float) -> str:
    """ROC and precision-recall curves on the test split, AUCs with bootstrap 95% CIs in the legend."""
    fig, (a1, a2) = new_figure(ncols=2, width=5.6, height=4.6)
    for m, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(*_thin(fpr, tpr), color=_color(m), label=_lab_ci(m, metrics[m], "roc_auc", 3))
        prec, rec, _ = precision_recall_curve(y, p)
        keep = rec >= 0.005                                    # below this, precision rests on a handful of rows
        a2.plot(*_thin(rec[keep], prec[keep]), color=_color(m), label=_lab_ci(m, metrics[m], "pr_auc", 4))
    reference_line(a1, [0, 1], [0, 1], "random (AUC 0.5)")
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    a1.set_title("ROC: looks strong because negatives dominate")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="lower right")
    reference_line(a2, [0, 1], [base_rate, base_rate], f"no skill (prevalence {base_rate:.4f})")
    a2.set(xlabel="Recall", ylabel="Precision (log scale)", xlim=(0, 1), yscale="log")
    a2.set_title("Precision-recall: the honest view for a 0.3% event")
    a2.legend(loc="lower left")
    return save_fig(fig, "predict_roc_pr", "Ranking quality of P(conversion | X) on the held-out test split",
                    "Same three models, two views. PR-AUC is judged against the prevalence line, not 0.5.")


def _lab_ci(m: str, met: dict, key: str, digits: int) -> str:
    lo, hi = met[f"{key}_ci"]
    return f"{LABELS[m]}  {met[key]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _loglog_reliability(ax, y, p, bins, color, label, lim) -> None:
    t = reliability_table(y, p, bins)
    ok = t["obs_rate"] > 0
    err = np.vstack([t["obs_rate"] - t["obs_lo"], t["obs_hi"] - t["obs_rate"]])[:, ok]
    ax.errorbar(t["mean_pred"][ok], t["obs_rate"][ok], yerr=np.clip(err, 0, None), fmt="o-", ms=4, lw=1.4,
                color=color, ecolor=color, elinewidth=0.9, capsize=0, label=label)
    reference_line(ax, lim, lim)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)


def _rel_limits(y, arrays: list[np.ndarray], bins: int) -> tuple[float, float]:
    vals = np.concatenate([reliability_table(y, p, bins)[["mean_pred", "obs_rate"]].to_numpy().ravel() for p in arrays])
    vals = vals[vals > 0]
    return float(vals.min() / 1.5), float(min(1.0, vals.max() * 1.5))


def plot_calibration(y: np.ndarray, scores: dict[str, np.ndarray], bins: int) -> str:
    """Reliability curves (equal-frequency bins, Wilson 95% CI) with predicted-probability histograms."""
    n = len(scores)
    fig, axes = new_figure(ncols=n, nrows=2, width=4.6, height=3.6, gridspec_kw={"height_ratios": [3, 1.3]})
    lim = _rel_limits(y, list(scores.values()), bins)
    for j, (m, p) in enumerate(scores.items()):
        _loglog_reliability(axes[0, j], y, p, bins, _color(m), LABELS[m], lim)
        axes[0, j].set_title(LABELS[m])
        axes[0, j].set_xlabel("")
        axes[0, j].set_ylabel("Observed conversion rate" if j == 0 else "")
        edges = np.geomspace(max(lim[0], 1e-6), 1.0, 50)
        axes[1, j].hist(np.clip(p, edges[0], 1.0), bins=edges, color=_color(m), alpha=0.85)
        axes[1, j].set(xscale="log", yscale="log", xlim=lim, xlabel="Predicted probability")
        axes[1, j].set_ylabel("Rows" if j == 0 else "")
    return save_fig(fig, "predict_calibration", "Are predicted probabilities the actual conversion rates?",
                    "Test split, log-log axes; dashed line = perfect calibration. Bars underneath show where the predictions sit.")


def plot_confusion(conf: dict, models: tuple[str, ...]) -> str:
    """Confusion matrices on test at thresholds chosen on validation (rows normalised by true class)."""
    rules = [("max_f1", "Threshold maximising F1 (validation)"), ("top_k", "Flag top 1% of users (validation quantile)")]
    fig, axes = new_figure(ncols=len(models), nrows=2, width=4.1, height=3.8)
    for i, (rule, rule_lab) in enumerate(rules):
        for j, m in enumerate(models):
            c = conf[(m, rule)]
            ax = axes[i, j]
            mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
            norm = mat / mat.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap=_CMAP, vmin=0, vmax=1)
            ax.grid(False)
            for r in range(2):
                for k in range(2):
                    ax.text(k, r, f"{int(mat[r, k]):,}\n({100 * norm[r, k]:.1f}%)", ha="center", va="center", fontsize=9,
                            color="white" if norm[r, k] > 0.5 else "#0b0b0b")
            ax.set_xticks([0, 1], ["no flag", "flagged"])
            ax.set_yticks([0, 1], ["no conversion", "converted"])
            ax.set_title(f"{LABELS[m]}", fontsize=10)
            ax.set_xlabel(f"precision {c['precision']:.3f}  recall {c['recall']:.3f}  F1 {c['f1']:.3f}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(rule_lab.split(" (")[0], fontsize=9)
    return save_fig(fig, "predict_confusion", "What do the operating points actually flag?",
                    "Test counts (share of the true class in brackets). Top row: max-F1 threshold; bottom row: top 1% flagged. Thresholds fit on validation.")


def plot_imbalance_tradeoff(metrics: dict, paired: dict) -> str:
    """Unweighted vs class-weighted LightGBM: ranking metrics and probability quality side by side."""
    models = ("lightgbm", "lightgbm_weighted")
    panels = [("roc_auc", "ROC-AUC", False), ("pr_auc", "PR-AUC", False), ("log_loss", "Log loss (lower is better)", True),
              ("mean_pred_over_base_rate", "Mean predicted / base rate (1 = unbiased)", True)]
    fig, axes = new_figure(ncols=4, width=3.3, height=4.0)
    for ax, (key, title, log_y) in zip(axes, panels):
        for i, m in enumerate(models):
            v = metrics[m][key]
            ci = metrics[m].get(f"{key}_ci")
            ax.errorbar([i], [v], yerr=None if ci is None else [[v - ci[0]], [ci[1] - v]], fmt="o", ms=8,
                        color=_color(m), capsize=4, lw=1.6)
        ax.set_xticks([0, 1], ["unweighted", "weighted"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=10)
        if log_y:
            ax.set_yscale("log")
    d = paired["lightgbm_weighted_minus_lightgbm"]
    # The title states whichever outcome the paired bootstrap supports, not the textbook expectation.
    ranking_hurt = d["roc_auc"]["hi"] < 0 and d["pr_auc"]["hi"] < 0
    title = ("Class weighting hurt ranking as well as calibration" if ranking_hurt
             else "Class weighting changes probabilities far more than it changes ranking")
    return save_fig(fig, "predict_imbalance_tradeoff", title,
                    f"LightGBM, same trees, test split, 95% CIs. Weighted minus unweighted: ROC-AUC {d['roc_auc']['diff']:+.4f}, PR-AUC {d['pr_auc']['diff']:+.4f}.")


def plot_calibration_methods(y: np.ndarray, cal_scores: dict, bins: int) -> str:
    """Uncalibrated vs Platt (sigmoid) vs isotonic, calibrators fit on validation, evaluated on test."""
    models = ["random_forest", "lightgbm", "lightgbm_weighted"]
    fig, axes = new_figure(ncols=len(models), width=4.7, height=4.4)
    allp = [p for m in models for p in cal_scores[m].values()]
    lim = _rel_limits(y, allp, bins)
    for ax, m in zip(axes, models):
        for meth, p in cal_scores[m].items():
            ll = row_log_loss(y, p).mean()
            _loglog_reliability(ax, y, p, bins, color_for(meth), f"{meth} (log loss {ll:.4f})", lim)
        ax.set_title(LABELS[m])
        ax.set_xlabel("Predicted probability")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Observed conversion rate")
    return save_fig(fig, "predict_calibration_methods", "Which calibration fix works, and for which model?",
                    "Test split, log-log axes, Wilson 95% CIs; dashed = perfect. Calibrators fit on the validation split only.")


def plot_cv(cv_summary: pd.DataFrame) -> str:
    """Cross-validation mean +/- SD across folds per configuration; the selected config per model is outlined."""
    df = cv_summary.reset_index(drop=True)
    labels = [f"{LABELS[r.model]}: {r.config}" for r in df.itertuples()]
    y = np.arange(len(df))[::-1]
    fig, axes = new_figure(ncols=2, width=5.2, height=0.42 * len(df) + 1.6, sharey=True)
    for ax, (key, title) in zip(axes, [("pr_auc", "PR-AUC (higher is better)"), ("log_loss", "Log loss (lower is better)")]):
        for yi, r in zip(y, df.itertuples()):
            ax.errorbar(getattr(r, f"{key}_mean"), yi, xerr=getattr(r, f"{key}_sd"), fmt="o", ms=9 if r.selected else 6,
                        color=_color(r.model), mfc=_color(r.model) if r.selected else "none", capsize=3, lw=1.4)
        ax.set_yticks(y, labels)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    return save_fig(fig, "predict_cv", "How much do hyperparameters matter, relative to fold-to-fold noise?",
                    "Stratified K-fold on a train subsample, features refit per fold. Dots = mean, bars = SD across folds, filled = selected by log loss.")


def plot_lr_coefficients(lr_tab: pd.DataFrame, k: int) -> str:
    """Forest plot of the k standardized logistic coefficients with the largest |z| (associations, not effects)."""
    top = lr_tab.reindex(lr_tab["z"].abs().sort_values(ascending=False).index).head(k).sort_values("coef")
    fig, ax = new_figure(width=6.6, height=0.32 * len(top) + 1.6)
    y = np.arange(len(top))
    cols = [SERIES[1] if c > 0 else SERIES[0] for c in top["coef"]]
    ax.hlines(y, top["ci_lo"], top["ci_hi"], color=cols, lw=1.6)
    ax.scatter(top["coef"], y, color=cols, zorder=3, s=28)
    ax.set_yticks(y, top["term"])
    zero_line(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel("Log-odds change per +1 SD (continuous, log-frequency) or per indicator (one-hot), 95% CI")
    return save_fig(fig, "predict_lr_coefficients", "Which inputs are associated with conversion in the logistic baseline?",
                    "Top terms by |z|, unpenalised GLM on a training subsample. Associations only; hashed features are anonymous.")


def plot_importance(imp: pd.DataFrame) -> str:
    """Three views of raw-feature importance for LightGBM: gain, permutation (validation), SHAP."""
    d = imp.sort_values("shap_mean_abs")
    y = np.arange(len(d))
    c = color_for("lightgbm")
    fig, axes = new_figure(ncols=3, width=4.3, height=4.6, sharey=True)
    axes[0].barh(y, d["gain_share"], color=c)
    axes[0].set_xlabel("Share of total split gain")
    axes[1].barh(y, d["perm_logloss_increase_mean"], xerr=d["perm_logloss_increase_sd"], color=c, error_kw={"lw": 1, "ecolor": INK_2})
    axes[1].set_xlabel("Log-loss increase when permuted (mean +/- SD)")
    axes[2].barh(y, d["shap_mean_abs"], color=c)
    axes[2].set_xlabel("Mean |SHAP| (log-odds, summed over encodings)")
    for ax, t in zip(axes, ["Gain", "Permutation (validation)", "SHAP"]):
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
    axes[0].set_yticks(y, d["raw_feature"])
    return save_fig(fig, "predict_importance", "Which raw features drive predicted conversion probability?",
                    "LightGBM on the tree design; encodings aggregated to raw features f0-f11. Predictive importance, not treatment-effect importance.")


def _display_names(cols: list[str]) -> dict[str, str]:
    return {c: (f"{c} (hashed id, unordered)" if c in ("f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11") else c) for c in cols}


def plot_shap_summary(shap_res: dict, top: int = 12) -> str:
    """SHAP beeswarm-style summary: each dot is a user; color = feature value percentile (ordered features only)."""
    rng = np.random.default_rng(0)
    order = shap_res["mean_abs"].sort_values(ascending=False).index[:top]
    names = _display_names(list(order))
    X, V, cols = shap_res["X"], shap_res["values"], shap_res["columns"]
    fig, ax = new_figure(width=7.2, height=0.42 * top + 1.8)
    for i, c in enumerate(order[::-1]):
        v = V[:, cols.index(c)]
        hist, edges = np.histogram(v, bins=60)
        dens = hist[np.clip(np.digitize(v, edges[1:-1]), 0, len(hist) - 1)] / max(hist.max(), 1)
        yy = i + rng.uniform(-1, 1, len(v)) * 0.38 * dens
        if names[c] == c:
            color = _CMAP(pd.Series(X[c].to_numpy()).rank(pct=True).to_numpy())
        else:
            color = MUTED
        ax.scatter(v, yy, c=color, s=4, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_yticks(range(top), [names[c] for c in order[::-1]])
    zero_line(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel("SHAP value (log-odds contribution to predicted conversion)")
    sm = plt.cm.ScalarMappable(cmap=_CMAP)
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.set_label("Feature value percentile", fontsize=9)
    cb.outline.set_visible(False)
    return save_fig(fig, "predict_shap_summary", "How do feature values push individual predictions up or down?",
                    f"TreeSHAP on {shap_res['n_rows']:,} validation rows, top {top} encoded columns. Gray = hashed ids whose numeric order is meaningless.")


def plot_shap_dependence(shap_res: dict, imp: pd.DataFrame) -> str:
    """Dependence of the total SHAP contribution of the most important raw feature on its value."""
    top = imp.sort_values("shap_mean_abs", ascending=False)["raw_feature"].iloc[0]
    cols = shap_res["columns"]
    members = [c for c in cols if c == top or c.startswith(f"{top}_")]
    total = shap_res["values"][:, [cols.index(c) for c in members]].sum(axis=1)
    categorical = f"{top}_logfreq" in cols and top in ("f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11")
    xcol = f"{top}_logfreq" if categorical else top
    x = shap_res["X"][xcol].to_numpy()
    fig, ax = new_figure(width=6.6, height=4.4)
    ax.scatter(x, total, s=5, alpha=0.35, color=MUTED, linewidths=0, rasterized=True)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    med = pd.DataFrame({"x": x, "s": total, "b": b}).groupby("b").agg(x=("x", "median"), s=("s", "median"))
    ax.plot(med["x"], med["s"], color=color_for("lightgbm"), lw=2, marker="o", ms=4, label="median per quantile bin")
    zero_line(ax)
    ax.set_xlabel(f"{xcol}" + (" (log count of the level in training; ordered)" if categorical else ""))
    ax.set_ylabel(f"Total SHAP of {top} (log-odds)")
    ax.legend(loc="best")
    return save_fig(fig, "predict_shap_dependence", f"How does the top feature ({top}) change predicted conversion?",
                    f"{shap_res['n_rows']:,} validation rows; each dot is a user. Association in the fitted model, not a causal effect.")
