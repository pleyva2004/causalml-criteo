"""Figures for the ``stats`` and ``causal`` stages (average effects, randomization checks, propensity).

Effects are drawn in percentage points. Estimates always carry their 95% interval; reference values
(the difference in means, the RCT benchmark, AUC = 0.5) are muted dashed lines. Colors follow
``src.visualization.plots`` entities: outcomes (visit / conversion), arms (control / treatment) and
models (lightgbm / logistic_regression). Biased "naive" contrasts are drawn in muted gray.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.visualization.plots import (
    GRID,
    INK,
    INK_2,
    MUTED,
    color_for,
    new_figure,
    reference_line,
    save_fig,
    zero_line,
)

PP = 100.0  # proportions -> percentage points


def _vline(ax, x: float, label: str | None = None, style: str = "--", color: str = MUTED) -> None:
    ax.axvline(x, color=color, linestyle=style, linewidth=1.2, label=label, zorder=2)


def plot_stats_effects(res: dict[str, Any], draws: dict[str, dict[str, np.ndarray]], outcomes: list[str],
                       alpha: float) -> str:
    """Bootstrap distribution of the absolute and relative effect per outcome, with analytic and percentile CIs."""
    fig, axes = new_figure(ncols=len(outcomes), nrows=2, width=4.8, height=3.0, squeeze=False)
    level = round((1 - alpha) * 100)
    for j, o in enumerate(outcomes):
        r = res[o]
        panels = [
            ("absolute", PP, r["absolute_effect"]["estimate"], r["absolute_effect"]["ci95_wald"],
             "Wald", "absolute effect (percentage points)"),
            ("relative", 100.0, r["relative_lift"]["estimate"], r["relative_lift"]["ci95_delta_log_rr"],
             "delta-method log RR", "relative lift (%)"),
        ]
        for i, (key, scale, point, ci, ci_name, xlabel) in enumerate(panels):
            ax = axes[i, j]
            x = draws[o][key] * scale
            ax.hist(x, bins=60, color=color_for(o), alpha=0.45, label="bootstrap replicates")
            boot_ci = r["bootstrap"][key]["ci95_percentile"]
            for k, v in enumerate(ci):
                _vline(ax, v * scale, f"analytic {level}% CI (Wald; delta-method log RR for lift)" if k == 0 else None)
            for k, v in enumerate(boot_ci):
                _vline(ax, v * scale, f"bootstrap {level}% percentile CI" if k == 0 else None, ":", INK_2)
            ax.axvline(point * scale, color=INK, linewidth=1.6, label="point estimate")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("replicates")
            ax.set_title(f"{o}: {point * scale:.3f}{' pp' if key == 'absolute' else '%'}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    b = next(iter(res.values()))["bootstrap"]["reps"]
    n_t = next(iter(res.values()))["rates"]["counts"]["n_t"]
    n_c = next(iter(res.values()))["rates"]["counts"]["n_c"]
    return save_fig(fig, "stats_effects",
                    "Do the exact bootstrap and the analytic intervals agree on the assignment effect?",
                    f"n_T = {n_t:,}, n_C = {n_c:,}; {b:,} stratified bootstrap replicates drawn as exact binomial counts")


def plot_stats_balance(c2st_plot: dict[str, Any], smd: pd.DataFrame, rand: dict[str, Any]) -> str:
    """ROC of the treatment classifiers, AUC permutation null, and per-feature SMDs against the noise band."""
    fig, axes = new_figure(ncols=3, nrows=1, width=4.4, height=4.0)
    ax = axes[0]
    for name, (grid, tpr, auc) in c2st_plot["roc"].items():
        lo, hi = auc["ci95_delong"]
        ax.plot(grid, tpr, color=color_for(name), linewidth=1.6,
                label=f"{name.replace('_', ' ')}: AUC {auc['auc']:.4f} [{lo:.4f}, {hi:.4f}]")
    reference_line(ax, [0, 1], [0, 1], "chance (AUC 0.5)")
    ax.set_xlabel("false positive rate (control scored as treated)")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on the held-out test split", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    p_txts = []
    for name, (null, obs, p) in c2st_plot["null"].items():
        floor = p <= 1.0 / (len(null) + 1) + 1e-12
        p_txts.append(f"p < {1.0 / len(null):.3g}" if floor else f"p = {p:.3g}")
        short = "LightGBM" if name == "lightgbm" else "logistic"
        ax.hist(null, bins=40, histtype="step", color=color_for(name), linewidth=1.4, label=f"{short} null")
        ax.axvline(obs, color=color_for(name), linewidth=2.0, label=f"{short} observed")
    top = ax.get_ylim()[1]
    reference_line(ax, [0.5, 0.5], [0, top], None)
    ax.set_ylim(0, top * 1.35)
    ax.set_xlabel("test-split AUC")
    ax.set_ylabel("permutations")
    n_perm = len(next(iter(c2st_plot["null"].values()))[0])
    ax.set_title(f"AUC vs {n_perm:,}-permutation null: {', '.join(p_txts)}", loc="left", fontsize=10)
    ax.legend(loc="upper center", fontsize=7.5, ncol=2)
    ax.ticklabel_format(axis="x", useOffset=False)

    ax = axes[2]
    s = smd.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(s))
    band = 1.96 * s["se_null"].iloc[0]
    ax.axvspan(-band, band, color=GRID, alpha=0.8, label="95% band under randomization")
    zero_line(ax, axis="x")
    ax.scatter(s["smd"], yy, color=INK, s=16, zorder=3, label="observed SMD")
    ax.set_yticks(yy, s["feature"], fontsize=7.5)
    ax.set_xlabel("standardized mean difference (T - C)")
    ax.set_title("Covariate balance, all rows", loc="left")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=2)
    lg = rand["c2st"]["models"]["lightgbm"]
    lrt = rand["c2st"]["joint_lrt"]
    return save_fig(fig, "stats_balance", "Can features predict treatment assignment?",
                    f"LightGBM held-out AUC {lg['auc']:.4f} (DeLong 95% CI {lg['ci95_delong'][0]:.4f}-"
                    f"{lg['ci95_delong'][1]:.4f}); joint logit LR test chi2({lrt['df']}) = {lrt['llr']:.0f}, "
                    f"p = {lrt['p_value']:.3g}; max |SMD| = {rand['smd']['max_abs_smd']:.4f} "
                    f"(noise SD {rand['smd']['noise_sd']:.5f})")


def _forest(ax, labels: list[str], est: np.ndarray, lo: np.ndarray, hi: np.ndarray, colors: list[str],
            hollow: list[bool] | None = None) -> None:
    """Points with 95% CI bars; rows with a missing CI (no valid SE) are drawn as a hollow point only."""
    yy = np.arange(len(labels))[::-1]
    for k, (x, y) in enumerate(zip(est, yy)):
        if np.isfinite(lo[k]) and np.isfinite(hi[k]):
            ax.errorbar([x], [y], xerr=[[x - lo[k]], [hi[k] - x]], fmt="none", ecolor=colors[k], elinewidth=1.6,
                        capsize=3, zorder=3)
        face = "white" if hollow and hollow[k] else colors[k]
        ax.scatter([x], [y], s=34, color=face, edgecolor=colors[k], linewidth=1.4, zorder=4)
    ax.set_yticks(yy, labels, fontsize=8)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)


def plot_causal_forest(table: pd.DataFrame, outcomes: list[str], primary: str | None = None) -> str:
    """Forest plot: every ATE estimator with its 95% CI, per outcome, against the difference in means."""
    fig, axes = new_figure(ncols=len(outcomes), nrows=1, width=5.2, height=4.6, squeeze=False)
    for j, o in enumerate(outcomes):
        ax = axes[0, j]
        g = table[table["outcome"] == o].reset_index(drop=True)
        dim = g[g["estimator"] == "dim"].iloc[0]
        ax.axvspan(dim["ci_low"] * PP, dim["ci_high"] * PP, color=GRID, alpha=0.7, label="DiM 95% CI")
        _vline(ax, dim["estimate"] * PP, "difference in means")
        labels = [f"{r.label}{' (no valid SE)' if r.estimator == 'g_comp_s_lgbm' else ''}" for r in g.itertuples()]
        colors = [MUTED if k == "ipw_hajek_lgbm_overfit" else color_for(o) for k in g["estimator"]]
        _forest(ax, labels, g["estimate"].to_numpy() * PP, g["ci_low"].astype(float).to_numpy() * PP,
                g["ci_high"].astype(float).to_numpy() * PP, colors, hollow=list(g["estimator"] == "g_comp_s_lgbm"))
        ax.set_xlabel(f"ATE on {o} (percentage points)")
        ax.set_title(o, loc="left")
        if j > 0:
            ax.set_yticks(ax.get_yticks(), [""] * len(ax.get_yticks()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.05), fontsize=8.5)
    samples = ", ".join(sorted(table["sample"].unique()))
    prim = "" if primary is None else f" Primary estimate: {table.loc[table['estimator'] == primary, 'label'].iloc[0]}."
    return save_fig(fig, "causal_forest", "Do covariate-adjusted and weighting estimators agree with the "
                    "difference in means?", f"95% CIs; {samples}. Hollow: point only (no valid SE). "
                    f"Gray: sensitivity check (overfit propensity).{prim}")


def plot_propensity(e_lgbm: np.ndarray, e_lr: np.ndarray, t: np.ndarray, prop: dict[str, Any],
                    e_overfit: np.ndarray | None = None) -> str:
    """Distribution of cross-fitted propensity scores by arm, against the design propensity."""
    t = np.asarray(t).astype(bool)
    panels = [("lightgbm", e_lgbm, "lightgbm", "LightGBM (early stopping)"),
              ("logistic_regression", e_lr, "logistic", "logistic regression")]
    if e_overfit is not None:
        panels.append(("lightgbm", e_overfit, "lightgbm_no_early_stopping", "LightGBM, no early stopping"))
    fig, axes = new_figure(ncols=len(panels), nrows=1, width=4.6, height=3.6)
    for ax, (name, e, key, title) in zip(axes, panels):
        lo, hi = np.quantile(e, [0.0005, 0.9995])
        pad = 0.1 * (hi - lo) + 1e-4
        bins = np.linspace(min(lo, prop["design_propensity"]) - pad, max(hi, prop["design_propensity"]) + pad, 80)
        for arm, mask in (("control", ~t), ("treatment", t)):
            ax.hist(np.clip(e[mask], bins[0], bins[-1]), bins=bins, density=True, histtype="stepfilled",
                    alpha=0.35, color=color_for(arm), label=arm)
            ax.hist(np.clip(e[mask], bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                    color=color_for(arm), linewidth=1.2)
        _vline(ax, prop["design_propensity"], "design e = 0.85")
        dg = prop[key]
        ax.set_title(f"{title}\nAUC {dg['auc_treatment_vs_control']:.4f}, control IPW ESS ratio "
                     f"{dg['by_arm']['control']['ipw_ess_ratio']:.3f}", loc="left", fontsize=10)
        ax.set_xlabel("cross-fitted propensity e(x) (0.05%-99.95% range shown)")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7.5)
    lg = prop["lightgbm"]
    q = lg["by_arm"]["treatment"]["quantiles"]
    return save_fig(fig, "causal_propensity", "How far do estimated propensities stray from the design value 0.85?",
                    f"{prop['folds']}-fold cross-fitted, {prop['sample']}; LightGBM e range {lg['min']:.3f}-{lg['max']:.3f} "
                    f"(treated q01-q99 {q['q01']:.3f}-{q['q99']:.3f}); clip bounds {prop['clip_bounds']}")


OBS_LABELS = {"naive_dim": "Naive difference in means", "lin": "Lin regression (linear X)",
              "ipw_hajek_known_selection": "Hajek IPW, known selection e", "ipw_hajek_lgbm": "Hajek IPW, LightGBM e",
              "aipw_lgbm": "AIPW, LightGBM mu + e"}


def plot_obs_demo(reps: pd.DataFrame, obs: dict[str, Any], outcomes: list[str]) -> str:
    """Per-seed estimates and CIs on the confounded sample vs the full-RCT benchmark."""
    fig, axes = new_figure(ncols=len(outcomes), nrows=1, width=5.2, height=4.0, squeeze=False)
    names = list(OBS_LABELS)
    for j, o in enumerate(outcomes):
        ax = axes[0, j]
        b = obs["benchmarks"]["dim"][o]
        ax.axvspan(b["ci95"][0] * PP, b["ci95"][1] * PP, color=GRID, alpha=0.7, label="full-data DiM 95% CI")
        _vline(ax, b["estimate"] * PP, "full-data difference in means")
        _vline(ax, obs["benchmarks"]["aipw"][o]["estimate"] * PP, "full-data AIPW", ":", INK)
        g = reps[reps["outcome"] == o]
        seeds = sorted(g["seed"].unique())
        offs = np.linspace(-0.3, 0.3, len(seeds)) if len(seeds) > 1 else [0.0]
        labels = []
        for k, name in enumerate(names):
            y0 = len(names) - 1 - k
            gg = g[g["estimator"] == name].set_index("seed")
            color = MUTED if name == "naive_dim" else color_for(o)
            for off, s in zip(offs, seeds):
                r = gg.loc[s]
                ax.plot([r["ci_low"] * PP, r["ci_high"] * PP], [y0 + off] * 2, color=color, linewidth=1.0, alpha=0.7)
                ax.scatter([r["estimate"] * PP], [y0 + off], s=12, color=color, zorder=3)
            sm = obs["summary"][o][name]
            labels.append(f"{OBS_LABELS[name]}\ncovers DiM {sm['vs_dim']['covered']}/{sm['replicates']}, "
                          f"AIPW {sm['vs_aipw']['covered']}/{sm['replicates']}")
        ax.set_yticks(np.arange(len(names))[::-1], labels, fontsize=7.5)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"ATE on {o} (percentage points)")
        ax.set_title(o, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    c = obs["confounder"]
    bal = obs["balance"]
    return save_fig(fig, "causal_obs_demo", "When treatment is confounded, which estimators recover the RCT effect?",
                    f"{len(bal)} seeds; selection on T and the ECDF of {c['feature']} (keeps ~{bal[0]['n_selected']:,} "
                    f"rows); one line per seed with its 95% CI")


def plot_exposure_iv(exposure: dict[str, Any], outcomes: list[str]) -> str:
    """Assignment effect, naive exposure contrasts, and the IV (Wald) effect of exposure on the exposed."""
    fig, axes = new_figure(ncols=len(outcomes), nrows=1, width=5.2, height=3.4, squeeze=False)
    for j, o in enumerate(outcomes):
        ax = axes[0, j]
        e = exposure[o]
        iv = e["iv_wald"]
        items = [
            ("ITT: effect of assignment", e["itt"]["estimate"], e["itt"]["ci95"], False),
            ("Naive: exposed vs control", e["naive_exposed_vs_control"]["estimate"],
             e["naive_exposed_vs_control"]["ci95"], True),
            ("Naive: exposed vs unexposed treated", e["naive_exposed_vs_unexposed_treated"]["estimate"],
             e["naive_exposed_vs_unexposed_treated"]["ci95"], True),
            ("IV (Wald): effect of exposure on exposed", iv["late"], iv["ci95_delta"], False),
        ]
        if "iv_wald_covariate_adjusted" in e:
            adj = e["iv_wald_covariate_adjusted"]
            items.append(("IV, covariate-adjusted (AIPW ratio)", adj["late"], adj["ci95"], False))
        yy = np.arange(len(items))[::-1]
        for (label, est, ci, naive), y in zip(items, yy):
            color = MUTED if naive else color_for(o)
            ax.errorbar([est * PP], [y], xerr=[[est * PP - ci[0] * PP], [ci[1] * PP - est * PP]], fmt="o",
                        color=color, ecolor=color, capsize=3, markersize=6)
            ax.annotate(f"{est * PP:.2f}", (est * PP, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=INK_2)
        zero_line(ax, axis="x")
        ax.set_yticks(yy, [it[0] for it in items], fontsize=8)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"effect on {o} (percentage points)")
        ax.set_title(f"{o}: first stage {iv['itt_d_first_stage']:.2%} exposed", loc="left")
        ax.margins(y=0.15)
    return save_fig(fig, "causal_exposure_iv", "How large is the effect of actually seeing an ad, and how wrong "
                    "are naive exposure comparisons?",
                    "Gray = naive contrasts (confounded by user activity); colored = assignment as the instrument, "
                    "unadjusted and covariate-adjusted. 95% CIs (delta method).")
