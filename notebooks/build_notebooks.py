"""Generate the five analysis notebooks (thin layers over src/ and results/).

Run ``uv run python notebooks/build_notebooks.py`` and then ``make notebooks`` to execute them. The notebooks
never re-implement modelling code: they call ``src`` functions on the development sample to show the
mechanics, and read the full-data results that ``make all`` wrote to ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """\
import json
import numpy as np
import pandas as pd
from IPython.display import Image, display

from src.utils import REPO_ROOT

FIG, TAB, MET = (REPO_ROOT / "results" / d for d in ("figures", "tables", "metrics"))
pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
pd.set_option("display.max_colwidth", 120)

def fig(name, width=950):
    display(Image(filename=str(FIG / f"{name}.png"), width=width))

def table(name):
    return pd.read_csv(TAB / f"{name}.csv")

def metrics(name):
    return json.loads((MET / f"{name}.json").read_text())"""


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS: dict[str, list[nbf.NotebookNode]] = {}

NOTEBOOKS["01_eda.ipynb"] = [
    md("""
# 01 · Data understanding and EDA

Criteo Uplift v2.1 ([Kaggle](https://www.kaggle.com/datasets/arashnic/uplift-modeling),
[Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)): one row per user from randomized
advertising incrementality tests. Full-data numbers come from `uv run causalml eda --mode full`; the code cells
below also run the same functions on the 5% development sample to show what they compute.

**Assignment is not exposure.** `treatment` is the randomized variable (eligible to be shown the advertiser's ads).
`exposure` (actually shown at least one ad) happens after randomization and depends on user activity, so it is an
outcome of treatment, not a second treatment.
"""),
    code(SETUP),
    code("""
from src.data.load import load_criteo
from src.data.preprocess import dataset_summary, exposure_funnel, feature_profile, structural_violations
from src.features.build_features import CATEGORICAL, CONTINUOUS

dev = load_criteo("dev")
print(f"development sample: {len(dev):,} rows")
print("structural constraints (T=0 => no exposure, no conversion without a visit):", structural_violations(dev))
dataset_summary(dev, CONTINUOUS, CATEGORICAL)"""),
    md("## Full dataset: summary, variable definitions, feature profile"),
    code("""
eda = metrics("eda")
print(f"rows: {eda['rows']:,}   features: {eda['n_features']}   treated share: {eda['treatment_share']:.4f}")
for k, v in eda["variable_definitions"].items():
    print(f"- {k}: {v}")
table("eda_feature_profile")"""),
    md("""
Four features are continuous; eight are hashed categoricals (60 to 3,743 levels) whose numeric values are random
projections, so only their frequency carries order. Most features have a large point mass at one value, and those
point masses come in coupled pairs (same rows), which looks like a shared default value rather than information.
"""),
    code('table("eda_mode_coupling")'),
    md("## Outcomes by randomized arm (descriptive)"),
    code('fig("eda_target_distribution"); fig("eda_rates_by_arm"); table("eda_rates_by_treatment")'),
    md("""
## Why exposure cannot be used as the treatment

Only a small share of treated users is ever shown an ad, and those users are far more active. Treated users who
were *not* exposed convert below the control group, which is impossible if exposure were random. Comparing exposed
with unexposed users therefore measures who gets shown ads, not what ads do.
"""),
    code('fig("eda_exposure_funnel"); table("eda_exposure_funnel")'),
    md("## Covariate balance between arms"),
    code("""
fig("eda_balance_love")
bal = table("eda_balance").sort_values("smd", key=abs, ascending=False)
print(eda["balance"]["interpretation"] if "interpretation" in eda["balance"] else "")
bal.head(12)"""),
    md("""
Every standardized difference is below the conventional 0.1 threshold, but at 14M rows the pure-randomization noise
level is about 0.0008, and the largest differences are ~40x that. The statistical-analysis and causal notebooks
show that this small imbalance is correlated with the outcomes and matters for the effect estimates.
"""),
    md("## Distributions, correlations and association with conversion"),
    code('fig("eda_continuous_by_arm"); fig("eda_categorical_by_arm"); fig("eda_correlation"); fig("eda_conversion_by_feature")'),
]

NOTEBOOKS["02_statistical_analysis.ipynb"] = [
    md("""
# 02 · Classical statistical analysis of the experiment

Before any model: is there a treatment effect on visits and conversions, how large, and how certain? Then: does
randomization actually hold in this pooled benchmark? Full-data results come from `uv run causalml stats --mode full`.
"""),
    code(SETUP),
    md("""
## Two-proportion tests

For each outcome, H0: p_treated = p_control against H1: p_treated ≠ p_control, pooled-SE z-test; unpooled Wald CI
for the difference; delta-method CI on the log risk ratio for the relative lift. The same function on the
development sample:
"""),
    code("""
from src.causal.ate import two_proportion_inference
from src.data.load import load_criteo

dev = load_criteo("dev", columns=["treatment", "visit", "conversion"])
for y in ["visit", "conversion"]:
    g = dev.groupby("treatment")[y].agg(["sum", "count"])
    r = two_proportion_inference(int(g.loc[1, "sum"]), int(g.loc[1, "count"]), int(g.loc[0, "sum"]), int(g.loc[0, "count"]))
    print(f"{y:10s} effect {r['absolute_effect']['estimate']:+.5f}  95% CI {np.round(r['absolute_effect']['ci95_wald'], 5)}  "
          f"z = {r['z_test']['statistic']:.2f}  p = {r['z_test']['p_value']:.2g}  relative lift {r['relative_lift']['estimate']:+.1%}")"""),
    md("## Full data: hypotheses, statistics, p-values, CIs, interpretation"),
    code("""
stats = metrics("stats")
rows = []
for outcome, r in stats["outcomes"].items():
    h = r["hypothesis_test"]
    rows.append({"outcome": outcome, "null": h["null"], "alternative": h["alternative"], "test": h["test"],
                 "statistic": h["statistic"], "p_value": h["p_value"], "log10_p": h["log10_p_value"], "ci95": h["ci95"],
                 "bootstrap_ci95": r["bootstrap"]["absolute"]["ci95_percentile"]})
display(pd.DataFrame(rows))
for outcome, r in stats["outcomes"].items():
    print(f"{outcome}: {r['hypothesis_test']['interpretation']}")"""),
    code('fig("stats_effects"); table("stats_effects")'),
    md("""
The bootstrap uses the exact equivalence between a stratified row bootstrap of a binary outcome and drawing each
arm's positive count from Binomial(n_arm, p̂_arm), so 2,000 replicates over 14M rows cost nothing.

## Randomization checks

Randomization is what makes the difference in means an unbiased ATE: if T is independent of (Y(0), Y(1)),
E[Y | T=1] − E[Y | T=0] = E[Y(1) − Y(0)]. Balance checks can refute that assumption but never prove it.
"""),
    code("""
rand = stats["randomization"]
print(json.dumps({k: v for k, v in rand.items() if not isinstance(v, (list,))}, indent=1, default=str)[:3000])"""),
    code('fig("stats_balance")'),
    md("""
A classifier two-sample test (can a model predict treatment from the features better than chance on held-out data?)
and a joint likelihood-ratio test both reject independence of T and X. The paper's original C2ST used far fewer rows.
The practical consequence, quantified in notebook 04: the adjusted ATE is materially smaller than the raw difference.
"""),
]

NOTEBOOKS["03_predictive_ml.ipynb"] = [
    md("""
# 03 · Predictive machine learning: P(conversion | X)

The "who converts" model, deliberately separate from "who converts *because of* the ad". Treatment is not a feature.
Models: logistic regression (interpretable baseline), random forest, LightGBM. Conversion is 0.29% positive, so
accuracy is meaningless (predicting "no" everywhere is 99.7% accurate). Full results: `uv run causalml predict --mode full`.
"""),
    code(SETUP),
    code("""
pred = metrics("predict")
print(pred["why_pr_auc"])
table("predict_test_metrics")"""),
    code('fig("predict_roc_pr")'),
    md("""
ROC-AUC looks excellent for every model because it averages over millions of easy negatives. PR-AUC is judged
against the prevalence line; it shows the models are ~75x better than random at ranking converters, and that the
three model families are close.

## Recomputing the headline metrics from the saved test predictions
"""),
    code("""
from src.models.evaluation import classification_metrics
from src.utils import load_config

p = pd.read_parquet(REPO_ROOT / "data/processed/predictions_full.parquet")
test = p[p.split == 2]
pd.DataFrame({m: classification_metrics(test.conversion.to_numpy(), test[f"p_{m}"].to_numpy())
              for m in ["logistic_regression", "random_forest", "lightgbm"]}).T"""),
    md("## Cross-validation and a small hyperparameter search"),
    code('fig("predict_cv"); table("predict_cv_comparison")'),
    md("## Class imbalance: reweighting is not free"),
    code('fig("predict_imbalance_tradeoff")'),
    md("""
## Calibration

Targeting decisions multiply probabilities by money (expected value = p × value − cost), so a model that
over-predicts by 2x doubles the perceived value of every user. The unweighted models are already close to calibrated;
Platt and isotonic calibration (fit on validation only) barely move log loss. The class-weighted model is badly
miscalibrated until recalibrated.
"""),
    code('fig("predict_calibration"); fig("predict_calibration_methods"); table("predict_calibration")'),
    md("## Operating points (thresholds chosen on validation)"),
    code('fig("predict_confusion"); table("predict_thresholds")'),
    md("""
## Interpretation

Coefficients and importances describe what *predicts* conversion. They say nothing about what the ad *changes*;
notebook 05 compares these rankings with the features that drive the treatment effect.
"""),
    code('fig("predict_lr_coefficients"); fig("predict_importance"); fig("predict_shap_summary"); fig("predict_shap_dependence")'),
]

NOTEBOOKS["04_causal_inference.ipynb"] = [
    md("""
# 04 · Causal inference: average effects, propensity scores, exposure

Estimand: ATE = E[Y(1) − Y(0)] of *assignment* to advertising (intention-to-treat), for visit and conversion.
Full-data results: `uv run causalml causal --mode full` (all 13,979,592 rows; estimation, not prediction).
"""),
    code(SETUP),
    code("""
causal = metrics("causal")
for est, info in causal["estimators"].items():
    if isinstance(info, dict):
        print(f"## {est}")
        for k in ("assumptions", "treatment", "outcome_model", "estimand", "limitations"):
            if k in info:
                print(f"  {k}: {info[k]}")"""),
    md("## Every estimator, both outcomes"),
    code('fig("causal_forest"); table("causal_ate_estimates")'),
    md("""
The difference in means and every IPW variant that uses the *design* propensity (0.85 for everyone) agree with each
other; every estimator that conditions on the features — Lin regression adjustment, IPW with an estimated
propensity, AIPW, g-computation — lands 13-28% lower. Because the estimated propensity is correlated with the
outcomes, the raw treated-vs-control comparison over-states the effect. The doubly robust AIPW estimate is primary.
"""),
    code('print(json.dumps(causal["implication"], indent=1, default=str)[:2500])'),
    md("## Propensity scores"),
    code('fig("causal_propensity"); print(json.dumps(causal["propensity"], indent=1, default=str)[:1500])'),
    md("""
In a clean randomized experiment the propensity is a known constant and modelling it only reduces variance. In
observational data it is the whole identification strategy: you must model who got treated, and you can only adjust
for what you measured. The demo below makes an observational dataset out of the experiment by dropping users with a
probability that depends on a strong predictor and on treatment, then checks which estimators recover the full-data
answer.
"""),
    code('fig("causal_obs_demo"); table("causal_obs_demo_replicates").groupby(["outcome", "estimator"]).mean(numeric_only=True)'),
    md("""
## The effect of actually seeing an ad

Randomized assignment is an instrument for exposure: it moves exposure (control users are never exposed) and plausibly
affects outcomes only through ads shown. With one-sided non-compliance, the Wald ratio ITT_Y / ITT_E is the average
effect of exposure on the users who were exposed.
"""),
    code('fig("causal_exposure_iv"); print(json.dumps(causal["exposure"], indent=1, default=str)[:2500])'),
    md("## The same estimators on the development sample, using the propensity from the CATE stage"),
    code("""
from src.causal.ate import difference_in_means, ipw_hajek

s = pd.read_parquet(REPO_ROOT / "data/processed/scores_dev.parquet")
s = s[s.split == 2]
for y in ["visit", "conversion"]:
    dim = difference_in_means(s[y].to_numpy(), s.treatment.to_numpy())
    haj = ipw_hajek(s[y].to_numpy(), s.treatment.to_numpy(), s.propensity.to_numpy())
    print(f"{y:10s} DiM {dim['estimate']:.5f}  Hajek-IPW {haj['estimate']:.5f}  (dev test rows: {len(s):,})")"""),
]

NOTEBOOKS["05_uplift_optimization.ipynb"] = [
    md("""
# 05 · Heterogeneous effects, uplift modeling, and targeting decisions

The central question: **does targeting by predicted uplift find more incremental outcomes than targeting the users
most likely to convert?** CATE(x) = E[Y(1) − Y(0) | X = x] is estimated with S-, T-, X-, DR-learners, a causal
forest and a transformed-outcome learner, all trained on the training split, selected on validation, and evaluated
once on the 2.8M-user test split. Uplift quantities are propensity-weighted (Hajek) because treatment is not
independent of the features here (notebook 04).
"""),
    code(SETUP),
    code("""
cate = metrics("cate")
print("selected on validation:", cate["selected_learner"])
m = table("cate_uplift_metrics")
m[(m.estimator == "ipw") & (m.metric == "qini_normalized")].pivot(index="scorer", columns="outcome", values="estimate")"""),
    code('fig("cate_qini_curves")'),
    md("## Uplift targeting vs conversion-probability targeting"),
    code("""
fig("cate_response_vs_uplift")
d = table("cate_uplift_differences")
d[d.estimator == "ipw"]"""),
    md("""
**Visits:** at small budgets the uplift model finds substantially more incremental visits than the response model,
because the users most likely to visit mostly visit anyway ("sure things"). **Conversions:** no uplift learner beats
the response model. The share of conversions that would have happened anyway is roughly flat across score groups, so
the ad scales conversion roughly in proportion to the baseline rate, and the likeliest converters are also the users
with the largest absolute uplift.
"""),
    code('fig("cate_sure_things")'),
    md("## Reproducing one number from the saved scores"),
    code("""
from src.causal.uplift_metrics import uplift_at_k

s = pd.read_parquet(REPO_ROOT / "data/processed/scores_full.parquet",
                    columns=["split", "treatment", "visit", "response_visit", "cate_x_learner_visit"])
s = s[s.split == 2]
for name in ["cate_x_learner_visit", "response_visit"]:
    print(f"{name:22s} uplift in the top 5% (unweighted): {uplift_at_k(s.visit.to_numpy(), s.treatment.to_numpy(), s[name].to_numpy(), 0.05):.4f}")"""),
    md("## Treatment-effect heterogeneity"),
    code('fig("cate_tau_distributions"); fig("cate_uplift_calibration"); print(json.dumps(cate["blp"], indent=1, default=str)[:1500])'),
    md("""
The BLP test (Chernozhukov et al.) asks whether the predicted effects carry real signal about the true effects; a
slope near 1 means they are well scaled. Predicted *negative* effects are a different story:
"""),
    code('fig("cate_negative_uplift"); table("cate_negative_uplift")'),
    code('fig("cate_segment_uplift"); fig("cate_rank_correlation"); fig("cate_effect_importance")'),
    md("""
## Predicting conversion is not predicting the treatment effect

Feature importance for the conversion model (notebook 03) versus for the effect model:
"""),
    code("""
pi = table("predict_feature_importance").set_index("raw_feature")[["shap_share"]]
pi.columns = ["conversion model: SHAP share"]
ei = table("cate_effect_importance")
ei = ei.assign(share=ei.importance / ei.groupby(["outcome", "method", "learner"]).importance.transform("sum"))
eff = ei.pivot_table(index="feature", columns=["outcome", "method"], values="share")
eff.columns = [f"effect ({o}): {m}" for o, m in eff.columns]
cmp = pd.concat([pi, eff], axis=1).sort_values(pi.columns[0], ascending=False)
print("Spearman rank correlation between conversion importance and each effect importance:")
print(cmp.corr(method="spearman").iloc[0, 1:].round(2).to_string())
cmp"""),
    md("""
## Decision optimization under a budget

Expected incremental value of treating a user = τ̂(x) × value − cost. Value and cost are illustrative assumptions
(the benchmark is sub-sampled, so absolute rates are not real-world). Policies are evaluated off-policy on the
randomized test split with propensity weights.
"""),
    code("""
tg = metrics("targeting")
print({k: tg[k] for k in ("value_per_conversion", "cost_per_treatment", "cost_value_ratio_default", "ate_test", "selected_learner")})
table("targeting_policy_values").head(30)"""),
    code('fig("targeting_incremental_vs_budget"); fig("targeting_policy_bars"); fig("targeting_net_value_vs_budget"); fig("targeting_cost_sensitivity")'),
    md("## Robustness: seeds, training halves, sample size"),
    code("""
if (MET / "robustness.json").exists():
    rb = metrics("robustness")
    print(json.dumps(rb["stability"], indent=1, default=str))
    fig("robustness_seed_stability"); fig("robustness_split_half"); fig("robustness_learning_curve")
    display(table("robustness_response_by_arm"))
else:
    print("run `make robustness` first")"""),
]


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbf.write(nb, HERE / name)


if __name__ == "__main__":
    main()
