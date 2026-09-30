"""Heterogeneous treatment effects (CATE) and uplift modelling: meta-learners, fitting, evaluation.

Estimand: ``tau(x) = E[Y(1) - Y(0) | X = x]`` for Y in {visit, conversion}, where T is the randomized
*assignment* to the advertiser's campaign (not exposure). Identification: T was randomized with
probability ~0.85 independently of X (Diemert et al. 2021; their classifier two-sample test found no
treatment predictability), so ``tau(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x]`` (unconfoundedness,
overlap and SUTVA hold by design). The EDA found small but systematic covariate imbalance, so a
cross-fitted propensity model e(x) is estimated anyway and used wherever a learner needs the
propensity (X-learner weights, DR and transformed-outcome pseudo-outcomes) and as a robustness
check of every held-out comparison (Hajek-weighted versions).

Learners (all base models come from the shared ``make_lgbm`` factory, tree design matrix):

* **S-learner** -- one outcome model ``mu(x, t)`` with T as a feature; ``tau = mu(x,1) - mu(x,0)``.
  Weakness: when T is a weak feature, regularisation (and split selection) shrinks tau towards 0.
* **T-learner** -- separate ``mu_1`` (treated) and ``mu_0`` (control); ``tau = mu_1 - mu_0``.
  Weakness: with an 85/15 split ``mu_0`` sees only 15% of the data, so the difference of two
  independently regularised models mostly reflects ``mu_0``'s noise.
* **X-learner** (Kunzel et al. 2019) -- impute individual effects with the *other* arm's model
  (``D1 = Y - mu_0(X)`` on treated, ``D0 = mu_1(X) - Y`` on control), regress each on X, and blend
  ``tau = g(x) tau_0(x) + (1 - g(x)) tau_1(x)`` with ``g = e(x)``. With 85% treated this puts most
  weight on ``tau_0``, whose imputations use the well-estimated ``mu_1`` (fit on 85% of the data);
  the noisy ``mu_0`` enters mainly through the down-weighted ``tau_1``. The imputations are
  out-of-sample by construction (each unit is scored by the model of the arm it is *not* in).
* **DR-learner** (Kennedy 2023) -- regress cross-fitted AIPW pseudo-outcomes
  ``phi = mu_1 - mu_0 + T (Y - mu_1) / e - (1 - T)(Y - mu_0) / (1 - e)`` on X; ``E[phi | X] = tau(X)``
  if either the outcome models or the propensity is correct (here e is ~known).
* **Transformed outcome / class transformation** (Athey & Imbens 2016; Jaskowski & Jaroszewicz
  2012) -- regress ``Z = Y (T - e) / (e (1 - e))`` on X; ``E[Z | X] = tau(X)``. The class-transformation
  label ``1[Y == T]`` is the special case e = 0.5; for rare outcomes and e != 0.5 the Z form has much
  lower variance (Y = 0 rows contribute exactly 0), so it is the version implemented.
* **Causal forest** -- see ``src.causal.causal_forest``.
* **Response baseline** -- ``P(Y = 1 | X)`` fit on all training rows ignoring T: "target the users
  most likely to convert". It is what uplift targeting has to beat.

Data use: every model is fit on the train split; the number of boosting rounds is chosen by early
stopping on one half of the validation split (a hash of ``row_id``), the other half selects the
learner (Qini coefficient); the test split is touched only for the final held-out evaluation.

Evaluation estimator: treatment is *not* independent of X in this sample -- on held-out data the
treated share rises from ~84.9% to ~86.1% in the top decile of predicted visit probability (see
``assignment_imbalance``), plausibly a side effect of the dataset's label-dependent sub-sampling.
Unweighted treated-vs-control differences inside a targeted group then mix the effect with that
imbalance, so every held-out comparison is reported primarily with propensity-weighted (Hajek)
arm means, and secondarily unweighted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.causal import uplift_metrics as um
from src.causal.causal_forest import fit_causal_forest, forest_ate, forest_importances, predict_cate
from src.causal.propensity import make_folds, propensity_diagnostics
from src.data.load import FEATURES, load_criteo
from src.features.build_features import CONTINUOUS, RAW_FEATURES, build_matrices
from src.models.tree_models import make_lgbm
from src.utils import get_logger, load_config, repo_path, save_json, save_table

log = get_logger(__name__)

META_LEARNERS = ["s_learner", "t_learner", "x_learner", "dr_learner", "class_transformation"]
ModelFactory = Callable[[], Any]


class ValSet(NamedTuple):
    """Early-stopping data: features, treatment, outcome and propensity of held-out rows."""
    X: pd.DataFrame
    t: np.ndarray
    y: np.ndarray
    e: np.ndarray


# ---------------------------------------------------------------------------- base-model helpers
def _fit(model: Any, X: pd.DataFrame, y: np.ndarray, eval_data: tuple[pd.DataFrame, np.ndarray] | None,
         rounds: int | None) -> Any:
    """Fit; LightGBM models early-stop on ``eval_data`` (other sklearn models ignore it)."""
    kw: dict[str, Any] = {}
    if eval_data is not None and rounds and isinstance(model, lgb.LGBMModel):
        kw = {"eval_X": eval_data[0], "eval_y": eval_data[1],
              "callbacks": [lgb.early_stopping(rounds, first_metric_only=True, verbose=False)]}
    model.fit(X, y, **kw)
    return model


def _proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _trees(model: Any) -> int | None:
    best = getattr(model, "best_iteration_", None)
    return int(best) if best else getattr(model, "n_estimators", None)


def _with_t(X: pd.DataFrame, t: np.ndarray | float) -> pd.DataFrame:
    out = X.copy()
    out["treatment"] = np.broadcast_to(np.asarray(t, dtype=np.float32), len(X))
    return out


def _arm(val: ValSet | None, arm: int) -> tuple[pd.DataFrame, np.ndarray] | None:
    if val is None:
        return None
    m = val.t == arm
    return val.X[m], val.y[m]


# ---------------------------------------------------------------------------- learners
@dataclass
class CATELearner:
    """Common interface: ``fit(X, t, y, e, val)`` then ``predict(X, e)`` returns tau-hat(x).

    ``outcome_model`` builds a fresh classifier for E[Y | ...]; ``effect_model`` a fresh regressor for
    effect targets. ``e`` is the propensity of the rows being fit / predicted (cross-fitted on train).
    """
    outcome_model: ModelFactory
    effect_model: ModelFactory | None = None
    early_stopping_rounds: int | None = None
    name: str = "cate"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, e: np.ndarray | None = None,
            val: ValSet | None = None) -> CATELearner:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, e: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {}


@dataclass
class SLearner(CATELearner):
    name: str = "s_learner"

    def fit(self, X, t, y, e=None, val=None):
        ev = None if val is None else (_with_t(val.X, val.t), val.y)
        self.model_ = _fit(self.outcome_model(), _with_t(X, t), y, ev, self.early_stopping_rounds)
        return self

    def predict(self, X, e=None):
        return _proba(self.model_, _with_t(X, 1.0)) - _proba(self.model_, _with_t(X, 0.0))

    def info(self):
        return {"trees": _trees(self.model_)}


@dataclass
class TLearner(CATELearner):
    name: str = "t_learner"

    def fit(self, X, t, y, e=None, val=None):
        t = np.asarray(t)
        self.mu1_ = _fit(self.outcome_model(), X[t == 1], y[t == 1], _arm(val, 1), self.early_stopping_rounds)
        self.mu0_ = _fit(self.outcome_model(), X[t == 0], y[t == 0], _arm(val, 0), self.early_stopping_rounds)
        return self

    def mu(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mu_0(x), mu_1(x))."""
        return _proba(self.mu0_, X), _proba(self.mu1_, X)

    def predict(self, X, e=None):
        m0, m1 = self.mu(X)
        return m1 - m0

    def info(self):
        return {"trees_mu1": _trees(self.mu1_), "trees_mu0": _trees(self.mu0_)}


@dataclass
class XLearner(CATELearner):
    """X-learner; reuses a fitted ``TLearner`` for the first stage when given one."""
    t_learner: TLearner | None = None
    name: str = "x_learner"

    def fit(self, X, t, y, e=None, val=None):
        t = np.asarray(t)
        if self.t_learner is None or not hasattr(self.t_learner, "mu1_"):
            self.t_learner = TLearner(self.outcome_model, early_stopping_rounds=self.early_stopping_rounds)
            self.t_learner.fit(X, t, y, val=val)
        tl = self.t_learner
        X1, X0 = X[t == 1], X[t == 0]
        d1 = y[t == 1] - _proba(tl.mu0_, X1)          # observed treated outcome - imputed control outcome
        d0 = _proba(tl.mu1_, X0) - y[t == 0]          # imputed treated outcome - observed control outcome
        ev1 = ev0 = None
        if val is not None:
            v1, v0 = val.t == 1, val.t == 0
            ev1 = (val.X[v1], val.y[v1] - _proba(tl.mu0_, val.X[v1]))
            ev0 = (val.X[v0], _proba(tl.mu1_, val.X[v0]) - val.y[v0])
        self.tau1_ = _fit(self.effect_model(), X1, d1, ev1, self.early_stopping_rounds)
        self.tau0_ = _fit(self.effect_model(), X0, d0, ev0, self.early_stopping_rounds)
        self.default_g_ = float(np.mean(t))
        return self

    def predict(self, X, e=None):
        g = self.default_g_ if e is None else np.asarray(e)
        return g * self.tau0_.predict(X) + (1.0 - g) * self.tau1_.predict(X)

    def info(self):
        return {"trees_tau1": _trees(self.tau1_), "trees_tau0": _trees(self.tau0_)}


@dataclass
class DRLearner(CATELearner):
    """DR-learner with K-fold cross-fitted outcome models (propensity supplied, already cross-fitted)."""
    n_folds: int = 2
    seed: int = 0
    name: str = "dr_learner"

    def pseudo_outcome(self, t: np.ndarray, y: np.ndarray, e: np.ndarray, m0: np.ndarray, m1: np.ndarray
                       ) -> np.ndarray:
        return m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)

    def fit(self, X, t, y, e=None, val=None):
        t, y = np.asarray(t), np.asarray(y)
        e = np.full(len(t), t.mean()) if e is None else np.asarray(e)
        folds = make_folds(2 * t + y, self.n_folds, self.seed)
        m0, m1 = np.empty(len(t)), np.empty(len(t))
        self.fold_models_: list[TLearner] = []
        for k in range(self.n_folds):
            tr, te = folds != k, folds == k
            tl = TLearner(self.outcome_model, early_stopping_rounds=self.early_stopping_rounds)
            tl.fit(X[tr], t[tr], y[tr], val=val)
            m0[te], m1[te] = tl.mu(X[te])
            self.fold_models_.append(tl)
        phi = self.pseudo_outcome(t, y, e, m0, m1)
        ev = None
        if val is not None:
            vm0, vm1 = self._mu_avg(val.X)
            ev = (val.X, self.pseudo_outcome(val.t, val.y, val.e, vm0, vm1))
        self.final_ = _fit(self.effect_model(), X, phi, ev, self.early_stopping_rounds)
        self.pseudo_sd_ = float(phi.std())
        return self

    def _mu_avg(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        preds = [tl.mu(X) for tl in self.fold_models_]
        return np.mean([p[0] for p in preds], axis=0), np.mean([p[1] for p in preds], axis=0)

    def predict(self, X, e=None):
        return self.final_.predict(X)

    def info(self):
        return {"trees_final": _trees(self.final_), "pseudo_outcome_sd": self.pseudo_sd_,
                "trees_nuisance": [tl.info() for tl in self.fold_models_]}


@dataclass
class TransformedOutcomeLearner(CATELearner):
    """Regression of the transformed outcome Z = Y (T - e) / (e (1 - e)) on X."""
    name: str = "class_transformation"

    @staticmethod
    def transform(t: np.ndarray, y: np.ndarray, e: np.ndarray) -> np.ndarray:
        return y * (t - e) / (e * (1 - e))

    def fit(self, X, t, y, e=None, val=None):
        t, y = np.asarray(t), np.asarray(y)
        e = np.full(len(t), t.mean()) if e is None else np.asarray(e)
        ev = None if val is None else (val.X, self.transform(val.t, val.y, val.e))
        self.model_ = _fit(self.effect_model(), X, self.transform(t, y, e), ev, self.early_stopping_rounds)
        return self

    def predict(self, X, e=None):
        return self.model_.predict(X)

    def info(self):
        return {"trees": _trees(self.model_)}


# ---------------------------------------------------------------------------- propensity
def crossfit_propensity(X: pd.DataFrame, t: np.ndarray, folds: np.ndarray, model: ModelFactory,
                        val: tuple[pd.DataFrame, np.ndarray] | None, rounds: int | None
                        ) -> tuple[np.ndarray, list[Any]]:
    """Out-of-fold e(x) on the training rows plus the fold models (average them for new rows)."""
    e = np.empty(len(t))
    models = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = _fit(model(), X[tr], t[tr], val, rounds)
        e[te] = _proba(m, X[te])
        models.append(m)
    return e, models


def predict_propensity(models: list[Any], X: pd.DataFrame) -> np.ndarray:
    return np.mean([_proba(m, X) for m in models], axis=0)


# ---------------------------------------------------------------------------- fitting pipeline
def selection_mask(row_id: np.ndarray, seed: int, share: float) -> np.ndarray:
    """True for validation rows reserved for model selection (never used for early stopping).

    A deterministic hash of ``row_id`` (splitmix64 finaliser), so the assignment does not depend on
    row order and can be recomputed by any stage.
    """
    with np.errstate(over="ignore"):
        z = np.asarray(row_id).astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(1_000_000)).astype(np.float64) / 1e6 < share


def scores_path(mode: str):
    return repo_path(load_config()["paths"]["processed_dir"]) / f"scores_{mode}.parquet"


def fit_meta_path(mode: str):
    return repo_path(load_config()["paths"]["processed_dir"]) / f"cate_fit_{mode}.json"


def _subsample(n: int, rows: int | None, seed: int) -> np.ndarray | None:
    if rows is None or rows >= n:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, size=rows, replace=False))


def fit_and_score(mode: str) -> dict[str, Any]:
    """Fit every learner on train, score val + test, write ``scores_{mode}.parquet``.

    Returns (and writes to ``cate_fit_{mode}.json``) the fit metadata: rows, timings, trees,
    causal-forest ATE and importances, propensity diagnostics.
    """
    cfg = load_config()
    ccfg, seed = cfg["cate"], cfg["seed"]
    rounds = ccfg["early_stopping_rounds"]
    labels = ["treatment", "visit", "conversion", "exposure"]
    cols = ["row_id", "split", *FEATURES, *labels]
    train = load_criteo(mode, "train", cols)
    sub = _subsample(len(train), cfg["causal"]["meta_learner_train_rows"], seed)
    if sub is not None:
        train = train.iloc[sub].reset_index(drop=True)
    val = load_criteo(mode, "val", cols).reset_index(drop=True)
    test = load_criteo(mode, "test", cols).reset_index(drop=True)
    log.info("rows: train %d, val %d, test %d", len(train), len(val), len(test))
    _, (Xtr, Xva, Xte) = build_matrices(train, val, test, kind="tree")
    feat_names = list(Xtr.columns)

    t_tr = train["treatment"].to_numpy(np.int64)
    t_va = val["treatment"].to_numpy(np.int64)
    sel = selection_mask(val["row_id"].to_numpy(), seed, ccfg["val_selection_share"])
    es = ~sel
    folds = make_folds(t_tr * 4 + train["visit"].to_numpy() * 2 + train["conversion"].to_numpy(),
                       ccfg["crossfit_folds"], seed)
    meta: dict[str, Any] = {"mode": mode, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                            "val_early_stopping_rows": int(es.sum()), "val_selection_rows": int(sel.sum()),
                            "treated_share_train": float(t_tr.mean()), "features": feat_names,
                            "timings_seconds": {}, "trees": {}}

    def outcome_model() -> Any:
        return make_lgbm("classifier", seed=seed, **ccfg["outcome_model"])

    def effect_model() -> Any:
        return make_lgbm("regressor", seed=seed, **ccfg["effect_model"])

    # ---- propensity e(x): cross-fitted on train, fold-average on val/test
    t0 = time.perf_counter()
    e_tr, e_models = crossfit_propensity(
        Xtr, t_tr, folds, lambda: make_lgbm("classifier", seed=seed, **ccfg["propensity_model"]),
        (Xva[es], t_va[es]), rounds)
    e_va, e_te = predict_propensity(e_models, Xva), predict_propensity(e_models, Xte)
    lo, hi = ccfg["propensity_clip"]
    clipped = {k: float(((v < lo) | (v > hi)).mean()) for k, v in (("train", e_tr), ("val", e_va), ("test", e_te))}
    e_tr, e_va, e_te = (np.clip(v, lo, hi) for v in (e_tr, e_va, e_te))
    meta["timings_seconds"]["propensity"] = time.perf_counter() - t0
    meta["propensity"] = {"test": propensity_diagnostics(e_te, test["treatment"].to_numpy(), (lo, hi),
                                                         design_p=float(t_tr.mean())),
                          "share_clipped": clipped, "trees": [_trees(m) for m in e_models]}
    log.info("propensity: test AUC %.4f, sd %.4f", meta["propensity"]["test"]["auc_treatment_vs_control"],
             float(e_te.std()))

    out_va: dict[str, np.ndarray] = {"propensity": e_va}
    out_te: dict[str, np.ndarray] = {"propensity": e_te}
    for outcome in cfg["causal"]["outcomes"]:
        y_tr = train[outcome].to_numpy(np.int64)
        y_va = val[outcome].to_numpy(np.int64)
        val_es = ValSet(Xva[es], t_va[es], y_va[es], e_va[es])

        t0 = time.perf_counter()
        resp = _fit(outcome_model(), Xtr, y_tr, (Xva[es], y_va[es]), rounds)
        meta["timings_seconds"][f"response_{outcome}_fit"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out_va[f"response_{outcome}"], out_te[f"response_{outcome}"] = _proba(resp, Xva), _proba(resp, Xte)
        meta["timings_seconds"][f"response_{outcome}_predict"] = time.perf_counter() - t0
        meta["trees"][f"response_{outcome}"] = _trees(resp)
        del resp

        tl = TLearner(outcome_model, early_stopping_rounds=rounds)
        learners: dict[str, CATELearner] = {
            "s_learner": SLearner(outcome_model, early_stopping_rounds=rounds),
            "t_learner": tl,
            "x_learner": XLearner(outcome_model, effect_model, rounds, t_learner=tl),
            "dr_learner": DRLearner(outcome_model, effect_model, rounds, n_folds=ccfg["crossfit_folds"], seed=seed),
            "class_transformation": TransformedOutcomeLearner(outcome_model, effect_model, rounds),
        }
        for name, learner in learners.items():
            t0 = time.perf_counter()
            learner.fit(Xtr, t_tr, y_tr, e_tr, val_es)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            out_va[f"cate_{name}_{outcome}"] = learner.predict(Xva, e_va)
            out_te[f"cate_{name}_{outcome}"] = learner.predict(Xte, e_te)
            pred_s = time.perf_counter() - t0
            meta["timings_seconds"][f"{name}_{outcome}_fit"] = fit_s
            meta["timings_seconds"][f"{name}_{outcome}_predict"] = pred_s
            meta["trees"][f"{name}_{outcome}"] = learner.info()
            log.info("%s/%s: fit %.1fs, predict %.1fs, info %s", name, outcome, fit_s, pred_s, learner.info())
        # X-learner reuses the T-learner's fit: its reported fit time excludes the first stage.
        del learners, tl

    # ---- causal forest on a subsample of the training rows
    cf_idx = _subsample(len(train), cfg["causal"]["causal_forest_train_rows"], seed + 1)
    cf_rows = len(train) if cf_idx is None else len(cf_idx)
    meta["causal_forest_train_rows"] = cf_rows
    meta["causal_forest"] = {}
    for outcome in cfg["causal"]["outcomes"]:
        Xcf = Xtr if cf_idx is None else Xtr.iloc[cf_idx]
        ycf = train[outcome].to_numpy(np.int64) if cf_idx is None else train[outcome].to_numpy(np.int64)[cf_idx]
        tcf = t_tr if cf_idx is None else t_tr[cf_idx]
        t0 = time.perf_counter()
        cf = fit_causal_forest(Xcf, tcf, ycf, seed)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        out_va[f"cate_causal_forest_{outcome}"] = predict_cate(cf, Xva, ccfg["predict_chunk_rows"])
        out_te[f"cate_causal_forest_{outcome}"] = predict_cate(cf, Xte, ccfg["predict_chunk_rows"])
        pred_s = time.perf_counter() - t0
        meta["timings_seconds"][f"causal_forest_{outcome}_fit"] = fit_s
        meta["timings_seconds"][f"causal_forest_{outcome}_predict"] = pred_s
        meta["causal_forest"][outcome] = {"ate_train_rows": forest_ate(cf),
                                          "importances": forest_importances(cf, feat_names).to_dict()}
        log.info("causal_forest/%s: fit %.1fs, predict %.1fs, ATE %s", outcome, fit_s, pred_s,
                 meta["causal_forest"][outcome]["ate_train_rows"])
        del cf

    frames = []
    for split_name, df, out in (("val", val, out_va), ("test", test, out_te)):
        f = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "split": df["split"].to_numpy()})
        for c in labels:
            f[c] = df[c].to_numpy()
        for k in sorted(out):
            f[k] = np.asarray(out[k], dtype=np.float32)
        frames.append(f)
    scores_df = pd.concat(frames, ignore_index=True)
    order = ["row_id", "split", *labels, "propensity",
             *[f"response_{o}" for o in cfg["causal"]["outcomes"]],
             *[f"cate_{ln}_{o}" for ln in ccfg["learners"] for o in cfg["causal"]["outcomes"]]]
    scores_df = scores_df[order]
    pq.write_table(pa.Table.from_pandas(scores_df, preserve_index=False), scores_path(mode), compression="zstd")
    log.info("wrote %s (%d rows, %d columns)", scores_path(mode), len(scores_df), scores_df.shape[1])
    fit_meta_path(mode).write_text(json.dumps(meta, indent=2, default=float))
    return meta


# ---------------------------------------------------------------------------- evaluation helpers
def _raw_feature(col: str) -> str:
    """Tree-design column -> raw Criteo feature (``f3_logfreq`` -> ``f3``)."""
    return col.split("_")[0]


def aggregate_to_raw(importance: dict[str, float]) -> pd.Series:
    s = pd.Series(importance, dtype=np.float64)
    agg = s.groupby(s.index.map(_raw_feature)).sum().reindex(RAW_FEATURES, fill_value=0.0)
    return agg / agg.sum() if agg.sum() > 0 else agg


def surrogate_shap(tau_fit: np.ndarray, X_fit: pd.DataFrame, X_eval: pd.DataFrame, seed: int
                   ) -> tuple[pd.Series, float]:
    """Mean |SHAP| of a LightGBM surrogate of tau-hat on the raw features, and its held-out R^2.

    Explains *what the fitted effect model's output depends on* (not a causal mechanism). The
    surrogate is fit on validation rows and evaluated on test rows; tau-hat is a model output, so no
    outcome label is involved.
    """
    model = make_lgbm("regressor", seed=seed, n_estimators=300, num_leaves=63, min_child_samples=200)
    model.fit(X_fit, tau_fit[0])
    pred = model.predict(X_eval)
    resid = tau_fit[1] - pred
    r2 = float(1.0 - resid.var() / tau_fit[1].var()) if tau_fit[1].var() > 0 else float("nan")
    contrib = model.predict(X_eval, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=list(X_eval.columns))
    return (imp / imp.sum() if imp.sum() > 0 else imp), r2


def segment_labels(values: pd.Series, feature: str, n_bins: int) -> pd.Series:
    """Label-free segments of one raw feature: quantile bins (continuous) or most frequent levels."""
    if feature in CONTINUOUS:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
        cats = bins.cat.categories
        names = {c: f"q{j + 1} [{c.left:.3g}, {c.right:.3g}]" for j, c in enumerate(cats)}
        return bins.map(names).astype(str)
    freq = values.value_counts(normalize=True)
    top = freq.index[: n_bins - 1]
    lab = pd.Series("other", index=values.index)
    for j, level in enumerate(top):
        share = f"{freq[level]:.0%}" if freq[level] >= 0.01 else "<1%"
        lab[values == level] = f"#{j + 1} ({share})"
    return lab


def assignment_imbalance(y: np.ndarray, t: np.ndarray, prognostic: np.ndarray, seed: int,
                         strata: tuple[int, ...] = (10, 50)) -> tuple[pd.DataFrame, dict[str, float]]:
    """Is T independent of the users' baseline outcome propensity on held-out data?

    ``prognostic`` is the response model's P(Y | X), fit on train ignoring T, so it is a fixed function
    of X on the test split. Under X-independent randomization the treated share is the same in
    every decile (up to binomial noise). The post-stratified ATE (DiM within score strata, weighted by
    stratum size) removes any imbalance *along this score* without a propensity model.
    """
    groups = um.rank_groups(prognostic, 10, seed)
    rows = []
    for k in range(10, 0, -1):                     # decile 1 = lowest predicted propensity
        m = groups == k
        share = float(t[m].mean())
        se = float(np.sqrt(share * (1 - share) / m.sum()))
        rows.append({"decile": 11 - k, "n": int(m.sum()), "treated_share": share, "ci_low": share - 1.96 * se,
                     "ci_high": share + 1.96 * se, "outcome_rate": float(y[m].mean())})
    imb = pd.DataFrame(rows)
    post: dict[str, float] = {"treated_share_other_deciles": float(t[groups != 1].mean())}
    for g in strata:
        gg = um.rank_groups(prognostic, g, seed)
        est = 0.0
        for k in range(1, g + 1):
            m = gg == k
            est += um.diff_in_means(y[m], t[m])["estimate"] * m.mean()
        post[f"ate_post_stratified_{g}"] = float(est)
    return imb, post


def _metric_dict(metrics: pd.DataFrame, scorer: str, names: list[str]) -> dict[str, dict[str, float]]:
    m = metrics[metrics["scorer"] == scorer].set_index("metric")
    return {k: {"estimate": float(m.at[k, "estimate"]), "ci_low": float(m.at[k, "ci_low"]),
                "ci_high": float(m.at[k, "ci_high"])} for k in names if k in m.index}


# ---------------------------------------------------------------------------- analysis
def analyze(mode: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Held-out evaluation of the saved scores (see the stage docstring for what is computed)."""
    from src.visualization import uplift_plots as up

    cfg = load_config()
    ccfg, ucfg, seed = cfg["cate"], cfg["uplift"], cfg["seed"]
    outcomes, learners = cfg["causal"]["outcomes"], ccfg["learners"]
    budgets, n_boot = ucfg["budgets"], ucfg["bootstrap_reps"]
    tags = [um.budget_tag(k) for k in budgets]
    sc = pd.read_parquet(scores_path(mode))
    va = sc[sc["split"] == 1].reset_index(drop=True)
    te = sc[sc["split"] == 2].reset_index(drop=True)
    del sc
    feats_te = load_criteo(mode, "test", ["row_id", *FEATURES])
    feats_te = te[["row_id"]].merge(feats_te, on="row_id", how="left", validate="one_to_one")[FEATURES]
    feats_va = load_criteo(mode, "val", ["row_id", *FEATURES])
    feats_va = va[["row_id"]].merge(feats_va, on="row_id", how="left", validate="one_to_one")[FEATURES]
    sel = selection_mask(va["row_id"].to_numpy(), seed, ccfg["val_selection_share"])
    t_te = te["treatment"].to_numpy(np.int64)
    e_te = te["propensity"].to_numpy(np.float64)
    ipw = um.stabilized_ipw_weights(t_te, e_te)
    random_score = np.random.default_rng(seed + ccfg["random_scorer_seed_offset"]).random(len(te))

    res: dict[str, Any] = {
        "mode": mode, "estimand": "CATE tau(x) = E[Y(1) - Y(0) | X = x] of randomized assignment T",
        "rows": {k: meta[k] for k in ("train_rows", "val_rows", "val_early_stopping_rows",
                                      "val_selection_rows", "test_rows", "causal_forest_train_rows")},
        "treated_share_train": meta["treated_share_train"],
        "timings_seconds": meta["timings_seconds"], "trees": meta["trees"],
        "propensity": meta["propensity"], "bootstrap_reps": n_boot, "budgets": budgets,
        "selected_learner": {}, "validation_qini": {}, "test": {}, "response_vs_uplift": {},
        "blp": {}, "negative_uplift": {}, "causal_forest": {}, "difference_in_means_test": {},
        "assignment_imbalance": {},
        "primary_estimator": ("ipw: every arm mean inside a targeted group is a Hajek estimator with stabilised "
                              "inverse-propensity weights from the cross-fitted e(x); 'test' holds this form, "
                              "'test_difference_in_means' the unweighted form"),
        "test_difference_in_means": {}, "validation_qini_difference_in_means": {},
    }
    tabs: dict[str, list[pd.DataFrame]] = {k: [] for k in (
        "metrics", "diffs", "val", "calib", "dist", "corr", "neg", "seg", "blp", "imp", "curves", "imb")}
    fig_data: dict[str, Any] = {"curves": {}, "bands": {}, "calib": {}, "diffs": {}, "corr": {}, "dist": {},
                                "neg": {}, "seg": {}, "imp": {}, "ate": {}}

    for outcome in outcomes:
        y_te = te[outcome].to_numpy(np.int64)
        y_va = va[outcome].to_numpy(np.int64)
        dim = um.diff_in_means(y_te, t_te)
        dim_ipw = um.diff_in_means(y_te, t_te, weights=ipw)
        res["difference_in_means_test"][outcome] = {"plain": dim, "ipw_hajek": dim_ipw}
        fig_data["ate"][outcome] = dim["estimate"]
        fig_data.setdefault("ate_ipw", {})[outcome] = dim_ipw["estimate"]
        imb, post = assignment_imbalance(y_te, t_te, te[f"response_{outcome}"].to_numpy(), seed)
        tabs["imb"].append(imb.assign(outcome=outcome))
        res["assignment_imbalance"][outcome] = {
            "treated_share_top_decile": imb.iloc[-1][["treated_share", "ci_low", "ci_high"]].to_dict(),
            "ate_difference_in_means": dim["estimate"], "ate_hajek_ipw": dim_ipw["estimate"],
            "ate_hajek_ipw_ci": [dim_ipw["ci_low"], dim_ipw["ci_high"]], **post}
        fig_data.setdefault("overall_anyway", {})[outcome] = dim_ipw["control_rate"] / dim_ipw["treated_rate"]
        cols = {ln: f"cate_{ln}_{outcome}" for ln in learners}

        # ---- model selection on the validation selection half (never used for early stopping)
        t_sel = va["treatment"].to_numpy()[sel]
        val_scores = {nm: va[c].to_numpy()[sel] for nm, c in [*cols.items(), ("response", f"response_{outcome}")]}
        vqs = {}
        for kind, w in (("ipw", um.stabilized_ipw_weights(t_sel, va["propensity"].to_numpy()[sel])), ("plain", None)):
            m = um.evaluate_scorings(y_va[sel], t_sel, val_scores, budgets=[], n_boot=0, seed=seed,
                                     grid_points=ucfg["curve_points"] * 10, sample_weight=w)["metrics"]
            vqs[kind] = m[m["metric"] == "qini_coefficient"].set_index("scorer")["estimate"].to_dict()
        vq = vqs["ipw"]
        selected = max(learners, key=lambda ln: vq[ln])
        res["selected_learner"][outcome] = selected
        res["validation_qini"][outcome] = vq
        res["validation_qini_difference_in_means"][outcome] = vqs["plain"]
        tabs["val"].append(pd.DataFrame({"outcome": outcome, "scorer": list(vq), "val_qini_coefficient": list(vq.values()),
                                         "val_qini_coefficient_dim": [vqs["plain"][n] for n in vq],
                                         "selected": [n == selected for n in vq]}))
        log.info("%s: validation Qini %s -> selected %s", outcome, {k: round(v, 6) for k, v in vq.items()}, selected)

        # ---- test evaluation: every scorer, plain and inverse-propensity (Hajek) weighted
        scorers = {ln: te[c].to_numpy(np.float64) for ln, c in cols.items()}
        scorers["response"] = te[f"response_{outcome}"].to_numpy(np.float64)
        scorers["random"] = random_score
        pairs = [(ln, "response") for ln in learners] + [("response", "random")]
        evals = {}
        for kind, w in (("plain", None), ("ipw", ipw)):
            evals[kind] = um.evaluate_scorings(y_te, t_te, scorers, budgets=budgets, n_boot=n_boot, seed=seed + 1,
                                               grid_points=ucfg["curve_points"] * 10, pairs=pairs, sample_weight=w)
            mt = evals[kind]["metrics"].assign(outcome=outcome, estimator=kind)
            dt = evals[kind]["differences"].assign(outcome=outcome, estimator=kind)
            tabs["metrics"].append(mt)
            tabs["diffs"].append(dt)
        ev = evals["ipw"]                                  # primary form (see res["primary_estimator"])
        headline = ["ate", "qini_coefficient", "qini_normalized", "auuc", "auuc_random", "auuc_normalized",
                    *[f"uplift_at_{g}" for g in tags], *[f"incremental_at_{g}" for g in tags]]
        res["test"][outcome] = {nm: _metric_dict(ev["metrics"], nm, headline) for nm in scorers}
        res["test_difference_in_means"][outcome] = {nm: _metric_dict(evals["plain"]["metrics"], nm, headline)
                                                    for nm in scorers}
        step = 10                      # evaluation grid has 10x curve_points intervals; tables/bands use curve_points
        for kind in ("ipw", "plain"):
            cv = evals[kind]["curves"]
            on_plot = np.isclose(cv["fraction"] * ucfg["curve_points"], np.round(cv["fraction"] * ucfg["curve_points"]))
            tabs["curves"].append(cv[on_plot].assign(outcome=outcome, estimator=kind))
        cv = ev["curves"]
        fig_data["curves"][outcome] = cv
        fig_data["bands"][outcome] = {nm: np.quantile(ev["boot_curves"][nm][:, ::step], [0.025, 0.975], axis=0)
                                      for nm in (selected, "response")}
        fig_data["grid_band"] = ev["grid"][::step]

        # ---- central experiment: response targeting vs uplift targeting (selected learner)
        rvu = {}
        for kind in ("plain", "ipw"):
            d = evals[kind]["differences"]
            m = evals[kind]["metrics"]
            d = d[(d["scorer_a"] == selected) & (d["scorer_b"] == "response")].set_index("metric")
            mm = m.set_index(["scorer", "metric"])
            per_budget = {}
            for g in tags:
                row = {}
                for met in ("incremental", "uplift", "treated_rate", "control_rate", "anyway_share"):
                    key = f"{met}_at_{g}"
                    row[met] = {nm: {c: float(mm.at[(nm, key), c]) for c in ("estimate", "ci_low", "ci_high")}
                                for nm in (selected, "response", "random")}
                    row[met]["difference_uplift_minus_response"] = {
                        c: float(d.at[key, c]) for c in ("estimate", "ci_low", "ci_high", "p_boot")}
                per_budget[g] = row
            qd = d.loc["qini_coefficient"]
            per_budget["qini_coefficient_difference"] = {c: float(qd[c]) for c in ("estimate", "ci_low", "ci_high",
                                                                                     "p_boot")}
            rvu[kind] = per_budget
        res["response_vs_uplift"][outcome] = {"uplift_scorer": selected, "primary": "ipw", **rvu}
        fig_data["diffs"][outcome] = {"selected": selected, "plain": rvu["plain"], "ipw": rvu["ipw"],
                                      "n_test": len(te)}

        # ---- CATE analysis: distributions, agreement, calibration, negative uplift, BLP
        qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        dist_rows = []
        for ln, c in cols.items():
            v = te[c].to_numpy(np.float64)
            dist_rows.append({"outcome": outcome, "learner": ln, "mean": v.mean(), "sd": v.std(),
                              "share_negative": float((v < 0).mean()), "share_exact_zero": float((v == 0).mean()),
                              "n_distinct": len(np.unique(v)),
                              **{f"q{int(q * 100):02d}": float(x) for q, x in zip(qs, np.quantile(v, qs))}})
        dist = pd.DataFrame(dist_rows)
        tabs["dist"].append(dist)
        fig_data["dist"][outcome] = dist
        ranked = pd.DataFrame({nm: te[c].rank().to_numpy() for nm, c in
                               [*cols.items(), ("response", f"response_{outcome}")]})
        corr = ranked.corr()
        tabs["corr"].append(corr.reset_index(names="scorer").assign(outcome=outcome))
        fig_data["corr"][outcome] = corr

        blp_rows, neg_rows = [], []
        for ln, c in cols.items():
            v = te[c].to_numpy(np.float64)
            cal = um.uplift_by_group(y_te, t_te, v, ccfg["calibration_groups"], seed, weights=ipw)
            cal_dim = um.uplift_by_group(y_te, t_te, v, ccfg["calibration_groups"], seed)
            for c_ in ("observed", "ci_low", "ci_high"):
                cal[f"{c_}_dim"] = cal_dim[c_].to_numpy()
            tabs["calib"].append(cal.assign(outcome=outcome, learner=ln))
            if ln == selected:
                fig_data["calib"][outcome] = (ln, cal)
            try:
                b = um.blp_heterogeneity_test(y_te, t_te, v, te[f"response_{outcome}"].to_numpy(), e_te)
            except ValueError as exc:
                log.warning("BLP skipped for %s/%s: %s", ln, outcome, exc)
                continue
            blp_rows.append({"outcome": outcome, "learner": ln, **b})
            neg = v < 0
            row = {"outcome": outcome, "learner": ln, "share_predicted_negative": float(neg.mean()),
                   "n_predicted_negative": int(neg.sum())}
            if neg.sum() > 0 and t_te[neg].min() == 0 and t_te[neg].max() == 1:
                d_plain = um.diff_in_means(y_te[neg], t_te[neg])
                d_ipw = um.diff_in_means(y_te[neg], t_te[neg], weights=ipw[neg])
                row.update({"mean_predicted": float(v[neg].mean()), "observed": d_ipw["estimate"],
                            "ci_low": d_ipw["ci_low"], "ci_high": d_ipw["ci_high"],
                            "observed_dim": d_plain["estimate"], "ci_low_dim": d_plain["ci_low"],
                            "ci_high_dim": d_plain["ci_high"], "treated_rate": d_ipw["treated_rate"],
                            "control_rate": d_ipw["control_rate"]})
            bottom = cal.iloc[-1]
            row.update({"bottom_decile_mean_predicted": bottom["mean_predicted"],
                        "bottom_decile_observed": bottom["observed"], "bottom_decile_ci_low": bottom["ci_low"],
                        "bottom_decile_ci_high": bottom["ci_high"]})
            neg_rows.append(row)
        blp = pd.DataFrame(blp_rows)
        neg = pd.DataFrame(neg_rows)
        tabs["blp"].append(blp)
        tabs["neg"].append(neg)
        res["blp"][outcome] = blp.set_index("learner").to_dict(orient="index")
        res["negative_uplift"][outcome] = neg.set_index("learner").to_dict(orient="index")
        fig_data["neg"][outcome] = neg

        # ---- importance of features for the EFFECT
        imp_rows = []
        cf_imp = aggregate_to_raw(meta["causal_forest"][outcome]["importances"])
        imp_rows.append(pd.DataFrame({"outcome": outcome, "method": "causal_forest_split_importance",
                                      "learner": "causal_forest", "feature": cf_imp.index,
                                      "importance": cf_imp.to_numpy(), "surrogate_r2": np.nan}))
        rng = np.random.default_rng(seed + 3)
        fit_idx = rng.choice(len(va), size=min(ccfg["surrogate_rows"], len(va)), replace=False)
        eval_idx = rng.choice(len(te), size=min(ccfg["shap_rows"], len(te)), replace=False)
        for ln, c in cols.items():
            imp, r2 = surrogate_shap((va[c].to_numpy()[fit_idx], te[c].to_numpy()[eval_idx]),
                                     feats_va.iloc[fit_idx], feats_te.iloc[eval_idx], seed)
            imp_rows.append(pd.DataFrame({"outcome": outcome, "method": "surrogate_tree_shap", "learner": ln,
                                          "feature": imp.index, "importance": imp.to_numpy(), "surrogate_r2": r2}))
        imp_df = pd.concat(imp_rows, ignore_index=True)
        tabs["imp"].append(imp_df)
        fig_data["imp"][outcome] = (selected, imp_df)

        # ---- heterogeneity across segments of the top features (by the selected learner's surrogate)
        top = (imp_df[(imp_df["learner"] == selected) & (imp_df["method"] == "surrogate_tree_shap")]
               .nlargest(ccfg["segment_features"], "importance")["feature"].tolist())
        tau_sel = te[cols[selected]].to_numpy(np.float64)
        seg_rows = []
        for f in top:
            lab = segment_labels(feats_te[f], f, ccfg["segment_bins"])
            for level in lab.unique():
                m = (lab == level).to_numpy()
                d = um.diff_in_means(y_te[m], t_te[m], weights=ipw[m])
                d0 = um.diff_in_means(y_te[m], t_te[m])
                seg_rows.append({"outcome": outcome, "feature": f, "segment": level, "n": int(m.sum()),
                                 "share": float(m.mean()), "observed": d["estimate"], "ci_low": d["ci_low"],
                                 "ci_high": d["ci_high"], "observed_dim": d0["estimate"], "ci_low_dim": d0["ci_low"],
                                 "ci_high_dim": d0["ci_high"], "mean_predicted": float(tau_sel[m].mean()),
                                 "learner": selected})
        seg = pd.DataFrame(seg_rows)
        tabs["seg"].append(seg)
        fig_data["seg"][outcome] = (selected, top, seg)

        cf_meta = meta["causal_forest"][outcome]["ate_train_rows"]
        res["causal_forest"][outcome] = {
            "ate_doubly_robust_on_forest_training_rows": cf_meta,
            "difference_in_means_test": dim,
            "mean_predicted_cate_test": float(te[cols["causal_forest"]].mean()) if "causal_forest" in cols else None,
        }

    # ---- tables
    save_table(pd.concat(tabs["metrics"], ignore_index=True)[
        ["outcome", "estimator", "scorer", "metric", "estimate", "ci_low", "ci_high"]], "cate_uplift_metrics")
    save_table(pd.concat(tabs["diffs"], ignore_index=True)[
        ["outcome", "estimator", "scorer_a", "scorer_b", "metric", "estimate", "ci_low", "ci_high", "p_boot"]],
        "cate_uplift_differences")
    save_table(pd.concat(tabs["val"], ignore_index=True), "cate_validation_selection")
    save_table(pd.concat(tabs["calib"], ignore_index=True), "cate_calibration_groups")
    save_table(pd.concat(tabs["dist"], ignore_index=True), "cate_tau_distribution")
    save_table(pd.concat(tabs["corr"], ignore_index=True), "cate_rank_correlation")
    save_table(pd.concat(tabs["neg"], ignore_index=True), "cate_negative_uplift")
    save_table(pd.concat(tabs["seg"], ignore_index=True), "cate_segment_uplift")
    save_table(pd.concat(tabs["blp"], ignore_index=True), "cate_blp")
    save_table(pd.concat(tabs["imp"], ignore_index=True), "cate_effect_importance")
    save_table(pd.concat(tabs["curves"], ignore_index=True), "cate_curves")
    save_table(pd.concat(tabs["imb"], ignore_index=True), "cate_imbalance_check")
    fit_rows = [{"model": k.rsplit("_", 1)[0], "phase": k.rsplit("_", 1)[1], "seconds": v}
                for k, v in meta["timings_seconds"].items() if k.endswith(("_fit", "_predict"))]
    save_table(pd.DataFrame(fit_rows), "cate_timings")

    # ---- figures
    fig_data["imbalance"] = pd.concat(tabs["imb"], ignore_index=True)
    fig_data["design_share"] = float(t_te.mean())
    res["figures"] = up.make_all(fig_data, outcomes, budgets, len(te), mode)
    save_json(res, "cate")
    return res


def run_cate(mode: str = "dev", refit: bool = True) -> dict[str, Any]:
    """Stage ``cate``: fit every learner and score val/test (``scores_{mode}.parquet``), then run the
    held-out evaluation. ``refit=False`` reuses existing scores (for re-running the analysis only)."""
    if refit or not scores_path(mode).exists() or not fit_meta_path(mode).exists():
        meta = fit_and_score(mode)
    else:
        meta = json.loads(fit_meta_path(mode).read_text())
        log.info("reusing %s (refit=False)", scores_path(mode))
    res = analyze(mode, meta)
    head: dict[str, Any] = {"selected_learner": res["selected_learner"]}
    for o, sel in res["selected_learner"].items():
        t = res["test"][o]
        head[f"{o}_qini_{sel}"] = t[sel]["qini_coefficient"]["estimate"]
        head[f"{o}_qini_response"] = t["response"]["qini_coefficient"]["estimate"]
        d = res["response_vs_uplift"][o]["ipw"]
        head[f"{o}_incremental_diff_at_10"] = d.get("10", {}).get("incremental", {}).get(
            "difference_uplift_minus_response", {}).get("estimate")
    return head
