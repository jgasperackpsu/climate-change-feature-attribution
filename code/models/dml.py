"""Phase 5: causal estimation with double/debiased machine learning (DoubleML partially linear model).

Usage:
    python code/models/dml.py <split_dir> <models_dir> <findings_dir> <visuals_dir> [--runs a,b,...] [--n-rep 20]
    e.g. python code/models/dml.py data/split_data models findings/dml visuals
<split_dir> is the output of code/data_preparation/split_panel.py. Run after fe_baseline.py and ml_baseline.py:
the comparisons read <models_dir>/fe_baseline/ and findings/baseline/shap_vs_causal.csv.

Model (partially linear regression, "partialling out" score), for each treatment d in turn:
    Y = theta_d * d + g(X_d) + e,      d = m(X_d) + v
X_d holds the other treatments and the controls, so fitting D_corporate puts D_household (and both spatial
lags) in X, and the reverse. The four treatments of the index versions are D_corporate, D_household,
W_D_corporate and W_D_household, so the own- and total-effect gaps of fe_baseline.py can be tested.

Fixed effects: state effects, year effects and the three EPA break dummies are absorbed first (each
variable is replaced by its residual from an OLS projection on state, year and break dummies; Frisch-Waugh-
Lovell). The nuisance learners therefore see within-state, within-year variation only, and the linear-learner
DML reproduces the two-way FE coefficient. The year means use every state, so they are not cross-fitted;
with 48 states and 18 years that leakage is negligible.

Split (data/split_data, whole-year blocks): estimation uses the train years only (1997-2014), the same
sample as the FE baseline. Hyperparameters and, for the "best" learner, the learner itself are chosen by
5-fold CV on the train years with whole states held out (the grouping the cross-fitting uses), by MSE only.
Validation (2015-2018) and test (2019-2022) report the chosen nuisance models' held-out R2 (fit on train,
each block absorbed within itself). They are not used for selection: absorbing state and year effects inside
a 4-year block leaves only ~8-14% of the train within-variation, and a first version that selected on
validation picked near-empty nuisance models (e.g. R2 ~0 for next year's index, which train CV explains at
~0.7), which failed to partial out the controls.

Cross-fitting: DoubleMLData with cluster_cols="postal_code" (the replacement for the deprecated
DoubleMLClusterData), so each fold holds out whole states and no state's rows are split across folds.
5 folds, N_REP repetitions with different state-to-fold draws; every treatment of a run shares the same draws,
so the scores of different treatments can be combined for the gap tests.

Inference: the cluster-robust variance is computed from DoubleML's score (psi / psi_deriv) with the
small-sample factor G / (G - 1) (DoubleML's own SE omits it; the FE baseline includes it), and p-values and
CIs use t(G - 1). Across repetitions the estimate is the median and the SE is the median of
sqrt(se_r^2 + (theta_r - theta)^2) (Chernozhukov et al. 2018). The gap (corporate - household, own and
total) uses the same per-repetition clustered covariance of the two scores.

Learners (one per nuisance model l = E[Y|X] and m = E[d|X]):
    linear   OLS                                   (checks that DML reproduces the FE baseline)
    lasso    degree-2 polynomial (squares and interactions) -> standardize -> Lasso, alpha tuned
    rf       random forest, 500 trees (Shen et al.'s learner), min_samples_leaf and max_features tuned
    xgb      XGBoost, n_estimators, max_depth and min_child_weight tuned, the rest fixed
    best     for each nuisance separately, the tuned learner above with the lowest CV MSE
             (chosen by prediction error only, never by the estimate)

Runs (version_learner):
    main_{linear,lasso,rf,xgb,best}  Y = log gross GHG per person; main_best is the main DML result
    co2_{xgb,best}                   Y = log EIA energy-related CO2 per person (no break dummies: EIA data)
    lvstk_{xgb,best}                 cattle moved to the corporate side (D_corporate_lvstk and its lag)
    noelec_best                      household index without residential electricity (D_household_noelec and
                                     its lag), which is partly tied to Y by arithmetic (data_sources.md)
    no_gdp_best, no_wy_best, no_ippu_best   the FE robustness versions
    no_nd_best                       North Dakota dropped (largest share of within-state household-index
                                     variation and second-largest corporate; see dml_overlap.png)
    placebo_{xgb,best}               treatments are NEXT year's indices, with this year's indices, lags and
                                     controls in X; leads are taken within each split, so the last year
                                     of each block is dropped; the estimates should be about zero
    drivers_{linear,best}            one driver at a time: 8 models, each driver (standardized to train SD 1)
                                     as the treatment with the other 7 drivers, the two index lags and the
                                     controls in X

Sensitivity (every run): DoubleML's omitted-variable-bias bounds (Chernozhukov et al. 2022) at
cf_y = cf_d = 0.03 and the robustness values RV (confounding strength that would move the estimate to 0)
and RVa (that would make it insignificant at 5%). main_best also benchmarks the strength of each control
(sensitivity_benchmark): an omitted confounder as strong as that observed variable.

Overlap: per state, the spread of each index over the train years and the spread left after absorbing the
fixed effects (the variation DML actually uses), with each state's share of that variation.

Writes:
    <models_dir>/dml/<run>/   settings.json, tuning.csv, coefficients.csv, gap_tests.csv, nuisance_r2.csv,
                              repetitions.csv, sensitivity.csv (and benchmarks.csv for main_best)
    <models_dir>/dml/summary.csv          every run's coefficients
    <findings_dir>/gap_tests.csv           every run's own and total gap tests
    <findings_dir>/learner_comparison.csv  main version: FE vs each DML learner
    <findings_dir>/drivers_comparison.csv  per-driver DML vs FE drivers spec
    <findings_dir>/shap_vs_dml.csv         SHAP vs DML driver importance
    <findings_dir>/sensitivity.csv         RV / RVa and bounds, every run
    <findings_dir>/overlap.csv             per-state spread of the indices
    <visuals_dir>/dml_learners.png      main specification across FE and every DML learner
    <visuals_dir>/dml_robustness.png    every version and the placebo
    <visuals_dir>/dml_drivers.png       one driver at a time, DML vs FE
    <visuals_dir>/dml_driver_ranks.png  driver importance ranks: SHAP vs FE vs DML
    <visuals_dir>/dml_sensitivity.png   omitted-variable-bias robustness values vs observed-variable benchmarks
    <visuals_dir>/dml_overlap.png       per-state spread of each index
Pass --runs "" to rebuild the summary files and figures from the runs already on disk.
"""
import argparse
import itertools
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import doubleml as dml  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))
sys.path.insert(0, str(ROOT / "analysis"))
from build_panel import CONTROLS, CORPORATE, HOUSEHOLD, LAGS, Y  # noqa: E402
from split_panel import load  # noqa: E402
from eda import AXIS, CORP, HH, INK, INK2, LABELS, MUTED, save, style  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="doubleml")
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 2026
N_FOLDS = 5
N_REP = 20
CF = 0.03
BREAKS = ["brk_mn_2010", "brk_mi_2010", "brk_md_2011"]
Y_NOIPPU = "y_log_gross_noippu_pc"
Y_CO2 = "y_log_co2_energy_pc"
NO_GDP = [c for c in CONTROLS if c != "log_gdp_real_pc"]
DRIVERS = CORPORATE + HOUSEHOLD
Z_DRIVERS = [f"z_{d}" for d in DRIVERS]
INDEX = ["D_corporate", "D_household"] + LAGS

XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
XGB_GRID = dict(n_estimators=[100, 300, 600], max_depth=[1, 2, 3], min_child_weight=[10, 30])
RF_FIXED = dict(n_estimators=500)
RF_GRID = dict(min_samples_leaf=[5, 20, 50], max_features=[0.33, 0.5, 1.0])
N_ALPHAS = 50
LEARNERS = ["linear", "lasso", "rf", "xgb"]


def version(y=Y, treat=INDEX, x=CONTROLS, breaks=True, drop=(), lead=False, corp="D_corporate",
            hh="D_household", lags=tuple(LAGS)):
    return dict(y=y, treat=list(treat), x=list(x), breaks=breaks, drop=list(drop), lead=lead, corp=corp, hh=hh,
                lags=list(lags))


VERSIONS = {
    "main": version(),
    "co2": version(y=Y_CO2, breaks=False),
    "lvstk": version(treat=["D_corporate_lvstk", "D_household", "W_D_corporate_lvstk", "W_D_household"],
                     corp="D_corporate_lvstk", lags=["W_D_corporate_lvstk", "W_D_household"]),
    "noelec": version(treat=["D_corporate", "D_household_noelec", "W_D_corporate", "W_D_household_noelec"],
                      hh="D_household_noelec", lags=["W_D_corporate", "W_D_household_noelec"]),
    "no_gdp": version(x=NO_GDP),
    "no_wy": version(drop=["WY"]),
    "no_nd": version(drop=["ND"]),
    "no_ippu": version(y=Y_NOIPPU, breaks=False),
    "placebo": version(treat=["lead_D_corporate", "lead_D_household"], x=INDEX + CONTROLS, lead=True,
                       corp="lead_D_corporate", hh="lead_D_household", lags=()),
    "drivers": version(treat=Z_DRIVERS, x=LAGS + CONTROLS, corp=None, hh=None, lags=()),
}
RUNS = (["main_linear", "main_lasso", "main_rf", "main_xgb", "main_best", "co2_xgb", "co2_best", "lvstk_xgb",
         "lvstk_best", "noelec_best", "no_gdp_best", "no_wy_best", "no_nd_best", "no_ippu_best", "placebo_xgb", "placebo_best",
         "drivers_linear", "drivers_best"])


# ---------------------------------------------------------------- data
def prepare(part, v, ref):
    """Main sample of one split with z_ drivers (train mean / SD) and, for the placebo, next year's indices
    taken within the split. Returns the rows used, sorted by state and year."""
    d = part[(part.in_main_sample == 1) & ~part.postal_code.isin(v["drop"])].sort_values(["postal_code", "year"])
    r = ref[(ref.in_main_sample == 1) & ~ref.postal_code.isin(v["drop"])]
    d = d.copy()
    for c in DRIVERS:
        d[f"z_{c}"] = (d[c] - r[c].mean()) / r[c].std()
    if v["lead"]:
        for c in ["D_corporate", "D_household"]:
            d[f"lead_{c}"] = d.groupby("postal_code")[c].shift(-1)
        d = d.dropna(subset=["lead_D_corporate"])
    return d.reset_index(drop=True)


def absorb(d, cols, breaks):
    """Residuals of every column from an OLS projection on state, year and (optionally) break dummies."""
    a = [pd.get_dummies(d.postal_code, dtype=float), pd.get_dummies(d.year, dtype=float).iloc[:, 1:]]
    if breaks:
        a.append(d[BREAKS].astype(float))
    a = pd.concat(a, axis=1).values
    x = d[cols].values.astype(float)
    res = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
    return pd.DataFrame(res, columns=cols).assign(postal_code=d.postal_code.values, year=d.year.values)


# ---------------------------------------------------------------- learners
def make(name, params=None):
    params = params or {}
    if name == "linear":
        return LinearRegression()
    if name == "lasso":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             Lasso(alpha=params["alpha"], max_iter=100_000))
    if name == "rf":
        return RandomForestRegressor(**RF_FIXED, **params, n_jobs=1, random_state=SEED)
    if name == "xgb":
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=SEED,
                            **XGB_FIXED, **params)
    raise ValueError(name)


def grid(name, x, y):
    if name == "linear":
        return [{}]
    if name == "lasso":
        z = StandardScaler().fit_transform(PolynomialFeatures(2, include_bias=False).fit_transform(x))
        amax = np.abs(z.T @ (y - y.mean())).max() / len(y)
        return [dict(alpha=float(a)) for a in np.geomspace(amax, amax * 1e-3, N_ALPHAS)]
    g = XGB_GRID if name == "xgb" else RF_GRID
    return [dict(zip(g, vals)) for vals in itertools.product(*g.values())]


def score_candidate(name, params, x, y, folds):
    """Mean squared error of one candidate over state-grouped CV folds of the train years."""
    se = 0.0
    for fit_idx, val_idx in folds:
        m = make(name, params).fit(x[fit_idx], y[fit_idx])
        se += float(((y[val_idx] - m.predict(x[val_idx])) ** 2).sum())
    mse = se / len(y)
    return dict(learner=name, params=params, cv_mse=mse, cv_r2=1 - mse / float(np.var(y)))


def tune(tr, targets, learners):
    """For each nuisance target (key -> (target col, X cols)): every candidate scored by 5-fold CV on the
    train years with whole states held out (the same grouping as the cross-fitting). Returns a DataFrame of
    all candidates and {key: best candidate per learner}."""
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(tr, groups=tr.postal_code))
    jobs = []
    for key, (target, xcols) in targets.items():
        x, y = tr[xcols].values, tr[target].values
        for name in learners:
            for params in grid(name, x, y):
                jobs.append((key, target, name, params, x, y))
    out = Parallel(n_jobs=-1)(delayed(score_candidate)(n, p, x, y, folds) for _, _, n, p, x, y in jobs)
    t = pd.DataFrame([dict(key=k, target=tg) | o for (k, tg, *_), o in zip(jobs, out)])
    best = {k: {n: g.loc[g.cv_mse.idxmin()].to_dict() for n, g in kg.groupby("learner")} for k, kg in t.groupby("key")}
    return t, best


def holdout_r2(model, x_fit, y_fit, x_out, y_out):
    pred = model.fit(x_fit, y_fit).predict(x_out)
    return 1 - float(np.mean((y_out - pred) ** 2)) / float(np.var(y_out))


# ---------------------------------------------------------------- inference
def t_inf(est, se, df):
    t = est / se
    p = 2 * stats.t.sf(np.abs(t), df)
    c = stats.t.ppf(0.975, df)
    return dict(estimate=est, se=se, t=t, df=df, p_two_sided=p, p_one_sided=stats.t.sf(t, df),
                ci_low=est - c * se, ci_high=est + c * se)


def influence(m, clusters):
    """Per-repetition cluster sums of the influence function -psi / J, shape (G, n_rep)."""
    J = m.psi_deriv[:, :, 0].mean(axis=0)
    inf = -m.psi[:, :, 0] / J
    return pd.DataFrame(inf).groupby(clusters).sum().values


def combine(models, weights, clusters, n):
    """theta and clustered SE (with G / (G - 1)) of sum_j w_j theta_j, per repetition, then aggregated."""
    theta_r = sum(w * models[k].all_coef[0] for k, w in weights.items())
    s = sum(w * influence(models[k], clusters) for k, w in weights.items())
    g = s.shape[0]
    se_r = np.sqrt((s ** 2).sum(axis=0) / n ** 2 * g / (g - 1))
    theta = np.median(theta_r)
    se = np.median(np.sqrt(se_r ** 2 + (theta_r - theta) ** 2))
    return theta, se, theta_r, se_r


# ---------------------------------------------------------------- one run
def run(name, parts, out_dir, n_rep, fe_coefs):
    vname, learner = name.rsplit("_", 1)
    v = VERSIONS[vname]
    t0 = time.time()
    cols = [v["y"]] + v["treat"] + v["x"]
    split = {k: prepare(parts[k], v, parts["train"]) for k in ["train", "validation", "test"]}
    a = {k: absorb(d, cols, v["breaks"]) for k, d in split.items()}
    tr, va, te = a["train"], a["validation"], a["test"]
    clusters = tr.postal_code.values
    n, g = len(tr), tr.postal_code.nunique()
    df = g - 1

    xsets = {d: [c for c in v["treat"] if c != d] + v["x"] for d in v["treat"]}
    targets = {}
    for d, xc in xsets.items():
        targets[f"l:{d}"] = (v["y"], xc)
        targets[f"m:{d}"] = (d, xc)
    tuning, best = tune(tr, targets, LEARNERS if learner == "best" else [learner])
    used = {}
    for key, cands in best.items():
        pick = min(cands.values(), key=lambda c: c["cv_mse"]) if learner == "best" else cands[learner]
        used[key] = dict(learner=pick["learner"], params=pick["params"], cv_mse=pick["cv_mse"], cv_r2=pick["cv_r2"])
    tuning["used"] = [used[k]["learner"] == ln and used[k]["params"] == p for k, ln, p in
                      zip(tuning.key, tuning.learner, tuning.params)]

    # cross-fitting: one PLR per treatment, all sharing the first one's state-level fold draws
    np.random.seed(SEED)
    models, first = {}, None
    for d, xc in xsets.items():
        data = dml.DoubleMLData(tr[[v["y"], d] + xc + ["postal_code"]], y_col=v["y"], d_cols=d, x_cols=xc,
                                cluster_cols="postal_code")
        ml_l = make(used[f"l:{d}"]["learner"], used[f"l:{d}"]["params"])
        ml_m = make(used[f"m:{d}"]["learner"], used[f"m:{d}"]["params"])
        if first is None:
            m = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=N_FOLDS, n_rep=n_rep)
            first = m
        else:
            m = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=N_FOLDS, n_rep=n_rep, draw_sample_splitting=False)
            m.set_sample_splitting(first.smpls, first.smpls_cluster)
        m.fit(n_jobs_cv=N_FOLDS, store_predictions=True)
        models[d] = m

    coefs, reps, nuis, sens = [], [], [], []
    for d, m in models.items():
        theta, se, theta_r, se_r = combine(models, {d: 1}, clusters, n)
        coefs.append(dict(term=d) | t_inf(theta, se, df) | dict(rep_min=theta_r.min(), rep_max=theta_r.max(),
                                                                se_doubleml=float(m.se[0])))
        reps += [dict(rep=r + 1, term=d, estimate=theta_r[r], se=se_r[r]) for r in range(n_rep)]
        for nu, target in [("l", v["y"]), ("m", d)]:
            yv = tr[target].values
            r2 = [1 - np.mean((yv - m.predictions[f"ml_{nu}"][:, r, 0]) ** 2) / np.var(yv) for r in range(n_rep)]
            # held-out checks of the chosen nuisance model: fit on all train rows, scored on validation and test
            # (each absorbed within its own block, which keeps only ~10% of the train within-variation)
            u = used[f"{nu}:{d}"]
            xt = tr[xsets[d]].values
            nuis.append(dict(treatment=d, nuisance=nu, target=target, learner=u["learner"],
                             params=json.dumps(u["params"]), cv_r2=u["cv_r2"], crossfit_r2_median=np.median(r2),
                             crossfit_r2_min=np.min(r2), crossfit_r2_max=np.max(r2),
                             val_r2=holdout_r2(make(u["learner"], u["params"]), xt, yv, va[xsets[d]].values,
                                               va[target].values),
                             test_r2=holdout_r2(make(u["learner"], u["params"]), xt, yv, te[xsets[d]].values,
                                                te[target].values)))
        m.sensitivity_analysis(cf_y=CF, cf_d=CF)
        sp = m.sensitivity_params
        sens.append(dict(run=name, term=d, estimate=theta, cf_y=CF, cf_d=CF,
                         theta_lower=float(sp["theta"]["lower"][0]), theta_upper=float(sp["theta"]["upper"][0]),
                         ci_lower=float(sp["ci"]["lower"][0]), ci_upper=float(sp["ci"]["upper"][0]),
                         rv=float(sp["rv"][0]), rva=float(sp["rva"][0])))
    coefs = pd.DataFrame(coefs)

    gaps = []
    if v["corp"]:
        combos = {"own": {v["corp"]: 1, v["hh"]: -1}}
        if v["lags"]:
            wc, wh = v["lags"]
            combos |= {"total": {v["corp"]: 1, wc: 1, v["hh"]: -1, wh: -1},
                       "total_corporate": {v["corp"]: 1, wc: 1}, "total_household": {v["hh"]: 1, wh: 1}}
        for cname, w in combos.items():
            theta, se, *_ = combine(models, w, clusters, n)
            gaps.append(dict(run=name, comparison=cname) | t_inf(theta, se, df))

    bench = []
    if name == "main_best":
        for d in ["D_corporate", "D_household"]:
            for b in v["x"] + [c for c in ["D_corporate", "D_household"] if c != d]:
                bm = models[d].sensitivity_benchmark(benchmarking_set=[b])
                bench.append(dict(term=d, benchmark=b, cf_y=float(bm["cf_y"].iloc[0]), cf_d=float(bm["cf_d"].iloc[0]),
                                  rho=float(bm["rho"].iloc[0]), delta_theta=float(bm["delta_theta"].iloc[0]),
                                  rv=sens[[s["term"] for s in sens].index(d)]["rv"]))

    o = out_dir / name
    o.mkdir(parents=True, exist_ok=True)
    tuning.assign(params=tuning.params.map(json.dumps)).to_csv(o / "tuning.csv", index=False)
    coefs.to_csv(o / "coefficients.csv", index=False)
    if gaps:
        pd.DataFrame(gaps).to_csv(o / "gap_tests.csv", index=False)
    pd.DataFrame(nuis).to_csv(o / "nuisance_r2.csv", index=False)
    pd.DataFrame(reps).to_csv(o / "repetitions.csv", index=False)
    pd.DataFrame(sens).to_csv(o / "sensitivity.csv", index=False)
    if bench:
        pd.DataFrame(bench).to_csv(o / "benchmarks.csv", index=False)
    (o / "settings.json").write_text(json.dumps(dict(
        run=name, version=vname, learner=learner, outcome=v["y"], treatments=v["treat"], x=v["x"],
        absorbed=["state FE", "year FE"] + (BREAKS if v["breaks"] else []), dropped_states=v["drop"],
        placebo_leads=v["lead"], sample="in_main_sample == 1, train years", n_obs=n, n_clusters=g,
        years=[int(tr.year.min()), int(tr.year.max())], inference_df=df, n_folds=N_FOLDS, n_rep=n_rep,
        seed=SEED, cluster="postal_code (DoubleMLData cluster_cols)", selection="5-fold state-grouped CV MSE on the train years",
        validation_years=[int(va.year.min()), int(va.year.max())], test_years=[int(te.year.min()), int(te.year.max())],
        nuisance_used=used, xgb_fixed=XGB_FIXED, xgb_grid=XGB_GRID, rf_fixed=RF_FIXED, rf_grid=RF_GRID,
        lasso=dict(features="degree-2 polynomial (no bias) -> standardize", n_alphas=N_ALPHAS, max_iter=100_000),
        sensitivity=dict(cf_y=CF, cf_d=CF), seconds=round(time.time() - t0, 1),
        versions=dict(python=platform.python_version(), doubleml=dml.__version__, numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                      xgboost=xgboost.__version__)), indent=2, default=str))
    show = coefs.set_index("term")[["estimate", "se", "p_two_sided"]]
    print(f"{name:<15} n={n} G={g} ({time.time() - t0:5.0f}s)")
    print("   " + show.to_string(float_format="%.4f").replace("\n", "\n   "))
    for gt in gaps:
        if gt["comparison"] in ("own", "total"):
            print(f"   {gt['comparison']} gap {gt['estimate']:+.4f} [{gt['ci_low']:+.4f}, {gt['ci_high']:+.4f}] "
                  f"p={gt['p_two_sided']:.3f} (one-sided {gt['p_one_sided']:.3f})")
    return coefs.assign(run=name, version=vname, learner=learner), pd.DataFrame(gaps), pd.DataFrame(sens)


# ---------------------------------------------------------------- comparisons and figures
def overlap(parts):
    v = VERSIONS["main"]
    d = prepare(parts["train"], v, parts["train"])
    a = absorb(d, ["D_corporate", "D_household"], True)
    rows = []
    for st, g in d.groupby("postal_code"):
        ag = a[a.postal_code == st]
        r = dict(postal_code=st)
        for c in ["D_corporate", "D_household"]:
            r |= {f"{c}_mean": g[c].mean(), f"{c}_min": g[c].min(), f"{c}_max": g[c].max(), f"{c}_sd": g[c].std(),
                  f"{c}_resid_sd": ag[c].std(), f"{c}_resid_var_share": (ag[c] ** 2).sum() / (a[c] ** 2).sum()}
        rows.append(r)
    return pd.DataFrame(rows)


def plot_overlap(ov, out):
    ov = ov.sort_values("D_corporate_mean")
    y = np.arange(len(ov))
    fig, axes = plt.subplots(1, 4, figsize=(11, 9), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1.3, 1, 1]))
    for ax, c, col in [(axes[0], "D_corporate", CORP), (axes[1], "D_household", HH)]:
        ax.hlines(y, ov[f"{c}_min"], ov[f"{c}_max"], color=col, lw=2, alpha=0.45)
        ax.scatter(ov[f"{c}_mean"], y, color=col, s=22, zorder=3, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{LABELS.get(c, c)}: range, 1997–2014")
        ax.set_xlabel("index (SD units), dot = state mean")
    for ax, c, col in [(axes[2], "D_corporate", CORP), (axes[3], "D_household", HH)]:
        ax.barh(y, ov[f"{c}_resid_var_share"] * 100, color=col, height=0.6)
        ax.axvline(100 / len(ov), color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{LABELS.get(c, c)}: share of within var.")
        ax.set_xlabel("% of residual variance")
        top = ov.nlargest(1, f"{c}_resid_var_share").iloc[0]
        ax.text(top[f"{c}_resid_var_share"] * 100, y[ov.postal_code.tolist().index(top.postal_code)],
                f" {top[f'{c}_resid_var_share'] * 100:.0f}%", va="center", fontsize=7.5, color=INK2)
    axes[0].set_yticks(y, ov.postal_code, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Overlap: every state contributes variation in both indices, but unevenly",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.01, "Left: raw index over the train years. Right: each state's share of the variation left after "
             "absorbing state, year and break effects, which is what the DML nuisance models and θ use; dashed line = equal "
             "share (1/48).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, out, "dml_overlap.png")


def plot_learners(lc, out):
    cols = [("D_corporate", "Corporate own effect", CORP), ("D_household", "Household own effect", HH),
            ("own", "Own gap (corp − hh)", INK2), ("total", "Total gap (with spillovers)", INK2)]
    est = lc.estimator.unique().tolist()
    y = np.arange(len(est))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, (term, title, col) in zip(axes, cols):
        s = lc[lc.term == term].set_index("estimator").loc[est]
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.hlines(y, s.ci_low, s.ci_high, color=col, lw=2, alpha=0.5)
        ax.scatter(s.estimate, y, color=col, s=36, zorder=3, edgecolor="white", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("log points of Y per SD")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, est)
    fig.suptitle("Main specification across estimators (95% CI, t(47), clustered by state)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "FE: two-way fixed effects (fe_baseline.py). DML rows: partially linear model on the same "
             "train years, nuisance learner as named; best = lowest state-grouped CV MSE per nuisance model.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, out, "dml_learners.png")


def plot_drivers(dc, out):
    dc = dc.sort_values("dml_estimate")
    y = np.arange(len(dc))
    colors = [CORP if t in CORPORATE else HH for t in dc.driver]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvline(0, color=AXIS, lw=0.9)
    off = 0.17
    ax.hlines(y + off, dc.dml_ci_low, dc.dml_ci_high, color=colors, lw=2, alpha=0.5)
    ax.scatter(dc.dml_estimate, y + off, c=colors, s=40, zorder=3, edgecolor="white", linewidth=1, label="DML (best)")
    ax.hlines(y - off, dc.fe_ci_low, dc.fe_ci_high, color=AXIS, lw=2)
    ax.scatter(dc.fe_estimate, y - off, facecolor="white", edgecolor=INK2, s=34, zorder=3, linewidth=1.3,
               label="Two-way FE")
    ax.set_yticks(y, [LABELS.get(t, t) for t in dc.driver])
    ax.set_xlabel("Δ log GHG per person, per SD of the driver (95% CI)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=CORP, label="Corporate driver, DML"),
               plt.Line2D([], [], marker="o", ls="", color=HH, label="Household driver, DML"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="Two-way FE")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("One driver at a time: DML vs. fixed effects, train years 1997–2014")
    fig.tight_layout()
    save(fig, out, "dml_drivers.png")


ROBUST_ROWS = [("main_best", "Main (best learner)"), ("main_xgb", "Main, XGBoost"),
               ("co2_best", "EIA energy CO₂ outcome"), ("lvstk_best", "Cattle on corporate side"),
               ("noelec_best", "Household index w/o res. electricity"), ("no_gdp_best", "No GDP control"), ("no_wy_best", "Without Wyoming"),
               ("no_nd_best", "Without North Dakota"), ("no_ippu_best", "Excl. industrial processes"),
               ("placebo_best", "Placebo: next year's index")]


def plot_robustness(coefs, gaps, out):
    rows = [(r, lab) for r, lab in ROBUST_ROWS if (coefs.run == r).any()]
    y = np.arange(len(rows))[::-1]
    panels = [("corp", "Corporate own effect", CORP), ("hh", "Household own effect", HH),
              ("own", "Own gap (corp − hh)", INK2), ("total", "Total gap (with spillovers)", INK2)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.6), sharey=True)
    for ax, (key, title, col) in zip(axes, panels):
        ax.axvline(0, color=AXIS, lw=0.9)
        for yi, (r, _) in zip(y, rows):
            v = VERSIONS[r.rsplit("_", 1)[0]]
            if key in ("corp", "hh"):
                s = coefs[(coefs.run == r) & (coefs.term == v[key])]
            else:
                s = gaps[(gaps.run == r) & (gaps.comparison == key)]
            if s.empty:
                continue
            s = s.iloc[0]
            placebo = r.startswith("placebo")
            ax.hlines(yi, s.ci_low, s.ci_high, color=MUTED if placebo else col, lw=2, alpha=0.5)
            ax.scatter(s.estimate, yi, s=34, zorder=3, linewidth=1.2, color=MUTED if placebo else col,
                       edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("log points of Y per SD")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lab for _, lab in rows])
    for ax in axes:
        ax.axhline(0.5, color=AXIS, lw=0.6, ls=":")
    fig.suptitle("DML across outcome, index, control and sample versions, plus the placebo",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.03, "95% CIs, t(G − 1), clustered by state, train years 1997–2014. Placebo (gray): the "
             "treatments are next year's indices with this year's in X, so its effects should be zero; "
             "it has no spillover terms, so no total gap.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    save(fig, out, "dml_robustness.png")


def plot_sensitivity(sens, bench, out):
    """Omitted-variable bias: a confounder inside the RV square (partial R2 with Y and with D both below RV)
    cannot move theta to zero; inside the RVa square it cannot make it insignificant. The dots place a
    confounder as strong as each observed variable (DoubleML sensitivity_benchmark)."""
    s = sens[sens.run == "main_best"].set_index("term")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, d, col in [(axes[0], "D_corporate", CORP), (axes[1], "D_household", HH)]:
        rv, rva = s.loc[d, "rv"], s.loc[d, "rva"]
        b = bench[bench.term == d]
        lim = max(rv, b[["cf_y", "cf_d"]].values.max()) * 1.25
        ax.add_patch(plt.Rectangle((0, 0), rv, rv, color=col, alpha=0.12, lw=0))
        ax.add_patch(plt.Rectangle((0, 0), rva, rva, color=col, alpha=0.28, lw=0))
        ax.text(rv, rv, f" RV = {rv:.2f}", fontsize=8, color=INK2, va="bottom")
        if rva > 0:
            ax.text(rva, rva * 0.5, f" RVa = {rva:.2f}", fontsize=8, color=INK2, va="center")
        ax.scatter(b.cf_d, b.cf_y, s=30, color=INK2, zorder=3, edgecolor="white", linewidth=1)
        # label benchmarks away from the origin directly; list the crowded ones in a note
        far = b[b[["cf_y", "cf_d"]].max(axis=1) >= 0.25 * rv]
        near = b[b[["cf_y", "cf_d"]].max(axis=1) < 0.25 * rv]
        for r in far.itertuples():
            ax.annotate(LABELS.get(r.benchmark, r.benchmark), (r.cf_d, r.cf_y), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        if len(near):
            ax.text(0.03, 0.62, "Near the origin (both < " + f"{0.25 * rv:.2f}):\n" +
                    "\n".join(LABELS.get(x, x) for x in near.benchmark), transform=ax.transAxes, ha="left",
                    va="top", fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("confounder's partial R² with the treatment (cf_d)")
        ax.set_ylabel("partial R² with Y (cf_y)")
        sig = "" if rva > 0 else " (already insignificant: RVa = 0)"
        ax.set_title(f"{LABELS.get(d, d)}: θ = {s.loc[d, 'estimate']:.3f}{sig}", fontsize=9.5)
    fig.suptitle("Sensitivity: how strong would an omitted confounder have to be?", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "main_best. Inside the light square a confounder cannot move θ to zero; inside the dark "
             "square it cannot make θ insignificant at 5%. Dots: a confounder as strong as that observed variable.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    save(fig, out, "dml_sensitivity.png")


def plot_ranks(sv, out):
    """Slope chart of driver importance ranks: SHAP (prediction), FE and DML (within-state)."""
    cols = [("shap_rank", "SHAP\n(XGBoost prediction)"), ("fe_rank", "Two-way FE\n(within-state)"),
            ("dml_rank", "DML\n(within-state)")]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in sv.itertuples():
        col = CORP if r.group == "corporate" else HH
        ranks = [getattr(r, c) for c, _ in cols]
        ax.plot(x, ranks, color=col, lw=2, alpha=0.75)
        ax.scatter(x, ranks, color=col, s=36, zorder=3, edgecolor="white", linewidth=1)
        name = LABELS.get(r.driver, r.driver)
        ax.text(x[0] - 0.08, ranks[0], f"{name}  {ranks[0]:.0f}", ha="right", va="center", fontsize=8, color=INK2)
        ax.text(x[-1] + 0.08, ranks[-1], f"{ranks[-1]:.0f}  {name}", ha="left", va="center", fontsize=8,
                color=INK2)
    ax.set_xticks(x, [lab for _, lab in cols])
    ax.set_ylim(8.6, 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.1, len(cols) - 0.0)
    ax.grid(False)
    handles = [plt.Line2D([], [], color=CORP, lw=2, label="Corporate driver"),
               plt.Line2D([], [], color=HH, lw=2, label="Household driver")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8)
    rho = sv.attrs.get("rho")
    sub = f" (Spearman ρ, SHAP vs DML = {rho.statistic:+.2f})" if rho is not None else ""
    ax.set_title(f"Driver rankings: prediction vs. within-state{sub}", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "dml_driver_ranks.png")


def comparisons(parts, models_dir, findings, visuals, coefs, gaps):
    fe = pd.read_csv(Path(models_dir) / "fe_baseline" / "coefficients.csv")
    fe_gaps = pd.read_csv(findings.parent / "baseline" / "gap_tests.csv")
    rows = []
    m = fe[fe.spec == "main"].set_index("term")
    for term in ["D_corporate", "D_household", "W_D_corporate", "W_D_household"]:
        r = m.loc[term]
        rows.append(dict(estimator="FE", term=term, estimate=r.coef, se=r.se, p_two_sided=r.p, ci_low=r.ci_low,
                         ci_high=r.ci_high))
    for cmp in ["own", "total"]:
        r = fe_gaps[(fe_gaps.spec == "main") & (fe_gaps.comparison == cmp)].iloc[0]
        rows.append(dict(estimator="FE", term=cmp, estimate=r.gap, se=r.gap_se, p_two_sided=r.p_two_sided,
                         p_one_sided=r.p_one_sided, ci_low=r.ci_low, ci_high=r.ci_high))
    for ln in ["linear", "lasso", "rf", "xgb", "best"]:
        run_ = f"main_{ln}"
        c = coefs[coefs.run == run_]
        if c.empty:
            continue
        label = f"DML {ln}" + (" (main)" if ln == "best" else "")
        rows += [dict(estimator=label, term=r.term, estimate=r.estimate, se=r.se, p_two_sided=r.p_two_sided,
                      ci_low=r.ci_low, ci_high=r.ci_high) for r in c.itertuples()]
        gg = gaps[(gaps.run == run_) & gaps.comparison.isin(["own", "total"])]
        rows += [dict(estimator=label, term=r.comparison, estimate=r.estimate, se=r.se, p_two_sided=r.p_two_sided,
                      p_one_sided=r.p_one_sided, ci_low=r.ci_low, ci_high=r.ci_high) for r in gg.itertuples()]
    lc = pd.DataFrame(rows)
    lc.to_csv(findings / "learner_comparison.csv", index=False)
    if lc.estimator.nunique() > 1:
        plot_learners(lc, visuals)

    dr = coefs[coefs.run == "drivers_best"]
    if not dr.empty:
        fd = fe[fe.spec == "drivers"].set_index("term")
        lin = coefs[coefs.run == "drivers_linear"].set_index("term")
        # importance in log points of Y, as in ml_baseline.compare: |theta| * mean |z - mean z| over all rows,
        # z standardized with the train main-sample mean and SD
        allp = pd.concat(parts.values())
        allp = allp[allp.in_main_sample == 1]
        tr = parts["train"][parts["train"].in_main_sample == 1]
        rows = []
        for r in dr.itertuples():
            j = r.term[2:]
            z = (allp[j] - tr[j].mean()) / tr[j].std()
            spread = (z - z.mean()).abs().mean()
            f = fd.loc[r.term]
            rows.append(dict(driver=j, group="corporate" if j in CORPORATE else "household",
                             dml_estimate=r.estimate, dml_se=r.se, dml_p=r.p_two_sided, dml_ci_low=r.ci_low,
                             dml_ci_high=r.ci_high, dml_linear_estimate=lin.loc[r.term].estimate if r.term in lin.index else np.nan,
                             fe_estimate=f.coef, fe_se=f.se, fe_p=f.p, fe_ci_low=f.ci_low, fe_ci_high=f.ci_high,
                             dml_importance=abs(r.estimate) * spread, fe_importance=abs(f.coef) * spread))
        dc = pd.DataFrame(rows)
        dc["dml_rank"] = dc.dml_importance.rank(ascending=False)
        dc["fe_rank"] = dc.fe_importance.rank(ascending=False)
        dc.to_csv(findings / "drivers_comparison.csv", index=False)
        plot_drivers(dc, visuals)
        shap = pd.read_csv(findings.parent / "baseline" / "shap_vs_causal.csv")
        shap = shap[(shap.level == "driver") & shap.term.isin(DRIVERS)][["term", "shap_importance", "shap_sign"]]
        sv = dc.merge(shap, left_on="driver", right_on="term").drop(columns="term")
        sv["shap_rank"] = sv.shap_importance.rank(ascending=False)
        sv["sign_agree"] = sv.shap_sign == np.sign(sv.dml_estimate)
        rho = stats.spearmanr(sv.shap_importance, sv.dml_importance)
        rho_fe = stats.spearmanr(sv.dml_importance, sv.fe_importance)
        sv = sv[["driver", "group", "shap_importance", "shap_rank", "shap_sign", "dml_estimate", "dml_p",
                 "dml_importance", "dml_rank", "fe_importance", "fe_rank", "sign_agree"]]
        sv.attrs["rho"] = rho
        sv.to_csv(findings / "shap_vs_dml.csv", index=False)
        plot_ranks(sv, visuals)
        print(sv.to_string(index=False, float_format="%.4f"))
        print(f"Spearman SHAP vs DML importance (8 drivers): {rho.statistic:+.3f} (p = {rho.pvalue:.3f}); "
              f"DML vs FE: {rho_fe.statistic:+.3f} (p = {rho_fe.pvalue:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split_dir")
    ap.add_argument("models_dir")
    ap.add_argument("findings_dir")
    ap.add_argument("visuals_dir")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--n-rep", type=int, default=N_REP)
    a = ap.parse_args()
    parts = load(a.split_dir)
    out = Path(a.models_dir) / "dml"
    findings, visuals = Path(a.findings_dir), Path(a.visuals_dir)
    findings.mkdir(parents=True, exist_ok=True)
    visuals.mkdir(parents=True, exist_ok=True)
    style()
    fe = pd.read_csv(Path(a.models_dir) / "fe_baseline" / "coefficients.csv")
    for name in filter(None, a.runs.split(",")):
        run(name, parts, out, a.n_rep, fe)

    # combine every run on disk (so a partial re-run with --runs keeps the others)
    names = [r for r in RUNS if (out / r / "coefficients.csv").exists()]
    coefs = pd.concat([pd.read_csv(out / r / "coefficients.csv").assign(run=r, version=r.rsplit("_", 1)[0],
                                                                       learner=r.rsplit("_", 1)[1]) for r in names])
    gaps = pd.concat([pd.read_csv(out / r / "gap_tests.csv") for r in names if (out / r / "gap_tests.csv").exists()])
    sens = pd.concat([pd.read_csv(out / r / "sensitivity.csv") for r in names])
    coefs.to_csv(out / "summary.csv", index=False)
    gaps.to_csv(findings / "gap_tests.csv", index=False)
    sens.to_csv(findings / "sensitivity.csv", index=False)
    ov = overlap(parts)
    ov.to_csv(findings / "overlap.csv", index=False)
    plot_overlap(ov, visuals)
    comparisons(parts, a.models_dir, findings, visuals, coefs, gaps)
    plot_robustness(coefs, gaps, visuals)
    if (out / "main_best" / "benchmarks.csv").exists():
        plot_sensitivity(sens, pd.read_csv(out / "main_best" / "benchmarks.csv"), visuals)
    print(f"wrote {out}/, {findings}/, {visuals}/dml_*.png")


if __name__ == "__main__":
    main()
