"""Phase 4: predictive ML baseline (XGBoost, 70/15/15 train / validation / test split) and SHAP attribution compared with the
fixed-effects estimates (Guo et al.-style prediction-based attribution vs. a within-state causal design).

Usage:
    python code/models/ml_baseline.py <split_dir> <models_dir> <findings_dir> <visuals_dir>
    e.g. python code/models/ml_baseline.py data/split_data models findings/baseline visuals
<split_dir> is the output of code/data_preparation/split_panel.py.
Run after fe_baseline.py: the comparison reads <models_dir>/fe_baseline/coefficients.csv.

Sample: contiguous 48 states (in_main_sample == 1), 1997-2022, Y = y_log_gross_pc.
Features ("all features"): the 8 index drivers, cattle per person, the controls, the two index spatial
lags and the three EPA break dummies. No state or year identifiers, so the model predicts from
observables only, as a prediction-based attribution would; the indices themselves are left out
because they are linear combinations of the drivers and would split SHAP credit with them.

Split (data/split_data, whole-year blocks): train 1997-2014, validation 2015-2018, test 2019-2022.
Tuning: random search (N_ITER draws), each draw fit on train and scored by RMSE on validation; the draw
with the lowest validation RMSE is kept. Every model is fit on train only, so validation and test are
both out of sample, and the test years are used for nothing but the final scores.

Benchmarks on the same split, all reported with R2, RMSE and MAE on train, validation and test:
    xgboost          tuned XGBoost on the features above (the ML baseline; SHAP is computed on this)
    xgboost_state    the same with state one-hot columns added (can it learn state levels?)
    pooled_linear    OLS on the same features
    persistence      each state's last training-year (2014) Y carried forward

SHAP: TreeExplainer on the xgboost model, for every state-year (train and test) and for test only.
Ranking comparison with the causal side, in the same units (log points of Y):
    SHAP importance     mean |phi_j|
    causal importance   |beta_j| * mean |z_j - mean z_j|, with beta_j the within-state effect per pooled SD
                        from fe_baseline's drivers spec (fit on train, SDs from train): the average absolute
                        contribution driver j would make if its within-state effect applied to its pooled spread
plus Spearman rank correlation and sign agreement (SHAP sign = sign of corr(phi_j, x_j)). The same is
done for the two indices, with SHAP summed over each index's drivers and the main-spec index coefficients.

Writes:
    <models_dir>/ml_baseline/xgboost.json             fitted model
    <models_dir>/ml_baseline/settings.json            split, tuned hyperparameters, features
    <models_dir>/ml_baseline/tuning.csv               every random-search draw with its validation RMSE
    <models_dir>/ml_baseline/metrics.csv              train / validation / test R2, RMSE, MAE for each benchmark
    <models_dir>/ml_baseline/shap_values.csv          per state-year SHAP values
    <findings_dir>/shap_vs_causal.csv                 per-driver and per-index comparison
    <visuals_dir>/shap_vs_causal.png, shap_beeswarm.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import ParameterSampler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))
sys.path.insert(0, str(ROOT / "analysis"))
from build_panel import CONTROLS, CORPORATE, HOUSEHOLD, LAGS, Y  # noqa: E402
from split_panel import load  # noqa: E402
from eda import AXIS, CORP, HH, INK, INK2, LABELS, MUTED, SEQ, save, style  # noqa: E402

BREAKS = ["brk_mn_2010", "brk_mi_2010", "brk_md_2011"]
DRIVERS = CORPORATE + HOUSEHOLD
FEATURES = DRIVERS + ["d_cattle_pc"] + CONTROLS + LAGS + BREAKS
GROUP = {**{d: "corporate" for d in CORPORATE}, **{d: "household" for d in HOUSEHOLD},
         "d_cattle_pc": "livestock", **{c: "control" for c in CONTROLS},
         **{lag: "spatial lag" for lag in LAGS}, **{b: "break dummy" for b in BREAKS}}
GROUP_COLOR = {"corporate": CORP, "household": HH, "livestock": "#1baf7a", "spatial lag": "#e87ba4",
               "control": MUTED, "break dummy": AXIS}
LABELS = LABELS | {"brk_mn_2010": "Break: MN 2010+", "brk_mi_2010": "Break: MI 2010+",
                   "brk_md_2011": "Break: MD 2011-12"}

N_ITER = 60
SEED = 340
GRID = {
    "n_estimators": [200, 400, 800, 1200],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1, 5, 10],
}


def scores(y, pred):
    return dict(r2=r2_score(y, pred), rmse=np.sqrt(mean_squared_error(y, pred)), mae=mean_absolute_error(y, pred))


def xgb(**params):
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=1, **params)


def tune(train, val, cols):
    """Random search: fit each draw on train, score RMSE on validation; returns (best params, all draws)."""
    def score(i, params):
        m = xgb(**params).fit(train[cols], train[Y])
        return params | dict(draw=i, val_rmse=np.sqrt(mean_squared_error(val[Y], m.predict(val[cols]))))

    # one thread per model, draws in parallel: far faster than multithreaded XGBoost on ~900 rows
    draws = Parallel(n_jobs=-1)(delayed(score)(i, params) for i, params in
                                enumerate(ParameterSampler(GRID, N_ITER, random_state=SEED)))
    draws = pd.DataFrame(draws).sort_values("val_rmse")
    best = {k: draws.iloc[0][k] for k in GRID}
    best = {k: int(v) if k in ("n_estimators", "max_depth", "min_child_weight") else float(v) for k, v in best.items()}
    return best, draws


def main(split_dir, models_dir, findings_dir, visuals_dir):
    parts = load(split_dir)
    p = pd.concat([parts[k].assign(split=k) for k in parts])
    d = p[p.in_main_sample == 1].reset_index(drop=True)
    assert d[FEATURES + [Y]].notna().all().all(), d[FEATURES + [Y]].isna().sum()
    states = pd.get_dummies(d.postal_code, prefix="st", dtype=float)
    d = pd.concat([d, states], axis=1)
    state_cols = FEATURES + list(states.columns)
    train, val, test = (d[d.split == k] for k in ["train", "validation", "test"])
    years = {k: f"{g.year.min()}-{g.year.max()}" for k, g in [("train", train), ("val", val), ("test", test)]}

    out = Path(models_dir) / "ml_baseline"
    out.mkdir(parents=True, exist_ok=True)
    print(f"tuning XGBoost: {N_ITER} draws, fit on train {years['train']}, scored on validation {years['val']} ...")
    best, draws = tune(train, val, FEATURES)
    best_state, _ = tune(train, val, state_cols)
    draws.to_csv(out / "tuning.csv", index=False)

    makers = {
        "xgboost": (FEATURES, lambda: xgb(**best)),
        "xgboost_state": (state_cols, lambda: xgb(**best_state)),
        "pooled_linear": (FEATURES, LinearRegression),
    }
    rows, models = [], {}
    for name, (cols, make) in makers.items():
        m = make().fit(train[cols], train[Y])
        models[name] = m
        rows.append(dict(model=name, n_features=len(cols))
                    | {f"train_{k}": v for k, v in scores(train[Y], m.predict(train[cols])).items()}
                    | {f"val_{k}": v for k, v in scores(val[Y], m.predict(val[cols])).items()}
                    | {f"test_{k}": v for k, v in scores(test[Y], m.predict(test[cols])).items()})
    last = train[train.year == train.year.max()].set_index("postal_code")[Y]
    rows.append(dict(model="persistence", n_features=0)
                | {f"val_{k}": v for k, v in scores(val[Y], val.postal_code.map(last)).items()}
                | {f"test_{k}": v for k, v in scores(test[Y], test.postal_code.map(last)).items()})
    metrics = pd.DataFrame(rows)
    for i, k in enumerate(["train", "val", "test"]):
        metrics.insert(1 + i, f"{k}_years", years[k])
    metrics.to_csv(out / "metrics.csv", index=False)
    models["xgboost"].save_model(out / "xgboost.json")
    (out / "settings.json").write_text(json.dumps(dict(
        outcome=Y, sample="in_main_sample == 1", split_dir=str(split_dir), split="70/15/15 whole-year blocks",
        **{f"{k}_years": [int(g.year.min()), int(g.year.max())] for k, g in [("train", train), ("val", val),
                                                                             ("test", test)]},
        n_rows={"train": len(train), "val": len(val), "test": len(test)},
        tuning="random search, fit on train, scored by validation RMSE", n_iter=N_ITER, seed=SEED, features=FEATURES, xgboost=best, xgboost_state=best_state), indent=2))
    print(metrics.to_string(index=False, float_format="%.4f"))

    # ---- SHAP on the predictive model
    explainer = shap.TreeExplainer(models["xgboost"])
    phi = pd.DataFrame(explainer.shap_values(d[FEATURES]), columns=FEATURES)
    phi.insert(0, "year", d.year.values)
    phi.insert(0, "postal_code", d.postal_code.values)
    phi.to_csv(out / "shap_values.csv", index=False)
    is_test = (d.split == "test").values

    comp = compare(d, phi, is_test, (d.split == "train").values, Path(models_dir) / "fe_baseline" / "coefficients.csv")
    f = Path(findings_dir)
    f.mkdir(parents=True, exist_ok=True)
    comp.to_csv(f / "shap_vs_causal.csv", index=False)
    drv = comp[comp.level == "driver"].dropna(subset=["causal_importance"])
    rho = stats.spearmanr(drv.shap_importance, drv.causal_importance)
    print(comp.to_string(index=False, float_format="%.4f"))
    print(f"Spearman rank correlation, 8 drivers (all rows): {rho.statistic:+.3f} (p = {rho.pvalue:.3f})")

    v = Path(visuals_dir)
    style()
    plot_comparison(comp, rho.statistic, v, years["train"], len(d))
    plot_beeswarm(phi[FEATURES].values, d[FEATURES], v)
    print(f"wrote {out}/, {f / 'shap_vs_causal.csv'}, {v / 'shap_vs_causal.png'}, {v / 'shap_beeswarm.png'}")


def compare(d, phi, is_test, is_train, fe_coefs_path):
    """Driver- and index-level SHAP vs. fixed-effects importance in log points of Y. The FE model is fit on
    train with drivers standardized by their train SD, so z uses the train mean and SD as well."""
    fe = pd.read_csv(fe_coefs_path).set_index(["spec", "term"])
    rows = []
    for j in FEATURES:
        x = d[j]
        z = (x - x[is_train].mean()) / x[is_train].std()
        r = dict(level="driver" if j in DRIVERS + ["d_cattle_pc"] else "other", term=j, group=GROUP[j],
                 shap_importance=phi[j].abs().mean(), shap_importance_test=phi.loc[is_test, j].abs().mean(),
                 shap_sign=np.sign(np.corrcoef(phi[j], x)[0, 1]) if x.std() > 0 else np.nan)
        if ("drivers", f"z_{j}") in fe.index:
            b = fe.loc[("drivers", f"z_{j}")]
            r |= dict(fe_coef_per_sd=b.coef, fe_ci_low=b.ci_low, fe_ci_high=b.ci_high, fe_p=b.p,
                      causal_importance=abs(b.coef) * (z - z.mean()).abs().mean(), fe_sign=np.sign(b.coef))
        rows.append(r)
    for idx, members in [("D_corporate", CORPORATE), ("D_household", HOUSEHOLD)]:
        s = phi[members].sum(axis=1)
        x = d[idx]
        b = fe.loc[("main", idx)]
        rows.append(dict(level="index", term=idx, group=idx.split("_")[1], shap_importance=s.abs().mean(),
                         shap_importance_test=s[is_test].abs().mean(), shap_sign=np.sign(np.corrcoef(s, x)[0, 1]),
                         fe_coef_per_sd=b.coef, fe_ci_low=b.ci_low, fe_ci_high=b.ci_high, fe_p=b.p,
                         causal_importance=abs(b.coef) * (x - x.mean()).abs().mean(), fe_sign=np.sign(b.coef)))
    c = pd.DataFrame(rows)
    for lvl in ["driver", "index"]:
        m = c.level == lvl
        c.loc[m, "shap_rank"] = c.loc[m, "shap_importance"].rank(ascending=False)
        has = m & c.causal_importance.notna()
        c.loc[has, "causal_rank"] = c.loc[has, "causal_importance"].rank(ascending=False)
    c["sign_agree"] = np.where(c.fe_sign.notna(), c.shap_sign == c.fe_sign, np.nan)
    return c


def plot_comparison(comp, rho, out, train_years, n_rows):
    drv = comp[comp.level == "driver"].sort_values("shap_importance")
    y = np.arange(len(drv))
    colors = [GROUP_COLOR[g] for g in drv.group]
    labels = [LABELS.get(t, t) for t in drv.term]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    a, b, c = axes
    b.sharex(a)  # same units: importance in log points of Y
    a.barh(y, drv.shap_importance, color=colors, height=0.6)
    a.set_title("Prediction: XGBoost mean |SHAP|")
    a.set_xlabel("log points of Y")
    imp = drv.causal_importance.fillna(0)
    b.barh(y, imp, color=colors, height=0.6)
    for yi, v in zip(y, drv.causal_importance):
        if pd.isna(v):
            b.text(0.002, yi, "not in FE model", va="center", fontsize=7.5, color=MUTED)
    b.set_title("Within-state: |β| × mean |z − z̄|")
    b.set_xlabel("log points of Y")
    has = drv.fe_coef_per_sd.notna()
    c.axvline(0, color=INK2, lw=0.8)
    c.errorbar(drv.fe_coef_per_sd[has], y[has.values],
               xerr=[(drv.fe_coef_per_sd - drv.fe_ci_low)[has], (drv.fe_ci_high - drv.fe_coef_per_sd)[has]],
               fmt="none", ecolor=AXIS, elinewidth=1.6, capsize=0)
    c.scatter(drv.fe_coef_per_sd[has], y[has.values], c=np.array(colors)[has.values], s=36, zorder=3,
              edgecolor="white", linewidth=1)
    c.set_title("Within-state effect per SD (95% CI)")
    c.set_xlabel("Δ log GHG per person, per pooled SD")
    a.set_yticks(y, labels)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ["corporate", "household", "livestock"]]
    fig.legend(handles, ["Corporate driver", "Household driver", "Livestock (robustness)"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.suptitle(f"Prediction-based vs. within-state driver importance (Spearman ρ = {rho:+.2f}, 8 drivers)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.1, f"SHAP: XGBoost trained {train_years.replace('-', '–')} on drivers, controls, spatial lags "
             f"and break dummies, no state or year identifiers; all {n_rows:,} state-years. Within-state: two-way FE "
             f"fit on {train_years.replace('-', '–')}, drivers spec, SEs clustered by state, t(47).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, out, "shap_vs_causal.png")


def plot_beeswarm(values, X, out):
    plt.figure()
    shap.summary_plot(values, X.rename(columns=LABELS), cmap=SEQ, show=False, plot_size=(7.5, 5.5),
                      color_bar_label="Feature value (low to high)")
    fig = plt.gcf()
    fig.axes[0].set_title("XGBoost SHAP values, every state-year", loc="left")
    fig.axes[0].set_xlabel("SHAP value (log points of Y)")
    save(fig, out, "shap_beeswarm.png")


if __name__ == "__main__":
    main(*sys.argv[1:5])
