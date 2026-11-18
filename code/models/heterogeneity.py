"""Phase 6: heterogeneity, tipping points and policy.

Usage:
    python code/models/heterogeneity.py <split_dir> <models_dir> <findings_dir> <visuals_dir> [--boot 500]
    e.g. python code/models/heterogeneity.py data/split_data models findings/heterogeneity visuals
Run after dml.py: the simulation and the rankings read findings/dml/drivers_comparison.csv and
learner_comparison.csv, and the comparison with prediction-based SHAP reads findings/baseline/shap_vs_causal.csv.

Sample: as in Phases 4 and 5, the contiguous 48 states in the train years (1997-2014, 864 state-years).
State effects, year effects and the three EPA break dummies are partialled out first (dml.absorb), so every
estimate below uses within-state, within-year variation only.

1. Causal forest (EconML CausalForestDML), one per index d in {D_corporate, D_household}:
       Y~ = theta(X) * d~ + g(X, W) + e,   d~ = m(X, W) + v      (~ = fixed effects partialled out)
   W: the other index, both spatial lags and the controls (partialled out).
   X, the effect modifiers:
       x_level  the state's mean of d over the train years. The state's persistent level, not this year's
                value: this year's value contains d~ itself, so the nuisance model m could predict the
                treatment from its own modifier and leave no treatment variation.
       x_gdp    log real GDP per person in that state-year.
   Nuisance learners: chosen per nuisance from Lasso / random forest / XGBoost by 5-fold state-grouped CV MSE
   on the train years (dml.tune). Cross-fitting holds out whole states (groups = state).
   Forest: 2,000 honest trees, min 10 rows per leaf, half-samples. Its pointwise CIs come from EconML's
   bootstrap of little bags, which samples rows, not states, so they ignore within-state correlation and are
   shown only as a guide. The formal test is cluster-robust:
   Linear heterogeneity test (best linear projection of theta(X)): OLS of the forest's cross-fitted
   residuals, Y~res on d~res, d~res * z(x_level), d~res * z(x_gdp) (z = standardized), with state-clustered SEs
   (CR1) and t(G - 1). The d~res coefficient is the average effect; each interaction is the change in the effect
   per SD of that modifier.

2. SHAP on the XGBoost nuisance model: XGBoost for E[Y~ | all 8 drivers~, both lags~, controls~] (the outcome
   nuisance of the one-driver-at-a-time DML with every driver in), tuned by state-grouped CV. TreeExplainer
   SHAP on the train rows, mean |SHAP| per driver (a within-state attribution, compared with the DML ranking and
   with the Phase 4 SHAP of the levels model), and dependence plots for the 4 drivers with the largest mean |SHAP|.

3. Tipping points (segmented regression), for the drivers with a significant DML effect and for both indices:
       Y = b1 x + b2 (x - c)+ + other regressors + state FE + year FE + breaks
   x is the standardized driver (other regressors: the other 7 drivers, both lags, controls, as in the FE drivers
   spec) or the index (other regressors: the main spec). c is searched over the 10th-90th percentiles of x
   (2.5-point steps) by least squares; b2 is the change in slope above c. A threshold is kept only if all three
   rules hold (fixed before the run):
       a. bootstrap: resampling whole states (N_BOOT draws) and re-searching c each time, the 95% percentile
          interval of b2 excludes 0;
       b. bootstrap: at least half of the re-estimated thresholds are within C_TOL = 0.25 SD of c;
       c. halves: re-estimated separately on 1997-2005 and 2006-2014, both halves give b2 with the same sign and
          a threshold within HALF_TOL = 0.5 SD of c.
   The naive clustered p-value of b2 at the selected c is also reported; it ignores the search over c and is
   anti-conservative.

4. Simulation: a 1-SD reduction in each driver with a significant DML effect (and in each index), applied to
   each state's 2022 gross GHG: avoided_s = gross_2022,s x (1 - exp(-theta)), in MMT CO2e, with theta and its 95%
   CI from Phase 5 (drivers_best and main_best). Gross = per-person emissions x state population. Two sizes:
   one pooled SD (the unit of theta, a large cross-state difference) and one within-state SD (a typical change in
   a state's own value over time, theta x within-SD / pooled-SD). For the indices, also the heterogeneous version
   with each state's mean forest CATE. 2022 is outside the estimation years: this is a scenario, not a forecast.
   Effects of different drivers are not additive (residential electricity and grid fossil share overlap by
   arithmetic).

5. Policy levers: each driver mapped to the state policy instruments that move it (policy_levers.csv).

6. Secondary analysis, local temperature: two-way FE (state-clustered, t(G - 1)) of the state temperature
   anomaly (temp_anomaly_f, vs. the state's 1991-2020 mean) on the indices and lags, and separately on the 8
   drivers, with log population density and log GDP per person. Degree days are left out: they are a function
   of temperature. Well-mixed GHGs from one state cannot measurably warm that state, so the expected result is
   null; a nonzero coefficient would point to reverse causation (weather driving energy demand).

Writes:
    <models_dir>/heterogeneity/forest_<index>/  settings.json, tuning.csv, cate.csv (every state-year)
    <models_dir>/heterogeneity/shap_nuisance/  settings.json, tuning.csv, shap_values.csv
    <findings_dir>/  blp_tests.csv, state_cate.csv, cate_curves.csv, shap_within.csv, thresholds.csv,
                     threshold_bootstrap.csv, avoided_national.csv, avoided_by_state.csv, policy_levers.csv,
                     temperature.csv
    <visuals_dir>/   het_cate.png, het_state_cate.png, het_shap_dependence.png, het_thresholds.png,
                     het_avoided.png
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from econml.dml import CausalForestDML  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ["data_preparation", "analysis", "models"]:
    sys.path.insert(0, str(ROOT / sub))
from build_panel import CONTROLS, CORPORATE, HOUSEHOLD, LAGS, Y  # noqa: E402
from split_panel import load  # noqa: E402
from eda import AXIS, CORP, HH, INK, INK2, LABELS, MUTED, SEQ, save, style  # noqa: E402
import dml as D  # noqa: E402
import fe_baseline as FE  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 2026
INDICES = ["D_corporate", "D_household"]
DRIVERS = CORPORATE + HOUSEHOLD
Z = [f"z_{d}" for d in DRIVERS]
BREAKS = D.BREAKS
FOREST = dict(n_estimators=2000, min_samples_leaf=10, max_samples=0.45, honest=True, criterion="het",
              inference=True, random_state=SEED, n_jobs=-1)
QGRID = np.arange(0.10, 0.9001, 0.025)
N_BOOT = 500
C_TOL, HALF_TOL = 0.25, 0.5
HALVES = [(1997, 2005), (2006, 2014)]
COLOR = {"D_corporate": CORP, "D_household": HH}
LABELS = LABELS | {f"z_{d}": LABELS[d] for d in DRIVERS}

LEVERS = {
    "d_res_elec_pc": ("Residential electricity / person", "household",
                      "Appliance and equipment efficiency standards; residential building energy codes; utility "
                      "energy-efficiency resource standards and programs; rate design",
                      "Partly arithmetic: power emissions are electricity sold x grid carbon intensity, so the payoff "
                      "falls as the grid decarbonizes; pair with grid policy"),
    "d_grid_fossil_share": ("Grid fossil share", "corporate",
                            "Renewable portfolio / clean energy standards (RPS / CES); carbon pricing on power "
                            "(cap-and-trade); coal retirement and new-capacity siting policy",
                            "Acts on every kWh sold in the state, household and business alike"),
    "d_fossil_prod_pc": ("Fossil production / person", "corporate",
                         "Methane rules for oil and gas (leak detection and repair, venting and flaring limits); "
                         "coal-mine methane capture; production policy (leasing, severance taxes)",
                         "Much of its effect is methane, which EIA's energy-CO2 outcome leaves out (Phase 5)"),
    "d_vmt_pc": ("Vehicle-miles / person", "household",
                 "Transit investment; land use and zoning (density, transit-oriented development); road pricing "
                 "and fuel taxes",
                 "Vehicle efficiency and EV policy lower emissions per mile, not miles, so they are not the lever "
                 "for this driver"),
    "d_sand_gravel_pc": ("Sand & gravel / person", "corporate", "None recommended on this evidence",
                         "No detectable within-state effect (Phase 5)"),
    "d_industrial_share": ("Industrial GDP share", "corporate", "None recommended on this evidence",
                           "No detectable within-state effect (Phase 5)"),
    "d_air_pax_pc": ("Air passenger-miles / person", "household", "None recommended on this evidence",
                     "No detectable within-state effect (Phase 5)"),
    "d_pce_nonenergy_pc": ("Real non-energy PCE / person", "household", "None recommended on this evidence",
                           "No detectable within-state effect (Phase 5)"),
}


# ---------------------------------------------------------------- helpers
def ols_cluster(y, x, groups):
    """OLS with CR1 cluster-robust covariance (G/(G-1) x (n-1)/(n-k)); returns b, V."""
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    e = y - x @ b
    s = pd.DataFrame(x * e[:, None]).groupby(groups).sum().values
    g = s.shape[0]
    v = xtx_inv @ (s.T @ s) @ xtx_inv * g / (g - 1) * (n - 1) / (n - k)
    return b, v


def t_row(est, se, df):
    t = est / se
    c = stats.t.ppf(0.975, df)
    return dict(estimate=est, se=se, t=t, df=df, p=2 * stats.t.sf(abs(t), df), ci_low=est - c * se,
                ci_high=est + c * se)


def best_learner(frame, feats, targets):
    """Per nuisance target, the Lasso / RF / XGBoost candidate with the lowest state-grouped CV MSE."""
    tuning, best = D.tune(frame, {k: (t, feats) for k, t in targets.items()}, ["lasso", "rf", "xgb"])
    pick = {k: min(v.values(), key=lambda r: r["cv_mse"]) for k, v in best.items()}
    return tuning, pick


# ---------------------------------------------------------------- 1. causal forest
def forest(tr, d, out):
    t0 = time.time()
    wcols = [c for c in D.INDEX if c != d] + CONTROLS
    a = D.absorb(tr, [Y, d] + wcols, breaks=True)
    groups = tr.postal_code.values
    lev = tr.groupby("postal_code")[d].transform("mean").values
    gdp = tr["log_gdp_real_pc"].values
    xm = np.column_stack([lev, gdp])
    feats = ["x_level", "x_gdp"] + wcols
    frame = pd.DataFrame(np.column_stack([xm, a[wcols].values]), columns=feats).assign(
        postal_code=groups, _y=a[Y].values, _t=a[d].values)
    tuning, pick = best_learner(frame, feats, {"y": "_y", "t": "_t"})
    est = CausalForestDML(model_y=D.make(pick["y"]["learner"], pick["y"]["params"]),
                          model_t=D.make(pick["t"]["learner"], pick["t"]["params"]), cv=D.N_FOLDS, **FOREST)
    est.fit(a[Y].values, a[d].values, X=xm, W=a[wcols].values, groups=groups, cache_values=True)
    yres, tres, _, _ = est.residuals_

    # cluster-robust linear heterogeneity test (best linear projection)
    zl, zg = (lev - lev.mean()) / lev.std(), (gdp - gdp.mean()) / gdp.std()
    xb = np.column_stack([np.ones_like(tres), tres, tres * zl, tres * zg])
    b, v = ols_cluster(yres, xb, groups)
    df = len(np.unique(groups)) - 1
    blp = [dict(index=d, term=t, **t_row(b[i], np.sqrt(v[i, i]), df))
           for i, t in [(1, "average effect"), (2, "x per SD of the state's mean level"),
                        (3, "x per SD of log GDP per person")]]

    cate = est.effect(xm)
    lo, hi = est.effect_interval(xm, alpha=0.05)
    rows = tr[["postal_code", "year"]].assign(index=d, x_level=lev, x_gdp=gdp, cate=cate, cate_lo=lo, cate_hi=hi)

    # CATE curves: one modifier over its range, the other at its median
    curves = []
    for mod, grid_vals, other in [("level", np.quantile(np.unique(lev), np.linspace(0.02, 0.98, 41)), np.median(gdp)),
                                  ("gdp", np.quantile(gdp, np.linspace(0.02, 0.98, 41)), np.median(lev))]:
        xg = np.column_stack([grid_vals, np.full_like(grid_vals, other)]) if mod == "level" else \
            np.column_stack([np.full_like(grid_vals, other), grid_vals])
        f = est.effect(xg)
        flo, fhi = est.effect_interval(xg, alpha=0.05)
        sd_mod = lev.std() if mod == "level" else gdp.std()
        mean_mod = lev.mean() if mod == "level" else gdp.mean()
        coef = 2 if mod == "level" else 3
        zz = (grid_vals - mean_mod) / sd_mod
        lin = b[1] + b[coef] * zz
        lin_se = np.sqrt(v[1, 1] + zz ** 2 * v[coef, coef] + 2 * zz * v[1, coef])
        c = stats.t.ppf(0.975, df)
        curves.append(pd.DataFrame(dict(index=d, modifier=mod, x=grid_vals, cate=f, cate_lo=flo, cate_hi=fhi,
                                        blp=lin, blp_lo=lin - c * lin_se, blp_hi=lin + c * lin_se)))

    o = out / f"forest_{d}"
    o.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(o / "tuning.csv", index=False)
    rows.to_csv(o / "cate.csv", index=False)
    (o / "settings.json").write_text(json.dumps(dict(
        treatment=d, w=wcols, x=["x_level: state mean of the index over the train years", "x_gdp: log_gdp_real_pc"],
        fixed_effects="state + year + break dummies partialled out first", n=len(tr), groups="postal_code",
        model_y={k: pick["y"][k] for k in ["learner", "params", "cv_r2"]},
        model_t={k: pick["t"][k] for k in ["learner", "params", "cv_r2"]},
        forest={k: v for k, v in FOREST.items() if k != "n_jobs"}, cv=D.N_FOLDS,
        forest_mean_cate=float(cate.mean()), seconds=round(time.time() - t0, 1)), indent=2, default=str))
    print(f"forest {d}: model_y {pick['y']['learner']} (CV R2 {pick['y']['cv_r2']:.2f}), model_t {pick['t']['learner']} "
          f"(CV R2 {pick['t']['cv_r2']:.2f}); mean CATE {cate.mean():.4f}; BLP average {b[1]:.4f}, "
          f"level {b[2]:+.4f} (p={blp[1]['p']:.3f}), gdp {b[3]:+.4f} (p={blp[2]['p']:.3f})")
    return rows, pd.DataFrame(blp), pd.concat(curves)


# ---------------------------------------------------------------- 2. SHAP on the XGBoost nuisance
def shap_nuisance(tr, out, findings):
    cols = Z + LAGS + CONTROLS
    a = D.absorb(tr, [Y] + cols, breaks=True)
    tuning, best = D.tune(a, {"y": (Y, cols)}, ["xgb"])
    p = best["y"]["xgb"]
    m = D.make("xgb", p["params"]).fit(a[cols].values, a[Y].values)
    phi = shap.TreeExplainer(m).shap_values(a[cols].values)
    imp = pd.Series(np.abs(phi).mean(axis=0), index=cols)

    dml_imp = pd.read_csv(Path("findings/dml/drivers_comparison.csv")).set_index("driver")
    ph4 = pd.read_csv(Path("findings/baseline/shap_vs_causal.csv")).set_index("term")
    rows = []
    for dname, z in zip(DRIVERS, Z):
        sign = np.sign(np.corrcoef(phi[:, cols.index(z)], a[z])[0, 1])
        rows.append(dict(driver=dname, group="corporate" if dname in CORPORATE else "household",
                         shap_within=imp[z], shap_within_sign=sign, dml_importance=dml_imp.loc[dname, "dml_importance"],
                         dml_estimate=dml_imp.loc[dname, "dml_estimate"],
                         shap_levels_phase4=ph4.loc[dname, "shap_importance"]))
    s = pd.DataFrame(rows)
    for c in ["shap_within", "dml_importance", "shap_levels_phase4"]:
        s[c.replace("importance", "").rstrip("_") + "_rank"] = s[c].rank(ascending=False)
    s["sign_agree_dml"] = s.shap_within_sign == np.sign(s.dml_estimate)
    rho_dml = stats.spearmanr(s.shap_within, s.dml_importance)
    rho_ph4 = stats.spearmanr(s.shap_within, s.shap_levels_phase4)
    s.attrs = dict(rho_dml=rho_dml, rho_ph4=rho_ph4)

    o = out / "shap_nuisance"
    o.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(o / "tuning.csv", index=False)
    pd.DataFrame(phi, columns=cols).assign(postal_code=a.postal_code, year=a.year).to_csv(o / "shap_values.csv",
                                                                                          index=False)
    (o / "settings.json").write_text(json.dumps(dict(
        target=f"{Y}, fixed effects and breaks partialled out", features=cols, params=p["params"],
        cv_r2=p["cv_r2"], cv="5-fold state-grouped, train years"), indent=2, default=str))
    s.to_csv(findings / "shap_within.csv", index=False)
    print(f"SHAP nuisance: XGBoost {p['params']}, CV R2 {p['cv_r2']:.3f}; within-SHAP vs DML rho "
          f"{rho_dml.statistic:+.2f} (p={rho_dml.pvalue:.3f}), vs Phase 4 SHAP {rho_ph4.statistic:+.2f}")
    return s, phi, a, cols


# ---------------------------------------------------------------- 3. segmented regression
def basis(m, tol=1e-9):
    u, sv, _ = np.linalg.svd(m, full_matrices=False)
    return u[:, sv > tol * sv.max()]


def profile(df, var, regs, cands, clusters):
    """Least-squares threshold search. Returns dict with c, b1, b2, naive clustered SE of b2, SSR profile."""
    absorb_cols = [pd.get_dummies(clusters, dtype=float), pd.get_dummies(df.year, dtype=float).iloc[:, 1:],
                   df[BREAKS].reset_index(drop=True).astype(float)]
    qa = basis(pd.concat([c.reset_index(drop=True) for c in absorb_cols], axis=1).values)
    res = lambda m: m - qa @ (qa.T @ m)  # noqa: E731
    x = df[var].values
    xr = res(df[regs].values.astype(float))
    qx = basis(xr)
    ey = res(df[Y].values.astype(float))
    ey = ey - qx @ (qx.T @ ey)
    ssr0 = float(ey @ ey)
    prof = []
    for c in cands:
        share_above = float((x > c).mean())
        if not 0.1 <= share_above <= 0.9:
            prof.append(np.nan)
            continue
        h = res(np.maximum(x - c, 0.0))
        h = h - qx @ (qx.T @ h)
        prof.append(ssr0 - (h @ ey) ** 2 / (h @ h))
    prof = np.array(prof)
    if np.all(np.isnan(prof)):
        return None
    i = int(np.nanargmin(prof))
    c = cands[i]
    h = res(np.maximum(x - c, 0.0))
    full = np.column_stack([xr, h])
    b, *_ = np.linalg.lstsq(full, res(df[Y].values.astype(float)), rcond=None)
    ht = h - qx @ (qx.T @ h)
    b2 = float(ht @ ey / (ht @ ht))
    e = ey - b2 * ht
    s = pd.Series(ht * e).groupby(np.asarray(clusters)).sum().values
    g, n, k = len(s), len(e), qa.shape[1] + xr.shape[1] + 1
    se = np.sqrt((s ** 2).sum() / (ht @ ht) ** 2 * g / (g - 1) * (n - 1) / (n - k))
    return dict(c=c, b1=float(b[regs.index(var)]), b2=b2, se_b2=se, df=g - 1, ssr0=ssr0, ssr=float(prof[i]),
                profile=prof)


def thresholds(tr, top, n_boot, findings):
    jobs = [(f"z_{d}", Z + LAGS + CONTROLS) for d in top] + [(d, D.INDEX + CONTROLS) for d in INDICES]
    rng = np.random.default_rng(SEED)
    states = np.array(sorted(tr.postal_code.unique()))
    draws = [rng.choice(states, size=len(states), replace=True) for _ in range(n_boot)]
    by_state = {s: g for s, g in tr.groupby("postal_code")}
    rows, boots, profiles = [], [], []
    for var, regs in jobs:
        cands = np.quantile(tr[var], QGRID)
        full = profile(tr, var, regs, cands, tr.postal_code.values)
        bs = []
        for bi, draw in enumerate(draws):
            parts = [by_state[s].assign(_cl=f"{s}_{j}") for j, s in enumerate(draw)]
            bd = pd.concat(parts, ignore_index=True)
            r = profile(bd, var, regs, cands, bd._cl.values)
            if r:
                bs.append(dict(variable=var, draw=bi, c=r["c"], b2=r["b2"]))
        bs = pd.DataFrame(bs)
        halves = {}
        for y0, y1 in HALVES:
            hd = tr[tr.year.between(y0, y1)].reset_index(drop=True)
            halves[f"{y0}-{y1}"] = profile(hd, var, regs, cands, hd.postal_code.values)
        lo, hi = np.percentile(bs.b2, [2.5, 97.5])
        near = float((np.abs(bs.c - full["c"]) <= C_TOL).mean())
        half_ok = all(h is not None and np.sign(h["b2"]) == np.sign(full["b2"]) and abs(h["c"] - full["c"]) <= HALF_TOL
                      for h in halves.values())
        rule_a, rule_b = bool(lo > 0 or hi < 0), near >= 0.5
        naive_p = 2 * stats.t.sf(abs(full["b2"] / full["se_b2"]), full["df"])
        row = dict(variable=var, threshold=full["c"], threshold_pct=float((tr[var] <= full["c"]).mean()),
                   slope_below=full["b1"], slope_change=full["b2"], slope_above=full["b1"] + full["b2"],
                   naive_p=naive_p, ssr_reduction_pct=100 * (1 - full["ssr"] / full["ssr0"]),
                   boot_b2_ci_low=lo, boot_b2_ci_high=hi, boot_share_c_within_tol=near,
                   boot_c_median=float(bs.c.median()), boot_c_iqr=float(bs.c.quantile(0.75) - bs.c.quantile(0.25)),
                   rule_a_slope_change=rule_a, rule_b_threshold_stable=rule_b, rule_c_halves=half_ok,
                   kept=rule_a and rule_b and half_ok, n_boot=len(bs))
        for k, h in halves.items():
            row |= {f"{k}_threshold": h["c"] if h else np.nan, f"{k}_slope_change": h["b2"] if h else np.nan,
                    f"{k}_naive_p": 2 * stats.t.sf(abs(h["b2"] / h["se_b2"]), h["df"]) if h else np.nan}
        rows.append(row)
        boots.append(bs)
        profiles.append(pd.DataFrame(dict(variable=var, c=cands, ssr_reduction_pct=100 * (1 - full["profile"] / full["ssr0"]))))
        print(f"threshold {var:>22}: c = {full['c']:+.2f} SD (pct {row['threshold_pct']:.2f}), slope {full['b1']:+.3f} -> "
              f"{full['b1'] + full['b2']:+.3f}, boot CI b2 [{lo:+.3f}, {hi:+.3f}], c stable {near:.2f}, "
              f"halves {half_ok} -> kept {row['kept']}")
    t = pd.DataFrame(rows)
    t.to_csv(findings / "thresholds.csv", index=False)
    pd.concat(boots).to_csv(findings / "threshold_bootstrap.csv", index=False)
    return t, pd.concat(boots), pd.concat(profiles)


# ---------------------------------------------------------------- 4. simulation
def one_sd_text(d, tr, scale=1.0):
    """The reduction in natural units: scale x the driver's pooled SD on the train years."""
    sd, med = tr[d].std() * scale, tr[d].median()
    if d in ("d_grid_fossil_share", "d_industrial_share"):
        return f"{100 * sd:.1f} percentage points lower"
    if d in ("d_vmt_pc", "d_res_elec_pc", "d_pce_nonenergy_pc"):
        return f"{100 * (1 - np.exp(-sd)):.0f}% lower per person"
    lo = max(np.expm1(med - sd), 0.0)
    return f"{100 * (1 - lo / np.expm1(med)):.0f}% lower per person at the median state ({np.expm1(med):,.1f} -> {lo:,.1f})"


def simulate(tr, base, top, state_cate, findings):
    drv = pd.read_csv("findings/dml/drivers_comparison.csv").set_index("driver")
    lc = pd.read_csv("findings/dml/learner_comparison.csv")
    lc = lc[lc.estimator == "DML best (main)"].set_index("term")
    total = base.ghg_gross_mmt.sum()
    rows, by_state = [], []
    items = [(d, drv.loc[d, "dml_estimate"], drv.loc[d, "dml_ci_low"], drv.loc[d, "dml_ci_high"], f"z_{d}") for d in top]
    items += [(d, lc.loc[d, "estimate"], lc.loc[d, "ci_low"], lc.loc[d, "ci_high"], d) for d in INDICES]
    for d, th, lo, hi, col in items:
        sd_pooled = tr[col].std()
        sd_within = (tr[col] - tr.groupby("postal_code")[col].transform("mean")).std()
        ratio = sd_within / sd_pooled
        for size, scale in [("1 pooled SD", 1.0), ("1 within-state SD", ratio)]:
            av = {k: base.ghg_gross_mmt * (1 - np.exp(-v * scale)) for k, v in [("mid", th), ("lo", lo), ("hi", hi)]}
            rows.append(dict(variable=d, size=size, theta_per_pooled_sd=th, ci_low=lo, ci_high=hi, scale_sd=scale,
                             change=one_sd_text(d, tr, scale) if d in DRIVERS else f"{scale:.2f} SD of the index",
                             avoided_mmt=av["mid"].sum(), avoided_mmt_low=av["lo"].sum(), avoided_mmt_high=av["hi"].sum(),
                             pct_of_48_state_gross=100 * av["mid"].sum() / total))
            if size == "1 pooled SD":
                by_state.append(base[["postal_code", "pop", "ghg_gross_pc_t", "ghg_gross_mmt"]].assign(
                    variable=d, theta=th, avoided_mmt=av["mid"].values,
                    avoided_t_per_person=(av["mid"] * 1e6 / base["pop"]).values))
    for d in INDICES:  # heterogeneous: each state's mean forest CATE
        c = state_cate[state_cate["index"] == d].set_index("postal_code").cate
        av = base.ghg_gross_mmt * (1 - np.exp(-base.postal_code.map(c).values))
        rows.append(dict(variable=d, size="1 pooled SD, state CATE (forest)", theta_per_pooled_sd=float(c.mean()),
                         scale_sd=1.0, change="1.00 SD of the index", avoided_mmt=av.sum(),
                         pct_of_48_state_gross=100 * av.sum() / total))
    nat = pd.DataFrame(rows)
    st = pd.concat(by_state)
    nat.to_csv(findings / "avoided_national.csv", index=False)
    st.to_csv(findings / "avoided_by_state.csv", index=False)
    print(f"simulation: 48-state gross 2022 = {total:,.0f} MMT")
    print(nat[["variable", "size", "avoided_mmt", "avoided_mmt_low", "avoided_mmt_high", "pct_of_48_state_gross"]]
          .round(2).to_string(index=False))
    return nat, st


# ---------------------------------------------------------------- 6. temperature
def temperature(train, findings):
    rows = []
    for name, regs in [("indices", D.INDEX + ["log_pop_density", "log_gdp_real_pc"]),
                       ("drivers", Z + LAGS + ["log_pop_density", "log_gdp_real_pc"])]:
        res, df = FE.fit(train, FE.spec("temp_anomaly_f", regs))
        for t in regs:
            if t.startswith(("D_", "W_", "z_")):
                rows.append(dict(spec=name, term=t, **t_row(res.params[t], res.std_errors[t], df)))
        drv = [t for t in regs if t.startswith(("D_", "W_", "z_"))]
        w = res.wald_test(formula=", ".join(f"{t} = 0" for t in drv))
        rows.append(dict(spec=name, term="joint F (all driver terms)", estimate=w.stat / len(drv),
                         p=stats.f.sf(w.stat / len(drv), len(drv), df), df=df))
    t = pd.DataFrame(rows)
    t.to_csv(findings / "temperature.csv", index=False)
    print("temperature anomaly (F):")
    print(t[["spec", "term", "estimate", "p"]].round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------- figures
def plot_cate(curves, state_cate, blp, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey="row")
    for i, d in enumerate(INDICES):
        for j, mod in enumerate(["level", "gdp"]):
            ax = axes[i, j]
            c = curves[(curves["index"] == d) & (curves.modifier == mod)]
            ax.fill_between(c.x, c.cate_lo, c.cate_hi, color=COLOR[d], alpha=0.15, linewidth=0)
            ax.plot(c.x, c.cate, color=COLOR[d], label="causal forest")
            ax.plot(c.x, c.blp, color=INK2, lw=1.2, ls="--", label="linear projection")
            ax.axhline(0, color=AXIS, lw=0.8)
            p = blp[(blp["index"] == d) & blp.term.str.contains("level" if mod == "level" else "GDP")].p.iloc[0]
            name = LABELS[d]
            ax.set_title(f"{name}: by {'state mean level' if mod == 'level' else 'GDP per person'} "
                         f"(slope p = {p:.2f})", fontsize=9.5)
            if mod == "level":
                lv = state_cate[state_cate["index"] == d].x_level
                ax.plot(lv, np.full(len(lv), ax.get_ylim()[0]), "|", color=MUTED, ms=6)
                ax.set_xlabel("state's mean index level, 1997–2014 (SD)")
            else:
                ax.set_xlabel("log real GDP per person")
            if j == 0:
                ax.set_ylabel("effect per SD of the index\n(log points of Y)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("How each index's within-state effect varies (causal forest, train years 1997–2014)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "Bands: forest 95% intervals (row bootstrap, not clustered by state: a guide only). Dashed: "
             "best linear projection; its slope p-value is clustered by state, t(47). Ticks on the level axis: the 48 states.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, out, "het_cate.png")


def plot_state_cate(state_cate, blp, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 8.6))
    for ax, d in zip(axes, INDICES):
        s = state_cate[state_cate["index"] == d].sort_values("cate")
        y = np.arange(len(s))
        ax.axvline(0, color=AXIS, lw=0.8)
        avg = blp[(blp["index"] == d) & (blp.term == "average effect")].estimate.iloc[0]
        ax.axvline(avg, color=INK2, lw=1, ls="--")
        ax.text(avg, len(s) + 0.3, f" average {avg:.3f}", fontsize=8, color=INK2, va="bottom")
        ax.scatter(s.cate, y, color=COLOR[d], s=22, zorder=3, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y, s.postal_code, fontsize=7)
        ax.set_ylim(-1, len(s) + 1.5)
        ax.set_title(LABELS[d])
        ax.set_xlabel("mean forest CATE, 1997–2014 (log points per SD)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Effect of a 1-SD rise in each index, by state", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "het_state_cate.png")


def plot_shap(s, phi, a, tr, cols, out):
    top = s.sort_values("shap_within", ascending=False).driver.head(4).tolist()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, d in zip(axes, top):
        z = f"z_{d}"
        sc = ax.scatter(a[z], phi[:, cols.index(z)], c=tr[z], cmap=SEQ, s=9, linewidths=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(LABELS[d], fontsize=9.5)
        ax.set_xlabel("within-state deviation (SD)")
    axes[0].set_ylabel("SHAP value (log points of Y,\nwithin-state)")
    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("driver level (SD)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("SHAP dependence, XGBoost outcome-nuisance model (fixed effects removed), top 4 drivers",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    save(fig, out, "het_shap_dependence.png")


def plot_thresholds(t, boots, profiles, out):
    vars_ = t.variable.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, v in zip(axes.ravel(), vars_):
        p = profiles[profiles.variable == v]
        r = t[t.variable == v].iloc[0]
        color = COLOR.get(v, CORP if any(v == f"z_{d}" for d in CORPORATE) else HH)
        ax.plot(p.c, p.ssr_reduction_pct, color=color)
        ax.axvline(r.threshold, color=INK2, lw=1, ls="--")
        b = boots[boots.variable == v].c
        ymin = 0
        ax.plot(b, np.full(len(b), ymin), "|", color=MUTED, alpha=0.15, ms=10)
        for k in [f"{y0}-{y1}_threshold" for y0, y1 in HALVES]:
            ax.plot(r[k], ymin, marker="^", color=INK, ms=6, zorder=4)
        verdict = "kept" if r.kept else "not kept"
        ax.set_title(f"{LABELS.get(v, v)}: c = {r.threshold:+.2f} ({verdict})", fontsize=9)
        ax.set_xlabel("candidate threshold c (SD)")
        ax.set_ylim(bottom=-0.05 * max(p.ssr_reduction_pct.max(), 0.1))
    for ax in axes[:, 0]:
        ax.set_ylabel("% of residual SS removed\nby the hinge at c")
    fig.suptitle("Segmented regression: candidate tipping points (train years, state and year FE)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "Dashed: selected c. Gray ticks: re-estimated c in 500 state bootstraps. Triangles: c re-estimated "
             "on 1997–2005 and 2006–2014. Kept only if the bootstrap slope-change CI excludes 0, half the bootstrap c's are "
             "within 0.25 SD, and both halves agree.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, out, "het_thresholds.png")


def plot_avoided(nat, out):
    sub = nat[nat["size"].isin(["1 pooled SD", "1 within-state SD"])]
    order = sub[sub["size"] == "1 pooled SD"].sort_values("avoided_mmt").variable.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, size in zip(axes, ["1 pooled SD", "1 within-state SD"]):
        s = sub[sub["size"] == size].set_index("variable").loc[order]
        y = np.arange(len(s))
        colors = [COLOR.get(v, CORP if v in CORPORATE else HH) for v in s.index]
        ax.barh(y, s.avoided_mmt, color=colors, height=0.6)
        ax.errorbar(s.avoided_mmt, y, xerr=[s.avoided_mmt - s.avoided_mmt_low, s.avoided_mmt_high - s.avoided_mmt],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
        for yi, v in zip(y, s.avoided_mmt):
            ax.text(max(v, 0) + 2, yi + 0.28, f"{v:,.0f}", fontsize=8, color=INK)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(f"Reduction of {size}")
        ax.set_xlabel("MMT CO₂e avoided per year, 48 states (2022 base)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)), [LABELS.get(v, v) for v in order])
    fig.suptitle("Simulated emissions avoided by a 1-SD reduction, one driver at a time",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.06, "θ and 95% CI from Phase 5 DML (drivers_best; indices: main_best), applied to each state's 2022 "
             "gross GHG. Not additive across drivers.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, out, "het_avoided.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("models_dir")
    ap.add_argument("findings_dir")
    ap.add_argument("visuals_dir")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    t0 = time.time()
    parts = load(args.split_dir)
    tr = D.prepare(parts["train"], D.VERSIONS["main"], parts["train"])
    out = Path(args.models_dir) / "heterogeneity"
    findings, vis = Path(args.findings_dir), Path(args.visuals_dir)
    for p in (out, findings, vis):
        p.mkdir(parents=True, exist_ok=True)

    cate, blp, curves = zip(*(forest(tr, d, out) for d in INDICES))
    cate, blp, curves = pd.concat(cate), pd.concat(blp), pd.concat(curves)
    state_cate = cate.groupby(["index", "postal_code"]).agg(cate=("cate", "mean"), x_level=("x_level", "first"),
                                                            x_gdp=("x_gdp", "mean")).reset_index()
    blp.to_csv(findings / "blp_tests.csv", index=False)
    state_cate.to_csv(findings / "state_cate.csv", index=False)
    curves.to_csv(findings / "cate_curves.csv", index=False)

    s, phi, a, cols = shap_nuisance(tr, out, findings)

    drv = pd.read_csv("findings/dml/drivers_comparison.csv")
    top = drv[drv.dml_p < 0.05].sort_values("dml_importance", ascending=False).driver.tolist()
    print(f"drivers with a significant DML effect: {top}")
    t, boots, profiles = thresholds(tr, top, args.boot, findings)

    base = parts["test"]
    base = base[(base.in_main_sample == 1) & (base.year == base.year.max())].reset_index(drop=True)
    nat, st = simulate(tr, base, top, state_cate, findings)

    pd.DataFrame([dict(driver=d, label=v[0], side=v[1], policy_levers=v[2], note=v[3],
                       dml_estimate=drv.set_index("driver").loc[d, "dml_estimate"],
                       dml_p=drv.set_index("driver").loc[d, "dml_p"]) for d, v in LEVERS.items()]
                 ).to_csv(findings / "policy_levers.csv", index=False)

    temperature(parts["train"], findings)

    style()
    plot_cate(curves, state_cate, blp, vis)
    plot_state_cate(state_cate, blp, vis)
    plot_shap(s, phi, a, tr, cols, vis)
    plot_thresholds(t, boots, profiles, vis)
    plot_avoided(nat, vis)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
