"""Reproduce parent paper 3: Munir, Rashid, Shams & Ahmed (2026), "Infrastructure, institutions, and the resilience
dividend: A hybrid Econometric-Machine learning analysis of CO2 emissions in Central and Eastern Europe",
Environmental Management: Economics and Policy 2, 100006.

Usage:
    python run_pp3.py [--data PANEL.csv] [--out DIR] [--no-fill] [--var-lag 2] [--dh-lag 2]
Defaults: --data ../../PP_data/pp3_data/pp3_panel_raw.csv (from fetch_wdi.py)   --out ../../results/pp3
--data can also point at a split file from TTV split/pp3_split/ (train.csv etc.).

Data: 18 CEE countries, 2000-2024, columns country, year, co2_pc, up, ic, eupe, gdp_pc, elec_kwh_pc, eo, ce
(see PP_data/pp3_data/README.md). Following the paper, missing years (mainly 2023-2024) are forward- then
back-filled within each country (--no-fill turns this off and drops incomplete rows), and CO2E, EC and IEI are
logged: lnco2e, lnec (ln_ec), lniei (ln_iei).

Writes Tables 2-20 as CSV, tables.md with all of them, and fig2_shap_rf.png / fig3_shap_gb.png.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

import econometrics as em  # noqa: E402
import ml_models as ml  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ["co2_pc", "up", "ic", "eupe", "gdp_pc", "elec_kwh_pc", "eo", "ce"]
Y = "lnco2e"
IVS = ["up", "ic", "eupe", "lnec", "lniei", "eo", "ce"]             # long-run equation order (section 4.3.1)
TABLE_ORDER = ["lnec", "lniei", "ce", "eo", "eupe", "ic", "up"]      # order used in Tables 7, 13, 14
CORR = ["lnco2e", "lnec", "lniei", "ce", "eo", "eupe", "ic", "up"]   # Tables 3-5, 9-11
ML_NAMES = {"lnec": "ln_ec", "lniei": "ln_iei", "ce": "ce", "eo": "eo", "eupe": "eupe", "ic": "ic", "up": "up"}


def prepare(path, fill=True):
    p = pd.read_csv(path)
    missing = [c for c in ["country", "year"] + RAW if c not in p]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    p = p.sort_values(["country", "year"]).reset_index(drop=True)
    before = p[RAW].isna().sum()
    if fill:
        p[RAW] = p.groupby("country")[RAW].transform(lambda s: s.ffill().bfill())
        filled = (before - p[RAW].isna().sum())
        print("filled within country (paper: forward fill / back fill):\n" + filled[filled > 0].to_string())
    p = p.dropna(subset=RAW)
    counts = p.groupby("country").size()
    if counts.nunique() > 1:
        keep = counts.max()
        print(f"unbalanced panel; keeping countries with all {keep} years: dropped {list(counts[counts < keep].index)}")
        p = p[p.country.isin(counts[counts == keep].index)]
    return p.assign(lnco2e=np.log(p.co2_pc), lnec=np.log(p.gdp_pc), lniei=np.log(p.elec_kwh_pc),
                    CO2E=p.co2_pc, EC=p.gdp_pc, IEI=p.elec_kwh_pc,
                    CE=p.ce, EO=p.eo, EUPE=p.eupe, IC=p.ic, UP=p.up).reset_index(drop=True)


def md(df, title, fmt="{:.3f}", index=True):
    d = df.copy()
    if index:
        d = d.reset_index()
    cells = [[fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in r]
             for r in d.itertuples(index=False)]
    lines = [f"## {title}", "", "| " + " | ".join(map(str, d.columns)) + " |", "|---" * d.shape[1] + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells] + [""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "PP_data" / "pp3_data" / "pp3_panel_raw.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "pp3")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--var-lag", type=int, default=2, help="VAR/ARDL lag (paper: 2, chosen by SC and FPE)")
    ap.add_argument("--dh-lag", type=int, default=2)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    d = prepare(a.data, fill=not a.no_fill)
    N, T = d.country.nunique(), d.year.nunique()
    print(f"panel: {N} countries x {T} years = {len(d)} rows ({d.year.min()}-{d.year.max()})")
    doc = [f"# Parent paper 3 replication ({N} countries, {d.year.min()}-{d.year.max()}, {len(d)} obs.)", ""]

    def save(df, name, title, **kw):
        df.to_csv(out / f"{name}.csv")
        doc.append(md(df, title, **kw))

    # Tables 2-6
    save(em.descriptives(d, ["CO2E", "CE", "EC", "EO", "EUPE", "IC", "IEI", "UP"]), "table02_descriptives",
         "Table 2. Descriptive statistics")
    r, pv = em.correlations(d, CORR)
    r.index = r.columns = pv.index = pv.columns = [c.upper() for c in CORR]
    r.to_csv(out / "table03_correlation_r.csv")
    pv.to_csv(out / "table03_correlation_p.csv")
    doc.append(md(r.where(np.tril(np.ones(r.shape, bool))), "Table 3. Correlation matrix (r; p-values in CSV)"))
    ur = []
    for v in CORR:
        row = {"Variable": v}
        for lab, tr, df_ in [("Level - No Trend", False, False), ("Level - With Trend", True, False),
                             ("1st Diff - No Trend", False, True), ("1st Diff - With Trend", True, True)]:
            res = em.panel_unit_root(d, v, trend=tr, diff=df_)
            row[lab] = f"{res['choi_z']:.3f} ({res['choi_p']:.3f})"
            row[lab + " Fisher"] = f"{res['fisher_chi2']:.2f} ({res['fisher_p']:.3f})"
        lvl = em.panel_unit_root(d, v)["choi_p"]
        row["Decision"] = "I(0)" if lvl < 0.05 else "I(1)"
        ur.append(row)
    save(pd.DataFrame(ur).set_index("Variable"), "table04_unit_root",
         "Table 4. Panel unit root (Choi Z with p; Fisher-ADF chi2 with p)")
    save(em.var_lag_selection(d, CORR).set_index("Lag"), "table05_var_lag_selection", "Table 5. VAR lag order selection")
    kao = em.kao_test(d, Y, IVS)
    save(pd.DataFrame([kao]).T.rename(columns={0: "value"}), "table06_kao", "Table 6. Kao residual cointegration test")

    # Tables 7-8
    ardl = em.pmg_ardl(d, Y, TABLE_ORDER, p=a.var_lag)
    t7 = pd.concat([ardl["long_run"].assign(part="Long run"), ardl["short_run"].assign(part="Short run")])
    save(t7[["part", "coef", "se", "t", "p"]], "table07_ardl_pmg", f"Table 7. Panel ARDL (PMG), ARDL({a.var_lag},1); "
         f"converged in {ardl['iterations']} iterations")
    save(em.bounds_wald(ardl).set_index("test"), "table08_bounds_wald", "Table 8. Bounds test (Wald, long-run = 0)")

    # Table 9
    dh = []
    for i, a_ in enumerate(CORR):
        for b_ in CORR[i + 1:]:
            for x, y in [(a_, b_), (b_, a_)]:
                r_ = em.dumitrescu_hurlin(d, x, y, K=a.dh_lag)
                dh.append({"Null Hypothesis": f"{x.upper()} does not homogeneously cause {y.upper()}",
                           "W-Stat.": r_["w_stat"], "Zbar-Stat.": r_["zbar_stat"], "Prob.": r_["p"]})
    save(pd.DataFrame(dh).set_index("Null Hypothesis"), "table09_dumitrescu_hurlin",
         f"Table 9. Dumitrescu-Hurlin panel causality (lags = {a.dh_lag})")

    # Tables 10-11
    irf, fevd = em.panel_var_irf_fevd(d, CORR, p=a.var_lag)
    save(irf, "table10_irf", "Table 10. Response of LNCO2E to Cholesky one S.D. innovations")
    save(fevd, "table11_fevd", "Table 11. Variance decomposition of LNCO2E")

    # Tables 12-15
    save(em.hausman(d, Y, TABLE_ORDER).set_index("test"), "table12_hausman", "Table 12. Hausman test")
    for cov in ["unadjusted", "kernel", "clustered"]:
        tab, fit = em.fixed_effects(d, Y, TABLE_ORDER, cov=cov)
        name = "table13_fixed_effects" + ("" if cov == "unadjusted" else f"_{cov}")
        fitrow = pd.DataFrame({k: [v] for k, v in fit.items()}).T.rename(columns={0: "coef"})
        save(pd.concat([tab, fitrow]), name, f"Table 13. Panel least squares (fixed effects), SE: {cov}")
    save(em.vif(d, TABLE_ORDER), "table14_vif", "Table 14. VIF")
    wt, rho = em.wooldridge_serial(d, Y, TABLE_ORDER)
    save(wt.set_index("test"), "table15_wooldridge", f"Table 15. Wooldridge serial correlation (rho = {rho:.3f})")

    # Tables 16-20
    drivers = TABLE_ORDER
    eff, stab = ml.dml_effects(d, Y, drivers)
    eff.variable = eff.variable.map(ML_NAMES)
    stab.variable = stab.variable.map(ML_NAMES)
    save(eff.set_index("variable"), "table16_dml", "Table 16. DML effect estimates (GB nuisance, per 1 SD of D)")
    save(stab.set_index("variable"), "table17_dml_fold_stability", "Table 17. DML stability across folds")
    cf = ml.counterfactuals(d, Y, drivers, eff.assign(variable=drivers))
    cf.variable = cf.variable.map(ML_NAMES)
    save(cf.set_index("variable"), "table18_counterfactual", "Table 18. Counterfactual: +1 SD in each driver")
    pm = ml.predictive_models(d, Y, drivers)
    for tab_no, key, title, fig in [(19, "random_forest", "Random forest", "fig2_shap_rf.png"),
                                    (20, "gradient_boosting", "Gradient boosting", "fig3_shap_gb.png")]:
        res = pm[key]
        imp = res["importances"].rename(index=ML_NAMES)
        save(imp, f"table{tab_no}_{key}", f"Table {tab_no}. {title}: test R2 = {res['r2_test']:.3f}, "
             f"RMSE = {res['rmse_test']:.3f} (train R2 {res['r2_train']:.3f})")
        plt.figure()
        shap.summary_plot(res["shap_values"], res["X_test"].rename(columns=ML_NAMES), show=False)
        plt.title(f"SHAP Beeswarm Plot - {title}")
        plt.savefig(out / fig, dpi=200, bbox_inches="tight")
        plt.close("all")

    (out / "tables.md").write_text("\n".join(doc))
    print(f"wrote {len(list(out.glob('*.csv')))} tables, 2 figures and tables.md to {out}")


if __name__ == "__main__":
    main()
