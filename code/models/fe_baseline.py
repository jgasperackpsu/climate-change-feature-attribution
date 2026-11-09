"""Phase 4: two-way fixed-effects baseline, the benchmark the DML estimates are compared against.

Usage:
    python code/models/fe_baseline.py <split_dir> <models_dir> <findings_dir>
    e.g. python code/models/fe_baseline.py data/split_data models findings/baseline
<split_dir> is the output of code/data_preparation/split_panel.py.

Split (70/15/15, whole-year blocks): every spec is estimated on the train years only (1997-2014), so all
coefficients, standard errors and tests below come from train. Validation (2015-2018) and test (2019-2022)
are used only to score out-of-sample prediction: Y_hat = X b + state effect + year effect of the last
training year (year effects for unseen years cannot be estimated, so the latest one is carried forward);
the pooled spec uses X b with its constant. z_ drivers in the validation and test years are standardized
with the train mean and SD.

Model (main sample: contiguous 48 states, train years):
    Y_st = b1 D_corporate + b2 D_household + g1 W_D_corporate + g2 W_D_household
           + controls + brk_mn_2010 + brk_mi_2010 + brk_md_2011 + state FE + year FE + e_st

Inference: cluster-robust covariance by state with the small-sample correction (linearmodels PanelOLS,
debiased=True), and t-distribution p-values and CIs with G - 1 degrees of freedom (G = number of states).
linearmodels' own p-values use t(df_resid) ~ t(1164), which is effectively normal, so they are recomputed.
The Driscoll-Kraay spec (main_dk) uses a Bartlett kernel over years and t(T - 1) inference instead
(T = number of years), since its variance is estimated from T cross-sectional averages.

Hypothesis test: corporate minus household, as a linear combination r'b of the coefficients with
    se = sqrt(r' V r)    from the clustered covariance,
tested two-sided (gap = 0) and one-sided (gap > 0, the hypothesis that corporate is larger). Two
comparisons, decided before DML (findings/baseline/baseline_results.md):
    own     (primary)    D_corporate - D_household
    total   (secondary)  (D_corporate + W_D_corporate) - (D_household + W_D_household); with
                         row-standardized W, own + spatial-lag coefficient is the average total effect

Specifications:
    main      as above
    no_gdp    drops log_gdp_real_pc (GDP per person may be a mediator of industrial activity)
    no_wy     drops Wyoming (largest within-state swings in D_corporate)
    no_ippu   Y = y_log_gross_noippu_pc (gross excluding industrial processes), without the break
              dummies: all three breaks are in iron and steel, which this Y already excludes
    pooled    main regressors plus a constant, no fixed effects (shows how much between-state and
              common-year differences distort the comparison)
    no_lags   main without the two spatial lags (shows what ignoring spillovers changes)
    main_dk   main with Driscoll-Kraay standard errors (Bartlett kernel, bandwidth DK_BANDWIDTH):
              robust to correlation across states within a year and over nearby years
    drivers   the 8 individual drivers (each standardized to pooled SD 1 over the main sample) in place
              of the two indices, with the index spatial lags, controls and breaks; the per-driver
              within-state comparator for the SHAP ranking in ml_baseline.py

Standard-error comparison for the main model (Driscoll-Kraay at bandwidths 1-6, two-way clustering by
state and year, and state clustering) goes to <findings_dir>/se_comparison.csv.

Writes, under <models_dir>/fe_baseline/:
    <spec>/model.pkl          fitted PanelEffectsResults (linearmodels)
    <spec>/coefficients.csv   term, coef, se, t, df, p, ci_low, ci_high   (t with G - 1 df)
    <spec>/metrics.csv        one row: fit statistics, Wald test on the drivers, gap test, and
                              out-of-sample val_/test_ r2, rmse, mae
    <spec>/summary.txt        linearmodels summary (its p-values use t(df_resid), see above)
    coefficients.csv, metrics.csv, table.md    all specs combined
and under <findings_dir>/:
    gap_tests.csv             gap, se, CI and p-values for D_corporate - D_household, every spec
                              that has both indices (and the total-effect gap where lags are in)
    inference_comparison.csv  normal vs t(G - 1) p-values and CIs for every coefficient

Each break dummy is nonzero in a single state, so with state clustering its standard error rests on
one cluster and is not reliable; the dummies are nuisance controls. The joint Wald test therefore
covers only the driver terms (the indices or individual drivers, and the spatial lags when present).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preparation"))
from build_panel import CONTROLS, CORPORATE, HOUSEHOLD, LAGS, Y  # noqa: E402
from split_panel import load  # noqa: E402

TREATMENTS = ["D_corporate", "D_household"]
BREAKS = ["brk_mn_2010", "brk_mi_2010", "brk_md_2011"]
DRIVERS = TREATMENTS + LAGS
Y_NOIPPU = "y_log_gross_noippu_pc"

DK_BANDWIDTH = 3  # ~ 4 (T / 100)^(2/9) = 2.96 for T = 26 (Newey-West rule of thumb, rounded up)
NO_GDP = [c for c in CONTROLS if c != "log_gdp_real_pc"]
Z_DRIVERS = [f"z_{d}" for d in CORPORATE + HOUSEHOLD]


def spec(outcome, regressors, drop=(), fe=True, cov="clustered"):
    return dict(outcome=outcome, regressors=list(regressors), drop=list(drop), fe=fe, cov=cov)


SPECS = {
    "main": spec(Y, DRIVERS + CONTROLS + BREAKS),
    "no_gdp": spec(Y, DRIVERS + NO_GDP + BREAKS),
    "no_wy": spec(Y, DRIVERS + CONTROLS + BREAKS, drop=["WY"]),
    "no_ippu": spec(Y_NOIPPU, DRIVERS + CONTROLS),
    "pooled": spec(Y, ["const"] + DRIVERS + CONTROLS + BREAKS, fe=False),
    "no_lags": spec(Y, TREATMENTS + CONTROLS + BREAKS),
    "main_dk": spec(Y, DRIVERS + CONTROLS + BREAKS, cov="dk"),
    "drivers": spec(Y, Z_DRIVERS + LAGS + CONTROLS + BREAKS),
}

# comparison: {term: weight} for the corporate side and the household side
COMPARISONS = {
    "own": ({"D_corporate": 1}, {"D_household": 1}),
    "total": ({"D_corporate": 1, "W_D_corporate": 1}, {"D_household": 1, "W_D_household": 1}),
}


def sample(p, drop=(), ref=None):
    """Main sample with standardized drivers (z_<driver>: pooled SD 1 over the main sample of `ref`, which
    defaults to p itself; pass the train panel to standardize validation / test with train statistics)."""
    d = p[(p.in_main_sample == 1) & ~p.postal_code.isin(drop)].copy()
    r = d if ref is None else ref[(ref.in_main_sample == 1) & ~ref.postal_code.isin(drop)]
    for c in CORPORATE + HOUSEHOLD:
        d[f"z_{c}"] = (d[c] - r[c].mean()) / r[c].std()
    return d.assign(const=1.0).set_index(["postal_code", "year"])


def split_effects(res):
    """Split the combined two-way fixed effects into state and year parts (alternating means; exact in one
    pass on a balanced panel). Returns (state effects, year effects)."""
    e = res.estimated_effects.iloc[:, 0]
    ent, tim = e.index.get_level_values(0), e.index.get_level_values(1)
    a = pd.Series(0.0, index=ent.unique())
    g = pd.Series(0.0, index=tim.unique())
    for _ in range(50):
        a = (e - g.reindex(tim).values).groupby(ent).mean()
        g_new = (e - a.reindex(ent).values).groupby(tim).mean()
        if np.allclose(g_new, g):
            break
        g = g_new
    return a, g_new


def predict_scores(res, s, train, part):
    """Out-of-sample R2, RMSE, MAE of a train-fit spec on a later part of the split."""
    d = sample(part, s["drop"], ref=train)
    xb = d[s["regressors"]] @ res.params
    if s["fe"]:
        a, g = split_effects(res)
        xb = xb + d.index.get_level_values(0).map(a).values + g.loc[g.index.max()]
    y, e = d[s["outcome"]], d[s["outcome"]] - xb
    return dict(r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), rmse=np.sqrt((e ** 2).mean()),
                mae=e.abs().mean())


def fit(p, s, bandwidth=DK_BANDWIDTH):
    """Fit one spec; returns (results, inference df).
    cov: clustered (by state, df G - 1), dk (Driscoll-Kraay, df T - 1), twoway (state and year, df min(G, T) - 1)."""
    d = sample(p, s["drop"])
    mod = PanelOLS(d[s["outcome"]], d[s["regressors"]], entity_effects=s["fe"], time_effects=s["fe"],
                   drop_absorbed=False)
    if s["cov"] == "dk":
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth, debiased=True)
        return res, int(res.time_info["total"]) - 1
    if s["cov"] == "twoway":
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True, debiased=True)
        return res, min(int(res.entity_info["total"]), int(res.time_info["total"])) - 1
    res = mod.fit(cov_type="clustered", cluster_entity=True, debiased=True)
    return res, int(res.entity_info["total"]) - 1


def t_inference(est, se, df):
    """Two-sided t(df) statistic, p-value and 95% CI."""
    t = est / se
    crit = stats.t.ppf(0.975, df)
    return t, 2 * stats.t.sf(np.abs(t), df), est - crit * se, est + crit * se


def combo(res, weights):
    """Linear combination r'b of the coefficients and its clustered SE sqrt(r' V r)."""
    r = pd.Series(weights, dtype=float).reindex(res.params.index, fill_value=0)
    return float(r @ res.params), float(np.sqrt(r @ res.cov @ r))


def gap_test(res, df, corp, hh):
    """corporate side - household side with its clustered SE; t(df) inference, two- and one-sided."""
    b_c, se_c = combo(res, corp)
    b_h, se_h = combo(res, hh)
    gap, se = combo(res, {t: corp.get(t, 0) - hh.get(t, 0) for t in corp | hh})
    t, p2, lo, hi = t_inference(gap, se, df)
    # cross-check against linearmodels' Wald test: for one restriction, W = t^2
    lhs = " + ".join(corp) + " - " + " - ".join(hh)
    w = res.wald_test(formula=f"{lhs} = 0")
    assert np.isclose(w.stat, t ** 2), (w.stat, t ** 2)
    # covariance between the two sides, from var(gap) = var(c) + var(h) - 2 cov(c, h)
    cov_ch = (se_c ** 2 + se_h ** 2 - se ** 2) / 2
    return dict(corporate=b_c, corporate_se=se_c, household=b_h, household_se=se_h,
                gap=gap, gap_se=se, t=t, df=df,
                p_two_sided=p2, p_one_sided=stats.t.sf(t, df), ci_low=lo, ci_high=hi,
                p_two_sided_normal=2 * stats.norm.sf(abs(t)),
                corr_corp_hh=cov_ch / (se_c * se_h), ratio=b_c / b_h)


def stars(pv):
    return "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""


def main(split_dir, models_dir, findings_dir):
    parts = load(split_dir)
    p = parts["train"]
    root = Path(models_dir) / "fe_baseline"
    coefs, metrics, gaps, comparison = [], [], [], []

    for spec, sp in SPECS.items():
        outcome, regressors, drop = sp["outcome"], sp["regressors"], sp["drop"]
        res, df = fit(p, sp)
        g = int(res.entity_info["total"])
        t, pv, lo, hi = t_inference(res.params.values, res.std_errors.values, df)
        c = pd.DataFrame(dict(term=res.params.index, coef=res.params.values, se=res.std_errors.values,
                              t=t, df=df, p=pv, ci_low=lo, ci_high=hi))
        z = 1.959964
        comparison.append(c[["term", "coef", "se"]].assign(
            spec=spec, p_normal=2 * stats.norm.sf(np.abs(c.t)), ci_low_normal=c.coef - z * c.se,
            ci_high_normal=c.coef + z * c.se, df=df, p_t=c.p, ci_low_t=c.ci_low, ci_high_t=c.ci_high))

        drivers = [t for t in DRIVERS + Z_DRIVERS if t in regressors]
        joint = res.wald_test(formula=", ".join(f"{t} = 0" for t in drivers))
        k = len(drivers)
        tests = {name: gap_test(res, df, corp, hh) for name, (corp, hh) in COMPARISONS.items()
                 if set(corp) | set(hh) <= set(regressors)}
        gaps += [dict(spec=spec, comparison=name, outcome=outcome, n_obs=int(res.nobs), n_clusters=g) | gt
                 for name, gt in tests.items()]
        oos = {f"{k}_{stat}": v for k, part in [("val", parts["validation"]), ("test", parts["test"])]
               for stat, v in predict_scores(res, sp, p, part).items()}
        m = dict(spec=spec, outcome=outcome, n_obs=int(res.nobs), n_states=g,
                 n_years=int(res.time_info["total"]),
                 train_years=f"{p.year.min()}-{p.year.max()}", fixed_effects="state + year" if sp["fe"] else "none",
                 cov=f"Driscoll-Kraay (Bartlett, bw {DK_BANDWIDTH})" if sp["cov"] == "dk" else "clustered by state",
                 n_clusters=g if sp["cov"] == "clustered" else np.nan, inference_df=df,
                 dropped_states=",".join(drop),
                 # res.rsquared is the within R2 with fixed effects and the ordinary R2 without; the pooled
                 # model's rsquared_within is a squared correlation, not comparable, so it is left blank
                 r2_within=res.rsquared if sp["fe"] else np.nan, r2=res.rsquared,
                 rmse=np.sqrt(res.resid_ss / res.df_resid),
                 # joint test as F(k, G - 1), the small-cluster analogue of the chi2(k) Wald test
                 wald_drivers_f=joint.stat / k, wald_drivers_df1=k, wald_drivers_df2=df,
                 wald_drivers_p=stats.f.sf(joint.stat / k, k, df),
                 **{f"{name}_{k}": gt[src] for name, gt in tests.items()
                    for k, src in [("corporate", "corporate"), ("household", "household"), ("gap", "gap"),
                                   ("gap_se", "gap_se"), ("gap_ci_low", "ci_low"), ("gap_ci_high", "ci_high"),
                                   ("gap_p_two_sided", "p_two_sided"), ("gap_p_one_sided", "p_one_sided")]},
                 **oos)

        out = root / spec
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "model.pkl", "wb") as f:
            pickle.dump(res, f)
        c.to_csv(out / "coefficients.csv", index=False)
        pd.DataFrame([m]).to_csv(out / "metrics.csv", index=False)
        (out / "summary.txt").write_text(str(res.summary))
        coefs.append(c.assign(spec=spec))
        metrics.append(m)
        if "D_corporate" in res.params:
            print(f"{spec:>8}: n={m['n_obs']:,}  D_corporate {res.params['D_corporate']:+.4f}  "
                  f"D_household {res.params['D_household']:+.4f}  t({df})  -> {out}")
        else:
            print(f"{spec:>8}: n={m['n_obs']:,}  joint F({k}, {df}) = {m['wald_drivers_f']:.2f}  -> {out}")
        print(f"{'':>10}out of sample RMSE: val {m['val_rmse']:.4f}, test {m['test_rmse']:.4f}")
        for name, gt in tests.items():
            print(f"{'':>10}{name:>5} gap {gt['gap']:+.4f} [{gt['ci_low']:+.4f}, {gt['ci_high']:+.4f}] "
                  f"p={gt['p_two_sided']:.3f} (one-sided {gt['p_one_sided']:.3f})")

    bw = []
    for b in list(range(1, 7)) + ["twoway"]:
        sp = SPECS["main_dk"] | (dict(cov="twoway") if b == "twoway" else {})
        res, df = fit(p, sp, bandwidth=b if b != "twoway" else None)
        gt = {name: gap_test(res, df, corp, hh) for name, (corp, hh) in COMPARISONS.items()}
        bw.append(dict(se=f"Driscoll-Kraay, bandwidth {b}" if b != "twoway" else "clustered by state and year",
                       df=df,
                       **{f"{t}_se": res.std_errors[t] for t in DRIVERS},
                       **{f"{n}_gap_se": g["gap_se"] for n, g in gt.items()},
                       **{f"{n}_gap_p_two_sided": g["p_two_sided"] for n, g in gt.items()}))
    bw.append(dict(se="clustered by state (main)", df=metrics[0]["inference_df"],
                   **{f"{t}_se": coefs[0].set_index("term").se[t] for t in DRIVERS},
                   **{f"{n}_gap_se": metrics[0][f"{n}_gap_se"] for n in COMPARISONS},
                   **{f"{n}_gap_p_two_sided": metrics[0][f"{n}_gap_p_two_sided"] for n in COMPARISONS}))

    coefs = pd.concat(coefs)[["spec", "term", "coef", "se", "t", "df", "p", "ci_low", "ci_high"]]
    metrics = pd.DataFrame(metrics)
    coefs.to_csv(root / "coefficients.csv", index=False)
    metrics.to_csv(root / "metrics.csv", index=False)
    (root / "table.md").write_text(table(coefs, metrics))

    f = Path(findings_dir)
    f.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gaps).to_csv(f / "gap_tests.csv", index=False)
    pd.DataFrame(bw).to_csv(f / "se_comparison.csv", index=False)
    pd.concat(comparison)[["spec", "term", "coef", "se", "p_normal", "ci_low_normal", "ci_high_normal",
                           "df", "p_t", "ci_low_t", "ci_high_t"]].to_csv(f / "inference_comparison.csv", index=False)
    print(f"wrote {root / 'coefficients.csv'}, {root / 'metrics.csv'}, {root / 'table.md'}, "
          f"{f / 'gap_tests.csv'}, {f / 'inference_comparison.csv'}, {f / 'se_comparison.csv'}")


def table(coefs, metrics):
    specs = list(metrics.spec)
    c = coefs.set_index(["spec", "term"])
    terms = list(dict.fromkeys(coefs.term))
    terms = [t for t in terms if t != "const"] + (["const"] if "const" in terms else [])
    lines = ["# Phase 4: two-way fixed-effects baseline and comparison specifications", "",
             f"Estimated on the train years ({metrics.train_years.iloc[0].replace('-', '–')}) of the 70/15/15 split "
             "(data/split_data); validation and test RMSE are out-of-sample predictions.", "",
             "| | " + " | ".join(specs) + " |", "|---" * (len(specs) + 1) + "|"]
    for t in terms:
        cells = [f"{c.loc[(s, t), 'coef']:.4f}{stars(c.loc[(s, t), 'p'])}<br>({c.loc[(s, t), 'se']:.4f})"
                 if (s, t) in c.index else "" for s in specs]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    m = metrics.set_index("spec")

    def has(s, n):
        return pd.notna(m.loc[s].get(f"{n}_gap", np.nan))

    rows = []
    for name, label in [("own", "Own-effect gap (primary): D_corporate − D_household"),
                        ("total", "Total-effect gap: (D + W·D)_corporate − (D + W·D)_household")]:
        rows += [(f"**{label}**", lambda s, n=name: "" if not has(s, n) else
                  f"{m.loc[s, f'{n}_gap']:.4f}{stars(m.loc[s, f'{n}_gap_p_two_sided'])}<br>({m.loc[s, f'{n}_gap_se']:.4f})"),
                 ("95% CI", lambda s, n=name: "" if not has(s, n) else
                  f"[{m.loc[s, f'{n}_gap_ci_low']:.4f}, {m.loc[s, f'{n}_gap_ci_high']:.4f}]"),
                 ("p (two-sided / one-sided)", lambda s, n=name: "" if not has(s, n) else
                  f"{m.loc[s, f'{n}_gap_p_two_sided']:.3f} / {m.loc[s, f'{n}_gap_p_one_sided']:.3f}")]
    rows += [("Outcome", lambda s: m.loc[s, "outcome"]),
             ("Fixed effects", lambda s: m.loc[s, "fixed_effects"]),
             ("Standard errors", lambda s: m.loc[s, "cov"]),
             ("Dropped states", lambda s: m.loc[s, "dropped_states"] or "none"),
             ("Observations", lambda s: f"{m.loc[s, 'n_obs']:,}"),
             ("Inference df", lambda s: f"{m.loc[s, 'inference_df']}"),
             ("Within R² / model R²", lambda s: ("—" if pd.isna(m.loc[s, 'r2_within']) else f"{m.loc[s, 'r2_within']:.3f}")
                                              + f" / {m.loc[s, 'r2']:.3f}"),
             ("RMSE (train)", lambda s: f"{m.loc[s, 'rmse']:.4f}"),
             ("RMSE (validation / test, out of sample)", lambda s: f"{m.loc[s, 'val_rmse']:.4f} / "
                                                                  f"{m.loc[s, 'test_rmse']:.4f}"),
             ("Joint F: driver terms = 0 (p)", lambda s: f"{m.loc[s, 'wald_drivers_f']:.2f} "
                                                        f"({m.loc[s, 'wald_drivers_p']:.2g})")]
    for name, f in rows:
        lines.append(f"| {name} | " + " | ".join(f(s) for s in specs) + " |")
    lines += ["", "Standard errors in parentheses: clustered by state (with small-sample correction), except main_dk "
              f"(Driscoll-Kraay, Bartlett kernel, bandwidth {DK_BANDWIDTH}). p-values and CIs use a t-distribution "
              "with G − 1 degrees of freedom for clustered SEs and T − 1 for Driscoll-Kraay. "
              "* p<0.1, ** p<0.05, *** p<0.01. One-sided gap p tests D_corporate > D_household.",
              "",
              "z_ terms in the drivers column are the individual drivers standardized to pooled SD 1 over the main "
              "sample, so each coefficient is the change in Y per SD of that driver, within a state.",
              "",
              "Each break dummy is nonzero in a single state, so its clustered standard error is identified from one "
              "cluster and should not be interpreted; the dummies are nuisance controls. no_ippu leaves them out: all "
              "three breaks are in iron and steel, which that outcome already excludes.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main(*sys.argv[1:4])
