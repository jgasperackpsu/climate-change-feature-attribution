"""Phase 5 addendum: full-sample confirmation of the FE main spec and DML main_best, 1997-2022.

Usage:
    python code/models/full_sample.py <panel.parquet> <split_dir> <models_dir> <findings_dir> [--check]
    e.g. python code/models/full_sample.py data/processed/panel.parquet data/split_data models findings/dml
--check first re-runs main_best on the train years with the helper below and compares it with the saved
models/dml/main_best/coefficients.csv, to confirm the helper is the same estimator.

Phases 4 and 5 estimate on the train years (1997-2014) only. This addendum re-estimates the two headline
models on every year of the corrected panel (contiguous 48 states, 1997-2022, 1,248 state-years), with
nothing else changed:
    FE    fe_baseline.py main spec: indices, spatial lags, controls, break dummies, state and year FE,
          SEs clustered by state with the small-sample correction, t(G - 1) = t(47)
    DML   dml.py main_best: state and year FE and break dummies partialled out, partially linear model per
          treatment (D_corporate, D_household, W_D_corporate, W_D_household), nuisance learner per nuisance
          chosen from OLS / Lasso / RF / XGBoost by 5-fold state-grouped CV MSE, 5 folds by state x 20
          repetitions sharing the same state-to-fold draws, clustered inference with G/(G - 1) and t(47)
Gaps: own = D_corporate - D_household; total = (D + W.D)_corporate - (D + W.D)_household. One-sided p tests
corporate > household (gaps) or effect > 0 (coefficients).

Selection note: in the main (1997-2014) results the validation and test years were never used to choose
DML learners; they only report held-out nuisance R2. In this full-sample run there is no held-out block:
the CV that chooses the learners runs on all 26 years, which is the only way to fit on all years.

Writes:
    <models_dir>/full_sample/  fe_main_coefficients.csv, dml_main_best_nuisance.csv, dml_main_best_tuning.csv
    <findings_dir>/full_sample.csv   FE and DML, 1997-2014 (saved Phase 4/5 results) and 1997-2022
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for sub in ["data_preparation", "analysis", "models"]:
    sys.path.insert(0, str(ROOT / sub))
from build_panel import CONTROLS, LAGS, Y  # noqa: E402
import dml as D  # noqa: E402
import fe_baseline as FE  # noqa: E402

TERMS = ["D_corporate", "D_household"]
GAPS = {"own": {"D_corporate": 1, "D_household": -1},
        "total": {"D_corporate": 1, "W_D_corporate": 1, "D_household": -1, "W_D_household": -1}}


def dml_fit(d, y, treats, x, breaks=True, n_rep=D.N_REP):
    """The dml.run estimator for a *_best run, on any sample: absorb FE (+ breaks), choose each nuisance
    learner by state-grouped CV MSE, fit one PLR per treatment on shared state-level fold draws.
    Returns (estimate(weights) -> t(G - 1) inference dict for sum_j w_j theta_j, nuisance table, tuning, info)."""
    a = D.absorb(d, [y] + treats + x, breaks)
    xsets = {t: [c for c in treats if c != t] + x for t in treats}
    targets = {}
    for t, xc in xsets.items():
        targets[f"l:{t}"] = (y, xc)
        targets[f"m:{t}"] = (t, xc)
    tuning, best = D.tune(a, targets, D.LEARNERS)
    used = {k: min(c.values(), key=lambda r: r["cv_mse"]) for k, c in best.items()}
    np.random.seed(D.SEED)
    models, first = {}, None
    for t, xc in xsets.items():
        data = DoubleMLData(a[[y, t] + xc + ["postal_code"]], y_col=y, d_cols=t, x_cols=xc, cluster_cols="postal_code")
        ml_l = D.make(used[f"l:{t}"]["learner"], used[f"l:{t}"]["params"])
        ml_m = D.make(used[f"m:{t}"]["learner"], used[f"m:{t}"]["params"])
        if first is None:
            m = DoubleMLPLR(data, ml_l, ml_m, n_folds=D.N_FOLDS, n_rep=n_rep)
            first = m
        else:
            m = DoubleMLPLR(data, ml_l, ml_m, n_folds=D.N_FOLDS, n_rep=n_rep, draw_sample_splitting=False)
            m.set_sample_splitting(first.smpls, first.smpls_cluster)
        m.fit(n_jobs_cv=D.N_FOLDS, store_predictions=True)
        models[t] = m
    clusters, n, g = a.postal_code.values, len(a), a.postal_code.nunique()

    def estimate(weights):
        theta, se, theta_r, _ = D.combine(models, weights, clusters, n)
        return D.t_inf(theta, se, g - 1) | dict(rep_min=float(theta_r.min()), rep_max=float(theta_r.max()))

    nuis = pd.DataFrame([dict(nuisance=k, learner=u["learner"], params=json.dumps(u["params"]), cv_r2=u["cv_r2"])
                         for k, u in used.items()])
    info = dict(n_obs=n, n_states=g, years=f"{int(a.year.min())}-{int(a.year.max())}")
    return estimate, nuis, tuning, info


def row(estimator, sample, term, r, n_obs):
    return dict(estimator=estimator, sample=sample, term=term, estimate=r["estimate"], se=r["se"], df=r["df"],
                p_two_sided=r["p_two_sided"], p_one_sided=r["p_one_sided"], ci_low=r["ci_low"],
                ci_high=r["ci_high"], n_obs=n_obs)


def fe_rows(p, sample):
    res, df = FE.fit(p, FE.SPECS["main"])
    n = int(res.nobs)
    out = []
    for t in TERMS:
        r = D.t_inf(float(res.params[t]), float(res.std_errors[t]), df)
        out.append(row("FE", sample, t, r, n))
    for name, (corp, hh) in FE.COMPARISONS.items():
        gt = FE.gap_test(res, df, corp, hh)
        out.append(row("FE", sample, f"{name} gap", dict(estimate=gt["gap"], se=gt["gap_se"], df=df,
                                                          p_two_sided=gt["p_two_sided"], p_one_sided=gt["p_one_sided"],
                                                          ci_low=gt["ci_low"], ci_high=gt["ci_high"]), n))
    return out, res


def saved_train_rows(models_dir, findings_base):
    """The 1997-2014 numbers as saved by fe_baseline.py and dml.py (not re-estimated here)."""
    out = []
    fe_c = pd.read_csv(Path(models_dir) / "fe_baseline/main/coefficients.csv").set_index("term")
    fe_m = pd.read_csv(Path(models_dir) / "fe_baseline/main/metrics.csv").iloc[0]
    for t in TERMS:
        r = fe_c.loc[t]
        out.append(row("FE", "1997-2014 (train)", t, dict(estimate=r.coef, se=r.se, df=r.df, p_two_sided=r.p,
                                                          p_one_sided=stats.t.sf(r.t, r.df), ci_low=r.ci_low,
                                                          ci_high=r.ci_high), int(fe_m.n_obs)))
    fe_g = pd.read_csv(Path(findings_base) / "gap_tests.csv")
    fe_g = fe_g[fe_g.spec == "main"].set_index("comparison")
    for name in ["own", "total"]:
        r = fe_g.loc[name]
        out.append(row("FE", "1997-2014 (train)", f"{name} gap", dict(estimate=r.gap, se=r.gap_se, df=r.df,
                                                                       p_two_sided=r.p_two_sided, p_one_sided=r.p_one_sided,
                                                                       ci_low=r.ci_low, ci_high=r.ci_high), int(r.n_obs)))
    dc = pd.read_csv(Path(models_dir) / "dml/main_best/coefficients.csv").set_index("term")
    dg = pd.read_csv(Path(models_dir) / "dml/main_best/gap_tests.csv").set_index("comparison")
    n = json.loads((Path(models_dir) / "dml/main_best/settings.json").read_text())["n_obs"]
    for t in TERMS:
        out.append(row("DML main_best", "1997-2014 (train)", t, dc.loc[t], n))
    for name in ["own", "total"]:
        out.append(row("DML main_best", "1997-2014 (train)", f"{name} gap", dg.loc[name], n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("split_dir")
    ap.add_argument("models_dir")
    ap.add_argument("findings_dir")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.models_dir) / "full_sample"
    out.mkdir(parents=True, exist_ok=True)
    findings = Path(args.findings_dir)

    if args.check:
        from split_panel import load
        tr = D.prepare(load(args.split_dir)["train"], D.VERSIONS["main"], load(args.split_dir)["train"])
        est, *_ = dml_fit(tr, Y, D.INDEX, CONTROLS)
        saved = pd.read_csv(Path(args.models_dir) / "dml/main_best/coefficients.csv").set_index("term")
        for t in D.INDEX:
            r = est({t: 1})
            print(f"check {t:>14}: helper {r['estimate']:.6f} (se {r['se']:.6f})  saved {saved.loc[t, 'estimate']:.6f} "
                  f"(se {saved.loc[t, 'se']:.6f})")

    p = pd.read_parquet(args.panel)
    p["state_fips"] = p["state_fips"].astype(str)
    full = D.prepare(p, D.VERSIONS["main"], p)
    assert len(full) == 1248 and full.postal_code.nunique() == 48, (len(full), full.postal_code.nunique())

    rows = saved_train_rows(args.models_dir, ROOT.parent / "findings/baseline")
    fe, res = fe_rows(p, "1997-2022 (full)")
    rows += fe
    pd.DataFrame(dict(term=res.params.index, coef=res.params.values, se=res.std_errors.values)).to_csv(
        out / "fe_main_coefficients.csv", index=False)

    est, nuis, tuning, info = dml_fit(full, Y, D.INDEX, CONTROLS)
    for t in TERMS:
        rows.append(row("DML main_best", "1997-2022 (full)", t, est({t: 1}), info["n_obs"]))
    for name, w in GAPS.items():
        rows.append(row("DML main_best", "1997-2022 (full)", f"{name} gap", est(w), info["n_obs"]))
    nuis.to_csv(out / "dml_main_best_nuisance.csv", index=False)
    tuning.assign(params=tuning.params.map(json.dumps)).to_csv(out / "dml_main_best_tuning.csv", index=False)

    t = pd.DataFrame(rows)
    t.to_csv(findings / "full_sample.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(nuis.to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
