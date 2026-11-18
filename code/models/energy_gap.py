"""Phase 6 addendum (exploratory): the corporate - household gap in energy states vs. other states.

Usage:
    python code/models/energy_gap.py <split_dir> <models_dir> <findings_dir> [--define-only]
    e.g. python code/models/energy_gap.py data/split_data models findings/heterogeneity
--define-only writes the energy-state list and stops; it is run first, before any model is estimated.

Definition (fixed before estimation, one definition only, no alternative splits): energy states are the top
quintile, 10 of the 48 main-sample states, by mean fossil fuel production per person (SEDS coal + crude +
natural gas, MMBtu per person) over the train years 1997-2014. Saved to <findings_dir>/energy_states.csv.

Models (train years 1997-2014, contiguous 48 states), with E = 1 for an energy state:
    FE    fe_baseline.py main spec plus D_corporate x E and D_household x E (E itself is absorbed by the state
          effects); SEs clustered by state with the small-sample correction, t(47)
    DML   the dml.py main_best estimator (full_sample.dml_fit: FE and breaks partialled out, CV-chosen learners,
          5 state folds x 20 repetitions) with six treatments: D_corporate, D_household, D_corporate x E,
          D_household x E, W_D_corporate, W_D_household, each fit with the other five and the controls in X
Reported, for each model:
    other states    own gap = b_corp - b_hh
    energy states   own gap = (b_corp + d_corp) - (b_hh + d_hh)
    difference      d_corp - d_hh  (energy-state gap minus other-state gap)
plus each side's effect within each group. Two-sided and one-sided (> 0) p-values, clustered t(47).
Exploratory: the split was chosen after Phase 6 showed the corporate effect falling with the state's level.

Writes:
    <findings_dir>/energy_states.csv   48 states: mean fossil production per person, rank, energy flag
    <findings_dir>/energy_gap.csv      every comparison for FE and DML
    <models_dir>/heterogeneity/energy_gap/  fe_coefficients.csv, dml_nuisance.csv, dml_tuning.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ["data_preparation", "analysis", "models"]:
    sys.path.insert(0, str(ROOT / sub))
from build_panel import CONTROLS, LAGS, Y  # noqa: E402
from split_panel import load  # noqa: E402
import dml as D  # noqa: E402
import fe_baseline as FE  # noqa: E402
from full_sample import dml_fit  # noqa: E402

N_ENERGY = 10
DC, DH = "D_corporate", "D_household"
DCE, DHE = "D_corporate_x_energy", "D_household_x_energy"
COMBOS = {
    "corporate effect, other states": {DC: 1},
    "household effect, other states": {DH: 1},
    "own gap, other states": {DC: 1, DH: -1},
    "corporate effect, energy states": {DC: 1, DCE: 1},
    "household effect, energy states": {DH: 1, DHE: 1},
    "own gap, energy states": {DC: 1, DCE: 1, DH: -1, DHE: -1},
    "difference in own gap (energy - other)": {DCE: 1, DHE: -1},
}


def define(train, findings):
    m = train[train.in_main_sample == 1]
    s = (m.groupby("postal_code")["fossil_prod_per_capita_mmbtu"].mean().sort_values(ascending=False)
         .rename("mean_fossil_prod_mmbtu_per_person").reset_index())
    s["rank"] = range(1, len(s) + 1)
    s["energy_state"] = (s["rank"] <= N_ENERGY).astype(int)
    s.to_csv(findings / "energy_states.csv", index=False)
    print(f"energy states ({N_ENERGY} of {len(s)}): "
          + ", ".join(f"{r.postal_code} {r.mean_fossil_prod_mmbtu_per_person:,.0f}" for r in s.head(N_ENERGY).itertuples())
          + f"; 11th: {s.iloc[N_ENERGY].postal_code} {s.iloc[N_ENERGY].mean_fossil_prod_mmbtu_per_person:,.0f}")
    return set(s.loc[s.energy_state == 1, "postal_code"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("models_dir")
    ap.add_argument("findings_dir")
    ap.add_argument("--define-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    findings = Path(args.findings_dir)
    findings.mkdir(parents=True, exist_ok=True)
    train = load(args.split_dir)["train"]
    energy = define(train, findings)
    if args.define_only:
        return

    p = train.copy()
    p["energy_state"] = p.postal_code.isin(energy).astype(float)
    p[DCE] = p[DC] * p.energy_state
    p[DHE] = p[DH] * p.energy_state
    out = Path(args.models_dir) / "heterogeneity" / "energy_gap"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # FE: main spec + index x energy interactions
    spec = FE.spec(Y, [DC, DH, DCE, DHE] + LAGS + CONTROLS + FE.BREAKS)
    res, df = FE.fit(p, spec)
    for name, w in COMBOS.items():
        est, se = FE.combo(res, w)
        rows.append(dict(estimator="FE", comparison=name) | D.t_inf(est, se, df) | dict(n_obs=int(res.nobs)))
    pd.DataFrame(dict(term=res.params.index, coef=res.params.values, se=res.std_errors.values)).to_csv(
        out / "fe_coefficients.csv", index=False)

    # DML: main_best estimator with the two interactions as extra treatments
    tr = D.prepare(p, D.VERSIONS["main"], p)
    est, nuis, tuning, info = dml_fit(tr, Y, [DC, DH, DCE, DHE] + LAGS, CONTROLS)
    for name, w in COMBOS.items():
        rows.append(dict(estimator="DML main_best", comparison=name) | est(w) | dict(n_obs=info["n_obs"]))
    nuis.to_csv(out / "dml_nuisance.csv", index=False)
    tuning.assign(params=tuning.params.map(json.dumps)).to_csv(out / "dml_tuning.csv", index=False)

    t = pd.DataFrame(rows)
    t.to_csv(findings / "energy_gap.csv", index=False)
    print(t[["estimator", "comparison", "estimate", "se", "p_two_sided", "p_one_sided", "ci_low", "ci_high"]]
          .round(4).to_string(index=False))
    print(nuis.to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
