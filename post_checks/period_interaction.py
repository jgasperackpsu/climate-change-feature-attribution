"""Post-check 1 (exploratory): did the index effects change between 1997-2014 and 2015-2022?

Usage (from the repository root):
    python post_checks/period_interaction.py data/processed/panel.parquet post_checks

Purpose: the own-effect gap is not significant on 1997-2014 but is on 1997-2022 (findings/dml/full_sample.md).
This tests whether the effects themselves differ between the two periods.

Sample: all years of the corrected panel, contiguous 48 states, 1997-2022 (1,248 state-years).
late = 1 for 2015-2022 (the validation and test years of the 70/15/15 split), 0 for 1997-2014.
Models, each with D_corporate x late and D_household x late added (late itself is absorbed by the year effects):
    FE    fe_baseline.py main spec (indices, spatial lags, controls, break dummies, state and year FE),
          SEs clustered by state with the small-sample correction, t(47)
    DML   the main_best estimator (code/models/full_sample.py dml_fit: FE and breaks partialled out,
          CV-chosen learners, 5 state folds x 20 repetitions, clustered t(47)) with six treatments:
          D_corporate, D_household, D_corporate x late, D_household x late, W_D_corporate, W_D_household
Reported: each index's effect in 1997-2014 and in 2015-2022 and the change; the own gap in each period and
its change. Two-sided and one-sided (> 0) p-values. Exploratory: the split was chosen after seeing that the
result depends on the period. The spatial lags are not interacted.

Writes: post_checks/period_interaction.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "code"
for sub in ["data_preparation", "analysis", "models"]:
    sys.path.insert(0, str(ROOT / sub))
from build_panel import CONTROLS, LAGS, Y  # noqa: E402
import dml as D  # noqa: E402
import fe_baseline as FE  # noqa: E402
from full_sample import dml_fit  # noqa: E402

LATE_START = 2015
DC, DH = "D_corporate", "D_household"
DCL, DHL = "D_corporate_x_late", "D_household_x_late"
COMBOS = {
    "corporate effect, 1997-2014": {DC: 1},
    "corporate effect, 2015-2022": {DC: 1, DCL: 1},
    "change in corporate effect (late - early)": {DCL: 1},
    "household effect, 1997-2014": {DH: 1},
    "household effect, 2015-2022": {DH: 1, DHL: 1},
    "change in household effect (late - early)": {DHL: 1},
    "own gap, 1997-2014": {DC: 1, DH: -1},
    "own gap, 2015-2022": {DC: 1, DCL: 1, DH: -1, DHL: -1},
    "change in own gap (late - early)": {DCL: 1, DHL: -1},
}


def main(panel_path, out_dir):
    t0 = time.time()
    p = pd.read_parquet(panel_path)
    p["late"] = (p.year >= LATE_START).astype(float)
    p[DCL] = p[DC] * p.late
    p[DHL] = p[DH] * p.late
    rows = []

    res, df = FE.fit(p, FE.spec(Y, [DC, DH, DCL, DHL] + LAGS + CONTROLS + FE.BREAKS))
    for name, w in COMBOS.items():
        est, se = FE.combo(res, w)
        rows.append(dict(estimator="FE", comparison=name) | D.t_inf(est, se, df) | dict(n_obs=int(res.nobs)))

    full = D.prepare(p, D.VERSIONS["main"], p)
    assert len(full) == 1248, len(full)
    est, nuis, _, info = dml_fit(full, Y, [DC, DH, DCL, DHL] + LAGS, CONTROLS)
    for name, w in COMBOS.items():
        rows.append(dict(estimator="DML main_best", comparison=name) | est(w) | dict(n_obs=info["n_obs"]))

    t = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "period_interaction.csv", index=False)
    print(t[["estimator", "comparison", "estimate", "se", "p_two_sided", "p_one_sided", "ci_low", "ci_high"]]
          .round(4).to_string(index=False))
    print(nuis.to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:3])
