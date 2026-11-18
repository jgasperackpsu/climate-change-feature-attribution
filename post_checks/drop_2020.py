"""Post-check 2: the 1997-2022 main DML without 2020, so COVID cannot drive the full-sample gap.

Usage (from the repository root):
    python post_checks/drop_2020.py data/processed/panel.parquet post_checks

Sample: the corrected panel, contiguous 48 states, 1997-2022 minus 2020 (25 years, 1,200 state-years).
Model: the main_best estimator exactly as in findings/dml/full_sample.md (code/models/full_sample.py
dml_fit: state and year FE and break dummies partialled out, CV-chosen learners, 5 state folds x 20
repetitions, clustered t(47)). The FE main spec on the same sample is reported alongside for reference.
Reported: D_corporate, D_household, own gap and total gap, with two- and one-sided p, next to the
full-sample (with 2020) results read from findings/dml/full_sample.csv.

Writes: post_checks/drop_2020.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ["data_preparation", "analysis", "models"]:
    sys.path.insert(0, str(ROOT / "code" / sub))
from build_panel import CONTROLS, Y  # noqa: E402
import dml as D  # noqa: E402
from full_sample import GAPS, TERMS, dml_fit, fe_rows, row  # noqa: E402

SAMPLE = "1997-2022 without 2020"


def main(panel_path, out_dir):
    t0 = time.time()
    p = pd.read_parquet(panel_path)
    p = p[p.year != 2020].reset_index(drop=True)
    full = D.prepare(p, D.VERSIONS["main"], p)
    assert len(full) == 1200 and 2020 not in set(full.year), len(full)

    prev = pd.read_csv(ROOT / "findings/dml/full_sample.csv")
    rows = prev[prev["sample"] == "1997-2022 (full)"].to_dict("records")
    fe, _ = fe_rows(p, SAMPLE)
    rows += fe
    est, nuis, _, info = dml_fit(full, Y, D.INDEX, CONTROLS)
    for t in TERMS:
        rows.append(row("DML main_best", SAMPLE, t, est({t: 1}), info["n_obs"]))
    for name, w in GAPS.items():
        rows.append(row("DML main_best", SAMPLE, f"{name} gap", est(w), info["n_obs"]))

    t = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "drop_2020.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(nuis.to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:3])
