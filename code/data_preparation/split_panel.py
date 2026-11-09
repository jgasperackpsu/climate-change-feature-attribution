"""70 / 15 / 15 train / validation / test split of the state-year panel.

Usage:
    python code/data_preparation/split_panel.py <panel.csv> <split_dir>
    e.g. python code/data_preparation/split_panel.py data/processed/panel.csv data/split_data

The data are a panel, so rows are not split at random: that would put a state's neighbouring years in both
train and test. The split is chronological, in blocks of whole years: the earliest years train, the next
validate, the latest test. With whole years the shares cannot be exactly 70/15/15; the year counts closest
to it are used (26 years, 1997-2022 -> 18 / 4 / 4 = 69 / 15 / 15%: train 1997-2014, validation 2015-2018,
test 2019-2022). Every state and every column is kept; the models select the main sample themselves.

Writes:
    <split_dir>/train/panel.csv, validation/panel.csv, test/panel.csv
    <split_dir>/split_summary.csv    rows, share, states and years of each part
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

RATIOS = (0.70, 0.15, 0.15)
PARTS = ["train", "validation", "test"]


def year_blocks(years, ratios=RATIOS):
    """Split sorted years into three consecutive blocks whose sizes are closest to the ratios."""
    n = len(years)
    a, b, _ = min(((a, b, n - a - b) for a, b in itertools.product(range(1, n - 1), repeat=2) if n - a - b >= 1),
                  key=lambda s: sum((s[i] / n - ratios[i]) ** 2 for i in range(3)))
    return years[:a], years[a:a + b], years[a + b:]


def load(split_dir):
    """{part: DataFrame} for the three saved parts."""
    return {k: pd.read_csv(Path(split_dir) / k / "panel.csv", dtype={"state_fips": str}) for k in PARTS}


def main(panel_path, split_dir):
    p = pd.read_csv(panel_path, dtype={"state_fips": str}).sort_values(["postal_code", "year"])
    out = Path(split_dir)
    rows = []
    for name, yrs in zip(PARTS, year_blocks(sorted(p.year.unique()))):
        part = p[p.year.isin(yrs)]
        f = out / name / "panel.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(f, index=False)
        main_n = int((part.in_main_sample == 1).sum())
        rows.append(dict(split=name, rows=len(part), share=len(part) / len(p), main_sample_rows=main_n,
                         states=part.postal_code.nunique(), years=len(yrs), first_year=min(yrs),
                         last_year=max(yrs), file=str(f), method="year blocks"))
        print(f"{name:<11} {len(part):>5} rows ({len(part) / len(p):5.1%}), {main_n} main-sample, "
              f"{min(yrs)}-{max(yrs)} -> {f}")
    pd.DataFrame(rows).to_csv(out / "split_summary.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
