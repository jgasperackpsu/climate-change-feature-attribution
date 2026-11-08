#!/usr/bin/env python3
"""
split_panel.py  (parent paper 3: 18 CEE countries, 2000-2024)

70 / 15 / 15 train / validation / test split of the paper's city-year panel, saved as
    TTV split/pp3_split/train/<name>.csv, validation/<name>.csv, test/<name>.csv  and  split_summary.csv

The data are a panel, so rows are not split at random (that would put the same country-year neighbours in train
and test). Two options:
    --by year    (default) chronological blocks of whole years, like the sequential split used for parent
                 paper 1: the earliest years train, the next validate, the latest test. With whole years the
                 shares cannot be exactly 70/15/15; the year counts closest to it are chosen and printed
                 (25 years -> 17 / 4 / 4 = 68 / 16 / 16%).
    --by entity  whole countries assigned at random (seeded) to train / validation / test, all years kept
                 together, so the shares are 70/15/15 of countries (18 -> 13 / 3 / 2).

Usage:
    python split_panel.py [--source ../../PP_data/pp3_data/pp3_panel_raw.csv] [--by year|entity] [--seed 340]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "PP_data" / "pp3_data" / "pp3_panel_raw.csv"
ENTITY, TIME = "country", "year"
RATIOS = (0.70, 0.15, 0.15)


def year_blocks(years, ratios=RATIOS):
    """Split a sorted list of years into three consecutive blocks whose sizes are closest to the ratios."""
    n = len(years)
    best = min(((a, b, n - a - b) for a, b in itertools.product(range(1, n - 1), repeat=2) if n - a - b >= 1),
               key=lambda s: sum((s[i] / n - ratios[i]) ** 2 for i in range(3)))
    a, b, _ = best
    return years[:a], years[a:a + b], years[a + b:]


def split(df, by, seed):
    if by == "year":
        tr, va, te = year_blocks(sorted(df[TIME].unique()))
        return df[df[TIME].isin(tr)], df[df[TIME].isin(va)], df[df[TIME].isin(te)]
    ids = np.array(sorted(df[ENTITY].unique()))
    np.random.default_rng(seed).shuffle(ids)
    n_tr = int(round(len(ids) * RATIOS[0]))
    n_va = int(round(len(ids) * RATIOS[1]))
    parts = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
    return tuple(df[df[ENTITY].isin(p)] for p in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output-dir", type=Path, default=HERE)
    ap.add_argument("--by", choices=["year", "entity"], default="year")
    ap.add_argument("--seed", type=int, default=340)
    a = ap.parse_args()
    if not a.source.exists():
        sys.exit(f"Error: '{a.source}' does not exist. Put the panel there first (see PP_data/*/README.md).")

    df = pd.read_csv(a.source).sort_values([ENTITY, TIME]).reset_index(drop=True)
    rows = []
    for name, part in zip(["train", "validation", "test"], split(df, a.by, a.seed)):
        out = a.output_dir / name / a.source.name
        out.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out, index=False)
        rows.append(dict(split=name, rows=len(part), share=len(part) / len(df), entities=part[ENTITY].nunique(),
                         first_year=part[TIME].min(), last_year=part[TIME].max(), file=str(out)))
        print(f"{name:<11} {len(part):>6,} rows ({len(part) / len(df):6.1%})  {part[ENTITY].nunique()} entities  "
              f"{part[TIME].min()}-{part[TIME].max()}  -> {out}")
    pd.DataFrame(rows).assign(method=a.by, seed=a.seed if a.by == "entity" else "").to_csv(
        a.output_dir / "split_summary.csv", index=False)


if __name__ == "__main__":
    main()
