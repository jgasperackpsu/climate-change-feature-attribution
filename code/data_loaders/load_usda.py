"""Step 5: turn the USDA NASS Quick Stats animals & products bulk file into data/interim/usda.csv.

Usage:
    python src/load/load_usda.py <path/to/qs.animals_products_YYYYMMDD.txt.gz> <path/to/state_crosswalk.csv> <out_dir>

Keeps annual state inventories from the SURVEY program (Census of Agriculture is only every 5 years):
    cattle_head        CATTLE, INCL CALVES - INVENTORY   as of Jan 1
    milk_cows_head     CATTLE, COWS, MILK - INVENTORY    as of Jan 1   (dairy: highest methane per head)
    hogs_head          HOGS - INVENTORY                  as of Dec 1
    sheep_head         SHEEP, INCL LAMBS - INVENTORY     as of Jan 1

Two quirks of the bulk file are handled:
  * the header joins two names ("ASD_DESCOUNTY_ANSI" = ASD_DESC + COUNTY_ANSI), so it is one name
    short of the data; read naively, every column after it shifts by one
  * values carry thousands separators ("1,234,000") and codes such as (D) withheld, (Z), (NA)
The 450+ MB file is read in chunks.
"""
import gzip
import sys
from pathlib import Path

import pandas as pd

TARGETS = {
    "CATTLE, INCL CALVES - INVENTORY": ("cattle_head", "FIRST OF JAN"),
    "CATTLE, COWS, MILK - INVENTORY": ("milk_cows_head", "FIRST OF JAN"),
    "HOGS - INVENTORY": ("hogs_head", "FIRST OF DEC"),
    "SHEEP, INCL LAMBS - INVENTORY": ("sheep_head", "FIRST OF JAN"),
}
USE = ["SOURCE_DESC", "SHORT_DESC", "DOMAIN_DESC", "AGG_LEVEL_DESC", "STATE_FIPS_CODE",
       "YEAR", "REFERENCE_PERIOD_DESC", "VALUE"]


def fixed_header(path):
    with gzip.open(path, "rt", encoding="latin-1") as f:
        names = f.readline().rstrip("\n").split("\t")
    out = []
    for n in names:
        out.extend(["ASD_DESC", "COUNTY_ANSI"] if n == "ASD_DESCOUNTY_ANSI" else [n])
    return out


def main(gz_path, crosswalk_path, out_dir, start=1997):
    keep = set(pd.read_csv(crosswalk_path, dtype=str)["fips_code"])
    names = fixed_header(gz_path)

    parts, seen = [], []
    reader = pd.read_csv(gz_path, sep="\t", header=None, skiprows=1, names=names, usecols=USE,
                         dtype=str, quoting=3, index_col=False, chunksize=1_000_000,
                         encoding="latin-1", on_bad_lines="warn")
    for c in reader:
        c = c[(c["SOURCE_DESC"] == "SURVEY") & (c["AGG_LEVEL_DESC"] == "STATE")
              & (c["DOMAIN_DESC"] == "TOTAL") & c["SHORT_DESC"].isin(TARGETS)]
        if c.empty:
            continue
        seen.append(c.groupby(["SHORT_DESC", "REFERENCE_PERIOD_DESC"]).size())
        want = c["SHORT_DESC"].map(lambda s: TARGETS[s][1])
        parts.append(c[c["REFERENCE_PERIOD_DESC"] == want])

    if seen:  # what the file actually has, so a renamed period is easy to spot
        print("reference periods found:\n" + pd.concat(seen).groupby(level=[0, 1]).sum().to_string() + "\n")
    d = pd.concat(parts)
    d["state_fips"] = d["STATE_FIPS_CODE"].str.strip().str.zfill(2)
    d = d[d["state_fips"].isin(keep)]
    d["year"] = d["YEAR"].astype(int)
    d = d[d["year"] >= int(start)]
    d["variable"] = d["SHORT_DESC"].map(lambda s: TARGETS[s][0])
    d["value"] = pd.to_numeric(d["VALUE"].str.replace(",", "").str.strip(), errors="coerce")

    dup = d.duplicated(["state_fips", "year", "variable"], keep=False)
    if dup.any():  # revised re-loads of the same estimate: keep the largest-loaded (last) row
        d = d.drop_duplicates(["state_fips", "year", "variable"], keep="last")

    tidy = d[["state_fips", "year", "variable", "value"]].sort_values(["state_fips", "year", "variable"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "usda.csv", index=False)

    cover = tidy.dropna().groupby("variable").agg(states=("state_fips", "nunique"),
                                                  first=("year", "min"), last=("year", "max"),
                                                  state_years=("value", "size"))
    print(cover.to_string())
    n_years = tidy["year"].nunique()
    print(f"\n(full coverage = 50 states x {n_years} years = {50 * n_years} state-years)")
    print(f"withheld/blank values: {int(tidy['value'].isna().sum())}")
    print(f"wrote {len(tidy):,} rows to {out / 'usda.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
