"""Step 7: turn USGS construction aggregates statistics into data/interim/usgs_minerals.csv.

Usage:
    python src/load/load_usgs_minerals.py <path/to/usgs_mineral_ext_1971_2023.xlsx> <path/to/state_crosswalk.csv> <out_dir>

The workbook (sheet Data_1971_2023) covers construction aggregates only: crushed stone and
construction sand and gravel. It does not contain cement or metal mining. Rows used are the
"State totals" (Table 2). Quantities are metric tons; values are current dollars.
    crushed_stone_t       crushed stone produced
    sand_gravel_t         construction sand and gravel produced
    aggregates_t          crushed stone + sand and gravel (missing if either is withheld)
'W' (withheld to protect company data) and '--' are treated as missing.
"""
import sys
from pathlib import Path

import pandas as pd

COMMODITIES = {"Stone, crushed": "crushed_stone_t",
               "Sand and gravel, construction": "sand_gravel_t"}


def main(xlsx, crosswalk_path, out_dir, start=1997):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    name_to_fips = dict(zip(xw["state_name"], xw["fips_code"]))

    d = pd.read_excel(xlsx, sheet_name="Data_1971_2023")
    d = d[(d["Data Description"] == "State totals") & d["Commodity"].isin(COMMODITIES)
          & (d["Year"] >= int(start))]
    d["state_fips"] = d["State Coverage"].str.strip().map(name_to_fips)
    d = d.dropna(subset=["state_fips"])
    d["variable"] = d["Commodity"].map(COMMODITIES)
    d["value"] = pd.to_numeric(d["Quantity"], errors="coerce")
    withheld = d["value"].isna()
    if withheld.any():
        print(f"withheld ('W') or blank values since {start}: {int(withheld.sum())}")
        print(d[withheld].groupby("variable")["state_fips"].nunique().rename("states affected").to_string())

    w = d.pivot_table(index=["state_fips", "Year"], columns="variable", values="value",
                      aggfunc="first", dropna=False)
    w["aggregates_t"] = w["crushed_stone_t"] + w["sand_gravel_t"]
    tidy = (w.reset_index().rename(columns={"Year": "year"})
             .melt(id_vars=["state_fips", "year"], var_name="variable", value_name="value")
             .sort_values(["state_fips", "year", "variable"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "usgs_minerals.csv", index=False)
    print(f"\n{tidy.state_fips.nunique()} states, {tidy.year.min()}-{tidy.year.max()}; "
          f"wrote {len(tidy):,} rows to {out / 'usgs_minerals.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
