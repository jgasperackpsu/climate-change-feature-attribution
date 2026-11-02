"""Step 3: turn BTS T-100 Segment (All Carriers) files into data/interim/bts.csv.

Usage:
    python src/load/load_bts.py <folder with bts_YYYY.csv> <path/to/state_crosswalk.csv> <out_dir>

For each state and year (by ORIGIN state, i.e. flights departing the state):
    air_pax_miles_millions      passengers x segment miles               -> household side
    air_freight_tonmiles_millions  freight (lb) / 2000 x segment miles    -> corporate side
    air_mail_tonmiles_millions     mail (lb) / 2000 x segment miles       -> kept separately
    air_departures              flights performed

Only domestic segments (REGION == 'D') are counted. EIA and EPA leave international aviation
fuel ("bunker fuels") out of state emissions, so counting international departures would
attribute travel to a state whose emissions don't include it.
Files are read in chunks, so 100+ MB years use little memory.
"""
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["PASSENGERS", "FREIGHT", "MAIL", "DISTANCE", "DEPARTURES_PERFORMED",
        "ORIGIN_STATE_FIPS", "REGION", "YEAR"]
CHUNK = 500_000


def read_year(path, keep_fips):
    totals = []
    years = set()
    for c in pd.read_csv(path, usecols=COLS, chunksize=CHUNK,
                         dtype={"ORIGIN_STATE_FIPS": str, "REGION": str}):
        years.update(c["YEAR"].dropna().astype(int).unique())
        c = c[(c["REGION"] == "D") & c["ORIGIN_STATE_FIPS"].str.zfill(2).isin(keep_fips)]
        c = c.assign(
            state_fips=c["ORIGIN_STATE_FIPS"].str.zfill(2),
            air_pax_miles_millions=c["PASSENGERS"] * c["DISTANCE"] / 1e6,
            air_freight_tonmiles_millions=c["FREIGHT"] / 2000 * c["DISTANCE"] / 1e6,
            air_mail_tonmiles_millions=c["MAIL"] / 2000 * c["DISTANCE"] / 1e6,
            air_departures=c["DEPARTURES_PERFORMED"],
        )
        totals.append(c.groupby("state_fips")[[
            "air_pax_miles_millions", "air_freight_tonmiles_millions",
            "air_mail_tonmiles_millions", "air_departures"]].sum())
    if len(years) != 1:
        raise ValueError(f"{path.name}: expected one YEAR value, found {sorted(years)}")
    out = pd.concat(totals).groupby(level=0).sum().reset_index()
    out["year"] = years.pop()
    return out


def main(folder, crosswalk_path, out_dir):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    keep = set(xw["fips_code"])  # 50 states; drops DC, Puerto Rico, territories

    frames = []
    for path in sorted(Path(folder).glob("bts_*.csv")):
        df = read_year(path, keep)
        yr = int(re.search(r"(\d{4})", path.name).group(1))
        if df["year"].iat[0] != yr:
            raise ValueError(f"{path.name}: file name says {yr} but YEAR column says {df['year'].iat[0]}")
        frames.append(df)
        print(f"{path.name}: {df.state_fips.nunique()} states, "
              f"{df.air_pax_miles_millions.sum()/1e3:,.0f} billion passenger-miles")

    w = pd.concat(frames, ignore_index=True)
    if w.duplicated(["state_fips", "year"]).any():
        raise ValueError("two files for the same year")
    tidy = (w.melt(id_vars=["state_fips", "year"], var_name="variable", value_name="value")
             .sort_values(["state_fips", "year", "variable"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "bts.csv", index=False)
    print(f"\nwrote {len(tidy):,} rows, years {w.year.min()}-{w.year.max()}, to {out / 'bts.csv'}")
    # a state with no domestic departures in a year would be missing; list any
    full = pd.MultiIndex.from_product([sorted(keep), sorted(w.year.unique())])
    gaps = full.difference(pd.MultiIndex.from_frame(w[["state_fips", "year"]]))
    if len(gaps):
        print(f"missing state-years: {list(gaps)[:10]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
