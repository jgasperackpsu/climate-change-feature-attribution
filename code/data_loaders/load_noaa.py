"""Step 6: turn NOAA nClimDiv statewide files into data/interim/noaa.csv.

Usage:
    python src/load/load_noaa.py <folder with climdiv-*st-* files> <path/to/state_crosswalk.csv> <out_dir>

Files used (the "st" = statewide versions; county "cy" and division "dv" files are ignored):
    climdiv-tmpcst-*   average temperature, deg F      -> annual mean
    climdiv-hddcst-*   heating degree days             -> annual sum
    climdiv-cddcst-*   cooling degree days             -> annual sum
    climdiv-pcpnst-*   precipitation, inches           -> annual sum

Line format: an ID of 10 digits (state 3, division 1, element 2, year 4) and 12 monthly values.
NOAA state codes are NOT FIPS (contiguous states 001-048 alphabetically, Alaska 050; no Hawaii);
they are mapped through the crosswalk's noaa_code column. Codes 101+ are regions and the U.S.,
and are dropped. A year with any missing month (e.g. the current, partial year) is left out.
Also writes temp_anomaly_f: each state's annual temperature minus its own 1991-2020 mean.
"""
import sys
from pathlib import Path

import pandas as pd

FILES = {  # file tag -> (variable, how to combine months, missing-value code)
    "tmpcst": ("temp_mean_f", "mean", -99.99),
    "hddcst": ("hdd", "sum", -9999.0),
    "cddcst": ("cdd", "sum", -9999.0),
    "pcpnst": ("precip_in", "sum", -9.99),
}


def read_climdiv(path, agg, missing):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 13 or len(parts[0]) != 10:
                continue
            ident, months = parts[0], [float(v) for v in parts[1:]]
            if ident[3] != "0":          # statewide records only
                continue
            if any(abs(v - missing) < 1e-6 for v in months):
                continue                  # incomplete year
            value = sum(months) / 12 if agg == "mean" else sum(months)
            rows.append({"noaa_code": ident[:3], "year": int(ident[6:10]), "value": value})
    return pd.DataFrame(rows)


def main(folder, crosswalk_path, out_dir, start=1997):
    xw = pd.read_csv(crosswalk_path, dtype=str).dropna(subset=["noaa_code"])
    noaa_to_fips = dict(zip(xw["noaa_code"].str.zfill(3), xw["fips_code"]))

    frames = []
    for tag, (var, agg, missing) in FILES.items():
        hits = sorted(Path(folder).glob(f"climdiv-{tag}-*"))
        if not hits:
            print(f"skipping {tag}: no climdiv-{tag}-* file found")
            continue
        df = read_climdiv(hits[-1], agg, missing)
        df["state_fips"] = df["noaa_code"].map(noaa_to_fips)
        df = df.dropna(subset=["state_fips"])        # drops regions and national codes
        df["variable"] = var
        frames.append(df)
        print(f"{hits[-1].name}: {df.state_fips.nunique()} states, {df.year.min()}-{df.year.max()}")

    d = pd.concat(frames, ignore_index=True)

    # anomaly vs each state's own 1991-2020 normal (NOAA's current baseline)
    t = d[d.variable == "temp_mean_f"].copy()
    base = t[t.year.between(1991, 2020)].groupby("state_fips")["value"].mean()
    t["value"] = t["value"] - t["state_fips"].map(base)
    t["variable"] = "temp_anomaly_f"
    d = pd.concat([d, t], ignore_index=True)

    d = d[d.year >= int(start)]
    tidy = d[["state_fips", "year", "variable", "value"]].sort_values(["state_fips", "year", "variable"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "noaa.csv", index=False)
    missing_states = sorted(set(xw["state_name"]) - set(xw.loc[xw.fips_code.isin(tidy.state_fips), "state_name"]))
    print(f"\nwrote {len(tidy):,} rows, {tidy.year.min()}-{tidy.year.max()}, to {out / 'noaa.csv'}")
    print(f"states with no data: {missing_states or 'none'} (Hawaii is not in nClimDiv)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
