"""Step 4: turn BEA regional tables into data/interim/bea.csv.

Usage:
    python src/load/load_bea.py <folder containing the BEA CSVs> <path/to/state_crosswalk.csv> <out_dir>

Files it looks for (anywhere under the folder):
    SAPCE2__ALL_AREAS_*.csv   per capita personal consumption expenditures by type, dollars
    SAGDP2__ALL_AREAS_*.csv   GDP by state and industry, millions of current dollars

Variables written (state_fips, year, variable, value):
    pce_per_capita_usd               total PCE per person (SAPCE2 line 1)            household side
    pce_energy_goods_per_capita_usd  gasoline and other energy goods (line 11)
    pce_nonenergy_per_capita_usd     total minus energy goods                       household side
    gdp_mining_share                 mining, quarrying, oil and gas / all industries (line 6 / 1)
    gdp_manufacturing_share          manufacturing / all industries (line 12 / 1)
    gdp_industrial_share             (mining + manufacturing) / all industries      corporate side

Use the non-energy PCE in the household index. Gasoline and home energy purchases ARE fuel use,
which EIA turns into CO2 directly, so including them would build the outcome into the treatment.
Shares use current dollars because chained (real) dollars do not add across industries.
"""
import sys
from pathlib import Path

import pandas as pd

PCE_LINES = {1: "pce_per_capita_usd", 11: "pce_energy_goods_per_capita_usd"}
GDP_LINES = {1: "gdp_all", 6: "gdp_mining", 12: "gdp_manufacturing"}


def find(folder, pattern):
    hits = sorted(Path(folder).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {folder}")
    return hits[-1]


def read_bea(path, lines, keep_fips):
    d = pd.read_csv(path, dtype=str, encoding="latin-1")
    d = d[d["GeoFIPS"].notna() & d["LineCode"].notna()]          # drops BEA's footer notes
    d["state_fips"] = d["GeoFIPS"].str.strip().str.strip('"').str[:2]
    d = d[d["GeoFIPS"].str.strip().str.strip('"').str.endswith("000") & d["state_fips"].isin(keep_fips)]
    d["line"] = pd.to_numeric(d["LineCode"], errors="coerce")
    d = d[d["line"].isin(lines)]
    years = [c for c in d.columns if c.isdigit()]
    long = d.melt(id_vars=["state_fips", "line"], value_vars=years, var_name="year", value_name="value")
    # (D) withheld, (NA) not available, (NM) not meaningful -> missing
    long["value"] = pd.to_numeric(long["value"].str.strip(), errors="coerce")
    long["year"] = long["year"].astype(int)
    long["variable"] = long["line"].map(lines)
    w = long.pivot_table(index=["state_fips", "year"], columns="variable", values="value", aggfunc="first")
    print(f"{path.name}: {w.index.get_level_values(0).nunique()} states, "
          f"{w.index.get_level_values(1).min()}-{w.index.get_level_values(1).max()}, "
          f"{int(w.isna().sum().sum())} withheld/missing cells")
    return w


def main(folder, crosswalk_path, out_dir):
    keep = set(pd.read_csv(crosswalk_path, dtype=str)["fips_code"])

    pce = read_bea(find(folder, "SAPCE2__ALL_AREAS_*.csv"), PCE_LINES, keep)
    pce["pce_nonenergy_per_capita_usd"] = pce["pce_per_capita_usd"] - pce["pce_energy_goods_per_capita_usd"]

    gdp = read_bea(find(folder, "SAGDP2__ALL_AREAS_*.csv"), GDP_LINES, keep)
    shares = pd.DataFrame(index=gdp.index)
    shares["gdp_mining_share"] = gdp["gdp_mining"] / gdp["gdp_all"]
    shares["gdp_manufacturing_share"] = gdp["gdp_manufacturing"] / gdp["gdp_all"]
    shares["gdp_industrial_share"] = (gdp["gdp_mining"] + gdp["gdp_manufacturing"]) / gdp["gdp_all"]

    w = pce.join(shares, how="outer")
    gaps = w[w.isna().any(axis=1)]
    if len(gaps):
        print("state-years with a missing value (BEA withheld a component):")
        print(gaps[gaps.isna().any(axis=1)].isna().sum().loc[lambda s: s > 0].to_string())
        print(sorted(set(gaps.index.get_level_values(0))))

    tidy = (w.reset_index().melt(id_vars=["state_fips", "year"], var_name="variable", value_name="value")
             .sort_values(["state_fips", "year", "variable"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "bea.csv", index=False)
    print(f"\nwrote {len(tidy):,} rows to {out / 'bea.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
