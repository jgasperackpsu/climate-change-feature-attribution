"""Outcome: turn EPA's All-State GHG Inventory workbook into data/interim/outcome/epa.csv.

Usage:
    python src/load/load_epa.py <path/to/AllStateGHGData...xlsx> <path/to/state_crosswalk.csv> <out_dir>

Source: Inventory of U.S. Greenhouse Gas Emissions and Sinks by State, 1990-2022 (EPA-430-R-24-006).
Reads the "Data by Econ Sect" tab. Units: Tg CO2e = million metric tons CO2e (AR5 100-yr GWPs).
Sinks are negative. The Econ Sect tab already leaves out memo items (bunker fuels, biomass CO2).

Variables written (state_fips, year, variable, value):
    ghg_gross_mmt                 all economic sectors except Land-use Change and Forestry   <- primary Y
    ghg_net_mmt                   gross + LULUCF (sinks negative)                           <- robustness
    ghg_<sector>_mmt              one per economic sector (residential, commercial, industry,
                                  transportation, electric_power, agriculture, lulucf)
    ghg_<gas>_mmt                 gross emissions by gas group: co2, ch4, n2o, fgas
    ghg_ippu_mmt                  IPCC sector "Industrial Processes and Product Use", all economic
                                  sectors (incl. ODS substitutes in residential/commercial/transport)
                                  <- gross minus this is the robustness Y without industrial processes
    ghg_iron_steel_mmt            IPPU iron and steel (incl. metallurgical coke) production; used to
                                  document the 2010 state-allocation break (see data_sources.md)

Rows for Territories, Federal Offshore ("FO") and "National" (not yet split to states) are dropped,
so the 50 states will sum to less than the national total; that gap is EPA's, not a loading error.
Within Transportation some mobile-source rows are negative by design (moved to Agriculture to
avoid double counting); summing by sector, as done here, handles them correctly.
"""
import re
import sys
from pathlib import Path

import pandas as pd


def slug(s):
    s = str(s).lower()
    if "lulucf" in s or "land-use" in s or "land use" in s:
        return "lulucf"            # "LULUCF Sector Net Total" in the workbook
    if "electric power" in s:
        return "electric_power"
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def gas_group(cat):
    c = str(cat).upper()
    if c.startswith("CO2") and "E" not in c[3:4]:
        return "co2"
    if c.startswith("CH4"):
        return "ch4"
    if c.startswith("N2O"):
        return "n2o"
    return "fgas"   # HFCs, PFCs, SF6, NF3, and ODS-substitute aggregates reported as "CO2e"


def main(xlsx, crosswalk_path, out_dir, start=1997):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    usps_to_fips = dict(zip(xw["postal_code"], xw["fips_code"]))

    xl = pd.ExcelFile(xlsx)
    sheet = next((s for s in xl.sheet_names if "econ" in s.lower()), None)
    if sheet is None:
        raise ValueError(f"no Econ Sect tab in {xl.sheet_names}")
    d = pd.read_excel(xl, sheet_name=sheet)
    # the README spells columns ECON_SECTOR / GEO_REF / GHG Category; the workbook uses lowercase
    d.columns = [str(c).strip().upper().replace(" ", "_") if not re.fullmatch(r"Y\d{4}", str(c).strip())
                 else str(c).strip() for c in d.columns]
    ycols = [c for c in d.columns if re.fullmatch(r"Y\d{4}", c)]
    gas_col = "GHG_CATEGORY"
    print(f"tab '{sheet}': {len(d):,} rows, {ycols[0]}-{ycols[-1]}")
    print("economic sectors: " + ", ".join(sorted(d["ECON_SECTOR"].dropna().unique())))

    d = d[d["GEO_REF"].isin(usps_to_fips)]
    d["sector"] = d["ECON_SECTOR"].map(slug)
    d["gas"] = d[gas_col].map(gas_group)
    d["ippu"] = d["SECTOR"].eq("Industrial Processes and Product Use")
    d["iron_steel"] = d["ippu"] & d["CATEGORY"].str.contains("Iron and Steel", na=False)
    long = d.melt(id_vars=["GEO_REF", "sector", "gas", "ippu", "iron_steel"], value_vars=ycols,
                  var_name="year", value_name="value")
    long["year"] = long["year"].str[1:].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce").fillna(0)
    long = long[long["year"] >= int(start)]
    long["state_fips"] = long["GEO_REF"].map(usps_to_fips)

    by_sector = long.pivot_table(index=["state_fips", "year"], columns="sector", values="value", aggfunc="sum", fill_value=0)
    by_sector.columns = [f"ghg_{c}_mmt" for c in by_sector.columns]
    lulucf = by_sector.get("ghg_lulucf_mmt", 0)
    out = by_sector.copy()
    out["ghg_net_mmt"] = by_sector.sum(axis=1)
    out["ghg_gross_mmt"] = out["ghg_net_mmt"] - lulucf

    gross = long[long["sector"] != "lulucf"]
    by_gas = gross.pivot_table(index=["state_fips", "year"], columns="gas", values="value", aggfunc="sum", fill_value=0)
    by_gas.columns = [f"ghg_{c}_mmt" for c in by_gas.columns]
    out = out.join(by_gas)
    for flag in ["ippu", "iron_steel"]:
        out[f"ghg_{flag}_mmt"] = gross[gross[flag]].groupby(["state_fips", "year"])["value"].sum()
        out[f"ghg_{flag}_mmt"] = out[f"ghg_{flag}_mmt"].fillna(0)

    gap = (out[[c for c in by_gas.columns]].sum(axis=1) - out["ghg_gross_mmt"]).abs().max()
    print(f"check: gas groups sum to gross within {gap:.4f} MMT")

    tidy = (out.reset_index().melt(id_vars=["state_fips", "year"], var_name="variable", value_name="value")
               .sort_values(["state_fips", "year", "variable"]))
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(o / "epa.csv", index=False)
    print(f"{tidy.state_fips.nunique()} states, {tidy.year.min()}-{tidy.year.max()}; wrote {len(tidy):,} rows to {o / 'epa.csv'}")
    tot = out["ghg_gross_mmt"].groupby(level="year").sum()
    print("50-state gross GHG, MMT CO2e:  " + "  ".join(f"{y}: {v:,.0f}" for y, v in tot.iloc[[0, -1]].items()))


if __name__ == "__main__":
    main(*sys.argv[1:4])