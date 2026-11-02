"""Step 1: turn EIA SEDS (eia_data.csv) into data/interim/eia.csv.

Usage:
    python src/load/load_eia.py <path/to/eia_data.csv> <path/to/state_crosswalk.csv> <out_dir>

Output is tidy: state_fips, year, variable, value (one row per state-year-variable).
"""
import sys
from pathlib import Path

import pandas as pd

START, END = 1997, 2024

# Raw SEDS series to keep: MSN code -> readable name (units in the name).
RAW = {
    # population
    "TPOPP": "pop_thousands",
    # outcome (robustness) and sector CO2, million metric tons
    "TETCE": "co2_total_mmt",
    "TERCE": "co2_residential_mmt",
    "TECCE": "co2_commercial_mmt",
    "TEICE": "co2_industrial_mmt",
    "TEACE": "co2_transport_mmt",
    "TEEIE": "co2_elecpower_mmt",
    # transportation sector CO2 by fuel, million metric tons (Phase 3 household/corporate transport split;
    # descriptive only, never a treatment). The nine fuels sum to TEACE.
    "MMACE": "co2_trans_gasoline_mmt",
    "DMACE": "co2_trans_distillate_mmt",
    "JFACE": "co2_trans_jetfuel_mmt",
    "RFACE": "co2_trans_residual_mmt",
    "AVACE": "co2_trans_avgas_mmt",
    "HLACE": "co2_trans_hgl_mmt",
    "LUACE": "co2_trans_lubricants_mmt",
    "NNACE": "co2_trans_natgas_mmt",
    "CLACE": "co2_trans_coal_mmt",
    # electricity sales by sector, billion Btu (used to allocate power-sector CO2)
    "ESRCB": "elec_sales_res_bbtu",
    "ESCCB": "elec_sales_com_bbtu",
    "ESICB": "elec_sales_ind_bbtu",
    "ESACB": "elec_sales_trans_bbtu",
    "ESTCB": "elec_sales_total_bbtu",
    # electric power sector inputs, billion Btu (for grid shares)
    "CLEIB": "elecpower_coal_bbtu",
    "NGEIB": "elecpower_natgas_bbtu",
    "PAEIB": "elecpower_petrol_bbtu",
    "NUEGB": "elecpower_nuclear_bbtu",
    "TEEIB": "elecpower_total_bbtu",
    # fossil fuel production, billion Btu
    "CLPRB": "prod_coal_bbtu",
    "PAPRB": "prod_crude_bbtu",
    "NGMPB": "prod_natgas_bbtu",
    # real GDP (control), million chained 2017 dollars; starts 1997
    "GDPRX": "gdp_real_musd2017",
}


def main(seds_path, crosswalk_path, out_dir):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    usps_to_fips = dict(zip(xw["postal_code"], xw["fips_code"]))

    d = pd.read_csv(seds_path, usecols=["MSN", "StateCode", "Year", "Data"],
                    dtype={"MSN": str, "StateCode": str})
    d = d[d["MSN"].isin(RAW) & d["Year"].between(START, END)]
    # keep the 50 states only (drops DC, US total, X3/X5 offshore)
    d = d[d["StateCode"].isin(usps_to_fips)]
    d["Data"] = pd.to_numeric(d["Data"], errors="coerce")

    w = d.pivot_table(index=["StateCode", "Year"], columns="MSN", values="Data",
                      aggfunc="first").rename(columns=RAW)

    # derived variables
    w["co2_total_per_capita_t"] = w["co2_total_mmt"] * 1e6 / (w["pop_thousands"] * 1e3)
    w["grid_fossil_share"] = (w["elecpower_coal_bbtu"] + w["elecpower_natgas_bbtu"]
                              + w["elecpower_petrol_bbtu"]) / w["elecpower_total_bbtu"]
    w["grid_nuclear_share"] = w["elecpower_nuclear_bbtu"] / w["elecpower_total_bbtu"]
    w["fossil_prod_per_capita_mmbtu"] = (w["prod_coal_bbtu"].fillna(0) + w["prod_crude_bbtu"].fillna(0)
                                         + w["prod_natgas_bbtu"].fillna(0)) * 1e3 / (w["pop_thousands"] * 1e3)
    w["res_elec_per_capita_mmbtu"] = w["elec_sales_res_bbtu"] * 1e3 / (w["pop_thousands"] * 1e3)
    w["gdp_real_per_capita_usd2017"] = w["gdp_real_musd2017"] * 1e6 / (w["pop_thousands"] * 1e3)

    # household vs corporate CO2 split (descriptive decomposition, Phase 3)
    sales = w["elec_sales_total_bbtu"]
    res_share = w["elec_sales_res_bbtu"] / sales
    w["co2_household_mmt"] = w["co2_residential_mmt"] + res_share * w["co2_elecpower_mmt"]
    w["co2_corporate_mmt"] = (w["co2_commercial_mmt"] + w["co2_industrial_mmt"]
                              + (1 - res_share) * w["co2_elecpower_mmt"])
    # transport CO2 stays its own column; split it later with FHWA/BTS shares

    tidy = (w.reset_index()
              .melt(id_vars=["StateCode", "Year"], var_name="variable", value_name="value"))
    tidy.insert(0, "state_fips", tidy.pop("StateCode").map(usps_to_fips))
    tidy = tidy.rename(columns={"Year": "year"}).sort_values(["state_fips", "year", "variable"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "eia.csv", index=False)

    # quick check: states x years with data, per variable
    cover = tidy.dropna().groupby("variable").agg(states=("state_fips", "nunique"),
                                                  first=("year", "min"), last=("year", "max"))
    print(cover.to_string())
    print(f"\nwrote {len(tidy):,} rows to {out / 'eia.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
