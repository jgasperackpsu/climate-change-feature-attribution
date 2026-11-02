"""Phase 3: split each state's EPA gross GHG into household (demand-side) and corporate (supply-side) parts.

Usage:
    python code/data_preparation/decompose_emissions.py <path/to/panel.csv> <out_csv>
    e.g. python code/data_preparation/decompose_emissions.py data/processed/panel.csv data/processed/decomposition.csv

Descriptive only: this is an accounting split of the outcome, not a model input. It uses fuel-based
shares (SEDS transport CO2 by fuel), which is fine here because nothing below becomes a treatment.

Allocation of the seven EPA economic sectors (gross = all sectors except LULUCF):
    residential       -> household
    commercial        -> corporate
    industry          -> corporate
    agriculture       -> corporate (livestock is a judgment call; main household index excludes cattle)
    electric power    -> household share = SEDS residential electricity sales / total sales; rest corporate
    transportation    -> household share = SEDS transport CO2 from (motor gasoline + passenger share of
                         jet fuel) / all transport CO2; rest (diesel, residual, natural gas, avgas, ...) corporate
Passenger share of jet fuel = passenger ton-miles / (passenger + freight + mail ton-miles), with a
passenger counted as 0.1 t including bags (BTS T-100, departing state). Freight is missing before 2003
(all-cargo undercount), so 1997-2002 use each state's 2003 share.

Writes a tidy file (state_fips, year, variable, value), 50 states x 1997-2022.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAX_TONS = 0.1   # tons per passenger including bags, for passenger ton-miles
TRANS_FUELS = ["gasoline", "distillate", "jetfuel", "residual", "avgas", "hgl", "lubricants", "natgas", "coal"]
HH_PARTS = ["ghg_hh_residential_mmt", "ghg_hh_electricity_mmt", "ghg_hh_transport_mmt"]
CORP_PARTS = ["ghg_corp_industry_mmt", "ghg_corp_commercial_mmt", "ghg_corp_agriculture_mmt",
              "ghg_corp_electricity_mmt", "ghg_corp_transport_mmt"]


def main(panel_path, out_csv):
    p = pd.read_csv(panel_path, dtype={"state_fips": str})
    d = p[["state_fips", "postal_code", "year"]].copy()

    d["elec_res_share"] = p["elec_sales_res_bbtu"] / p["elec_sales_total_bbtu"]

    pax_tm = p["air_pax_miles_millions"] * PAX_TONS
    jet_pax = pax_tm / (pax_tm + p["air_freight_tonmiles_millions"] + p["air_mail_tonmiles_millions"])
    d["jet_pax_share"] = jet_pax.groupby(p.state_fips).bfill()   # 1997-2002 take the 2003 share

    fuels = p[[f"co2_trans_{f}_mmt" for f in TRANS_FUELS]].sum(axis=1)
    d["transport_hh_share"] = (p["co2_trans_gasoline_mmt"] + d["jet_pax_share"] * p["co2_trans_jetfuel_mmt"]) / fuels

    d["ghg_hh_residential_mmt"] = p["ghg_residential_mmt"]
    d["ghg_hh_electricity_mmt"] = d["elec_res_share"] * p["ghg_electric_power_mmt"]
    d["ghg_hh_transport_mmt"] = d["transport_hh_share"] * p["ghg_transportation_mmt"]
    d["ghg_corp_industry_mmt"] = p["ghg_industry_mmt"]
    d["ghg_corp_commercial_mmt"] = p["ghg_commercial_mmt"]
    d["ghg_corp_agriculture_mmt"] = p["ghg_agriculture_mmt"]
    d["ghg_corp_electricity_mmt"] = (1 - d["elec_res_share"]) * p["ghg_electric_power_mmt"]
    d["ghg_corp_transport_mmt"] = (1 - d["transport_hh_share"]) * p["ghg_transportation_mmt"]
    d["ghg_household_mmt"] = d[HH_PARTS].sum(axis=1)
    d["ghg_corporate_mmt"] = d[CORP_PARTS].sum(axis=1)
    d["household_share"] = d["ghg_household_mmt"] / p["ghg_gross_mmt"]
    # robustness: agriculture (livestock) on the household side
    d["household_share_ag_hh"] = (d["ghg_household_mmt"] + p["ghg_agriculture_mmt"]) / p["ghg_gross_mmt"]

    # --- checks ---
    print("================ CHECKS ================")
    ok_grid = len(d) == 1300 and d.state_fips.nunique() == 50 and d.year.nunique() == 26
    print(f"rows {len(d):,} = {d.state_fips.nunique()} states x {d.year.nunique()} years: {'OK' if ok_grid else 'FAIL'}")
    gap = (d.ghg_household_mmt + d.ghg_corporate_mmt - p.ghg_gross_mmt).abs().max()
    print(f"household + corporate = EPA gross within {gap:.2e} MMT")
    tgap = (fuels - p.co2_transport_mmt).abs().max()
    print(f"SEDS transport fuels sum to SEDS transport CO2 within {tgap:.4f} MMT")
    shares = ["elec_res_share", "jet_pax_share", "transport_hh_share", "household_share"]
    for s in shares:
        print(f"  {s:20s} missing {d[s].isna().sum():3d}   range {d[s].min():.3f}-{d[s].max():.3f}")
    bad = d[shares].isna().any(axis=1) | ((d[shares] < 0) | (d[shares] > 1)).any(axis=1)
    print(f"rows with a share missing or outside [0, 1]: {bad.sum()}")
    # Delaware had no scheduled service 1997-1999: its jet share comes from 2003
    print("Delaware jet_pax_share 1997-2003: "
          + ", ".join(f"{v:.3f}" for v in d.loc[d.postal_code == "DE"].set_index("year").jet_pax_share.loc[1997:2003]))

    nat = d.groupby("year")[HH_PARTS + CORP_PARTS + ["ghg_household_mmt", "ghg_corporate_mmt"]].sum()
    nat["gross"] = p.groupby("year")["ghg_gross_mmt"].sum()
    print("\n50-state totals (MMT CO2e) and household share:")
    for y in (1997, 2005, 2019, 2020, 2022):
        r = nat.loc[y]
        print(f"  {y}: gross {r.gross:,.0f} = household {r.ghg_household_mmt:,.0f} + corporate {r.ghg_corporate_mmt:,.0f}"
              f"  -> household share {r.ghg_household_mmt / r.gross:.1%}")
    comp = (nat[HH_PARTS + CORP_PARTS].div(nat.gross, axis=0)).loc[[1997, 2022]].T
    comp.index = [c.replace("ghg_", "").replace("_mmt", "") for c in comp.index]
    print("\ncomponent shares of 50-state gross:")
    print((comp * 100).round(1).to_string())
    ag = (nat.ghg_household_mmt + d.groupby("year").ghg_corp_agriculture_mmt.sum()) / nat.gross
    print(f"robustness (agriculture -> household): household share {ag.loc[1997]:.1%} (1997), {ag.loc[2022]:.1%} (2022)")

    s22 = d[d.year == 2022].set_index("postal_code").household_share.sort_values()
    print("\n2022 household share, lowest: " + ", ".join(f"{k} {v:.0%}" for k, v in s22.head(5).items()))
    print("2022 household share, highest: " + ", ".join(f"{k} {v:.0%}" for k, v in s22.tail(5).items()))

    out_vars = shares[:3] + HH_PARTS + CORP_PARTS + ["ghg_household_mmt", "ghg_corporate_mmt",
                                                    "household_share", "household_share_ag_hh"]
    tidy = (d.melt(id_vars=["state_fips", "year"], value_vars=out_vars, var_name="variable", value_name="value")
             .sort_values(["state_fips", "year", "variable"]))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    print(f"\nwrote {len(tidy):,} rows to {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
