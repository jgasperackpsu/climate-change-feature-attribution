"""Join every tidy interim file into the state x year analysis panel and build the driver indices.

Usage:
    python code/data_preparation/build_panel.py <interim_dir> <path/to/state_crosswalk.csv> <path/to/state_neighbors.csv> <out_dir>
    e.g. python code/data_preparation/build_panel.py data/interim data/state_crosswalk.csv data/state_neighbors.csv data/processed

state_neighbors.csv is the row-standardized queen-contiguity edge list from build_spatial_weights.py.

Reads every CSV under <interim_dir> (recursively). Each must have exactly the columns
state_fips, year, variable, value; a variable name that appears in two files is an error.

Writes:
    <out_dir>/panel.csv             one row per state-year, 50 states x 1997-2022 (1,300 rows)
    <out_dir>/data_dictionary.csv   variable, description, unit, source, role

Data fixes applied here (documented in data_sources.md):
    Indiana 2022 VMT   one-year +22% spike that reverses in 2023 -> mean of 2021 and 2023, flag vmt_imputed
    Delaware 1997-1999 no scheduled air service, so BTS has no rows -> air variables set to 0
    BTS air freight    undercounts all-cargo carriers before 2003 -> set to missing before 2003
    EPA iron & steel   state-allocation breaks (MN, MI from 2010; MD 2011-2012) -> Y left as is,
                       flag y_break plus one step dummy per break (brk_*) for the model; robustness
                       Y y_log_gross_noippu_pc drops all industrial process emissions

Main sample (in_main_sample = 1): the contiguous 48 states, 1997-2022 (1,248 rows). nClimDiv has
no Hawaii and no Alaska degree days, so HDD/CDD are missing there.

Naming: y_ = outcomes, d_ = treatment components, D_ = treatment indices, W_ = spatial lags of
the indices, brk_ = EPA break step dummies (extra controls); the controls (X in DML notation) are log_pop, log_gdp_real_pc, hdd_k, cdd_k.

Indices (z = standardized with the mean and SD of the main sample; AK and HI are scored on the same scale):
    D_corporate = z(mean of z(industrial GDP share), z(log1p fossil production per person),
                    z(grid fossil share), z(log1p sand & gravel per person))
    D_household = z(mean of z(log VMT per person), z(log1p air passenger-miles per person),
                    z(log non-energy PCE per person), z(log residential electricity per person))
Robustness variants:
    D_household_lvstk   household index plus cattle per person (livestock on the household side)
    D_corporate_lvstk   corporate index plus cattle per person (livestock on the corporate side)
    D_household_noelec  household index without residential electricity (it overlaps the outcome:
                        electric power emissions ~ electricity sold x grid fossil share)
    D_corporate_freight corporate index plus air freight, 2003+ only
Spatial lags W_<index> = row-standardized neighbor average of each index in the same year;
Alaska and Hawaii have no land neighbors, so their lag is 0 (has_neighbors = 0).
No fuel-consumption variable enters any index: EIA/EPA compute CO2 from fuel use.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START, END = 1997, 2022
COLS = ["state_fips", "year", "variable", "value"]

CORPORATE = ["d_industrial_share", "d_fossil_prod_pc", "d_grid_fossil_share", "d_sand_gravel_pc"]
HOUSEHOLD = ["d_vmt_pc", "d_air_pax_pc", "d_pce_nonenergy_pc", "d_res_elec_pc"]
CONTROLS = ["log_pop", "log_gdp_real_pc", "hdd_k", "cdd_k"]
Y = "y_log_gross_pc"

# EPA iron and steel breaks (Phase 3 review): dummy -> (postal code, first year, last year)
BREAKS = {
    "brk_mn_2010": ("MN", 2010, END),
    "brk_mi_2010": ("MI", 2010, END),
    "brk_md_2011": ("MD", 2011, 2012),
}

# variable: (description, unit, role)
DERIVED = {
    "in_main_sample": ("1 if in the main model sample (contiguous 48 states; HDD/CDD available)", "0/1", "flag"),
    "pop": ("population", "persons", "control input"),
    "ghg_gross_pc_t": ("gross GHG (excl. LULUCF) per person", "t CO2e / person", "outcome"),
    "ghg_net_pc_t": ("net GHG (gross + LULUCF) per person", "t CO2e / person", "outcome (robustness)"),
    "y_log_gross_pc": ("log gross GHG per person: PRIMARY Y", "log t CO2e / person", "outcome"),
    "y_log_co2_energy_pc": ("log EIA energy-related CO2 per person", "log t CO2 / person", "outcome (robustness)"),
    "ghg_gross_noippu_pc_t": ("gross GHG excluding industrial processes and product use, per person", "t CO2e / person", "outcome (robustness)"),
    "y_log_gross_noippu_pc": ("log gross GHG excl. industrial processes per person (iron & steel break robustness)", "log t CO2e / person", "outcome (robustness)"),
    "y_break": ("1 in state-years after an EPA iron & steel allocation break (MN, MI 2010+; MD 2011-2012)", "0/1", "flag"),
    "brk_mn_2010": ("step dummy: Minnesota 2010+ (EPA iron & steel allocation break)", "0/1", "break dummy"),
    "brk_mi_2010": ("step dummy: Michigan 2010+ (EPA iron & steel allocation break)", "0/1", "break dummy"),
    "brk_md_2011": ("step dummy: Maryland 2011-2012 (EPA iron & steel spike before Sparrows Point closure)", "0/1", "break dummy"),
    "d_industrial_share": ("(mining + manufacturing) / all-industry GDP, current $", "share", "corporate driver"),
    "d_fossil_prod_pc": ("log1p fossil fuel production per person (coal + crude + gas)", "log1p MMBtu / person", "corporate driver"),
    "d_grid_fossil_share": ("fossil share of electric power sector energy input", "share", "corporate driver"),
    "d_sand_gravel_pc": ("log1p construction sand & gravel production per person", "log1p t / person", "corporate driver"),
    "d_air_freight_pc": ("log1p domestic air freight ton-miles per person; missing before 2003", "log1p ton-mi / person", "corporate driver (robustness)"),
    "d_vmt_pc": ("log vehicle-miles traveled per person", "log mi / person", "household driver"),
    "d_air_pax_pc": ("log1p domestic air passenger-miles per person (departing state)", "log1p mi / person", "household driver"),
    "d_pce_nonenergy_pc": ("log non-energy personal consumption expenditures per person, current $", "log $ / person", "household driver"),
    "d_res_elec_pc": ("log residential electricity purchases per person; overlaps electric power emissions", "log MMBtu / person", "household driver"),
    "d_cattle_pc": ("log1p cattle and calves per person; little within-state variation", "log1p head / person", "driver (robustness only)"),
    "log_pop": ("log population (= log density given state FE)", "log persons", "control"),
    "log_gdp_real_pc": ("log real GDP per person", "log 2017 $ / person", "control"),
    "hdd_k": ("heating degree days / 1,000", "thousand degree-days F", "control"),
    "cdd_k": ("cooling degree days / 1,000", "thousand degree-days F", "control"),
    "vmt_imputed": ("1 if VMT was imputed (Indiana 2022)", "0/1", "flag"),
    "D_corporate": ("corporate supply-side driver index", "SD", "treatment"),
    "D_household": ("household demand-side driver index", "SD", "treatment"),
    "D_household_lvstk": ("household index plus cattle (robustness)", "SD", "treatment (robustness)"),
    "D_corporate_lvstk": ("corporate index plus cattle (robustness)", "SD", "treatment (robustness)"),
    "D_household_noelec": ("household index without residential electricity (robustness)", "SD", "treatment (robustness)"),
    "D_corporate_freight": ("corporate index plus air freight, 2003+ (robustness)", "SD", "treatment (robustness)"),
    "has_neighbors": ("1 if the state has queen-contiguity neighbors (0 for AK, HI)", "0/1", "flag"),
}
VARIANTS = {
    "D_corporate": CORPORATE,
    "D_household": HOUSEHOLD,
    "D_household_lvstk": HOUSEHOLD + ["d_cattle_pc"],
    "D_corporate_lvstk": CORPORATE + ["d_cattle_pc"],
    "D_household_noelec": [c for c in HOUSEHOLD if c != "d_res_elec_pc"],
    "D_corporate_freight": CORPORATE + ["d_air_freight_pc"],
}
for _d in VARIANTS:
    DERIVED[f"W_{_d}"] = (f"spatial lag: neighbor average of {_d} (0 for AK, HI)", "SD", "spatial control")
LAGS = ["W_D_corporate", "W_D_household"]


def read_interim(interim_dir):
    frames, owner = [], {}
    for f in sorted(Path(interim_dir).rglob("*.csv")):
        d = pd.read_csv(f, dtype={"state_fips": str})
        if list(d.columns) != COLS:
            raise ValueError(f"{f}: columns {list(d.columns)}, expected {COLS}")
        for v in d["variable"].unique():
            if v in owner:
                raise ValueError(f"variable '{v}' is in both {owner[v]} and {f}")
            owner[v] = f
        if d.duplicated(["state_fips", "year", "variable"]).any():
            raise ValueError(f"{f}: duplicate state_fips-year-variable rows")
        frames.append(d)
        print(f"read {f}: {len(d):,} rows, {d.variable.nunique()} variables")
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(index=["state_fips", "year"], columns="variable", values="value")
    wide.columns.name = None
    return wide, owner


def z(s, ref):
    """Standardize s with the mean and SD of s on the reference rows (boolean mask)."""
    return (s - s[ref].mean()) / s[ref].std()


def index(df, cols, ref):
    """Standardized mean of standardized components; missing if any component is missing."""
    return z(df[cols].apply(z, ref=ref).mean(axis=1, skipna=False), ref)


def within(df, col):
    """Two-way (state and year) demeaned series, for correlations net of the fixed effects."""
    s = df[col]
    return s - s.groupby(df.state_fips).transform("mean") - s.groupby(df.year).transform("mean") + s.mean()


def spatial_lag(p, edges, col):
    """Row-standardized neighbor average of col in the same year; 0 where a state has no neighbors."""
    nb = edges.merge(p[["state_fips", "year", col]].rename(columns={"state_fips": "neighbor_fips"}), on="neighbor_fips")
    nb["wx"] = nb["weight"] * nb[col]
    # min_count=1 keeps a lag missing when every neighbor value is missing (e.g. freight index before 2003)
    lag = nb.groupby(["state_fips", "year"])["wx"].sum(min_count=1)
    if nb[col].isna().any():
        # a lag built from only some neighbors would not be a neighbor average; require all of them
        lag[nb[col].isna().groupby([nb.state_fips, nb.year]).any()] = np.nan
    out = p.set_index(["state_fips", "year"]).index.map(lag)
    return pd.Series(np.where(p["has_neighbors"] == 1, out, 0.0), index=p.index)


def main(interim_dir, crosswalk_path, neighbors_path, out_dir):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    wide, owner = read_interim(interim_dir)

    # --- fixes that need years outside 1997-2022, applied before trimming ---
    ind = "18"
    vmt_cols = ["vmt_total_millions", "vmt_rural_millions", "vmt_urban_millions"]
    before = wide.loc[(ind, 2022), "vmt_total_millions"]
    for c in vmt_cols:
        wide.loc[(ind, 2022), c] = wide.loc[[(ind, 2021), (ind, 2023)], c].mean()
    wide.loc[(ind, 2022), "urban_vmt_share"] = wide.loc[(ind, 2022), "vmt_urban_millions"] / wide.loc[(ind, 2022), "vmt_total_millions"]
    wide["vmt_imputed"] = 0
    wide.loc[(ind, 2022), "vmt_imputed"] = 1
    print(f"fix: Indiana 2022 VMT {before:,.0f} -> {wide.loc[(ind, 2022), 'vmt_total_millions']:,.0f} million")

    # --- full 50-state x 1997-2022 grid ---
    grid = pd.MultiIndex.from_product([xw["fips_code"], range(START, END + 1)], names=["state_fips", "year"])
    p = wide.reindex(grid).reset_index()
    p = p.merge(xw[["fips_code", "postal_code", "state_name"]].rename(columns={"fips_code": "state_fips"}), on="state_fips")

    air = ["air_departures", "air_pax_miles_millions", "air_freight_tonmiles_millions", "air_mail_tonmiles_millions"]
    de = (p.state_fips == "10") & p.year.between(1997, 1999)
    if p.loc[de, air].notna().any().any():
        raise ValueError("Delaware 1997-1999 air rows are no longer empty; recheck the BTS fix")
    p.loc[de, air] = 0.0
    if p[air].isna().any().any():
        raise ValueError("unexpected missing BTS rows besides Delaware 1997-1999")
    print("fix: Delaware 1997-1999 air variables set to 0 (no scheduled service)")
    p.loc[p.year < 2003, "air_freight_tonmiles_millions"] = np.nan
    print("fix: air freight set to missing before 2003 (all-cargo undercount)")

    # --- derived variables ---
    p["pop"] = p["pop_thousands"] * 1e3
    p["ghg_gross_pc_t"] = p["ghg_gross_mmt"] * 1e6 / p["pop"]
    p["ghg_net_pc_t"] = p["ghg_net_mmt"] * 1e6 / p["pop"]
    p["y_log_gross_pc"] = np.log(p["ghg_gross_pc_t"])
    p["y_log_co2_energy_pc"] = np.log(p["co2_total_mmt"] * 1e6 / p["pop"])
    p["ghg_gross_noippu_pc_t"] = (p["ghg_gross_mmt"] - p["ghg_ippu_mmt"]) * 1e6 / p["pop"]
    p["y_log_gross_noippu_pc"] = np.log(p["ghg_gross_noippu_pc_t"])

    for b, (st, y0, y1) in BREAKS.items():
        p[b] = ((p.postal_code == st) & p.year.between(y0, y1)).astype(int)
    p["y_break"] = p[list(BREAKS)].max(axis=1)

    p["d_industrial_share"] = p["gdp_industrial_share"]
    p["d_fossil_prod_pc"] = np.log1p(p["fossil_prod_per_capita_mmbtu"])
    p["d_grid_fossil_share"] = p["grid_fossil_share"]
    p["d_sand_gravel_pc"] = np.log1p(p["sand_gravel_t"] / p["pop"])
    p["d_air_freight_pc"] = np.log1p(p["air_freight_tonmiles_millions"] * 1e6 / p["pop"])
    p["d_vmt_pc"] = np.log(p["vmt_total_millions"] * 1e6 / p["pop"])
    p["d_air_pax_pc"] = np.log1p(p["air_pax_miles_millions"] * 1e6 / p["pop"])
    p["d_pce_nonenergy_pc"] = np.log(p["pce_nonenergy_per_capita_usd"])
    p["d_res_elec_pc"] = np.log(p["res_elec_per_capita_mmbtu"])
    p["d_cattle_pc"] = np.log1p(p["cattle_head"] / p["pop"])

    p["log_pop"] = np.log(p["pop"])
    p["log_gdp_real_pc"] = np.log(p["gdp_real_per_capita_usd2017"])
    p["hdd_k"] = p["hdd"] / 1e3
    p["cdd_k"] = p["cdd"] / 1e3

    model_inputs = [Y] + [c for cols in VARIANTS.values() for c in cols if c != "d_air_freight_pc"] + CONTROLS
    p["in_main_sample"] = p[model_inputs].notna().all(axis=1).astype(int)
    main_rows = p.in_main_sample == 1

    late = p.year >= 2003
    for d, cols in VARIANTS.items():
        rows = late if d == "D_corporate_freight" else pd.Series(True, index=p.index)
        p.loc[rows, d] = index(p[rows], cols, main_rows[rows])
        p.loc[~rows, d] = np.nan

    edges = pd.read_csv(neighbors_path, dtype={"state_fips": str, "neighbor_fips": str})
    p["has_neighbors"] = p.state_fips.isin(edges.state_fips).astype(int)
    for d in VARIANTS:
        p[f"W_{d}"] = spatial_lag(p, edges, d)
        if d == "D_corporate_freight":
            p.loc[~late, f"W_{d}"] = np.nan

    # --- column order and write ---
    ids = ["state_fips", "postal_code", "state_name", "year"]
    derived = list(DERIVED)
    raw = sorted(c for c in p.columns if c not in ids + derived)
    p = p[ids + derived + raw].sort_values(["state_fips", "year"]).reset_index(drop=True)

    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    p.to_csv(o / "panel.csv", index=False)
    dd = [dict(variable=k, description=v[0], unit=v[1], source="derived (build_panel.py)", role=v[2]) for k, v in DERIVED.items()]
    dd += [dict(variable=c, description="as loaded", unit="see name", source=str(Path(owner[c]).relative_to(interim_dir)), role="input")
           for c in raw]
    pd.DataFrame(dd).to_csv(o / "data_dictionary.csv", index=False)
    print(f"wrote {o / 'panel.csv'} ({len(p):,} rows x {p.shape[1]} cols) and {o / 'data_dictionary.csv'}")

    report(p)


def report(p):
    print("\n================ REPORT ================")
    n_states, years = p.state_fips.nunique(), p.year.nunique()
    print(f"rows {len(p):,} = {n_states} states x {years} years: {'OK' if len(p) == 50 * 26 == n_states * years else 'FAIL'}")

    epa22 = p.loc[p.year == 2022, "ghg_gross_mmt"].sum()
    print(f"check: 50-state gross GHG 2022 = {epa22:,.0f} MMT (expected ~6,308; EPA national 6,344)")
    net_gap = (p.ghg_gross_mmt + p.ghg_lulucf_mmt - p.ghg_net_mmt).abs().max()
    sectors = [c for c in p.columns if c.startswith("ghg_") and c.endswith("_mmt")
               and c not in ("ghg_gross_mmt", "ghg_net_mmt", "ghg_lulucf_mmt", "ghg_co2_mmt", "ghg_ch4_mmt", "ghg_n2o_mmt", "ghg_fgas_mmt",
                             "ghg_ippu_mmt", "ghg_iron_steel_mmt")]
    sec_gap = (p[sectors].sum(axis=1) - p.ghg_gross_mmt).abs().max()
    print(f"check: gross + LULUCF = net within {net_gap:.2e}; sectors sum to gross within {sec_gap:.2e} MMT")
    print(f"check: 50-state population 2022 = {p.loc[p.year == 2022, 'pop'].sum() / 1e6:,.1f} million")

    print("\nEPA iron & steel breaks (MMT; log-point change in Y vs. Y without industrial processes):")
    for b, (st, y0, y1) in BREAKS.items():
        s = p[p.postal_code == st].set_index("year")
        yrs = [y0 - 1, y0] if y1 == END else list(range(y0 - 1, y1 + 2))
        iron = " -> ".join(f"{s.loc[y, 'ghg_iron_steel_mmt']:.2f}" for y in yrs)
        dy = " / ".join(f"{s.loc[y, Y] - s.loc[y - 1, Y]:+.3f}" for y in yrs[1:])
        dn = " / ".join(f"{s.loc[y, 'y_log_gross_noippu_pc'] - s.loc[y - 1, 'y_log_gross_noippu_pc']:+.3f}" for y in yrs[1:])
        print(f"  {b}: {st} {y0}-{y1} ({int(s[b].sum())} yrs)  iron & steel {'/'.join(map(str, yrs))}: {iron}  "
              f"dY {dy}  dY_noippu {dn}")
    print(f"  y_break = 1 in {int(p.y_break.sum())} state-years; "
          f"IPPU share of 50-state gross 2022: {p.loc[p.year == 2022, 'ghg_ippu_mmt'].sum() / epa22:.1%}")

    model = [Y] + CORPORATE + HOUSEHOLD + CONTROLS + ["D_corporate", "D_household"] + LAGS
    miss = p[model].isna().groupby(p.postal_code).sum()
    miss = miss.loc[miss.sum(axis=1) > 0, miss.sum() > 0]
    print("\nmissing values in model variables, by state:")
    print(miss.to_string() if len(miss) else "  none")
    c = p[p.in_main_sample == 1]
    out = sorted(p.loc[p.in_main_sample == 0, "postal_code"].unique())
    ok = len(c) == 48 * 26 and p.loc[c.index, model].notna().all().all()
    print(f"main sample: {len(c):,} rows, {c.state_fips.nunique()} states (excluded: {', '.join(out)}): "
          f"{'OK' if ok else 'FAIL'}")
    stats = c[["D_corporate", "D_household"]].agg(["mean", "std"]).round(6)
    print(f"check: indices standardized on main sample (mean, SD): "
          + "; ".join(f"{k} {stats.loc['mean', k]:.3f}, {stats.loc['std', k]:.3f}" for k in stats))
    print(f"controls in thousands of degree-days: hdd_k {c.hdd_k.min():.2f}-{c.hdd_k.max():.2f}, "
          f"cdd_k {c.cdd_k.min():.2f}-{c.cdd_k.max():.2f}")

    print("\ncattle_head coverage 1997-2022: "
          f"{p.cattle_head.notna().sum()} of {len(p)}; hogs {p.hogs_head.notna().sum()}, sheep {p.sheep_head.notna().sum()} "
          "(hogs and sheep are not used)")

    rows = []
    for x in CORPORATE + HOUSEHOLD + ["d_cattle_pc"] + list(VARIANTS):
        cx = c[c[x].notna()]
        rows.append(dict(variable=x, r_pooled=cx[x].corr(cx[Y]), r_within=within(cx, x).corr(within(cx, Y)),
                         sd_within=within(cx, x).std() / cx[x].std(), n=len(cx)))
    print("\ncorrelation with Y (log gross GHG per person), main sample; sd_within = share of SD left after FE:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nindex pairs (main sample): pooled / within state+year correlation")
    for a, b in [("D_corporate", "D_household"), ("D_corporate", "D_household_lvstk"),
                 ("D_corporate_lvstk", "D_household"), ("D_corporate", "D_household_noelec"),
                 ("D_household", "D_household_lvstk"), ("D_household", "D_household_noelec"),
                 ("d_res_elec_pc", "d_grid_fossil_share")]:
        print(f"  {a:>20} vs {b:<20} {c[a].corr(c[b]):6.3f} / {within(c, a).corr(within(c, b)):6.3f}")

    print("\nspatial lags: states with no neighbors (lag = 0): "
          + ", ".join(sorted(p.loc[p.has_neighbors == 0, "postal_code"].unique()))
          + f"; in main sample: {c.loc[c.has_neighbors == 0, 'postal_code'].nunique()}")
    for d in ["D_corporate", "D_household"]:
        print(f"  corr({d}, W_{d}): pooled {c[d].corr(c['W_' + d]):.3f}, "
              f"within {within(c, d).corr(within(c, 'W_' + d)):.3f}")
    # hand check: Maine's only neighbor is New Hampshire, so W for ME must equal NH's index
    me = p[p.postal_code == "ME"].set_index("year")["W_D_corporate"]
    nh = p[p.postal_code == "NH"].set_index("year")["D_corporate"]
    print(f"  check: W_D_corporate(ME) == D_corporate(NH): {np.allclose(me, nh)}")

    top = p.loc[p.year == 2022].nlargest(3, "ghg_gross_pc_t")[["postal_code", "ghg_gross_pc_t"]]
    print("\nhighest gross GHG per person, 2022: " + ", ".join(f"{r.postal_code} {r.ghg_gross_pc_t:.1f} t" for r in top.itertuples()))


if __name__ == "__main__":
    main(*sys.argv[1:5])
