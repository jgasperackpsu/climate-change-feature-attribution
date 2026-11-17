"""Land area: turn the Census TIGER/Line state shapefile into data/interim/shared/tiger.csv.

Usage:
    python code/data_loaders/load_tiger.py <path/to/tl_2022_us_state.zip> <path/to/state_crosswalk.csv> <out_dir>
    e.g. python code/data_loaders/load_tiger.py data/raw/shared/TIGER/tl_2022_us_state.zip data/state_crosswalk.csv data/interim/shared

Variable written (state_fips, year, variable, value):
    land_area_sqmi    ALAND (square meters of land, inland and coastal water excluded) / 2,589,988.11

TIGER gives one set of 2022 boundaries, so the area is repeated for every year 1997-2022; state land
area changes by far less than 1% over the period. build_panel.py uses it for per-square-mile scaling
(population density). The same shapefile gives the spatial weights (build_spatial_weights.py).
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

START, END = 1997, 2022
SQM_PER_SQMI = 2_589_988.11


def main(shp, crosswalk_path, out_dir):
    keep = set(pd.read_csv(crosswalk_path, dtype=str)["fips_code"])
    g = pd.DataFrame(gpd.read_file(shp, ignore_geometry=True))
    g = g[g["STATEFP"].isin(keep)]
    if len(g) != 50:
        raise ValueError(f"expected 50 states in {shp}, found {len(g)}")
    area = g.set_index("STATEFP")["ALAND"].astype(float) / SQM_PER_SQMI
    tidy = pd.DataFrame([dict(state_fips=s, year=y, variable="land_area_sqmi", value=a)
                         for s, a in area.items() for y in range(START, END + 1)]).sort_values(["state_fips", "year"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "tiger.csv", index=False)
    print(f"land area: {area.min():,.0f} (smallest) to {area.max():,.0f} sq mi; "
          f"50-state total {area.sum():,.0f} sq mi; wrote {len(tidy):,} rows to {out / 'tiger.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
