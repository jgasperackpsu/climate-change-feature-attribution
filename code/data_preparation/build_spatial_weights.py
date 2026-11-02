"""Queen-contiguity spatial weights for the 50 states from the Census TIGER/Line state shapefile.

Usage:
    python code/data_preparation/build_spatial_weights.py <path/to/tl_2022_us_state.zip> <path/to/state_crosswalk.csv> <out_csv>
    e.g. python code/data_preparation/build_spatial_weights.py data/raw/shared/TIGER/tl_2022_us_state.zip data/state_crosswalk.csv data/state_neighbors.csv

Writes an edge list (state_fips, neighbor_fips, weight), row-standardized so each state's weights
sum to 1. DC and territories are dropped before computing contiguity. Alaska and Hawaii have no
neighbors, so they have no rows; build_panel.py gives them a spatial lag of 0.

Queen contiguity counts shared corners, so the Four Corners pairs (AZ-CO, UT-NM) are neighbors.
TIGER/Line state polygons include inland and Great Lakes water, so some links run across water
only (IL-MI, MI-MN, NY-RI); these are kept. The script prints spot checks for these pairs.
"""
import sys

import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen


def main(shp, crosswalk_path, out_csv):
    xw = pd.read_csv(crosswalk_path, dtype=str)
    g = gpd.read_file(shp)
    g = g[g["STATEFP"].isin(xw["fips_code"])].sort_values("STATEFP").reset_index(drop=True)
    if len(g) != 50:
        raise ValueError(f"expected 50 states in {shp}, found {len(g)}")

    w = Queen.from_dataframe(g, ids=g["STATEFP"].tolist(), use_index=False, silence_warnings=True)
    w.transform = "r"
    edges = pd.DataFrame([(i, j, wt) for i in w.neighbors for j, wt in zip(w.neighbors[i], w.weights[i])],
                         columns=["state_fips", "neighbor_fips", "weight"]).sort_values(["state_fips", "neighbor_fips"])

    post = dict(zip(xw["fips_code"], xw["postal_code"]))
    pairs = {tuple(sorted((post[a], post[b]))) for a, b in zip(edges.state_fips, edges.neighbor_fips)}
    islands = sorted(post[i] for i in w.islands)
    sym = all(((edges.state_fips == b) & (edges.neighbor_fips == a)).any() for a, b in zip(edges.state_fips, edges.neighbor_fips))
    rowsum = edges.groupby("state_fips")["weight"].sum()
    print(f"{len(pairs)} neighbor pairs; islands: {islands}; symmetric: {sym}; "
          f"row sums in [{rowsum.min():.6f}, {rowsum.max():.6f}]")
    for a, b in [("AZ", "CO"), ("NM", "UT"), ("MI", "WI"), ("IL", "MI"), ("MI", "MN"), ("NY", "RI"), ("DE", "NJ")]:
        print(f"  {a}-{b}: {'yes' if (a, b) in pairs else 'no'}")
    deg = edges.groupby("state_fips").size().rename(index=post).sort_values()
    print("neighbors per state: min " + ", ".join(f"{s} {n}" for s, n in deg.head(3).items())
          + "; max " + ", ".join(f"{s} {n}" for s, n in deg.tail(3).items()))

    edges.to_csv(out_csv, index=False)
    print(f"wrote {len(edges)} rows to {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
