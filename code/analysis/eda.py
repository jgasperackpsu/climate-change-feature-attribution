"""Phase 3: descriptive EDA - emission shares over time, missingness, outliers, driver correlations, VIF,
index correlation.

Usage:
    python code/analysis/eda.py <path/to/panel.csv> <path/to/decomposition.csv> <visuals_dir> <tables_dir>
    e.g. python code/analysis/eda.py data/processed/panel.csv data/processed/decomposition.csv visuals data/processed

Writes PNG figures to <visuals_dir> and vif.csv, correlations.csv, missingness.csv, outliers.csv to <tables_dir>.

Missingness: for every model input and robustness variable, missing state-years in the 50-state panel and
in the main sample, with the states and years affected.
Outliers: one-year jumps. For Y, the robustness outcomes, every driver and log(1 + MMT) of each EPA sector
and of IPPU and iron and steel, the year-over-year change within a state is scored with a robust z (median and
1.4826 x MAD of all changes of that variable); |z| > OUTLIER_Z is flagged, with whether the jump reverses the
next year (a one-off spike) or persists (a level shift). The EPA iron and steel allocation breaks (MN, MI 2010;
MD 2011-12) show up in the sector series: they are too small a share of total gross GHG to stand out in Y.
Correlations and VIFs use the main sample (in_main_sample = 1: contiguous 48 states, 1,248 rows).
"Within" = two-way (state and year) demeaned, i.e. the variation left for a fixed-effects model.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preparation"))
from build_panel import CONTROLS, CORPORATE, HOUSEHOLD, LAGS, Y, within  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-2, chrome, blue ramp, blue<->red diverging
HH, CORP = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("bluered", ["#184f95", "#5598e7", "#f0efec", "#e98585", "#b83232"])

LABELS = {
    "d_industrial_share": "Industrial GDP share", "d_fossil_prod_pc": "Fossil production / person",
    "d_grid_fossil_share": "Grid fossil share", "d_sand_gravel_pc": "Sand & gravel / person",
    "d_vmt_pc": "VMT / person", "d_air_pax_pc": "Air passenger-miles / person",
    "d_pce_nonenergy_pc": "Real non-energy PCE / person", "d_res_elec_pc": "Residential electricity / person",
    "d_cattle_pc": "Cattle / person", "log_pop_density": "log population density", "log_gdp_real_pc": "log real GDP / person",
    "hdd_k": "HDD (thousands)", "cdd_k": "CDD (thousands)", "D_corporate": "D corporate", "D_household": "D household",
    "W_D_corporate": "W · D corporate", "W_D_household": "W · D household",
}
COMPONENTS = {
    "ghg_hh_residential_mmt": "Residential (direct)", "ghg_hh_electricity_mmt": "Residential electricity",
    "ghg_hh_transport_mmt": "Personal transport", "ghg_corp_industry_mmt": "Industry",
    "ghg_corp_commercial_mmt": "Commercial", "ghg_corp_agriculture_mmt": "Agriculture",
    "ghg_corp_electricity_mmt": "Business electricity", "ghg_corp_transport_mmt": "Freight & other transport",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


OUTLIER_Z = 5
SCANNED = [Y, "y_log_co2_energy_pc", "y_log_gross_noippu_pc"] + CORPORATE + HOUSEHOLD + ["d_cattle_pc"]
SECTORS = ["ghg_residential_mmt", "ghg_commercial_mmt", "ghg_industry_mmt", "ghg_transportation_mmt",
           "ghg_electric_power_mmt", "ghg_agriculture_mmt", "ghg_ippu_mmt", "ghg_iron_steel_mmt"]
CHECKED = SCANNED + ["d_air_freight_pc"] + CONTROLS + LAGS + ["D_corporate", "D_household"]


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"  wrote {out / name}")


def vif(df):
    """VIF_j = j-th diagonal of the inverse correlation matrix (= 1 / (1 - R^2_j) with an intercept)."""
    return pd.Series(np.diag(np.linalg.inv(df.corr().values)), index=df.columns)


def fig_national_share(dec, out):
    nat = dec.groupby("year")[["ghg_household_mmt", "ghg_corporate_mmt"]].sum()
    share = nat.ghg_household_mmt / nat.sum(axis=1)
    q = dec.groupby("year").household_share.quantile([0.1, 0.9]).unstack()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for side, s, color in [("Household", share, HH), ("Corporate", 1 - share, CORP)]:
        lo, hi = (q[0.1], q[0.9]) if side == "Household" else (1 - q[0.9], 1 - q[0.1])
        ax.fill_between(q.index, lo * 100, hi * 100, color=color, alpha=0.15, linewidth=0)
        ax.plot(s.index, s * 100, color=color)
        ax.text(2022.4, s.iloc[-1] * 100, f"{side}\n{s.iloc[-1]:.0%}", va="center", color=INK, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(1997, 2022)
    ax.set_ylabel("share of gross GHG (%)")
    ax.set_title("Household and corporate shares of gross GHG emissions, 1997–2022")
    ax.annotate("Lines: 50-state totals. Bands: 10th–90th percentile across the 50 states. Electricity split by "
                "sales shares; transport by SEDS fuel (gasoline + passenger jet fuel).",
                (0, -0.16), xycoords="axes fraction", fontsize=7.5, color=MUTED)
    save(fig, out, "decomp_household_share_national.png")
    return share


def fig_components(dec, out):
    nat = dec.groupby("year")[list(COMPONENTS)].sum()
    nat = nat.div(nat.sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(2, 5, figsize=(12, 4.8), sharex=True, sharey=True)
    hh = [c for c in COMPONENTS if "_hh_" in c]
    corp = [c for c in COMPONENTS if "_corp_" in c]
    for row, (cols, color, side) in enumerate([(hh, HH, "Household"), (corp, CORP, "Corporate")]):
        for j in range(5):
            ax = axes[row, j]
            if j >= len(cols):
                ax.set_visible(False)
                continue
            c = cols[j]
            ax.plot(nat.index, nat[c], color=color)
            ax.set_title(COMPONENTS[c], fontsize=9)
            ax.text(2022, nat[c].iloc[-1] + 1.5, f"{nat[c].iloc[-1]:.1f}%", ha="right", fontsize=8, color=INK)
        axes[row, 0].set_ylabel(f"{side}\n% of gross", color=INK)
    axes[0, 0].set_ylim(0, 30)
    axes[0, 0].set_xticks([2000, 2010, 2020])
    handles = [plt.Line2D([], [], color=HH), plt.Line2D([], [], color=CORP)]
    fig.legend(handles, ["Household", "Corporate"], loc="center", bbox_to_anchor=(0.8, 0.75), ncol=1)
    fig.suptitle("Components of 50-state gross GHG emissions, share of total", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "decomp_components_national.png")


def fig_state_heatmap(dec, out):
    m = dec.pivot(index="postal_code", columns="year", values="household_share") * 100
    m = m.loc[m[2022].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ, vmin=0, vmax=65)
    ax.set_yticks(range(len(m)), m.index, fontsize=7)
    xt = [i for i, y in enumerate(m.columns) if y % 5 == 0 or y == 2022]
    ax.set_xticks(xt, [m.columns[i] for i in xt])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("household share of gross GHG (%)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Household share by state (sorted by 2022)")
    save(fig, out, "decomp_household_share_states.png")


def missingness(p):
    """Missing state-years per variable: 50-state panel and main sample, with the states and years affected."""
    rows = []
    for c in CHECKED:
        miss = p[c].isna()
        main = miss & (p.in_main_sample == 1)
        rows.append(dict(variable=c, n=len(p), missing=int(miss.sum()), missing_share=miss.mean(),
                         missing_main_sample=int(main.sum()),
                         states=",".join(sorted(p.loc[miss, "postal_code"].unique())),
                         years=(f"{p.loc[miss, 'year'].min()}-{p.loc[miss, 'year'].max()}" if miss.any() else "")))
    return pd.DataFrame(rows)


def outliers(p, z_cut=OUTLIER_Z):
    """One-year jumps scored by a robust z over all within-state year-over-year changes of each variable."""
    rows = []
    q = p.sort_values(["postal_code", "year"]).copy()
    for c in SECTORS:
        q[f"log1p_{c}"] = np.log1p(q[c].clip(lower=0))
    for c in SCANNED + [f"log1p_{c}" for c in SECTORS]:
        dx = q.groupby("postal_code")[c].diff()
        nxt = q.groupby("postal_code")[c].diff().groupby(q.postal_code).shift(-1)
        med = dx.median()
        mad = 1.4826 * (dx - med).abs().median()
        z = (dx - med) / mad
        hit = z.abs() > z_cut
        for i in q.index[hit]:
            reverses = pd.notna(nxt[i]) and np.sign(nxt[i]) == -np.sign(dx[i]) and abs(nxt[i]) > 0.5 * abs(dx[i])
            rows.append(dict(variable=c, postal_code=q.at[i, "postal_code"], year=int(q.at[i, "year"]),
                             change=dx[i], robust_z=z[i], next_change=nxt[i],
                             pattern="spike (reverses)" if reverses else "level shift",
                             in_main_sample=int(q.at[i, "in_main_sample"])))
    return pd.DataFrame(rows).sort_values(["variable", "robust_z"], key=lambda s: s.abs() if s.name == "robust_z" else s,
                                          ascending=[True, False])


def fig_corr(pooled, within_c, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    labels = [LABELS[c] for c in pooled.columns]
    for ax, c, title in [(axes[0], pooled, "Pooled"), (axes[1], within_c, "Within state and year (after fixed effects)")]:
        im = ax.imshow(c.values, cmap=DIV, vmin=-1, vmax=1)
        n = len(c)
        for i in range(n):
            for j in range(n):
                if i != j:
                    v = c.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if abs(v) > 0.6 else INK)
        ax.set_xticks(range(n), labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(n), labels if ax is axes[0] else [], fontsize=7.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        # separate corporate / household / controls blocks
        for k in (len(CORPORATE) - 0.5, len(CORPORATE) + len(HOUSEHOLD) - 0.5):
            ax.axhline(k, color=SURFACE, lw=2)
            ax.axvline(k, color=SURFACE, lw=2)
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Pearson correlation", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Driver and control correlations, main sample (48 states, 1997–2022)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, out, "driver_correlations.png")


def fig_index_scatter(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    wc, wh = within(m, "D_corporate"), within(m, "D_household")
    for ax, (x, y, title) in zip(axes, [(m.D_corporate, m.D_household, "Pooled"),
                                        (wc, wh, "Within state and year")]):
        wy = m.postal_code == "WY"
        ax.scatter(x[~wy], y[~wy], s=10, color=HH, alpha=0.45, linewidths=0, label="Other 47 states")
        ax.scatter(x[wy], y[wy], s=14, color=CORP, linewidths=0, label="Wyoming")
        ax.set_title(f"{title}: r = {x.corr(y):.2f}")
        ax.set_xlabel("D corporate (SD)")
        ax.set_ylabel("D household (SD)")
    axes[0].legend(loc="lower right", fontsize=8, markerscale=1.5)
    fig.suptitle("Corporate vs household driver index, state-years (48 states, 1997–2022)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "index_scatter.png")


def fig_vif(v, out):
    specs = ["drivers + controls, within", "model regressors, within"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [12, 8]})
    for ax, spec in zip(axes, specs):
        s = v[v.spec == spec].set_index("variable").vif.sort_values()
        ax.barh([LABELS[i] for i in s.index], s.values, color=HH, height=0.6)
        for i, val in enumerate(s.values):
            ax.text(val + 0.1, i, f"{val:.1f}", va="center", fontsize=8, color=INK)
        ax.axvline(5, color=MUTED, lw=1)
        ax.text(5.05, len(s) - 0.4, "5", color=MUTED, fontsize=8)
        ax.set_xlim(0, max(6, s.max() * 1.15))
        ax.set_title(spec.capitalize())
        ax.set_xlabel("variance inflation factor")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Variance inflation factors after state and year fixed effects (48 states)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "vif.png")


def main(panel_path, decomp_path, visuals_dir, tables_dir):
    style()
    out, tab = Path(visuals_dir), Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(panel_path, dtype={"state_fips": str})
    dec = (pd.read_csv(decomp_path, dtype={"state_fips": str})
             .pivot(index=["state_fips", "year"], columns="variable", values="value").reset_index())
    dec = dec.merge(p[["state_fips", "year", "postal_code"]], on=["state_fips", "year"])
    if len(dec) != 1300:
        raise ValueError(f"decomposition has {len(dec)} rows, expected 1,300")

    print("figures:")
    fig_national_share(dec, out)
    fig_components(dec, out)
    fig_state_heatmap(dec, out)

    miss, outl = missingness(p), outliers(p)
    miss.to_csv(tab / "missingness.csv", index=False)
    outl.to_csv(tab / "outliers.csv", index=False)
    print(f"  wrote {tab / 'missingness.csv'} and {tab / 'outliers.csv'}")

    m = p[p.in_main_sample == 1].reset_index(drop=True)
    if len(m) != 1248 or m.state_fips.nunique() != 48:
        raise ValueError(f"main sample is {len(m)} rows / {m.state_fips.nunique()} states, expected 1,248 / 48")
    xs = CORPORATE + HOUSEHOLD + CONTROLS
    w = pd.DataFrame({c: within(m, c) for c in xs + ["D_corporate", "D_household"] + LAGS})
    pooled_c, within_c = m[xs].corr(), w[xs].corr()
    fig_corr(pooled_c, within_c, out)
    fig_index_scatter(m, out)

    model = ["D_corporate", "D_household"] + LAGS + CONTROLS
    v = pd.concat([
        vif(m[xs]).rename("vif").to_frame().assign(spec="drivers + controls, pooled"),
        vif(w[xs]).rename("vif").to_frame().assign(spec="drivers + controls, within"),
        vif(m[model]).rename("vif").to_frame().assign(spec="model regressors, pooled"),
        vif(w[model]).rename("vif").to_frame().assign(spec="model regressors, within"),
    ]).rename_axis("variable").reset_index()[["spec", "variable", "vif"]]
    fig_vif(v, out)
    v.to_csv(tab / "vif.csv", index=False)

    corr_long = pd.concat([
        pooled_c.stack().rename("r").reset_index().assign(spec="pooled"),
        within_c.stack().rename("r").reset_index().assign(spec="within"),
    ])
    corr_long.columns = ["var1", "var2", "r", "spec"]
    corr_long[corr_long.var1 < corr_long.var2].to_csv(tab / "correlations.csv", index=False)
    print(f"  wrote {tab / 'vif.csv'} and {tab / 'correlations.csv'}")

    # --- report ---
    print("\n================ REPORT ================")
    print(f"main sample: {len(m):,} rows, {m.state_fips.nunique()} states")
    print("\nmissing state-years (50-state panel / main sample):")
    print(miss[miss.missing > 0][["variable", "missing", "missing_main_sample", "states", "years"]].to_string(index=False))
    print(f"\none-year jumps with |robust z| > {OUTLIER_Z}, outcomes:")
    print(outl[outl.variable.str.startswith("y_")].round(3).to_string(index=False))
    print("EPA iron & steel and IPPU series, 10 largest (all in outliers.csv):")
    print(outl[outl.variable.str.contains("iron_steel|ippu_mmt") & (outl.in_main_sample == 1)]
          .head(10).round(3).to_string(index=False))
    print(f"jumps per driver: " + ", ".join(f"{k} {v}" for k, v in outl[outl.variable.str.startswith("d_")]
                                              .variable.value_counts().items()))
    print("\nVIF (rule of thumb: > 5 worth a look, > 10 a problem):")
    print(v.pivot(index="variable", columns="spec", values="vif").round(2).to_string())
    pairs = within_c.where(np.triu(np.ones(within_c.shape, bool), 1)).stack()
    top = pairs.abs().sort_values(ascending=False).head(5)
    print("\nlargest within correlations among drivers and controls:")
    for (a, b), _ in top.items():
        print(f"  {a:22s} {b:22s} {pairs[(a, b)]:+.3f} (pooled {pooled_c.loc[a, b]:+.3f})")
    print(f"\ncorr(D_corporate, D_household): pooled {m.D_corporate.corr(m.D_household):.3f}, "
          f"within {w.D_corporate.corr(w.D_household):.3f}")
    for d in ["D_corporate", "D_household"]:
        print(f"corr({d}, Y): pooled {m[d].corr(m[Y]):.3f}, within {w[d].corr(within(m, Y)):.3f}")
    hs = dec.groupby("year")[["ghg_household_mmt"]].sum().ghg_household_mmt / \
        dec.groupby("year")[["ghg_household_mmt", "ghg_corporate_mmt"]].sum().sum(axis=1)
    print(f"\n50-state household share of gross GHG: {hs.min():.1%} ({hs.idxmin()}) to {hs.max():.1%} ({hs.idxmax()})")
    print(f"corr(state household share, Y) in 2022: "
          f"{dec[dec.year == 2022].set_index('state_fips').household_share.corr(p[p.year == 2022].set_index('state_fips')[Y]):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
