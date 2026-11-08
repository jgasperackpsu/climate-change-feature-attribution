# Data for parent paper 3

Munir, S., Rashid, M., Shams, H. A., & Ahmed, S. (2026). Infrastructure, institutions, and the resilience dividend: A hybrid Econometric–Machine learning analysis of CO2 emissions in Central and Eastern Europe. *Environmental Management: Economics and Policy*, 2, 100006. https://doi.org/10.1016/j.jemecp.2026.100006

All data are public: World Bank World Development Indicators (WDI) and Worldwide Governance Indicators (WGI).

## File to put here

**`pp3_panel_raw.csv`**: one row per country-year, 18 countries × 2000–2024 (450 rows). Blanks are fine: following the paper, `run_pp3.py` forward-fills and then back-fills missing years within each country.

The easiest route is the download script, which writes this file directly:

```
cd parent_paper_repo_material/codes/code_pp3
python fetch_wdi.py
```

Or download by hand from https://databank.worldbank.org and save with these columns:

| Column | World Bank code | Database | Paper variable |
|---|---|---|---|
| `country` | | | Country name |
| `iso3` | | | ISO3 code (optional) |
| `year` | | | 2000–2024 |
| `co2_pc` | EN.GHG.CO2.PC.CE.AR5 (older code EN.ATM.CO2E.PC) | WDI | CO2E: CO2 per capita excl. LULUCF, t |
| `up` | SP.URB.TOTL.IN.ZS | WDI | UP: urban population, % of total |
| `ic` | GOV_WGI_GE.EST (older code GE.EST) | WGI | IC: government effectiveness, estimate |
| `eupe` | GOV_WGI_RQ.EST (older code RQ.EST) | WGI | EUPE: regulatory quality, estimate |
| `gdp_pc` | NY.GDP.PCAP.KD | WDI | EC: GDP per capita, constant US$ |
| `elec_kwh_pc` | EG.USE.ELEC.KH.PC | WDI | IEI: electric power consumption, kWh per capita |
| `eo` | NE.TRD.GNFS.ZS | WDI | EO: trade, % of GDP |
| `ce` | GOV_WGI_VA.EST (older code VA.EST) | WGI | CE: voice and accountability, estimate |

Countries (ISO3): ALB, BIH, BGR, HRV, CZE, EST, HUN, LVA, LTU, MKD, POL, ROU, SVK, SVN, MDA, UKR, ARM, GEO.

## Coverage (downloaded 2026-11-08)

All 18 countries × 2000–2024 (450 rows). The five WDI series are complete. The three WGI series are missing 2001 for every country, because the WGI were published every other year before 2002. `run_pp3.py` fills that year forward from 2000, following the paper's fill rule. The WGI codes gained a `GOV_WGI_` prefix in the 2024 revision. The `.EST` series are still on the −2.5 to 2.5 scale the paper uses.

The filled data match the paper's Table 2 closely, with several values exact: CO₂ median 4.918 and minimum 0.903, GDP per capita median 8,756.7 and minimum 1,229.215, trade minimum 48.521, and electricity maximum 7,354.893 and minimum 1,172.261. Means differ slightly (for example, the governance estimates are about 0.04 lower and the CO₂ maximum is 17.93 rather than 17.39), which is consistent with later World Bank revisions.

## Run order

```
cd parent_paper_repo_material/codes/code_pp3
python fetch_wdi.py                                  # -> PP_data/pp3_data/pp3_panel_raw.csv
python run_pp3.py                                    # -> results/pp3/table02..table20, figs, tables.md
python "../../TTV split/pp3_split/split_panel.py"   # 70/15/15 split -> TTV split/pp3_split/
```
