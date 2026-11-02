# Data Sources

This document lists every dataset used in this project, what it provides, and how to obtain it. Raw data is **not** stored in this repository because several files exceed GitHub's size limits. The cleaned versions **are** stored: data/interim/ holds one tidy file per source (state_fips, year, variable, value) and data/processed/ holds the final state-year panel. To reproduce the data from scratch, download each source below into data/raw/ and run the loaders in code/data_loaders/.

All sources are U.S. federal agencies, so the data is in the public domain.

## Overview

| Role | Source | Agency | Years used | Loader |
|---|---|---|---|---|
| **Response** | State GHG Inventory (gross emissions) | U.S. EPA | 1997–2022 | load_epa.py |
| Response (robustness) | SEDS (energy-related CO₂) | EIA | 1997–2022 | load_eia.py |
| Response (robustness) | GHG Reporting Program | U.S. EPA | 2010–2022 | not yet built |
| **Control** | nCLIMDIV heating and cooling degree days | NOAA NCEI | 1997–2022 | load_noaa.py |
| Control | SEDS (population, real GDP) | EIA | 1997–2022 | load_eia.py |
| Corporate | SEDS (fossil fuel production, grid mix) | EIA | 1997–2022 | load_eia.py |
| Corporate | GDP by State and Industry (SAGDP2) | BEA | 1997–2022 | load_bea.py |
| Corporate | Aggregates time series (sand and gravel) | USGS | 1997–2022 | load_usgs_minerals.py |
| Citizen Behaviors | Highway Statistics, Tables VM-2 and PS-1 | FHWA | 1997–2022 | load_fhwa.py |
| Citizen Behaviors | T-100 Segment (domestic) | BTS | 1997–2022 | load_bts.py |
| Citizen Behaviors | SEDS (residential electricity) | EIA | 1997–2022 | load_eia.py |
| Citizen Behaviors | Personal Consumption Expenditures by State (SAPCE2) | BEA | 1997–2022 | load_bea.py |
| Citizen Behaviors | Quick Stats (livestock inventory) | USDA NASS | 1997–2022 | load_usda.py |
| Spatial weights | TIGER/Line state boundaries (queen contiguity) | Census Bureau | 2022 boundaries | build_spatial_weights.py |
| Secondary analysis | nCLIMDIV temperature anomalies | NOAA NCEI | 1997–2022 | load_noaa.py |
| Secondary analysis | MODIS MOD11A1 land surface temperature | NASA LP DAAC | 2001–2022 | not yet built |
| Secondary analysis | NLCD land cover | MRLC | 2015–2024 | not yet built |
| Secondary analysis | Crop irrigation withdrawals | USGS | 2000–2020 | not yet built |
| Robustness only | FIA / EVALIDATOR | USDA Forest Service (USFS) | periodic | not yet built |

### Dropped Data

| Role | Source | Agency | Reason|
|---|---|---|---|
| Corporate | Forms EIA-860 / EIA-923 | EIA | Only 2025–2026 downloaded; SEDS already gives the state generation mix back to 1960 |
| Corporate | Company Facts API (10-K CapEx) | SEC EDGAR | Firm-level national totals; can’t be split by state, so year fixed effects would absorb it |
| Citizen Behaviors | Monthly Retail Trade | Census | National totals only; replaced by BEA state consumer spending (SAPCE2) |

Every loader also reads data/state_crosswalk.csv (state name, postal code, FIPS code, NOAA state code), which is committed to the repository.

---

## Response Variable

**Pivot:** The original response, year-over-year change in state temperature anomaly, was replaced with state gross greenhouse gas emissions per person. Greenhouse gases mix globally, so one state's emissions have almost no measurable effect on that same state's temperature in the same year, and year-over-year differencing removes the warming trend and leaves mostly weather noise. Emissions are causally linked to the drivers in both feature sets. Temperature is kept for a secondary, local-drivers analysis (see NOAA below).

### U.S. EPA: Inventory of U.S. Greenhouse Gas Emissions and Sinks by State

- **Provides:** Annual state-level greenhouse gas emissions and removals, 1990–2022, by economic sector (residential, commercial, industry, transportation, electric power, agriculture, land use) and by gas (CO₂, CH₄, N₂O, fluorinated gases), in million metric tons of CO₂ equivalent. This gives us our main response variable, gross emissions per person, which excludes land use and forestry (LULUCF):

$$Y_{s,t} = \ln\left(\frac{\text{Gross GHG}_{s,t}}{\text{Population}_{s,t}}\right)$$

  Net emissions (gross plus forest and land sinks) are kept as a robustness check. They are not the primary response because heavily forested states can come out near zero or negative (Maine is −0.3 MMT in 2022), which would mostly measure forest cover.
- **Citation:**
  U.S. Environmental Protection Agency. Inventory of U.S. Greenhouse Gas Emissions and Sinks by State: 1990–2022. Consolidated All-State GHG Data (AllStateGHGData90-22_v082924). Retrieved on Nov 2, 2026 from https://www.epa.gov/ghgemissions/state-ghg-emissions-and-removals

### U.S. EPA: Greenhouse Gas Reporting Program *(secondary analysis)*

- **Provides:** Facility-reported greenhouse gas data for large emitters (2023 data summary spreadsheets).
- **Citation:**
  U.S. Environmental Protection Agency. Greenhouse Gas Reporting System. 2023 Data Summary Spreadsheets. Retrieved on Oct 29, 2026 from https://www.epa.gov/ghgreporting/data-sets.

---

## Controls and Secondary Analysis

### NOAA NCEI Climate Divisional Database (nCLIMDIV)

- **Provides:** Monthly statewide temperature, heating degree days (HDD), cooling degree days (CDD), and precipitation, dating back over a century. HDD and CDD are controls for how much heating and cooling each year required. The temperature anomaly (each state's annual mean minus its own 1991–2020 normal) is the response for the secondary analysis:

$$\text{Anomaly}_{s,t} = \bar{T}_{s,t} - \bar{T}_{s,\,1991\text{–}2020}$$

- **Citation:**
  Vose, Russell S.; Applequist, Scott; Squires, Mike; Durre, Imke; Menne, Matthew J.; Williams, Claude N., Jr.; Fenimore, Chris; Gleason, Karin; Arndt, Derek (2014): NOAA Monthly U.S. Climate Divisional Database (NClimDiv). /data/nclimdiv-monthly/archive. NOAA National Climatic Data Center. doi:10.7289/V5M32STR. Oct 19, 2026.
- **Notes:** **Hawaii is not in nCLIMDIV, and nCLIMDIV gives Alaska temperature and precipitation but no heating or cooling degree days.** Because HDD and CDD are controls, the main model runs on the **contiguous 48 states (1,248 state-years)**. Alaska and Hawaii stay in the panel with missing HDD/CDD (in_main_sample = 0). HDD and CDD enter the models as hdd_k and cdd_k (thousands of degree-days) so their coefficients are readable.

No link currently available. The source is undergoing data migration and the metadata XML is unretrievable.

### MODIS Thermal Data *(secondary analysis)*

- **Provides:** Land surface temperature and emissivity, used to derive state-level Urban Heat Island spatial index metrics for the secondary (local temperature) analysis.
- **Citation:**
  National Aeronautics and Space Administration (NASA) Land Processes Distributed Active Archive Center (LP DAAC), MODIS/061/MOD11A1 Land Surface Temperature and Emissivity Daily L3 Global 1km. Retrieved on Oct 19, 2026 from Google Earth Engine.

### USGS Water Use Data and NLCD *(secondary analysis)*

- **Provides:** Secondary analysis only. These affect local temperature, not greenhouse gas emissions.
  - USGS Water Use in the United States: state-level irrigation withdrawal data, used to estimate irrigation density.
  - NLCD: classification of developed open space, high-density urban areas, and forest cover across every state.
- **Citation 1 (USGS):**
  Haynes, J.V., Read, A.L., Chan, A.Y., Martin, D.J., Regan, R.S., Henson, W.R., Niswonger, R.G., and Stewart, J.S., 2023, Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000-2020 within the conterminous United States (ver. 2.0, September 2024): U.S. Geological Survey data release, https://doi.org/10.5066/P9LGISUM. Retrieved on Oct 30, 2026.
- **Citation 2 (NLCD):**
  Multi-Resolution Land Characteristics Consortium. (2024). National Land Cover Database. Land Cover (CONUS), 2015-2024. Retrieved on Oct 30, 2026 from https://www.mrlc.gov/data.

---

## Feature Set A: Corporate and Supply-Side Metrics

### EIA: State Energy Data System (SEDS)

- **Provides:** Annual state-level fossil fuel production (coal, crude oil, natural gas), the fossil fuel share of electric power sector energy input, population, real GDP, and energy-related CO₂ emissions by sector (robustness response).
- **Citation:**
  U.S. Energy Information Administration. (2026). State Energy Data System. Consolidated data file, Consumption, price, expenditure, production, indicators, and CO2 emissions estimates. Retrieved on Oct 29, 2026 from https://www.eia.gov/state/seds/seds-data-complete.php#CompleteDataFile.

### BEA: Gross Domestic Product by State and Industry (SAGDP2)

- **Provides:** State GDP by industry in current dollars, used to compute the industrial share of the economy: (mining + manufacturing) ÷ all-industry GDP. Current dollars are used because chained (real) dollars do not add correctly across industries.
- **Citation:**
  U.S. Bureau of Economic Analysis. Gross Domestic Product by State and Industry (SAGDP2), 1997–2025. Retrieved on Nov 2, 2026 from https://apps.bea.gov/regional/downloadzip.htm.

### USGS: Construction Aggregates

- **Provides:** Annual state-level production of construction sand and gravel and crushed stone (metric tons), used as a measure of construction activity. Construction sand and gravel is the variable used, because USGS withholds crushed stone figures for Delaware and Louisiana in most years.
- **Citation:**
  U.S. Geological Survey (USGS) National Minerals Information Center, Aggregates Time Series Data by State, Type, and End Use, 1971–2023. Retrieved on Oct 19, 2026 from https://www.usgs.gov/media/files/usgs-aggregates-time-series-data-state-type-and-end-use.

### U.S. Forest Service: Forest Inventory and Analysis (FIA) *(robustness only)*

- **Provides:** State-level forest area change, carbon density, and timber harvest.
- **Citation:**
  U.S. Department of Agriculture, Forest Service. (2024). Forest Inventory and Analysis EVALIDATOR (Version 2.0) [Web application]. U.S. Department of Agriculture, Forest Service, Forest Inventory and Analysis Program. Retrieved on Oct 19, 2026 from https://research.fs.usda.gov/products/dataandtools/evalidator-and-fiadb-api.

---

## Feature Set B: Citizen and Consumer Demand Metrics

### FHWA: Highway Statistics Series

- **Provides:** Annual vehicle-miles traveled (VMT) by state, rural and urban, converted to VMT per person.
- **Citation:**
  U.S. Department of Transportation. (2024). Federal Highway Administration, Office of Highway Policy Information. Highway Statistics Series. Retrieved on Oct 29, 2026 from https://www.fhwa.dot.gov/policyinformation/statistics.cfm.

### BTS: Air Carrier Statistics (T-100 Segment Data)

- **Provides:** Domestic air passenger-miles (passengers × segment distance) and freight ton-miles, by state of departure.
- **Citation:**
  U.S. Department of Transportation. (2026). Bureau of Transportation Statistics. Air Carriers: T-100 Segment (All Carriers). Retrieved on Oct 29, 2026 from https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FIM.


### EIA: State Energy Data System (Residential Electricity)

- **Provides:** Residential electricity purchases per person, and electricity sales by sector, used to split electric power emissions between households and businesses in the descriptive analysis.
- **Notes:** Residential electricity is household demand, not fuel combustion, but it comes close to the no-fuel-consumption rule. Electric power emissions are roughly electricity sold × the grid's fossil share, so this driver is tied to the outcome by arithmetic, especially with grid fossil share in the corporate index. It stays in the main household index; D_household_noelec drops it as a robustness check, and the overlap must be discussed in the writeup.
- **Citation:**
  Same SEDS file as Feature Set A.

### BEA: Personal Consumption Expenditures by State (SAPCE2)

- **Provides:** Personal consumption expenditures per person by type of product, used as a proxy for consumer demand. Spending on gasoline and other energy goods is subtracted, because that spending is fuel use.
- **Citation:**
  U.S. Bureau of Economic Analysis. Per Capita Personal Consumption Expenditures by State (SAPCE2), 1997–2025. Retrieved on Nov 2, 2026 from https://apps.bea.gov/regional/downloadzip.htm.

### USDA National Agricultural Statistics Service (NASS)

- **Provides:** State annual livestock inventories from the NASS survey program (cattle and calves, milk cows, hogs, sheep), used to compute cattle per person. Livestock is the largest source of agricultural methane.
- **Notes:** Cattle per person is **robustness only**, not part of the main household index. After state and year fixed effects only 6% of its variation is left, and its within-state correlation with Y is −0.16 even though the pooled correlation is +0.71 (driven by WY and the Dakotas). Whether livestock is household or corporate activity is a judgment call, so both placements are run as robustness checks (D_household_lvstk, D_corporate_lvstk). Hogs (1,281 of 1,300 rows) and sheep (848 of 1,300; no data for AR, DE, FL, GA, HI, MS, RI, SC) are loaded but not used.
- **Citation:**
  U.S. Department of Agriculture. (2026). National Agriculture Statistics Service, Data & Statistics: Quick Stats Large Datasets. Retrieved on Oct 29, 2026 from https://www.nass.usda.gov/datasets/.

---

## Spatial Weights

### U.S. Census Bureau: TIGER/Line Shapefiles, States (2022)

- **Provides:** State boundary polygons, used to build a queen-contiguity neighbor matrix for the 50 states (DC and territories dropped). Queen contiguity counts shared corners, so the Four Corners pairs (AZ–CO, UT–NM) are neighbors. The matrix is row-standardized and saved as an edge list in data/state_neighbors.csv (state_fips, neighbor_fips, weight): 110 neighbor pairs, 220 rows. build_panel.py uses it to add each index's neighbor average in the same year (W_D_corporate, W_D_household, and the robustness indices).
- **Limitations:** Alaska and Hawaii have no neighbors, so their spatial lag is set to 0 (flag has_neighbors = 0). Neither is in the main sample (no degree days), so this matters only for a 50-state specification without HDD/CDD. TIGER/Line state polygons include Great Lakes and coastal water, so three links cross water only: IL–MI, MI–MN and NY–RI. They are kept.
- **Citation:**
  U.S. Census Bureau. (2022). TIGER/Line Shapefiles: States (tl_2022_us_state). Retrieved on Nov 2, 2026 from https://www2.census.gov/geo/tiger/TIGER2022/STATE/.

---

## Analysis Panel: Main Sample and Indices

build_panel.py writes data/processed/panel.csv (50 states × 1997–2022 = 1,300 rows). The **main sample** (in_main_sample = 1) is the contiguous 48 states, 1,248 rows. Naming: y_ = outcomes, d_ = treatment components, D_ = treatment indices, W_ = spatial lags; the controls (X in DML notation) are log_pop, log_gdp_real_pc, hdd_k and cdd_k.

Each index is the standardized mean of standardized components. Means and SDs come from the main sample, so 1 SD means 1 SD across the 48 states used in estimation; AK and HI are scored on the same scale.

| Index | Components | Role |
|---|---|---|
| D_corporate | d_industrial_share, d_fossil_prod_pc, d_grid_fossil_share, d_sand_gravel_pc | main |
| D_household | d_vmt_pc, d_air_pax_pc, d_pce_nonenergy_pc, d_res_elec_pc | main |
| D_household_lvstk | D_household components + d_cattle_pc | robustness: livestock on household side |
| D_corporate_lvstk | D_corporate components + d_cattle_pc | robustness: livestock on corporate side |
| D_household_noelec | D_household components − d_res_elec_pc | robustness: overlap with electric power emissions |
| D_corporate_freight | D_corporate components + d_air_freight_pc (2003+) | robustness: air freight |

On the main sample, D_corporate and D_household correlate 0.12 pooled and 0.18 within state and year. D_household with and without cattle correlate 0.99 within, confirming cattle adds almost nothing once fixed effects are in.

---

## Data Fixes Applied in build_panel.py

These are applied when the panel is built, not in the loaders, so the interim files hold the data as published.

| Issue | Fix | Flag |
|---|---|---|
| Indiana 2022 VMT is a one-year +22% spike (95,684 million miles) that reverses in 2023 | Replaced total, rural and urban VMT with the mean of 2021 and 2023 (81,925 million miles); urban share recomputed | vmt_imputed = 1 |
| Delaware had no scheduled air service in 1997–1999, so BTS has no rows | All air variables set to 0 | none |
| BTS air freight before 2003 undercounts all-cargo carriers (FedEx in TN, UPS in KY, Anchorage in AK) | Air freight set to missing before 2003; it is used only in the 2003+ robustness index D_corporate_freight | none |

---

## Reproducing the Data

1. Clone the repository. The cleaned data in data/interim/ and data/processed/ is already included, so **steps 2–5 are only needed to rebuild it from the original sources**.
2. Download each source from the links above into the matching data/raw/ subfolder (data/raw/outcome/, data/raw/climate/, data/raw/corporate/, data/raw/citizen/). File names and retrieval dates are listed with each citation.
3. Run each loader in code/data_loaders/. Each one takes the raw file or folder, data/state_crosswalk.csv, and an output folder, and writes one tidy file to data/interim/:

| Loader | Writes |
|---|---|
| load_epa.py | data/interim/outcome/epa.csv |
| load_eia.py | data/interim/shared/eia.csv |
| load_bts.py | data/interim/shared/bts.csv |
| load_bea.py | data/interim/shared/bea.csv |
| load_noaa.py | data/interim/shared/noaa.csv |
| load_fhwa.py | data/interim/civilian/fhwa.csv |
| load_usda.py | data/interim/corporate/usda.csv |
| load_usgs_minerals.py | data/interim/corporate/usgs_minerals.csv |

   Example: `python3 code/data_loaders/load_epa.py data/raw/outcome/allstateghgdata90-22_v082924/AllStateGHGData90-22_v082924.xlsx data/state_crosswalk.csv data/interim/outcome`

4. Download tl_2022_us_state.zip (see Spatial Weights) into data/raw/shared/TIGER/ and build the neighbor matrix: `python3 code/data_preparation/build_spatial_weights.py data/raw/shared/TIGER/tl_2022_us_state.zip data/state_crosswalk.csv data/state_neighbors.csv`. The output is committed, so this step is only needed to rebuild it.
5. Run build_panel.py (`python3 code/data_preparation/build_panel.py data/interim data/state_crosswalk.csv data/state_neighbors.csv data/processed`) to join every interim file into data/processed/panel.csv (one row per state-year, 50 states × 1997–2022 = 1,300 rows) and data/processed/data_dictionary.csv. It prints a report with coverage checks, missingness, and driver correlations.
6. data/raw/ is git-ignored, so downloaded data will not be committed.
