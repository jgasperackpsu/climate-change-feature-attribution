# Data Sources

This document lists every dataset used in this project, what it provides, and how to obtain it. Raw data is **not** stored in this repository. To reproduce the data, download each source below and/or run the pull scripts included in data/raw/ for the subdirectories BEA and SEC EDGAR.

## Overview

| Role | Source | Agency |
|---|---|---|
| **Response** | nCLIMDIV temperature anomalies | NOAA NCEI |
| **Control** | MODIS MOD11A1 land surface temperature | NASA LP DAAC |
| Corporate  | SEDS (extraction, generation) | EIA |
| Corporate  | EIA-860 / EIA-923 | EIA |
| Corporate  | Minerals Yearbook (aggregates) | USGS |
| Corporate  | FIA / EVALIDATOR | USDA Forest Service (USFS) |
| Corporate  | Company Facts API (10-K CapEx) | SEC EDGAR |
| Corporate  | Quick Stats (livestock) | USDA NASS |
| Citizen Behaviors | Highway Statistics, Table VM-2 | FHWA |
| Citizen Behaviors | T-100 Domestic Segment | BTS |
| Citizen Behaviors | SEDS (residential consumption) | EIA |
| Citizen Behaviors | GHG Reporting Program | U.S. EPA |
| Citizen Behaviors | Monthly Retail Trade | Census / BEA |
| Citizen Behaviors | Crop irrigation withdrawals | USGS |
| Citizen Behaviors | NLCD land cover | MRLC |

---

## Climate Response and Controls

### NOAA NCEI Climate Divisional Database (nCLIMDIV)

- **Provides:** Monthly and annual surface temperature anomalies by state and climate division, dating back over a century. This will also give us our main response variable, year-over-year change in temperature:

$$TYoY_t = \text{Anomaly}_t − \text{Anomaly}_{(t−1)}$$

- **Citation:**
  Vose, Russell S.; Applequist, Scott; Squires, Mike; Durre, Imke; Menne, Matthew J.; Williams, Claude N., Jr.; Fenimore, Chris; Gleason, Karin; Arndt, Derek (2014): NOAA Monthly U.S. Climate Divisional Database (NClimDiv). /data/nclimdiv-monthly/archive. NOAA National Climatic Data Center. doi:10.7289/V5M32STR. Oct 19, 2026.

No link currently available. The source is undergoing data migration and the metadata XML is unretrievable.

### MODIS Thermal Data

- **Provides:** Land surface temperature and emissivity, used to derive state-level Urban Heat Island spatial index metrics and to control for localized microclimate heating.
- **Citation:**
  National Aeronautics and Space Administration (NASA) Land Processes Distributed Active Archive Center (LP DAAC), MODIS/061/MOD11A1 Land Surface Temperature and Emissivity Daily L3 Global 1km. Retrieved on Oct 19, 2026 from Google Earth Engine.

---

## Feature Set A: Corporate and Supply-Side Metrics

### EIA: State Energy Data System (SEDS)

- **Provides:** Annual state-level fossil fuel extraction volumes (coal, crude oil, natural gas) and grid-level fossil fuel power generation (kWh).
- **Citation:**
  U.S. Energy Information Administration (EIA), State Energy Data System (SEDS). Retrieved on Oct 19, 2026 from https://www.eia.gov/state/seds/seds-data-complete.php?sid=US.

### EIA: Forms EIA-860 and EIA-923

- **Provides:** Plant-level and state-aggregated power generation sources and fossil fuel infrastructure capacities.
- **Citation:**
  U.S. Energy Information Administration (EIA), Forms EIA-860 and EIA-923 Data Files. Retrieved on Oct 19, 2026 from https://www.eia.gov/electricity/data/eia860/ and https://www.eia.gov/electricity/data/eia923/.

### USGS: Minerals Information

- **Provides:** Annual state-level industrial output data for cement production, raw steel tonnage, mining output, and chemical mineral extraction (Minerals Yearbook).
- **Citation:**
  U.S. Geological Survey (USGS) National Minerals Information Center, Minerals Yearbook: Area Reports: Domestic. Retrieved on Oct 19, 2026 from https://www.usgs.gov/media/files/usgs-aggregates-time-series-data-state-type-and-end-use.

### U.S. Forest Service: Forest Inventory and Analysis (FIA) / Global Forest Watch

- **Provides:** State-level deforestation acreage, timber harvest rates, and land conversion tracking.
- **Citation:**
  U.S. Department of Agriculture, Forest Service. (2024). Forest Inventory and Analysis EVALIDATOR (Version 2.0) [Web application]. U.S. Department of Agriculture, Forest Service, Forest Inventory and Analysis Program. Retrieved on Oct 19, 2026 from https://research.fs.usda.gov/products/dataandtools/evalidator-and-fiadb-api.

### SEC EDGAR and Climate Disclosure Databases (CDP / Bloomberg ESG)

- **Provides:** Annual capital expenditures (CapEx) on fossil fuel exploration, production, and infrastructure expansion from SEC Form 10-K filings, covering S&P 500 energy and utility corporations, aggregated or mapped by state operating presence.
- **Citation:**
  Data were programmatically extracted from the U.S. Securities and Exchange Commission (SEC) EDGAR Company Facts API (data.sec.gov/api/xbrl/companyfacts/) across target S&P 500 CIK identifiers. Code can be found in **data/raw/corporate/SEC EDGAR/pull_data.py**.

### USDA National Agricultural Statistics Service (NASS)

- **Provides:** State-by-state annual livestock headcount (cattle, swine, poultry), used to compute livestock density per capita.
- **Citation:**
  U.S. Department of Agriculture. (2026). National Agriculture Statistics Service, Data & Statistics: Quick Facts Large Datasets. Retrieved on Oct 29, 2026 from https://www.nass.usda.gov/datasets/.

---

## Feature Set B: Citizen and Consumer Demand Metrics

### FHWA: Highway Statistics Series (Table VM-2)

- **Provides:** Annual Vehicle Miles Traveled (VMT) per capita for all 50 states.
- **Citation:**
  U.S. Department of Transportation. (2024). Federal Highway Administration, Office of Highway Policy Information. Highway Statistics Series. Retrieved on Oct 29, 2026 from https://www.fhwa.dot.gov/policyinformation/statistics.cfm.

### BTS: Air Carrier Statistics (T-100 Segment Data)

- **Provides:** State-level passenger flight departures, origin-destination passenger counts, and passenger flight miles.
- **Citation:**
  U.S. Department of Transportation. (2026). Bureau of Transportation Statistics. Air Carriers: T-100 Domestic Segment (U.S. Carriers). Retrieved on Oct 29, 2026 from https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FIM.

### EIA: Residential Energy Consumption Survey (RECS) and SEDS

- **Provides:** State-aggregated household electricity usage (kWh per household) and residential natural gas / fuel oil consumption per capita.
- **Citation:**
  U.S. Energy Information Administration. (2026). State Energy Data System. Consolidated data file, Consumption, price, expenditure, production, indicators, and CO2 emissions estimates. Retrieved on Oct 29, 2026 from https://www.eia.gov/state/seds/seds-data-complete.php#CompleteDataFile.

### U.S. EPA: Greenhouse Gas Reporting Program

- **Provides:** Facility-reported greenhouse gas data (2023 data summary spreadsheets), including landfill methane.
- **Citation:**
  U.S. Environmental Protection Agency. Greenhouse Gas Reporting System. 2023 Data Summary Spreadsheets. Retrieved on Oct 29, 2026 from https://www.epa.gov/ghgreporting/data-sets.

**Note:** The original plan used state-level municipal solid waste (MSW) generation per capita (EPA State Measurement Framework and WARM tool). **That measure was replaced with consumer spending** due to not enough data (see the next entry).

#### Consumer Spending

- **Provides:** Estimates of monthly retail and food services sales by kind of business, January 1992 through August 2026, used as a proxy for consumer demand.
- **Citation:**
  U.S. Bureau of Economic Analysis. U.S. Census Bureau, Annual Retail Trade Survey Revision Reports. Estimates of Monthly Retail and Food Services Sales by Kind of Business: January 1992 Through August 2026. Retrieved on Oct 29, 2026 from https://www.bea.gov/data/consumer-spending/main.

### USGS Water Use Data and NLCD (Turfgrass and Irrigation Density)

- **Provides:**
  - USGS Water Use in the United States: state-level irrigation withdrawal data, used to estimate turfgrass and golf course irrigation density.
  - NLCD: classification of developed open space, high-density urban areas, and turfgrass footprints across every state.
- **Citation 1 (USGS):**
  Haynes, J.V., Read, A.L., Chan, A.Y., Martin, D.J., Regan, R.S., Henson, W.R., Niswonger, R.G., and Stewart, J.S., 2023, Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000-2020 within the conterminous United States (ver. 2.0, September 2024): U.S. Geological Survey data release, https://doi.org/10.5066/P9LGISUM. Retrieved on Oct 30, 2026.
- **Citation 2 (NLCD):**
  Multi-Resolution Land Characteristics Consortium. (2024). National Land Cover Database. Land Cover (CONUS), 2015-2024. Retrieved on Oct 30, 2026 from https://www.mrlc.gov/data.

---

## Reproducing the Data

1. Clone the repository.
2. Run the pull scripts in data/raw/ (for example, data/raw/corporate/SEC EDGAR/pull_data.py).
3. For sources without a script, download the files from the links above into the matching data/raw/ subfolder.
4. data/raw/ is git-ignored except for .py files, so downloaded data will not be committed.
