# corporate-carbon-ml-prediction

Which matters more for a U.S. state's greenhouse gas emissions per person: corporate supply-side activity or household demand? This project builds a state-year panel (contiguous 48 states, 1997–2022) and compares two kinds of attribution:

- **Prediction-based:** XGBoost and SHAP, as in the parent paper's machine-learning attribution.
- **Causal, within-state:** two-way fixed effects and double/debiased machine learning (DoubleML), which use only changes within a state over time.

**Outcome:** log gross GHG per person (EPA State GHG Inventory).

**Treatments:** two equal-weight indices of standardized drivers.

- **D_corporate:** industrial GDP share, fossil fuel production per person, grid fossil share, sand and gravel per person.
- **D_household:** vehicle-miles per person, air passenger-miles per person, non-energy consumption per person, residential electricity per person.

Each index also enters as a spatial lag (the neighbor average) to capture spillovers across state lines. Sources: [data_sources.md](data_sources.md).

## Current findings (Phase 5)

All models are fit on the train years (1997–2014) of a 70/15/15 split by whole years ([data/split_data/](data/split_data/)).

- **The corporate index has a positive within-state effect.** Per SD of the index, emissions per person rise by 0.053 log points (about 5.5%; p = 0.023) in the main DML result. That holds with every nuisance learner and in all but one robustness version.
- **The household effect is smaller and borderline** (0.035, p = 0.080), and it is mostly residential electricity.
- **"Corporate > household" is not supported.** The gap is 0.019 (p = 0.56; one-sided 0.28), and it is not significant with any learner or in any version. Counting spillovers does not change that.
- **Placebo and sensitivity checks:** the placebo passes (next year's index has no effect). An omitted confounder would need a partial R² of 0.17 to remove the corporate effect.
- **Drivers:** the within-state effect is carried by residential electricity, fossil production and grid fossil share. SHAP instead ranks drivers by how well they separate states (fossil production first, residential electricity sixth).

Details: [findings/dml/dml_results.md](findings/dml/dml_results.md) (Phase 5) and [findings/baseline/baseline_results.md](findings/baseline/baseline_results.md) (Phase 4).

## Project status

| Phase | What | Write-up |
|---|---|---|
| Data | Loaders for EPA, EIA, BEA, BTS, FHWA, NOAA, USDA, USGS; panel and spatial weights | [data_sources.md](data_sources.md) |
| 3 | Exploratory analysis, emissions decomposition, EPA iron and steel breaks | [findings/EDA/eda_review.md](findings/EDA/eda_review.md) |
| 4 | Fixed-effects baseline (8 specifications) and XGBoost + SHAP baseline, on the 70/15/15 split | [findings/baseline/baseline_results.md](findings/baseline/baseline_results.md) |
| 5 | DoubleML partially linear models: 5 learners, robustness versions, placebo, sensitivity, overlap, 8 one-driver models | [findings/dml/dml_results.md](findings/dml/dml_results.md) |

## Repository layout

| Path | Contents |
|---|---|
| `code/data_loaders/` | One loader per source → tidy files in `data/interim/` |
| `code/data_preparation/` | `build_spatial_weights.py`, `build_panel.py` (→ `data/processed/panel.csv`), `decompose_emissions.py`, `split_panel.py` (→ `data/split_data/`) |
| `code/analysis/eda.py` | Phase 3 figures and tables; also the shared chart style |
| `code/models/` | `fe_baseline.py`, `ml_baseline.py` (Phase 4), `dml.py` (Phase 5) |
| `data/` | `interim/` and `processed/` panel files, `split_data/` (train / validation / test), state crosswalk and neighbor list. Raw data is not stored; see [data_sources.md](data_sources.md) |
| `models/` | Fitted models and per-run outputs: `fe_baseline/`, `ml_baseline/`, `dml/` |
| `findings/` | Write-ups and result tables per phase: `EDA/`, `baseline/`, `dml/` |
| `visuals/` | All figures (`dml_*.png` for Phase 5) |
| `parent_paper_repo_material/` | Replication of the parent paper (Munir et al. 2026); see its [README](parent_paper_repo_material/README.md) |

## Run

From the repository root, with the project environment (`pip install -r requirements.txt`; DoubleML ≥ 0.11):

```
# panel (after the loaders have filled data/interim/)
python code/data_preparation/build_panel.py data/interim data/state_crosswalk.csv data/state_neighbors.csv data/processed
python code/data_preparation/decompose_emissions.py data/processed/panel.csv data/processed/decomposition.csv
python code/analysis/eda.py data/processed/panel.csv data/processed/decomposition.csv visuals data/processed

# 70/15/15 split by whole years: train 1997-2014, validation 2015-2018, test 2019-2022
python code/data_preparation/split_panel.py data/processed/panel.csv data/split_data

# Phase 4 baselines (FE first: the ML comparison reads its coefficients)
python code/models/fe_baseline.py data/split_data models findings/baseline
python code/models/ml_baseline.py data/split_data models findings/baseline visuals

# Phase 5 DML (about 25 minutes on 8 cores; --runs "" rebuilds summaries and figures only)
python code/models/dml.py data/split_data models findings/dml visuals
```

Each script's docstring documents its modelling choices and outputs.

## How the split is used

| Part | Years | Fixed effects | XGBoost (Phase 4) | DML (Phase 5) |
|---|---|---|---|---|
| Train | 1997–2014 | estimation | fitting | estimation; nuisance learners chosen by state-grouped CV |
| Validation | 2015–2018 | out-of-sample scores | hyperparameter tuning | held-out nuisance R² only |
| Test | 2019–2022 | out-of-sample scores | final scores | held-out nuisance R² only |

DML does not choose learners on validation: absorbing state and year effects inside a 4-year block leaves too little variation to score within-state nuisance models (see the learner-selection note in [dml_results.md](findings/dml/dml_results.md)).
