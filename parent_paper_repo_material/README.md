# Parent paper replication

Main parent paper: Munir, S., Rashid, M., Shams, H. A., & Ahmed, S. (2026). Infrastructure, institutions, and the resilience dividend: A hybrid Econometric–Machine learning analysis of CO2 emissions in Central and Eastern Europe. *Environmental Management: Economics and Policy*, 2, 100006. https://doi.org/10.1016/j.jemecp.2026.100006

The paper combines panel econometrics (unit roots, cointegration, PMG-ARDL, Granger causality, panel VAR, fixed effects) with double machine learning, tree ensembles and SHAP attribution. That mix of causal and prediction-based attribution is what this project builds on.

| Folder | Contents |
|---|---|
| `codes/code_pp3/` | `fetch_wdi.py` (downloads the data), `econometrics.py` (Tables 2–15), `ml_models.py` (Tables 16–20), `run_pp3.py` (runs everything) |
| `PP_data/pp3_data/` | `pp3_panel_raw.csv`: 18 CEE countries × 2000–2024 from World Bank WDI and WGI. See `PP_data/pp3_data/README.md` |
| `TTV split/pp3_split/` | `split_panel.py`: 70/15/15 train / validation / test split |
| `results/pp3/` | Output tables (CSV and `tables.md`) and SHAP figures |

## Run

From the repository root, using the project environment (it has `doubleml`, `statsmodels`, `linearmodels` and `shap`):

```
source .venv314/bin/activate
python parent_paper_repo_material/codes/code_pp3/fetch_wdi.py               # data -> PP_data/pp3_data/
python parent_paper_repo_material/codes/code_pp3/run_pp3.py                 # tables + figures -> results/pp3/
python "parent_paper_repo_material/TTV split/pp3_split/split_panel.py"      # 70/15/15 split
```

Each script's docstring lists the modelling choices the paper does not report. The split script splits by whole years in time order by default (`--by entity` splits by whole countries instead), because a random row split would leak information across a panel. Every analysis script takes `--data`, so it can run on the full panel or on a split file.

## Replication status (2026-11-08)

Reproduces closely:
- Descriptive statistics, several values exactly (e.g. CO₂ median 4.918, electricity maximum 7,354.893)
- Random forest test R² (0.943 vs 0.931)
- SHAP and permutation importance of ln electricity (0.432 / 1.460 vs 0.407 / 1.558)
- Fixed-effects coefficients on ln electricity, ln GDP and trade
- DML effects of ln electricity and civic engagement
- Hausman test

Does not reproduce:
- PMG-ARDL long-run coefficients
- Kao cointegration test
- DML signs for ln GDP, regulatory quality and urbanization

Likely causes:
- Unreported EViews settings
- World Bank data revisions (the 2024 WGI revision; real 2023–2024 values instead of the paper's filled values)
- Strong collinearity among the governance variables
- Internal inconsistencies in the paper (e.g. ARDL t-statistics above 50, and a bounds-test F(7, 137) that doesn't fit 18 countries × 25 years)
