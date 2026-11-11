# Phase 5: Causal Estimation with Double Machine Learning

DoubleML partially linear models of log gross GHG per person on the corporate and household driver indices, fit on the 70/15/15 split's train years (1997–2014, 48 states, 864 state-years), the same sample as the Phase 4 fixed-effects baseline. Script: [code/models/dml.py](../../code/models/dml.py). Phase 4 results: [findings/baseline/baseline_results.md](../baseline/baseline_results.md).

## Summary

1. **The corporate index has a positive within-state effect.** In the main result, a 1 SD rise in the corporate index raises emissions per person by 0.053 log points (about 5.5%; 95% CI 0.008 to 0.099; p = 0.023). It is positive and significant at 5% with every learner (0.053–0.068) and in every version but one. The exception is the EIA energy-CO₂ outcome (0.042, p = 0.15).
2. **The household index's effect is smaller and borderline:** 0.035 (95% CI −0.004 to 0.074; p = 0.080). It is significant at 5% with the random forest and XGBoost learners and in two of the eight versions, and it is mostly residential electricity. Without that driver the household index's effect falls to 0.017 (p = 0.25).
3. **The hypothesis that corporate activity has the larger effect is not supported.** The own-effect gap is 0.019 (95% CI −0.045 to 0.082; p = 0.56 two-sided, 0.28 one-sided). The gap including spillovers is 0.029 (p = 0.54). No learner and no version makes either gap significant at 5%, two-sided or one-sided. The smallest one-sided p is 0.079, for the total-effect gap without Wyoming.
4. **DML reproduces the fixed-effects baseline.** With linear learners it returns the FE coefficients within cross-fitting noise. With flexible learners the estimates move by less than one standard error. The one-driver-at-a-time models match FE driver by driver (rank correlation 0.98).
5. **The placebo passes.** Next year's indices have no effect on this year's emissions (0.007, p = 0.76; 0.011, p = 0.55).
6. **The corporate effect is moderately robust to omitted variables.** A confounder would need a partial R² of 0.17 with both emissions and the index to remove it, and 0.05 to make it insignificant. Every observed control is weaker than that, except GDP per person, which is strongly related to the index but only weakly to emissions.
7. **Prediction and causal rankings still disagree.** SHAP from the Phase 4 XGBoost model ranks fossil production first and residential electricity sixth. DML ranks residential electricity first. The rank correlation between SHAP and DML is +0.33 (p = 0.42).

## Design

| Choice | What was done | Why |
|---|---|---|
| Model | Partially linear: Y = θ·D + g(X) + ε, D = m(X) + v ("partialling out" score) | The plan's model: one causal coefficient per treatment, flexible nuisance functions |
| Treatments | D_corporate, D_household, W_D_corporate, W_D_household. Each is fit in turn with the other three and the controls in X | D₁ with D₂ in X and the reverse, as planned. The spatial lags as treatments give the total-effect gap |
| Controls (X) | log population, log real GDP per person, heating and cooling degree days | Same as the FE main model |
| Fixed effects | State effects, year effects and the 3 EPA break dummies are partialled out of every variable first (OLS residuals) | The learners then see only within-state, within-year variation. With linear learners DML equals FE (checked: the in-sample FWL estimate is 0.0614, the FE value) |
| Cross-fitting | 5 folds, whole states held out (`DoubleMLData(cluster_cols="postal_code")`, the replacement for the deprecated `DoubleMLClusterData`), 20 repetitions with different state-to-fold draws | No state's rows are split across folds, as planned |
| Learner choice | Each nuisance model's hyperparameters and, for "best", the learner type are chosen by 5-fold state-grouped CV MSE on the train years | Prediction error only, never the estimate |
| Inference | Clustered by state from DoubleML's score with the G/(G − 1) correction. t(47) p-values and CIs. Median over repetitions, SE adjusted for spread across repetitions | Matches the FE baseline's inference |
| Validation / test years | Report the chosen nuisance models' held-out R² only (`nuisance_r2.csv`) | See the note on learner selection below |

**Note on learner selection.** The first version chose learners by prediction error on the validation years (2015–2018). That failed. Absorbing state and year effects inside a 4-year block leaves only 8–14% of the train years' within-variation, and models fit on train predicted it worse than a constant. A linear model explains 74% of next year's corporate index within train but gets R² = −1.18 on validation. Selection then favoured near-empty models that did not partial out the controls. In the placebo this produced spurious borderline effects (p ≈ 0.08 for both leads). Choosing by state-grouped CV on the train years, which is what the plan specifies ("best learner by CV"), fixed it: the lead's nuisance model now has R² = 0.70, and the placebo is clean. Every result below uses CV-based selection. Validation and test R² are still reported but should not be over-read: the low R² there reflects the 4-year blocks, not nuisance-model failure.

## Main result and learners

[visuals/dml_learners.png](../../visuals/dml_learners.png) · [learner_comparison.csv](learner_comparison.csv)

| Estimator | D_corporate | D_household | Own gap | p (2-sided / 1-sided) | Total gap | p (2-sided) |
|---|---|---|---|---|---|---|
| FE (Phase 4) | 0.061 (0.011) | 0.034 (0.094) | 0.027 | 0.43 / 0.21 | 0.053 | 0.28 |
| DML linear | 0.054 (0.024) | 0.034 (0.092) | 0.020 | 0.56 / 0.28 | 0.047 | 0.31 |
| DML Lasso (squares and interactions) | 0.053 (0.023) | 0.040 (0.055) | 0.013 | 0.68 / 0.34 | 0.031 | 0.50 |
| DML random forest | 0.068 (0.008) | 0.056 (0.011) | 0.012 | 0.71 / 0.35 | 0.014 | 0.78 |
| DML XGBoost | 0.065 (0.007) | 0.051 (0.012) | 0.013 | 0.67 / 0.33 | 0.015 | 0.74 |
| **DML best by CV (main)** | **0.053 (0.023)** | **0.035 (0.080)** | **0.019** | **0.56 / 0.28** | **0.029** | **0.54** |

p-values in parentheses. Units: log points of emissions per person per SD of the index.

- **Linear DML vs. FE:** the household coefficient and both spillover coefficients match (0.034, 0.062, 0.035 vs. 0.034, 0.061, 0.036). The corporate coefficient is 0.054 vs. 0.061. The difference comes from cross-fitting: Wyoming and North Dakota carry 45% of the corporate index's within-state variation (see Overlap), so the repetitions where they are held out move the estimate most (range across repetitions 0.038–0.070).
- **Flexible learners** (random forest, XGBoost) raise the household coefficient to about 0.05, now significant, and the corporate one slightly. The gap stays near 0.01. Curvature in the nuisance functions does not favour either side.
- **Learners chosen in the main result:** Lasso for both of D_corporate's nuisance models, OLS and Lasso for D_household's, and random forest or XGBoost for the spatial lags. That is why main_best and main_lasso give the same D_corporate estimate. Cross-fitted within-state R² of the nuisance models is 0.17–0.33.
- **Totals with spillovers** (main_best): corporate 0.095 (p = 0.016), household 0.064 (p = 0.052). Neither spillover coefficient is significant (W_D_corporate 0.039, p = 0.24; W_D_household 0.030, p = 0.37). The FE model's significant corporate spillover (0.061, p = 0.031) survives only with linear learners (0.062, p = 0.045). The spillover result is learner- and sample-dependent and should not be presented as a finding.

## Robustness

[visuals/dml_robustness.png](../../visuals/dml_robustness.png) · [gap_tests.csv](gap_tests.csv) · [models/dml/summary.csv](../../models/dml/summary.csv)

All rows use the best learner by CV.

| Version | Corporate | Household | Own gap (p) | Total gap (p) |
|---|---|---|---|---|
| main | 0.053 (0.023) | 0.035 (0.080) | 0.019 (0.56) | 0.029 (0.54) |
| EIA energy CO₂ outcome | 0.042 (0.15) | 0.043 (0.066) | −0.003 (0.94) | 0.020 (0.70) |
| Cattle on the corporate side | 0.069 (0.018) | 0.039 (0.060) | 0.027 (0.46) | 0.049 (0.36) |
| Household index without residential electricity | 0.056 (0.019) | 0.017 (0.25) | 0.041 (0.17) | 0.063 (0.20) |
| No GDP control | 0.085 (0.001) | 0.049 (0.019) | 0.038 (0.21) | 0.039 (0.45) |
| Without Wyoming | 0.067 (0.013) | 0.027 (0.15) | 0.040 (0.21) | 0.066 (0.16) |
| Without North Dakota | 0.055 (0.014) | 0.035 (0.11) | 0.023 (0.46) | 0.043 (0.35) |
| Excl. industrial processes | 0.059 (0.018) | 0.057 (0.009) | −0.001 (0.98) | 0.043 (0.37) |

- **EIA energy CO₂:** the corporate effect weakens (0.042, not significant) and the household effect is about the same as in main. A plausible reason is that EIA's energy-related CO₂ leaves out methane and process emissions. Fossil production, a corporate driver, emits much of its GHG as methane from wells, mines and pipelines.
- **Without residential electricity:** the household index's effect falls to 0.017 (p = 0.25), and the gap widens to 0.041 but is still not significant (p = 0.17, one-sided 0.084). Residential electricity carries the household side. It is also the driver most tied to the outcome by arithmetic, since electric power emissions are roughly electricity sold × grid fossil share.
- **Excluding industrial processes:** the two effects are equal (gap −0.001). The FE baseline's one significant result, the total-effect gap in this version (0.095, p = 0.039), does not survive DML (0.043, p = 0.37).
- **Without Wyoming or North Dakota:** the corporate effect holds (0.067 and 0.055, both p < 0.02), so it does not rest on either state alone.
- **XGBoost versions** (`co2_xgb`, `lvstk_xgb`, `main_xgb`) give the same pattern; see `summary.csv`.

## Placebo: next year's index

| Learner | Next year's corporate index | Next year's household index | Gap |
|---|---|---|---|
| best by CV | 0.007 (p = 0.76) | 0.011 (p = 0.55) | −0.006 (p = 0.83) |
| XGBoost | 0.010 (p = 0.71) | 0.017 (p = 0.56) | −0.006 (p = 0.80) |

The treatments are next year's indices, with this year's indices, the spatial lags and the controls in X (the last year of the train block is dropped, 816 state-years). Future values of the drivers do not predict current emissions once current values are held fixed, which is evidence against reverse causation and against slow-moving confounders that track the indices. Note that a placebo can only fail loudly: it does not rule out confounders that move with the current index alone.

## Sensitivity to omitted variables

[visuals/dml_sensitivity.png](../../visuals/dml_sensitivity.png) · [sensitivity.csv](sensitivity.csv) · [models/dml/main_best/benchmarks.csv](../../models/dml/main_best/benchmarks.csv)

DoubleML's bounds (Chernozhukov et al. 2022), main result:

| Treatment | θ | Bounds at cf_y = cf_d = 0.03 | CI at those bounds | RV | RVa |
|---|---|---|---|---|---|
| D_corporate | 0.053 | [0.045, 0.062] | [0.008, 0.097] | **0.17** | **0.05** |
| D_household | 0.035 | [0.027, 0.042] | [−0.003, 0.073] | 0.14 | 0.02 |

- **RV** is the strength (partial R² with both the outcome and the treatment, after the controls) that an omitted confounder would need to push θ to zero. **RVa** is the strength that would make θ insignificant at 5%.
- **Benchmarks** (an omitted confounder as strong as one of the observed variables):
  - For D_corporate, every control and the household index have a partial R² below 0.035 on both axes, inside the RVa square, so a confounder like any of them could not make the effect insignificant.
  - GDP per person is the exception: it explains 31% of the corporate index's residual variation but only 2.5% of emissions'. Dropping it raises the estimate (no_gdp: 0.085), so its omission biases upward.
  - For D_household, the effect is already insignificant, and a confounder as strong as heating degree days (partial R² 0.035 with emissions) would exceed its RVa.
- **Reading:** robust to moderate confounding but not to strong confounding. A 0.17 partial R² is a large share of the within-state variation that is left (the best nuisance models explain 0.25–0.33 of it). Strong unmeasured within-state drivers of emissions that also track the corporate index, such as plant openings and closures or state energy policy, remain the main threat.

## Overlap

[visuals/dml_overlap.png](../../visuals/dml_overlap.png) · [overlap.csv](overlap.csv)

Every state's indices vary over 1997–2014, so there is no state with zero variation in the treatment. But the variation that DML and FE actually use, what is left after absorbing state and year effects, is concentrated:

- **Corporate index:** Wyoming 27% and North Dakota 18% (45% together), then Oklahoma and Louisiana (5% each). 41 of the 48 states contribute less than an equal share.
- **Household index:** North Dakota 24.5% and Delaware 10%. 32 states contribute less than an equal share.

The without-Wyoming and without-North-Dakota versions show that the corporate effect does not depend on either state alone (0.067 and 0.055). The effective sample for the corporate index is still closer to a handful of energy states than to 48. That should be said when generalising.

## One driver at a time

[visuals/dml_drivers.png](../../visuals/dml_drivers.png) · [drivers_comparison.csv](drivers_comparison.csv)

8 models: each standardized driver as the treatment, with the other 7 drivers, both index spatial lags and the controls in X.

| Driver | Index | DML effect per SD | 95% CI | p | FE effect |
|---|---|---|---|---|---|
| Residential electricity / person | household | **0.119** | [0.062, 0.176] | <0.001 | 0.120 |
| Fossil production / person | corporate | **0.089** | [0.023, 0.156] | 0.010 | 0.089 |
| Grid fossil share | corporate | **0.084** | [0.045, 0.124] | <0.001 | 0.086 |
| VMT / person | household | **0.024** | [0.004, 0.043] | 0.019 | 0.026 |
| Sand & gravel / person | corporate | 0.024 | [−0.010, 0.058] | 0.15 | 0.025 |
| Air passenger-miles / person | household | −0.005 | [−0.030, 0.020] | 0.69 | −0.004 |
| Industrial GDP share | corporate | −0.008 | [−0.026, 0.011] | 0.40 | −0.005 |
| Non-energy PCE / person | household | −0.057 | [−0.143, 0.030] | 0.19 | −0.061 |

- Each index is carried by a few drivers. Fossil production and grid fossil share carry the corporate index. Residential electricity carries the household index, with a small VMT effect. Industrial GDP share, sand and gravel, air travel and non-energy consumption have no detectable effect.
- Two of the three largest effects are about electricity (grid fossil share and residential electricity). Electric power emissions are roughly electricity sold × grid fossil share, so part of that effect is accounting rather than behaviour.
- DML and FE agree almost exactly (Spearman ρ = 0.98 on importance). For these drivers, flexible nuisance models change nothing; the linear FE specification was not hiding curvature.

## SHAP vs. DML driver rankings

[visuals/dml_driver_ranks.png](../../visuals/dml_driver_ranks.png) · [shap_vs_dml.csv](shap_vs_dml.csv)

Importance in log points of Y. SHAP uses mean |φ| from the Phase 4 XGBoost model. DML uses |θ| × the mean absolute deviation of the standardized driver.

| Driver | SHAP rank | DML rank |
|---|---|---|
| Fossil production | 1 | 2 |
| Grid fossil share | 2 | 3 |
| VMT | 3 | 6 |
| Industrial GDP share | 4 | 7 |
| Non-energy PCE | 5 | 4 |
| Residential electricity | 6 | **1** |
| Air passenger-miles | 7 | 8 |
| Sand & gravel | 8 | 5 |

Spearman ρ = +0.33 (p = 0.42). Signs agree for 7 of 8 drivers; industrial GDP share is positive in SHAP and slightly negative (not significant) in DML. This replaces the FE column of the Phase 4 SHAP comparison, with the same conclusion. A prediction-based attribution rewards drivers that separate states (fossil production, industrial share, VMT), while the within-state causal design puts household electricity use first. Using SHAP rankings as causal attributions would misstate which drivers to target.

## Caveats

- **Train years only.** Every estimate uses 1997–2014 (864 state-years), to stay comparable with the FE baseline and keep 2015–2022 held out. Precision is lower than a full-sample estimate would give, and the Phase 4 comparison showed that some results (the spillovers in particular) change between periods.
- **Partialling out the fixed effects first.** The year means use all states, so they are not cross-fitted. With 48 states this leakage is negligible, but the nuisance learners only ever see within-state, within-year variation and cannot exploit interactions with the state's level.
- **Partially linear model.** θ is a single average within-state effect per SD. Heterogeneous effects (e.g., larger in energy states) are not estimated.
- **Inference with 48 clusters,** using t(47) and the small-sample correction. Wyoming and North Dakota dominate the identifying variation (see Overlap).

## Files

| File | Contents |
|---|---|
| models/dml/&lt;run&gt;/ | settings.json (sample, learners and parameters used, versions), tuning.csv (every CV candidate), coefficients.csv, gap_tests.csv, nuisance_r2.csv (CV, cross-fit, validation and test R²), repetitions.csv, sensitivity.csv; benchmarks.csv for main_best |
| [models/dml/summary.csv](../../models/dml/summary.csv) | Every run's coefficients |
| [gap_tests.csv](gap_tests.csv) | Own, total, total-corporate and total-household combinations for every run |
| [learner_comparison.csv](learner_comparison.csv) | Main specification: FE and each DML learner |
| [sensitivity.csv](sensitivity.csv) | Bounds at cf_y = cf_d = 0.03, RV and RVa for every run and term |
| [overlap.csv](overlap.csv) | Per state: mean, range, SD and within-state residual SD of each index, and its share of the residual variance |
| [drivers_comparison.csv](drivers_comparison.csv) | Per driver: DML (best and linear) and FE estimates, importance and ranks |
| [shap_vs_dml.csv](shap_vs_dml.csv) | Per driver: SHAP and DML importance, ranks and sign agreement |
| visuals/dml_*.png | learners, robustness, drivers, driver_ranks, sensitivity, overlap |

Runs: main_{linear, lasso, rf, xgb, best}, co2_{xgb, best}, lvstk_{xgb, best}, noelec_best, no_gdp_best, no_wy_best, no_nd_best, no_ippu_best, placebo_{xgb, best}, drivers_{linear, best}. Reproduce with `python code/models/dml.py data/split_data models findings/dml visuals` (about 25 minutes on 8 cores). `--runs ""` rebuilds the summary files and figures from saved runs.
