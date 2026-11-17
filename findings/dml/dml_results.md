# Phase 5: Causal Estimation with Double Machine Learning

DoubleML partially linear models of log gross GHG per person on the corporate and household driver indices, fit on the 70/15/15 split's train years (1997–2014, 48 states, 864 state-years), the same sample as the Phase 4 fixed-effects baseline. Script: [code/models/dml.py](../../code/models/dml.py). Phase 4 results: [findings/baseline/baseline_results.md](../baseline/baseline_results.md).

**Re-run 2026-11-17** on the corrected panel: non-energy consumer spending in real 2017 dollars and population per square mile as the size control (see "Panel corrections" in the Phase 4 write-up). The corporate results are essentially unchanged. The household index changed, and its main-result effect rose from 0.035 (p = 0.080) to 0.042 (p = 0.032). The own-effect gap shrank from 0.019 to 0.012.

## Summary

1. **The corporate index has a positive within-state effect.** In the main result, a 1 SD rise in the corporate index raises emissions per person by 0.052 log points (about 5.4%; 95% CI 0.006 to 0.099; p = 0.027). It is positive and significant at 5% with every learner (0.052–0.067) and in every version but one. The exception is the EIA energy-CO₂ outcome (0.037, p = 0.20).
2. **The household index also has a positive effect, of similar size:** 0.042 (95% CI 0.004 to 0.080; p = 0.032). It is significant at 5% with the random forest, XGBoost and best learners, and in two of the seven robustness versions. It is less robust than the corporate effect: it is not significant without Wyoming (0.020) or North Dakota (0.029), and it is mostly residential electricity. Without that driver the household index's effect falls to 0.026 (p = 0.12).
3. **The hypothesis that corporate activity has the larger effect is not supported.** The own-effect gap is 0.012 (95% CI −0.052 to 0.075; p = 0.71 two-sided, 0.36 one-sided). The gap including spillovers is 0.032 (p = 0.48). No learner and no version makes the own-effect gap significant, two-sided or one-sided. Of all the gap tests, only one clears 5% one-sided: the total-effect gap without Wyoming (0.087; one-sided p = 0.029, two-sided p = 0.059). That version drops the state that dominates the corporate index's variation, and it is one of about 18 gap tests, so it is not evidence for the hypothesis.
4. **DML reproduces the fixed-effects baseline.** With linear learners it returns the FE coefficients within cross-fitting noise. With flexible learners the estimates move by less than one standard error. The one-driver-at-a-time models match FE driver by driver (rank correlation 0.98).
5. **The placebo passes.** Next year's indices have no effect on this year's emissions (0.008, p = 0.73; 0.007, p = 0.68).
6. **Both effects are moderately robust to omitted variables.** A confounder would need a partial R² of 0.17 (corporate) or 0.18 (household) with both emissions and the index to remove the effect, and 0.05–0.06 to make it insignificant. Every observed control is weaker than that, except GDP per person, which is strongly related to the indices but only weakly to emissions.
7. **Prediction and causal rankings still disagree.** SHAP from the Phase 4 XGBoost model ranks fossil production first and residential electricity fifth of the 8 drivers. DML ranks residential electricity first. The rank correlation between SHAP and DML is +0.55 (p = 0.16).

## Design

| Choice | What was done | Why |
|---|---|---|
| Model | Partially linear: Y = θ·D + g(X) + ε, D = m(X) + v ("partialling out" score) | The plan's model: one causal coefficient per treatment, flexible nuisance functions |
| Treatments | D_corporate, D_household, W_D_corporate, W_D_household. Each is fit in turn with the other three and the controls in X | D₁ with D₂ in X and the reverse, as planned. The spatial lags as treatments give the total-effect gap |
| Controls (X) | log population per square mile, log real GDP per person, heating and cooling degree days | Same as the FE main model |
| Fixed effects | State effects, year effects and the 3 EPA break dummies are partialled out of every variable first (OLS residuals) | The learners then see only within-state, within-year variation. With linear learners DML equals FE (the in-sample FWL estimate equals the FE value) |
| Cross-fitting | 5 folds, whole states held out (`DoubleMLData(cluster_cols="postal_code")`, the replacement for the deprecated `DoubleMLClusterData`), 20 repetitions with different state-to-fold draws | No state's rows are split across folds, as planned |
| Learner choice | Each nuisance model's hyperparameters and, for "best", the learner type are chosen by 5-fold state-grouped CV MSE on the train years | Prediction error only, never the estimate |
| Inference | Clustered by state from DoubleML's score with the G/(G − 1) correction. t(47) p-values and CIs. Median over repetitions, SE adjusted for spread across repetitions | Matches the FE baseline's inference |
| Validation / test years | Report the chosen nuisance models' held-out R² only (`nuisance_r2.csv`) | See the note on learner selection below |

**Note on learner selection.** The first version chose learners by prediction error on the validation years (2015–2018). That failed. Absorbing state and year effects inside a 4-year block leaves only 8–14% of the train years' within-variation, and models fit on train predicted it worse than a constant. A linear model explains 74% of next year's corporate index within train but gets R² = −1.18 on validation. Selection then favoured near-empty models that did not partial out the controls. In the placebo this produced spurious borderline effects (p ≈ 0.08 for both leads). Choosing by state-grouped CV on the train years, which is what the plan specifies ("best learner by CV"), fixed it, and the placebo is clean. Every result below uses CV-based selection. Validation and test R² are still reported but should not be over-read: the low R² there reflects the 4-year blocks, not nuisance-model failure.

## Main result and learners

[visuals/dml_learners.png](../../visuals/dml_learners.png) · [learner_comparison.csv](learner_comparison.csv)

| Estimator | D_corporate | D_household | Own gap | p (2-sided / 1-sided) | Total gap | p (2-sided) |
|---|---|---|---|---|---|---|
| FE (Phase 4) | 0.061 (0.013) | 0.028 (0.108) | 0.033 | 0.31 / 0.16 | 0.074 | 0.11 |
| DML linear | 0.054 (0.028) | 0.028 (0.098) | 0.025 | 0.43 / 0.22 | 0.070 | 0.11 |
| DML Lasso (squares and interactions) | 0.052 (0.027) | 0.033 (0.065) | 0.020 | 0.52 / 0.26 | 0.054 | 0.21 |
| DML random forest | 0.067 (0.010) | 0.048 (0.009) | 0.020 | 0.51 / 0.26 | 0.043 | 0.35 |
| DML XGBoost | 0.066 (0.008) | 0.043 (0.010) | 0.023 | 0.45 / 0.22 | 0.036 | 0.40 |
| **DML best by CV (main)** | **0.052 (0.027)** | **0.042 (0.032)** | **0.012** | **0.71 / 0.36** | **0.032** | **0.48** |

p-values in parentheses. Units: log points of emissions per person per SD of the index.

- **Linear DML vs. FE:** the household coefficient and both spillover coefficients match (0.028, 0.062, 0.018 vs. 0.028, 0.061, 0.020). The corporate coefficient is 0.054 vs. 0.061. The difference comes from cross-fitting: Wyoming and North Dakota carry 45% of the corporate index's within-state variation (see Overlap), so the repetitions where they are held out move the estimate most (range across repetitions 0.038–0.070).
- **Flexible learners** (random forest, XGBoost) raise the household coefficient to about 0.045, now significant, and the corporate one slightly. The gap stays near 0.02. Curvature in the nuisance functions does not favour either side.
- **Learners chosen in the main result:** Lasso for both of D_corporate's nuisance models, random forest (outcome) and Lasso (treatment) for D_household's, and XGBoost or random forest for the spatial lags. That is why main_best and main_lasso give the same D_corporate estimate. Cross-fitted within-state R² of the nuisance models is 0.18–0.33.
- **Totals with spillovers** (main_best): corporate 0.090 (p = 0.022), household 0.057 (p = 0.046). Neither spillover coefficient is significant (W_D_corporate 0.037, p = 0.24; W_D_household 0.014, p = 0.62). The FE model's significant corporate spillover (0.061, p = 0.034) survives only with linear learners (0.062, p = 0.048). The spillover result is learner- and sample-dependent and should not be presented as a finding.

## Robustness

[visuals/dml_robustness.png](../../visuals/dml_robustness.png) · [gap_tests.csv](gap_tests.csv) · [models/dml/summary.csv](../../models/dml/summary.csv)

All rows use the best learner by CV.

| Version | Corporate | Household | Own gap (p) | Total gap (p) |
|---|---|---|---|---|
| main | 0.052 (0.027) | 0.042 (0.032) | 0.012 (0.71) | 0.032 (0.48) |
| EIA energy CO₂ outcome | 0.037 (0.20) | 0.032 (0.098) | 0.004 (0.92) | 0.044 (0.37) |
| Cattle on the corporate side | 0.067 (0.022) | 0.033 (0.067) | 0.033 (0.35) | 0.053 (0.32) |
| Household index without residential electricity | 0.056 (0.021) | 0.026 (0.12) | 0.029 (0.35) | 0.062 (0.19) |
| No GDP control | 0.084 (0.001) | 0.046 (0.008) | 0.039 (0.18) | 0.057 (0.21) |
| Without Wyoming | 0.066 (0.017) | 0.020 (0.19) | 0.047 (0.13) | 0.087 (0.059) |
| Without North Dakota | 0.055 (0.015) | 0.029 (0.13) | 0.028 (0.34) | 0.043 (0.30) |
| Excl. industrial processes | 0.059 (0.020) | 0.044 (0.015) | 0.012 (0.70) | 0.070 (0.12) |

- **EIA energy CO₂:** the corporate effect weakens (0.037, not significant), and so does the household effect (0.032, p = 0.098). A plausible reason for the corporate drop is that EIA's energy-related CO₂ leaves out methane and process emissions. Fossil production, a corporate driver, emits much of its GHG as methane from wells, mines and pipelines.
- **Without residential electricity:** the household index's effect falls to 0.026 (p = 0.12), and the gap widens to 0.029 but is still not significant (p = 0.35). Residential electricity carries the household side. It is also the driver most tied to the outcome by arithmetic, since electric power emissions are roughly electricity sold × grid fossil share.
- **Without Wyoming or North Dakota:** the corporate effect holds (0.066 and 0.055, both p < 0.02), so it does not rest on either state alone. The household effect does not: it falls to 0.020 and 0.029, neither significant. North Dakota carries a quarter of the household index's within-state variation (see Overlap). Without Wyoming the total-effect gap is 0.087 (two-sided p = 0.059, one-sided 0.029), the only gap test in the DML that clears 5% one-sided.
- **Excluding industrial processes:** the own effects are about equal (gap 0.012). The FE baseline's significant total-effect gap in this version (0.114, p = 0.011) does not survive DML (0.070, p = 0.12).
- **XGBoost versions** (`co2_xgb`, `lvstk_xgb`, `main_xgb`) give the same pattern; see `summary.csv`.

## Placebo: next year's index

| Learner | Next year's corporate index | Next year's household index | Gap |
|---|---|---|---|
| best by CV | 0.008 (p = 0.73) | 0.007 (p = 0.68) | 0.001 (p = 0.96) |
| XGBoost | 0.007 (p = 0.77) | 0.012 (p = 0.56) | −0.004 (p = 0.85) |

The treatments are next year's indices, with this year's indices, the spatial lags and the controls in X (the last year of the train block is dropped, 816 state-years). Future values of the drivers do not predict current emissions once current values are held fixed, which is evidence against reverse causation and against slow-moving confounders that track the indices. Note that a placebo can only fail loudly: it does not rule out confounders that move with the current index alone.

## Sensitivity to omitted variables

[visuals/dml_sensitivity.png](../../visuals/dml_sensitivity.png) · [sensitivity.csv](sensitivity.csv) · [models/dml/main_best/benchmarks.csv](../../models/dml/main_best/benchmarks.csv)

DoubleML's bounds (Chernozhukov et al. 2022), main result:

| Treatment | θ | Bounds at cf_y = cf_d = 0.03 | CI at those bounds | RV | RVa |
|---|---|---|---|---|---|
| D_corporate | 0.052 | [0.044, 0.061] | [0.007, 0.097] | **0.17** | **0.05** |
| D_household | 0.042 | [0.036, 0.048] | [0.006, 0.078] | **0.18** | **0.06** |

- **RV** is the strength (partial R² with both the outcome and the treatment, after the controls) that an omitted confounder would need to push θ to zero. **RVa** is the strength that would make θ insignificant at 5%.
- **Benchmarks** (an omitted confounder as strong as one of the observed variables):
  - For both indices, every control and the other index have partial R² of at most 0.047 with emissions and at most 0.008 with the index, inside the RVa square. A confounder like any of them could not make either effect insignificant.
  - GDP per person is the exception: it explains 29% of the corporate index's residual variation (15% of the household index's) but only about 2% of emissions'. Dropping it raises both estimates (no_gdp: 0.084 and 0.046), so its omission biases upward.
- **Reading:** robust to moderate confounding but not to strong confounding. A 0.17 partial R² is a large share of the within-state variation that is left (the best nuisance models explain 0.18–0.33 of it). Strong unmeasured within-state drivers of emissions that also track an index, such as plant openings and closures or state energy policy, remain the main threat. The household effect's RVa is larger than before the panel correction (0.06, was 0.02), but it is still the less robust of the two across versions (see Robustness).

## Overlap

[visuals/dml_overlap.png](../../visuals/dml_overlap.png) · [overlap.csv](overlap.csv)

Every state's indices vary over 1997–2014, so there is no state with zero variation in the treatment. But the variation that DML and FE actually use, what is left after absorbing state and year effects, is concentrated:

- **Corporate index:** Wyoming 27% and North Dakota 18% (45% together), then Oklahoma and Louisiana (5% each). 41 of the 48 states contribute less than an equal share.
- **Household index:** North Dakota 26%, then Delaware 8%, Nevada 5% and Oregon 5%. 33 states contribute less than an equal share.

The without-Wyoming and without-North-Dakota versions show that the corporate effect does not depend on either state alone (0.066 and 0.055). The household effect is not significant without either one. The effective sample for each index is closer to a handful of states than to 48. That should be said when generalising.

## One driver at a time

[visuals/dml_drivers.png](../../visuals/dml_drivers.png) · [drivers_comparison.csv](drivers_comparison.csv)

8 models: each standardized driver as the treatment, with the other 7 drivers, both index spatial lags and the controls in X.

| Driver | Index | DML effect per SD | 95% CI | p | FE effect |
|---|---|---|---|---|---|
| Residential electricity / person | household | **0.120** | [0.064, 0.176] | <0.001 | 0.122 |
| Fossil production / person | corporate | **0.092** | [0.026, 0.158] | 0.007 | 0.091 |
| Grid fossil share | corporate | **0.084** | [0.045, 0.123] | <0.001 | 0.085 |
| VMT / person | household | **0.024** | [0.004, 0.043] | 0.020 | 0.027 |
| Sand & gravel / person | corporate | 0.025 | [−0.010, 0.059] | 0.16 | 0.025 |
| Air passenger-miles / person | household | −0.005 | [−0.030, 0.020] | 0.68 | −0.004 |
| Industrial GDP share | corporate | −0.008 | [−0.027, 0.010] | 0.37 | −0.005 |
| Real non-energy PCE / person | household | −0.037 | [−0.094, 0.021] | 0.20 | −0.039 |

- Each index is carried by a few drivers. Fossil production and grid fossil share carry the corporate index. Residential electricity carries the household index, with a small VMT effect. Industrial GDP share, sand and gravel, air travel and real non-energy consumption have no detectable effect.
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
| Real non-energy PCE | 4 | 4 |
| Residential electricity | 5 | **1** |
| Industrial GDP share | 6 | 7 |
| Air passenger-miles | 7 | 8 |
| Sand & gravel | 8 | 5 |

Spearman ρ = +0.55 (p = 0.16). Signs agree for 7 of 8 drivers; industrial GDP share is positive in SHAP and slightly negative (not significant) in DML. This replaces the FE column of the Phase 4 SHAP comparison, with the same conclusion. A prediction-based attribution rewards drivers that separate states (fossil production, VMT, industrial share), while the within-state causal design puts household electricity use first. Using SHAP rankings as causal attributions would misstate which drivers to target.

## Caveats

- **Train years only.** Every estimate uses 1997–2014 (864 state-years), to stay comparable with the FE baseline and keep 2015–2022 held out. Precision is lower than a full-sample estimate would give, and the Phase 4 comparison showed that some results (the spillovers in particular) change between periods.
- **Partialling out the fixed effects first.** The year means use all states, so they are not cross-fitted. With 48 states this leakage is negligible, but the nuisance learners only ever see within-state, within-year variation and cannot exploit interactions with the state's level.
- **Partially linear model.** θ is a single average within-state effect per SD. Heterogeneous effects (e.g., larger in energy states) are not estimated.
- **Inference with 48 clusters,** using t(47) and the small-sample correction. Wyoming and North Dakota dominate the identifying variation (see Overlap).
- **Deflator.** Consumer spending is deflated with the national GDP price deflator, since no PCE price index was available locally. Year fixed effects absorb any national deflator, so the choice affects only the driver's scale in the standardized index.

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
