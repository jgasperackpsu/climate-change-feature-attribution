# Phase 4 Baseline: Fixed-Effects Results

Two-way (state and year) fixed-effects regression of Y on D_corporate, D_household, their spatial lags, the controls and the three EPA break dummies, plus four comparison specifications (pooled OLS, no spatial lags, Driscoll–Kraay SEs, individual drivers) and a predictive XGBoost + SHAP baseline (sections at the end). Sample: the contiguous 48 states. Standard errors are clustered by state. Inference uses a t-distribution with G − 1 degrees of freedom.

Full table: [models/fe_baseline/table.md](../../models/fe_baseline/table.md). Script: [code/models/fe_baseline.py](../../code/models/fe_baseline.py).

## Train / validation / test split (added 2026-11-09)

Every model in this file is now fit on a 70/15/15 split saved in [data/split_data/](../../data/split_data/) (made by [code/data_preparation/split_panel.py](../../code/data_preparation/split_panel.py)). The split is by whole years, so no state's neighbouring years appear in both train and test:

| Part | Years | Main-sample state-years | Share |
|---|---|---|---|
| Train | 1997–2014 | 864 | 69% |
| Validation | 2015–2018 | 192 | 15% |
| Test | 2019–2022 | 192 | 15% |

- **FE models** are estimated on train only, so every coefficient, SE and test below comes from 1997–2014 (864 state-years, 18 years). Validation and test are used only to score out-of-sample prediction.
- **XGBoost** is tuned by fitting on train and scoring on validation. Test is used only for the final scores.

**What changed from the earlier full-sample run (1997–2022, 1,248 state-years):**

- The corporate effect fell from 0.105 to 0.061. The household effect barely moved (0.028 to 0.034).
- The own-effect gap shrank from 0.077 to 0.027. Its one-sided p went from 0.039 to 0.21, so **the pre-specified corporate > household result no longer holds**, under either test.
- The spillover result reversed. Neighbours' corporate activity is now the significant lag, and neighbours' household activity is not.
- Pooled OLS, the SHAP-vs-FE driver comparison and the "persistence beats XGBoost" result are qualitatively unchanged.

## Pre-specified comparison (decided 2026-11-03, before any DML model was run)

With row-standardized spatial lags, the average total effect of a driver is its own coefficient plus its spatial-lag coefficient (β + θ). The comparison was fixed before Phase 5:

- **Primary: own effects,** β_corporate − β_household. States make climate policy for their own territory, and the hypothesis is about drivers of a state's own emissions.
- **Secondary: total effects,** (β + θ)_corporate − (β + θ)_household. Always reported alongside the primary result, as a main result, not a footnote.
- **Direction:** corporate > household. It was stated in the midterm presentation and the implementation plan, both written before these models were run. That is the basis for reporting a one-sided p-value. The two-sided p-value is always reported first.

The DML in Phase 5 reports both comparisons the same way.

## Headline: the corporate effect is positive; whether it is larger is not established

Across all four versions, a 1 SD increase in the corporate index is associated with 0.061–0.082 log points (about 6–8.5%) higher emissions per person. Every 95% CI is above zero:

| Version | D_corporate | 95% CI | p |
|---|---|---|---|
| main | 0.061 | [0.015, 0.108] | 0.011 |
| no_gdp | 0.082 | [0.038, 0.125] | 0.0004 |
| no_wy | 0.078 | [0.024, 0.132] | 0.006 |
| no_ippu | 0.066 | [0.016, 0.116] | 0.011 |

The household index's own effect is 0.034–0.045. That is about half the corporate effect, but no longer clearly zero: it is significant at 5% when GDP is dropped (p = 0.040) and in no_ippu (p = 0.042), and borderline in main (p = 0.094) and no_wy (p = 0.10).

## Comparison 1 (primary): own effects

$$\text{gap} = \beta_{corp} - \beta_{hh}, \qquad \text{SE(gap)} = \sqrt{V_{cc} + V_{hh} - 2V_{ch}}$$

V is the clustered covariance matrix. The two estimates are negatively correlated (about −0.22 in the main model), which makes the SE of the gap larger than if they were independent. The script checks that gap/SE squared equals linearmodels' Wald test statistic for the same restriction.

| Version | Corporate | Household | Gap | SE | 95% CI | p (two-sided) | p (one-sided, gap > 0)* |
|---|---|---|---|---|---|---|---|
| main | 0.061 | 0.034 | **0.027** | 0.034 | [−0.041, 0.095] | **0.426** | 0.213 |
| no_gdp | 0.082 | 0.045 | 0.037 | 0.036 | [−0.036, 0.110] | 0.311 | 0.155 |
| no_wy | 0.078 | 0.034 | 0.044 | 0.037 | [−0.030, 0.118] | 0.240 | 0.120 |
| no_ippu | 0.066 | 0.041 | 0.025 | 0.035 | [−0.045, 0.095] | 0.480 | 0.240 |

\* One-sided because the direction (corporate > household) was stated in the midterm presentation and implementation plan, before these models were run.

- **Direction:** the corporate point estimate is larger in all four versions, by 0.025–0.044 log points per SD (about 2.5–4.5% more emissions per person).
- **Two-sided:** not significant in any version (p = 0.24–0.48). Every 95% CI includes zero.
- **One-sided:** not significant in any version either (p = 0.12–0.24).
- **no_ippu:** run without the break dummies (see below). The gap is about the same as main (0.025 vs 0.027).
- **The ratio:** corporate is about 1.8 times household in the main model (0.061 / 0.034). That is fine as a description but not as a formal result. The household coefficient's CI includes zero, so any CI for the ratio is unbounded. The gap is the quantity tested.

## Comparison 2 (secondary): total effects

$$\text{gap}_{total} = (\beta + \theta)_{corp} - (\beta + \theta)_{hh}, \qquad \text{SE} = \sqrt{r' V r}, \quad r = (1, -1, 1, -1) \text{ on } (\beta_{corp}, \beta_{hh}, \theta_{corp}, \theta_{hh})$$

Main model:

| | Own (β) | Spillover (θ) | Total (β + θ) | SE of total |
|---|---|---|---|---|
| Corporate | 0.061 | 0.061 | 0.123 | 0.033 |
| Household | 0.034 | 0.036 | 0.070 | 0.032 |
| Gap | 0.027 | | **0.053** | 0.048 |

| Version | Corporate total | Household total | Gap | SE | 95% CI | p (two-sided) | p (one-sided) |
|---|---|---|---|---|---|---|---|
| main | 0.123 | 0.070 | **0.053** | 0.048 | [−0.043, 0.149] | **0.276** | 0.138 |
| no_gdp | 0.148 | 0.088 | 0.059 | 0.050 | [−0.042, 0.161] | 0.247 | 0.123 |
| no_wy | 0.132 | 0.066 | 0.066 | 0.049 | [−0.033, 0.165] | 0.187 | 0.094 |
| no_ippu | 0.149 | 0.054 | 0.095 | 0.045 | [0.005, 0.185] | 0.039 | 0.019 |

Both total effects are positive. Counting spillovers now *widens* the gap: it about doubles, from 0.027 to 0.053 in the main model. In the full-sample run it closed it (0.077 → 0.010). The total gap is still not significant in main, no_gdp or no_wy. It clears 5% only in no_ippu (p = 0.039). That is one of eight gap tests and a secondary comparison, so on its own it should not be read as support for the hypothesis.

**Main result, stated together:** "The corporate index has the larger point estimate for a state's own emissions in every version, but the difference is small and not significant (gap 0.027, 95% CI −0.041 to 0.095; p = 0.43 two-sided, 0.21 one-sided). Counting cross-border spillovers, the gap is larger but still not significant (0.053, 95% CI −0.043 to 0.149)."

## The neighbor effect: W_D_corporate

When corporate activity in neighboring states rises by 1 SD, a state's own emissions per person rise by about 5.5–8.5% (0.054–0.083; p = 0.004–0.036 in main, no_gdp and no_ippu, p = 0.054 without Wyoming). Household activity in neighboring states has no detectable effect (0.012–0.044, p > 0.18).

This is the reverse of the full-sample run, where the household lag was 0.095 (p = 0.011) and the corporate lag was not significant. Since the two runs share the 1997–2014 years, the earlier household spillover was coming largely from 2015–2022. The spillover estimates are sample-dependent, so neither should be presented as a stable finding.

**Possible readings of the corporate lag:**

- **Shared regional energy and industrial cycles:** booms in fossil production and fossil-heavy power are regional (the Bakken in ND and MT, the Marcellus and Utica in PA, WV and OH). A neighbor's production boom can raise the home state's emissions through pipelines, processing, transport and power plants serving the same region.
- **Regional shocks:** year fixed effects remove national shocks but not regional ones (a regional recession, a cold Midwest winter, a regional price spike). Those move neighbors' corporate activity and the state's own emissions together. That would produce the same coefficient without any cross-border mechanism.

Both lags stay in the DML. Without them, neighbors' activity would be left in the error term, which is spatially correlated with each state's own drivers. A one-driver-at-a-time DML with per-component lags could check whether the corporate lag comes through fossil production or grid fossil share.

## Controls

- **log_pop = −0.33** (−0.31 to −0.42 across versions; p = 0.06 in main, 0.02 in no_ippu). Y is per person, so total emissions grow with an elasticity of about 1 − 0.33 = 0.67 to population: within a state, a 10% larger population gives about 6.7% more total emissions, holding the per-person drivers fixed. On train alone this is less precisely estimated than before.
- **log GDP per person = +0.19** (p = 0.046 in main).
- **Heating degree days:** 1,000 more heating degree days raise emissions per person by about 3.7–4.4% (p < 0.001). Cooling degree days have a small positive effect that is not significant (0.007–0.013, p > 0.2).

## Break dummies

The main, no_gdp and no_wy versions include three step dummies for the EPA iron and steel breaks (brk_mn_2010, brk_mi_2010, brk_md_2011). Each is nonzero in a single state, so its clustered SE rests on one cluster and should not be interpreted. They are only there as controls.

no_ippu leaves the dummies out. All three breaks are in iron and steel, which that outcome already excludes. When the full-sample run kept the dummies in no_ippu, they picked up real changes rather than artifacts: brk_md_2011 was absorbing the real 7.5% drop in Maryland's other emissions in 2011.

## Inference: t with G − 1 degrees of freedom

With only 48 clusters, p-values and CIs use a t-distribution with G − 1 = 47 degrees of freedom (46 when Wyoming is dropped). The critical value is 2.012, not 1.96. The clustered covariance includes the usual small-sample correction.

Note on linearmodels: fitting with `debiased=True` applies the covariance correction, but its p-values use t with the residual degrees of freedom (about 790 on the train years). That is effectively normal, not t(47). The script therefore recomputes p-values and CIs with t(G − 1). The p-values in each model's summary.txt are linearmodels' own and should not be quoted.

The effect on the main model ([inference_comparison.csv](inference_comparison.csv) has every term and version):

| Term | Coef | SE | p (normal) | p (t, 47 df) | 95% CI (t, 47 df) |
|---|---|---|---|---|---|
| D_corporate | 0.061 | 0.023 | 0.008 | 0.011 | [0.015, 0.108] |
| D_household | 0.034 | 0.020 | 0.087 | 0.094 | [−0.006, 0.074] |
| W_D_corporate | 0.061 | 0.028 | 0.026 | 0.031 | [0.006, 0.117] |
| W_D_household | 0.036 | 0.033 | 0.279 | 0.284 | [−0.031, 0.102] |

No conclusion changes. Four significance stars drop one level in the state-clustered versions:

- D_corporate in main and in no_ippu, from 1% to 5% (p = 0.011 in both)
- D_household in no_wy, from 10% to not significant (p = 0.102)
- W_D_corporate in no_wy, from 5% to 10% (p = 0.054)

## Comparison specifications (added 2026-11-08)

Four more columns in [table.md](../../models/fe_baseline/table.md), each changing one thing about the main model:

| Version | What changes | D_corporate | D_household | Own gap (p, two-sided) | Total gap (p) |
|---|---|---|---|---|---|
| main | (reference) | 0.061 | 0.034 | 0.027 (0.426) | 0.053 (0.276) |
| pooled | no fixed effects | **0.429** | 0.020 | **0.409 (<0.001)** | 0.342 (<0.001) |
| no_lags | no spatial lags | 0.073 | 0.041 | 0.032 (0.379) | — |
| main_dk | Driscoll–Kraay SEs | 0.061 | 0.034 | 0.027 (0.305) | 0.053 (0.298) |

### Pooled OLS: differences between states inflate the corporate effect seven-fold

Without fixed effects the corporate coefficient is 0.429, seven times the within-state estimate. The gap becomes 0.41 and is overwhelmingly significant. That is the cross-section talking: Wyoming, North Dakota, West Virginia and Louisiana have high corporate indices *and* high emissions per person for reasons that do not change from year to year (resource endowments, the age of the power fleet, long-standing industrial structure). The household coefficient barely moves (0.020), so all of the distortion is on the corporate side.

GDP per person flips sign (+0.19 → −0.23). Pooled R² is 0.81, but RMSE is six times the FE model's (0.265 vs 0.044 on train), because the pooled model is fitting levels across states, not changes within them. Out of sample it is the worst model (test RMSE 0.290).

**Takeaway:** with this split, the within-state design finds no significant gap, while a cross-sectional comparison would report a decisive corporate > household result. This is the strongest argument for fixed effects (and for state-clustered cross-fitting in the DML).

### Ignoring spillovers: both own effects rise a little

Dropping the two spatial lags raises both own coefficients: corporate from 0.061 to 0.073, household from 0.034 to 0.041 (now p ≈ 0.05). The gap barely changes (0.027 → 0.032, p = 0.38). A state's activity is correlated with its neighbors', so leaving out the lags loads part of the neighbor effect onto the state's own indices. In the full-sample run this mostly inflated the household coefficient; here it inflates both.

### Driscoll–Kraay: smaller standard errors, same conclusion

DK SEs are about 30–55% smaller than state-clustered ones. Even so, the own-effect gap is not significant at any bandwidth (p = 0.24–0.30 for bandwidths 1–6; [se_comparison.csv](se_comparison.csv)), and neither is the total-effect gap (p = 0.22–0.31). In the full-sample run DK made the own gap significant; on train alone it does not.

State-clustered SEs remain the primary inference:

- Clustering by state *and* year gives almost the same SEs as clustering by state (own-gap SE 0.034 for both). Correlation across states within a year therefore adds little.
- What DK gives up is within-state serial correlation beyond the bandwidth. Clustering by state allows any serial correlation over the full 18 training years; DK with bandwidth 3 cuts it off after three. The residuals of a state-level emissions model are persistent (slow-moving power fleets and industrial mix), so that cutoff is what makes the SEs shrink.
- DK's justification is asymptotic in T, and T = 18 is short (its inference uses t(17)).

### Drivers one at a time (within-state)

Replacing the two indices with the 8 standardized drivers (per pooled SD, same lags, controls and breaks):

| Driver | Index | Effect per SD | 95% CI | p |
|---|---|---|---|---|
| Residential electricity / person | household | **0.120** | [0.066, 0.174] | <0.001 |
| Fossil production / person | corporate | **0.089** | [0.013, 0.164] | 0.022 |
| Grid fossil share | corporate | **0.086** | [0.046, 0.125] | <0.001 |
| Non-energy PCE / person | household | −0.061 | [−0.157, 0.034] | 0.20 |
| VMT / person | household | **0.026** | [0.008, 0.044] | 0.005 |
| Sand & gravel / person | corporate | 0.025 | [−0.007, 0.056] | 0.13 |
| Industrial GDP share | corporate | −0.005 | [−0.022, 0.013] | 0.59 |
| Air passenger-miles / person | household | −0.004 | [−0.027, 0.019] | 0.73 |

- Three drivers carry most of the effect: residential electricity on the household side, grid fossil share and fossil production on the corporate side. **Two of the three are about electricity.** VMT is also significant, but its effect is a quarter of the size.
- Within R² rises from 0.409 (indices) to 0.537 (drivers), and the drivers model predicts the held-out years much better (test RMSE 0.100 vs 0.150). The equal-weight indices throw away information, since they average strong drivers with ones that have no detectable within-state effect. This supports the EDA concern about whether each index is one thing, and makes the Phase 5 one-driver-at-a-time DML necessary.
- Keep in mind the overlap noted in data_sources.md: electric power emissions are roughly electricity sold × grid fossil share, so these two drivers are partly tied to Y by arithmetic. D_household_noelec exists for this reason and should be run in Phase 5.

## ML baseline: XGBoost on the 70/15/15 split (added 2026-11-08, re-run 2026-11-09)

Script: [code/models/ml_baseline.py](../../code/models/ml_baseline.py). Features: the 8 drivers, cattle per person, the 4 controls, the 2 index spatial lags and the 3 break dummies (18 total), with no state or year identifiers. Hyperparameters were chosen by a 60-draw random search: each draw was fit on train (1997–2014) and scored by validation RMSE (2015–2018). Test (2019–2022) was used only for the final scores.

| Model | Train RMSE | Val RMSE | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| **XGBoost (ML baseline)** | 0.003 | 0.108 | 0.930 | 0.169 | 0.133 |
| XGBoost + state one-hot | 0.002 | 0.104 | 0.934 | 0.163 | 0.127 |
| Pooled linear (same features) | 0.157 | 0.173 | 0.934 | 0.163 | 0.127 |
| Persistence (each state's 2014 value) | — | **0.068** | **0.946** | **0.148** | 0.130 |
| *FE main (for reference)* | 0.044 | 0.082 | 0.944 | 0.150 | 0.129 |
| *FE drivers (for reference)* | 0.039 | 0.068 | 0.975 | **0.100** | **0.080** |

Y is in logs, so a test RMSE of 0.17 means predictions are off by about 17% per state-year. Test errors are larger than validation errors for every model, since 2019–2022 is further from the training years and includes 2020.

- **XGBoost memorizes the training years** (train RMSE 0.003, R² = 1.00). The tuned model (depth 6, 400 trees, learning rate 0.1) was the best of 60 draws on validation, but the gap between train and validation shows it fits noise.
- **Simply carrying each state's 2014 value forward beats XGBoost** on both validation (0.068 vs 0.108) and test (0.148 vs 0.169). XGBoost does no better than a linear regression on the same features. The drivers do not predict *changes* better than "nothing changes". The predictive model's accuracy comes from levels across states, which are exactly what the causal design throws away.
- **High R² is not the point.** R² is high because the cross-state spread of emissions per person is large (Wyoming is about 9× New York). Any model that knows roughly which state it is looking at gets most of it.
- Giving XGBoost state identifiers helps only a little (0.163), because trees cannot extrapolate a state's level shift past the training years.
- The FE drivers model is the only one that clearly beats persistence on test (0.100). It has state effects for levels and within-state driver effects for changes.

## SHAP vs. the within-state estimates (Guo et al.-style comparison)

Figures: [shap_vs_causal.png](../../visuals/shap_vs_causal.png), [shap_beeswarm.png](../../visuals/shap_beeswarm.png). Table: [shap_vs_causal.csv](shap_vs_causal.csv).

To put both on the same scale (log points of Y):

- SHAP importance = mean |φ_j| over all 1,248 state-years (train, validation and test), from the XGBoost model fit on train
- Within-state importance = |β_j| × mean |z_j − z̄_j|, with β_j from the FE drivers spec fit on train. This is the average contribution driver j would make if its within-state effect applied to its full spread.

**Index level: same direction, different size.** Summing SHAP over each index's drivers gives corporate 0.296 vs household 0.078 (ratio 3.8). FE gives 0.050 vs 0.027 (ratio 1.9). Both say corporate > household, but SHAP doubles the ratio and its magnitudes are 3–6× larger, because they include differences between states.

**Driver level: partial agreement** (Spearman ρ = +0.45, p = 0.26, 8 drivers; signs agree for 7 of 8).

| Driver | SHAP rank | Within-state rank | Comment |
|---|---|---|---|
| Fossil production | **1** | 2 | SHAP importance (0.245) is three times the within-state importance (0.081): SHAP also picks up that producing states (WY, ND, WV) are high emitters |
| Grid fossil share | 3 | 3 | Agree |
| Residential electricity | 7 | **1** | Largest within-state effect, but weak at separating states, so prediction ranks it low |
| VMT | 4 | 5 | Close |
| Industrial GDP share | 5 | 7 | **Opposite signs:** SHAP positive (industrial states emit more), FE slightly negative and not significant |
| Non-energy PCE | 6 | 4 | Both negative; the FE CI is wide |
| Sand & gravel | 9 | 6 | Small in both |
| Air passenger-miles | 8 | 8 | Negligible in both |

Three more contrasts that matter for the writeup:

- **Cattle per person is SHAP's #2 feature** (0.112). Within states its correlation with Y is negative (−0.16; data_sources.md). SHAP is picking up the Plains states, not an effect of cattle.
- **Spillovers agree in direction.** SHAP gives W_D_corporate its third-largest importance (0.095) and W_D_household almost none (0.005). The FE model now ranks the lags the same way: the corporate lag is 0.061 (p = 0.031), and the household lag is not significant. SHAP's share is still much larger. Neighbors of energy-producing states are themselves high emitters, which a predictive model rewards.
- **Break dummies get about zero SHAP.** The trees never split on them, which is expected and harmless.

**Takeaway for the paper:** a prediction-based attribution would rank fossil production, cattle and neighbors' corporate activity as the top drivers, and put residential electricity near the bottom. Much of that ranking reflects *differences between states*. The within-state design puts residential electricity first, with fossil production and the grid's fossil share next. That contrast is the motivation for using DML rather than SHAP alone. Phase 5's one-driver-at-a-time DML will replace the FE column of this comparison.

## Files

| File | Contents |
|---|---|
| [data/split_data/](../../data/split_data/) | train/, validation/, test/ panel.csv and split_summary.csv (rows, share, states and years per part) |
| [se_comparison.csv](se_comparison.csv) | Main model: driver SEs and gap p-values under Driscoll–Kraay (bandwidths 1–6), two-way (state + year) clustering and state clustering |
| [shap_vs_causal.csv](shap_vs_causal.csv) | Per feature and per index: SHAP importance (all rows and test only), SHAP sign, FE coefficient per SD with CI, within-state importance, ranks, sign agreement |
| models/ml_baseline/ | xgboost.json, settings.json (split, tuned hyperparameters, features), tuning.csv (all 60 draws with validation RMSE), metrics.csv (train/validation/test for every benchmark), shap_values.csv |
| [gap_tests.csv](gap_tests.csv) | One row per version × comparison (own, total): each side's estimate and SE, the gap, its SE, t, df, 95% CI, two- and one-sided p (and the normal p for comparison), the correlation between the two sides, and the ratio |
| [inference_comparison.csv](inference_comparison.csv) | Every coefficient in every version: normal vs. t(G − 1) p-values and CIs |
| models/fe_baseline/&lt;version&gt;/ | Fitted model, coefficients.csv (t(G − 1) inference), metrics.csv (includes both gap tests and validation/test scores), summary.txt |
