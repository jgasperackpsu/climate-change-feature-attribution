# Phase 4 Baseline: Fixed-Effects Results

Two-way (state and year) fixed-effects regression of Y on D_corporate, D_household, their spatial lags, the controls and the three EPA break dummies, plus four comparison specifications (pooled OLS, no spatial lags, Driscoll–Kraay SEs, individual drivers) and a predictive XGBoost + SHAP baseline (sections at the end). Sample: the contiguous 48 states, 1997–2022 (1,248 state-years). Standard errors are clustered by state. Inference uses a t-distribution with G − 1 degrees of freedom.

Full table: [models/fe_baseline/table.md](../../models/fe_baseline/table.md). Script: [code/models/fe_baseline.py](../../code/models/fe_baseline.py).

## Pre-specified comparison (decided 2026-11-03, before any DML model was run)

With row-standardized spatial lags, the average total effect of a driver is its own coefficient plus its spatial-lag coefficient (β + θ). Own and total effects give opposite answers to "which is larger" (see below), so the comparison was fixed before Phase 5:

- **Primary: own effects,** β_corporate − β_household. States make climate policy for their own territory, and the hypothesis is about drivers of a state's own emissions.
- **Secondary: total effects,** (β + θ)_corporate − (β + θ)_household. Always reported alongside the primary result, as a main result, not a footnote.
- **Direction:** corporate > household. It was stated in the midterm presentation and the implementation plan, both written before these models were run. That is the basis for reporting a one-sided p-value. The two-sided p-value is always reported first.

The DML in Phase 5 reports both comparisons the same way.

## Headline: the corporate effect is stable and positive

Across all four versions, a 1 SD increase in the corporate index is associated with 0.105–0.125 log points (about 11–13%) higher emissions per person. Every 95% CI is above zero:

| Version | D_corporate | 95% CI | p |
|---|---|---|---|
| main | 0.105 | [0.042, 0.168] | 0.002 |
| no_gdp | 0.121 | [0.061, 0.180] | 0.0002 |
| no_wy | 0.125 | [0.060, 0.189] | 0.0003 |
| no_ippu | 0.107 | [0.039, 0.174] | 0.003 |

That the corporate effect is positive is not in doubt. The uncertain part is whether it is *larger* than the household effect.

The household index's own effect is 0.028–0.038 and not significant at 5% in any version (p = 0.09–0.20).

## Comparison 1 (primary): own effects

$$\text{gap} = \beta_{corp} - \beta_{hh}, \qquad \text{SE(gap)} = \sqrt{V_{cc} + V_{hh} - 2V_{ch}}$$

V is the clustered covariance matrix. The two estimates are negatively correlated (about −0.28), which makes the SE of the gap larger than if they were independent. The script checks that gap/SE squared equals linearmodels' Wald test statistic for the same restriction.

| Version | Corporate | Household | Gap | SE | 95% CI | p (two-sided) | p (one-sided, gap > 0)* |
|---|---|---|---|---|---|---|---|
| main | 0.105 | 0.028 | **0.077** | 0.043 | [−0.009, 0.163] | **0.079** | 0.039 |
| no_gdp | 0.121 | 0.036 | 0.084 | 0.041 | [0.001, 0.167] | 0.047 | 0.023 |
| no_wy | 0.125 | 0.031 | 0.093 | 0.046 | [0.002, 0.185] | 0.046 | 0.023 |
| no_ippu | 0.107 | 0.038 | 0.069 | 0.045 | [−0.021, 0.158] | 0.131 | 0.066 |

\* One-sided because the direction (corporate > household) was stated in the midterm presentation and implementation plan, before these models were run.

- **Direction:** the corporate effect is larger in all four versions, by 0.07–0.09 log points per SD (about 7–10% more emissions per person).
- **Two-sided:** the gap is not significant in the main model (p = 0.079; the 95% CI includes zero). It clears 5% only when GDP is dropped or Wyoming is left out, and only just (p ≈ 0.047).
- **One-sided:** significant at 5% in three of the four versions, including the main model (p = 0.039). no_ippu is the exception (p = 0.066).
- **no_ippu:** now run without the break dummies (see below). The gap is a little smaller (0.069 vs 0.077), as expected: removing industrial processes takes away part of what the corporate index drives. The corporate coefficient itself barely changes (0.105 → 0.107). The household coefficient rises slightly (0.028 → 0.038).
- **The ratio:** "3.7 times" (0.105 / 0.028 in the main model) is fine as a description, but not as a formal result. The household coefficient's CI includes zero, so any CI for the ratio is unbounded. The gap is the quantity tested.

## Comparison 2 (secondary): total effects

$$\text{gap}_{total} = (\beta + \theta)_{corp} - (\beta + \theta)_{hh}, \qquad \text{SE} = \sqrt{r' V r}, \quad r = (1, -1, 1, -1) \text{ on } (\beta_{corp}, \beta_{hh}, \theta_{corp}, \theta_{hh})$$

Main model:

| | Own (β) | Spillover (θ) | Total (β + θ) | SE of total |
|---|---|---|---|---|
| Corporate | 0.105 | 0.028 | 0.133 | 0.036 |
| Household | 0.028 | 0.095 | 0.123 | 0.035 |
| Gap | 0.077 | | **0.010** | 0.046 |

| Version | Corporate total | Household total | Gap | SE | 95% CI | p (two-sided) | p (one-sided) |
|---|---|---|---|---|---|---|---|
| main | 0.133 | 0.123 | **0.010** | 0.046 | [−0.082, 0.102] | **0.834** | 0.417 |
| no_gdp | 0.154 | 0.139 | 0.015 | 0.046 | [−0.076, 0.107] | 0.740 | 0.370 |
| no_wy | 0.147 | 0.122 | 0.026 | 0.048 | [−0.071, 0.122] | 0.594 | 0.297 |
| no_ippu | 0.146 | 0.122 | 0.024 | 0.048 | [−0.072, 0.120] | 0.622 | 0.311 |

Both total effects are clearly positive (each CI is well above zero). The gap between them is close to zero in every version.

**Main result, stated together:** "Corporate activity has the larger effect on a state's own emissions (gap 0.077, 95% CI −0.009 to 0.163; p = 0.08 two-sided, 0.04 one-sided). Once cross-border spillovers are counted, the two are similar (gap 0.010, 95% CI −0.082 to 0.102)."

The reversal comes from where each side's effect lands. The corporate effect lands almost entirely in the state where the activity happens. Household activity seems to raise emissions mainly in neighboring states, through power plants and roads across the state line.

## The neighbor effect: W_D_household

When household activity in neighboring states rises by 1 SD, a state's own emissions per person rise by about 8–10% (p ≈ 0.01–0.02 in every version). Corporate activity in neighboring states has no detectable effect (0.02–0.04, p > 0.29).

**Suggested sentence for the results:** "The positive effect of neighbors' household activity likely reflects emissions crossing state lines through trade and travel. Power plants in one state partly meet electricity demand in neighboring states, and commuting and through-traffic put part of a neighboring state's travel demand on the home state's roads and fuel sales."

This is the reason to keep both spatial lags in the DML. Without them, neighbors' demand would be left in the error term, which is spatially correlated with each state's own drivers.

**Caveats:**

- **Regional shocks:** year fixed effects remove national shocks but not regional ones, such as a regional recession, a cold winter across the Midwest, or a regional gas price spike. Those move neighbors' household activity and the state's own emissions together. That would produce the same positive coefficient without any trade.
- **Possible test:** the electricity explanation predicts that the effect comes through the residential electricity component and should be stronger for states that export power (e.g., WV, WY, ND, PA, AL). The commuting explanation predicts it comes through VMT. The one-driver-at-a-time DML in Phase 5 could include spatial lags for each component and check this.

## Controls

- **log_pop = −0.33** (−0.33 to −0.38 across versions, p < 0.02). Y is per person, so total emissions grow with an elasticity of about 1 − 0.33 = 0.67 to population: within a state, a 10% larger population gives about 6.7% more total emissions, holding the per-person drivers fixed.
- **Heating degree days:** 1,000 more heating degree days raise emissions per person by about 3.6–4.0% (p < 0.001). Cooling degree days have a smaller, negative and only marginally significant effect.

## Break dummies

The main, no_gdp and no_wy versions include three step dummies for the EPA iron and steel breaks (brk_mn_2010, brk_mi_2010, brk_md_2011). Each is nonzero in a single state, so its clustered SE rests on one cluster and should not be interpreted. They are only there as controls.

no_ippu leaves the dummies out. All three breaks are in iron and steel, which that outcome already excludes. With the dummies in, they picked up real changes rather than artifacts: brk_md_2011 (−0.019, p = 0.04) was absorbing the real 7.5% drop in Maryland's other emissions in 2011.

## Inference: t with G − 1 degrees of freedom

With only 48 clusters, p-values and CIs use a t-distribution with G − 1 = 47 degrees of freedom (46 when Wyoming is dropped). The critical value is 2.012, not 1.96. The clustered covariance includes the usual small-sample correction.

Note on linearmodels: fitting with `debiased=True` applies the covariance correction, but its p-values use t with the residual degrees of freedom (1,164 here). That is effectively normal (critical value 1.962), not t(47). The script therefore recomputes p-values and CIs with t(G − 1). The p-values in each model's summary.txt are linearmodels' own and should not be quoted.

The effect on the main model ([inference_comparison.csv](inference_comparison.csv) has every term and version):

| Term | Coef | SE | p (normal) | p (t, 47 df) | 95% CI (t, 47 df) |
|---|---|---|---|---|---|
| D_corporate | 0.105 | 0.031 | 0.0008 | 0.002 | [0.042, 0.168] |
| D_household | 0.028 | 0.021 | 0.190 | 0.196 | [−0.015, 0.071] |
| W_D_corporate | 0.028 | 0.035 | 0.431 | 0.435 | [−0.043, 0.099] |
| W_D_household | 0.095 | 0.036 | 0.008 | 0.011 | [0.023, 0.167] |

No conclusion changes. Three significance stars drop one level:

- W_D_household in the main model, from 1% to 5% (p = 0.011)
- D_household in no_gdp, from 10% to not significant (p = 0.102)
- log_pop in no_gdp, from 1% to 5% (p = 0.011)

## Comparison specifications (added 2026-11-08)

Four more columns in [table.md](../../models/fe_baseline/table.md), each changing one thing about the main model:

| Version | What changes | D_corporate | D_household | Own gap (p, two-sided) | Total gap (p) |
|---|---|---|---|---|---|
| main | (reference) | 0.105 | 0.028 | 0.077 (0.079) | 0.010 (0.834) |
| pooled | no fixed effects | **0.426** | 0.029 | **0.397 (<0.001)** | 0.381 (<0.001) |
| no_lags | no spatial lags | 0.109 | **0.050** | 0.059 (0.192) | — |
| main_dk | Driscoll–Kraay SEs | 0.105 | 0.028 | 0.077 (**0.005**) | 0.010 (0.843) |

### Pooled OLS: differences between states inflate the corporate effect four-fold

Without fixed effects the corporate coefficient is 0.426, four times the within-state estimate. The gap becomes 0.40 and is overwhelmingly significant. That is the cross-section talking: Wyoming, North Dakota, West Virginia and Louisiana have high corporate indices *and* high emissions per person for reasons that do not change from year to year (resource endowments, the age of the power fleet, long-standing industrial structure). The household coefficient barely moves (0.029), so all of the distortion is on the corporate side.

The controls show the same problem. GDP per person flips sign (+0.17 → −0.21) and cooling degree days flip from negative to positive. Pooled R² is 0.81, but RMSE is five times the FE model's (0.269 vs 0.054), because the pooled model is fitting levels across states, not changes within them.

**Takeaway:** a cross-sectional comparison would report a decisive corporate > household result that the within-state design does not support. This is the strongest argument for fixed effects (and for state-clustered cross-fitting in the DML).

### Ignoring spillovers: neighbors' demand gets credited to the state's own households

Dropping the two spatial lags raises the household coefficient from 0.028 to 0.050 (now p = 0.021) and leaves the corporate coefficient almost unchanged (0.105 → 0.109). The own-effect gap shrinks from 0.077 to 0.059 (p = 0.19).

That fits the W_D_household result above. A state's household activity is correlated with its neighbors' (shared regional economy). Leaving out neighbors' demand loads part of the cross-border effect onto the state's own household index. Without lags, household activity looks more important for a state's own emissions than it is, and the gap is understated.

### Driscoll–Kraay: smaller standard errors, but not more trustworthy here

DK SEs are 35–60% smaller than state-clustered ones. The own-effect gap becomes significant at any bandwidth (p = 0.0006 to 0.011 for bandwidths 1–6; [se_comparison.csv](se_comparison.csv)). The total-effect gap stays near zero (p ≈ 0.84) either way.

That does **not** mean the clustered SEs were too cautious:

- Clustering by state *and* year gives almost the same SEs as clustering by state (own-gap SE 0.042 vs 0.043). Correlation across states within a year therefore adds little.
- What DK gives up is within-state serial correlation beyond the bandwidth. Clustering by state allows any serial correlation over the full 26 years; DK with bandwidth 3 cuts it off after three. The residuals of a state-level emissions model are persistent (slow-moving power fleets and industrial mix), so that cutoff is what makes the SEs shrink.
- DK's justification is asymptotic in T, and T = 26 is short.

**Decision:** state-clustered SEs with t(G − 1) stay the primary inference. DK is reported as a robustness column: corporate > household holds under it at the 1% level, but it is not the headline p-value.

### Drivers one at a time (within-state)

Replacing the two indices with the 8 standardized drivers (per pooled SD, same lags, controls and breaks):

| Driver | Index | Effect per SD | 95% CI | p |
|---|---|---|---|---|
| Grid fossil share | corporate | **0.110** | [0.066, 0.155] | <0.001 |
| Residential electricity / person | household | **0.109** | [0.043, 0.175] | 0.002 |
| Fossil production / person | corporate | 0.065 | [−0.003, 0.132] | 0.061 |
| Non-energy PCE / person | household | −0.056 | [−0.197, 0.085] | 0.43 |
| Air passenger-miles / person | household | 0.018 | [−0.028, 0.064] | 0.43 |
| Industrial GDP share | corporate | 0.013 | [−0.008, 0.035] | 0.22 |
| VMT / person | household | 0.009 | [−0.018, 0.036] | 0.49 |
| Sand & gravel / person | corporate | 0.002 | [−0.035, 0.039] | 0.91 |

- Two drivers carry almost everything: grid fossil share on the corporate side and residential electricity on the household side. **Both are about electricity.** Fossil production is the only other driver near significance.
- Within R² rises from 0.393 (indices) to 0.489 (drivers). The equal-weight indices throw away information, since they average strong drivers with ones that have no detectable within-state effect. This supports the EDA concern about whether each index is one thing, and makes the Phase 5 one-driver-at-a-time DML necessary.
- Keep in mind the overlap noted in data_sources.md: electric power emissions are roughly electricity sold × grid fossil share, so these two drivers are partly tied to Y by arithmetic. D_household_noelec exists for this reason and should be run in Phase 5.

## ML baseline: XGBoost with a time-blocked split (added 2026-11-08)

Script: [code/models/ml_baseline.py](../../code/models/ml_baseline.py). Features: the 8 drivers, cattle per person, the 4 controls, the 2 index spatial lags and the 3 break dummies (18 total), with no state or year identifiers. Train 1997–2016, test 2017–2022. Hyperparameters chosen by a 60-draw random search with expanding-window CV inside the training years (validation blocks 2009–10, 11–12, 13–14, 15–16).

| Model | CV RMSE | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| **XGBoost (ML baseline)** | 0.087 | **0.964** | **0.120** | **0.091** |
| XGBoost + state one-hot | 0.081 | 0.969 | 0.113 | 0.084 |
| Pooled linear (same features) | 0.168 | 0.935 | 0.162 | 0.124 |
| Persistence (each state's 2016 value) | — | 0.983 | 0.083 | 0.063 |

Y is in logs, so a test RMSE of 0.12 means predictions are off by about 12% per state-year.

- **High R² is not the point.** R² is high because the cross-state spread of emissions per person is large (Wyoming is about 9× New York). Any model that knows roughly which state it is looking at gets most of it.
- **Simply carrying each state's 2016 value forward beats XGBoost** (RMSE 0.083 vs 0.120). The drivers do not predict *changes* better than "nothing changes", which matches the FE model's modest within R² (0.39). The predictive model's accuracy comes from levels across states, which are exactly what the causal design throws away.
- Giving XGBoost state identifiers helps only a little (0.113), because trees cannot extrapolate a state's level shift past the training years.

## SHAP vs. the within-state estimates (Guo et al.-style comparison)

Figures: [shap_vs_causal.png](../../visuals/shap_vs_causal.png), [shap_beeswarm.png](../../visuals/shap_beeswarm.png). Table: [shap_vs_causal.csv](shap_vs_causal.csv).

To put both on the same scale (log points of Y):

- SHAP importance = mean |φ_j| over all 1,248 state-years
- Within-state importance = |β_j| × mean |z_j − z̄_j|. This is the average contribution driver j would make if its within-state effect applied to its full spread.

**Index level: same direction, different size.** Summing SHAP over each index's drivers, corporate 0.266 vs household 0.127 (ratio 2.1). FE: 0.086 vs 0.022 (ratio 3.9). Both say corporate > household. SHAP magnitudes are 3–6× larger because they include differences between states.

**Driver level: the rankings largely disagree** (Spearman ρ = 0.38, p = 0.35, 8 drivers).

| Driver | SHAP rank | Within-state rank | Why they differ |
|---|---|---|---|
| Fossil production | **1** | 3 | SHAP picks up that producing states (WY, ND, WV) are high emitters; the within-state effect is about a third as large and its CI includes zero |
| Grid fossil share | 3 | **2** | Close |
| Residential electricity | 7 | **1** | Largest within-state effect, but weak at separating states, so prediction ranks it low |
| Non-energy PCE | 6 | 4 | Both negative; the FE CI is very wide |
| Industrial GDP share | 4 | 6 | SHAP rewards the cross-section |
| VMT | 5 | 7 | Same |
| Air passenger-miles | 8 | 5 | **Opposite signs:** SHAP negative (hub states have low emissions per person), FE positive but not significant |
| Sand & gravel | 9 | 8 | Negligible in both |

Three more contrasts that matter for the writeup:

- **Cattle per person is SHAP's #2 feature** (0.097). Within states its correlation with Y is negative (−0.16; data_sources.md). SHAP is picking up the Plains states, not an effect of cattle.
- **Spillovers flip.** SHAP gives W_D_corporate its second-largest importance (0.098) and W_D_household almost none (0.006). The FE model finds the reverse: the corporate lag is not significant, and the household lag is 0.095 (p = 0.011). Neighbors of energy-producing states are themselves high emitters, which a predictive model rewards.
- **Break dummies get about zero SHAP.** The trees never split on them, which is expected and harmless.

**Takeaway for the paper:** a prediction-based attribution would rank fossil production, cattle and neighbors' corporate activity as the top drivers. All three are mostly *differences between states*. The within-state design instead points to electricity: the grid's fossil share and household electricity use. That contrast is the motivation for using DML rather than SHAP alone. Phase 5's one-driver-at-a-time DML will replace the FE column of this comparison.

## Files

| File | Contents |
|---|---|
| [se_comparison.csv](se_comparison.csv) | Main model: driver SEs and gap p-values under Driscoll–Kraay (bandwidths 1–6), two-way (state + year) clustering and state clustering |
| [shap_vs_causal.csv](shap_vs_causal.csv) | Per feature and per index: SHAP importance (all rows and test only), SHAP sign, FE coefficient per SD with CI, within-state importance, ranks, sign agreement |
| models/ml_baseline/ | xgboost.json, settings.json (split, tuned hyperparameters, features), tuning.csv (all 60 draws), metrics.csv (train/CV/test for every benchmark), shap_values.csv |
| [gap_tests.csv](gap_tests.csv) | One row per version × comparison (own, total): each side's estimate and SE, the gap, its SE, t, df, 95% CI, two- and one-sided p (and the normal p for comparison), the correlation between the two sides, and the ratio |
| [inference_comparison.csv](inference_comparison.csv) | Every coefficient in every version: normal vs. t(G − 1) p-values and CIs |
| models/fe_baseline/&lt;version&gt;/ | Fitted model, coefficients.csv (t(G − 1) inference), metrics.csv (includes both gap tests), summary.txt |
