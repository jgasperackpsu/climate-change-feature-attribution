# Phase 6: Heterogeneity, Tipping Points and Policy

How the within-state effects from Phase 5 vary across states, whether any driver has a tipping point, what a reduction in each driver would be worth in tons, and which policies move each driver. Script: [code/models/heterogeneity.py](../../code/models/heterogeneity.py). Sample: the contiguous 48 states, train years 1997–2014 (864 state-years), with state effects, year effects and the EPA break dummies partialled out, as in Phases 4 and 5.

## Summary

1. **The corporate effect is smaller where corporate activity is already high.** Per SD of a state's average corporate level, the effect falls by 0.035 log points (p = 0.017, clustered by state). In Wyoming and North Dakota the forest's effect is about 0.03 per SD, against 0.1–0.27 in low-corporate states. GDP per person does not change it (p = 0.48).
2. **The household effect is larger in high-household and richer states.** It rises by 0.045 per SD of the state's average household level (p < 0.001) and by 0.023 per SD of log GDP per person (p = 0.012). It is near zero in West Virginia, Rhode Island, New York and Delaware, and about 0.13–0.14 in Nevada, Maryland, Washington and Minnesota.
3. **Phase 5's corporate estimate is a weighted average dominated by the energy states.** The forest's unweighted average corporate effect is 0.087. Weighting each state by its share of the within-state variation, as DML implicitly does, gives 0.049, close to DML's 0.052, because Wyoming and North Dakota carry 45% of that variation and have small effects. The typical state's corporate effect is larger than the headline number. The household side does not reconcile this way: the forest's weighted average (0.068) is above DML's 0.042, a gap of about 1.4 DML standard errors.
4. **With fixed effects removed, SHAP agrees with the causal ranking.** SHAP on an XGBoost model of within-state emissions ranks drivers almost as DML does (ρ = +0.83, p = 0.010), unlike SHAP on the levels model in Phase 4 (+0.55 against DML). Prediction-based attribution mis-ranks drivers because it is fit on levels, not because it is machine learning.
5. **No tipping point survives.** Six candidate thresholds were tested (the four drivers with a significant DML effect and both indices). None meets all three stability rules. The closest is the corporate index: its slope flattens above about −0.36 SD, and that holds across bootstrap samples, but the threshold moves to +1.09 SD in 1997–2005.
6. **A realistic within-state reduction is worth 1–2.5% of 48-state emissions per driver.** One within-state SD less residential electricity per person (6%) avoids about 154 MMT CO₂e a year (95% CI 83–224) on the 2022 base. A grid fossil share 4.6 points lower avoids 103 MMT (55–150). Fossil production 26% lower at the median state avoids 56 MMT (16–95). VMT 4% lower avoids 37 MMT (6–67). The 48-state total in 2022 was 6,250 MMT. These are not additive.
7. **Local temperature: the indices show nothing; some drivers do, which points to regional confounding.** The indices have no effect on a state's own temperature anomaly (joint p = 0.17), as expected for well-mixed gases. In the driver specification the joint test rejects (p = 0.013), driven by real consumer spending and the spatial lags. That pattern looks like regional weather moving with regional economic activity, a reminder that year effects remove national shocks but not regional ones.

## 1. Causal forest: how each index's effect varies

[visuals/het_cate.png](../../visuals/het_cate.png) · [visuals/het_state_cate.png](../../visuals/het_state_cate.png) · [blp_tests.csv](blp_tests.csv) · [state_cate.csv](state_cate.csv) · [cate_curves.csv](cate_curves.csv)

**Setup.** One EconML `CausalForestDML` per index:

| Choice | What was done | Why |
|---|---|---|
| Outcome and treatment | log gross GHG per person and the index, both with state and year effects and break dummies partialled out | Same within-state variation as Phases 4 and 5 |
| Controls (W) | the other index, both spatial lags, log population density, log real GDP per person, HDD, CDD (partialled out) | Same as the Phase 5 main model |
| Effect modifiers (X) | the state's mean of the index over 1997–2014, and log real GDP per person | The plan's two modifiers. The state mean, not this year's value, is used for "own level": this year's value contains the treatment itself, so the nuisance model could predict the treatment from its own modifier |
| Nuisance learners | chosen by state-grouped 5-fold CV MSE from Lasso, random forest, XGBoost. Corporate: Lasso for both (CV R² 0.28, 0.39). Household: random forest for Y, Lasso for the index (0.28, 0.43) | Prediction error only, as in Phase 5 |
| Cross-fitting | 5 folds, whole states held out (`groups = state`) | No state split across folds |
| Forest | 2,000 honest trees, at least 10 state-years per leaf, half-samples | EconML defaults for inference, larger forest |
| Formal test | Best linear projection: the forest's cross-fitted residuals, Ỹ on T̃, T̃ × z(level), T̃ × z(GDP), with state-clustered SEs and t(47) | The forest's own intervals resample rows, not states, so they ignore within-state correlation |

**Linear heterogeneity tests** (clustered by state, t(47)):

| Index | Average effect | Change per SD of the state's mean level | Change per SD of log GDP per person |
|---|---|---|---|
| D_corporate | 0.087 (p = 0.002) | **−0.035** (95% CI −0.063 to −0.006; p = 0.017) | +0.010 (p = 0.48) |
| D_household | 0.057 (p < 0.001) | **+0.045** (0.022 to 0.067; p < 0.001) | **+0.023** (0.005 to 0.040; p = 0.012) |

**Reading:**

- **Corporate: diminishing effect at high levels.** States whose economies are already built around fossil production and fossil power (WY, ND, OK, LA, MT, NM) have the smallest corporate effects (about 0–0.03 per SD). States with low corporate levels (MA, WA, MD, VA, NH) have the largest (0.24–0.27). One reading is saturation: in an energy state, a further rise in the index mostly scales up activity whose emissions are already high per person, so the log change is small. The low-level states are also where the index varies least within the state, so their forest estimates lean on extrapolation; the linear projection's slope is the reliable summary, not the extreme state values.
- **Household: larger in high-use and richer states.** The household effect is close to zero in WV, RI, NY and DE and about 0.13–0.14 in NV, MD, WA and MN. Where household activity (mostly residential electricity) is high, a further 1-SD rise adds more emissions. The GDP gradient points the same way.
- **The forest averages differ from Phase 5,** and for corporate the reason is the weighting. DML's partially linear θ weights each state-year by its residual treatment variance. Weighting the forest's state effects by each state's share of the within-state variation (the overlap table in Phase 5) gives 0.049 for corporate, matching DML's 0.052. For household the same weighting gives 0.068, above DML's 0.042; the household forest uses a random-forest outcome model and the extra modifiers as features, and the difference is about 1.4 DML standard errors, so treat the forest's household level as less certain than its slope.
- **Ten states have a negative corporate point estimate** (IA, AZ, WI, SD, NV, TX, CO, OH, KY, KS) and four a negative household one. None of these is individually reliable (see the forest bands); they show the spread, not state-specific findings.

## 2. SHAP on the XGBoost nuisance model

[visuals/het_shap_dependence.png](../../visuals/het_shap_dependence.png) · [shap_within.csv](shap_within.csv)

XGBoost fit to within-state emissions (fixed effects and breaks partialled out) from all 8 drivers, both spatial lags and the controls, all also partialled out. This is the outcome nuisance model of the one-driver-at-a-time DML with every driver included. Tuned by state-grouped CV: 300 depth-1 trees, CV R² 0.37.

| Driver | Within-state SHAP rank | DML rank (Phase 5) | Levels SHAP rank (Phase 4) |
|---|---|---|---|
| Grid fossil share | **1** | 3 | 2 |
| Residential electricity | **2** | **1** | 5 |
| Fossil production | 3 | 2 | **1** |
| Sand & gravel | 4 | 5 | 8 |
| VMT | 5 | 6 | 3 |
| Real non-energy PCE | 6 | 4 | 4 |
| Air passenger-miles | 7 | 8 | 7 |
| Industrial GDP share | 8 | 7 | 6 |

- **Rank correlation with DML: +0.83 (p = 0.010).** With the levels model of Phase 4 it was +0.55. The same algorithm, explained the same way, gives a near-causal ranking once the between-state differences are removed. The disagreement in Phases 4 and 5 came from what the model was fit on, not from SHAP.
- **Signs agree for 7 of 8 drivers.** The exception is real non-energy spending (SHAP positive, DML negative and not significant).
- **Dependence plots** (top 4 by within-state SHAP: grid fossil share, residential electricity, fossil production, sand and gravel) are monotone and roughly linear over the bulk of the data, flattening only at the extremes. Depth-1 trees cannot show interactions, but CV preferred them to deeper trees, which is itself evidence that within-state effects are close to additive. Nothing here suggests a threshold.

## 3. Tipping points: segmented regression

[visuals/het_thresholds.png](../../visuals/het_thresholds.png) · [thresholds.csv](thresholds.csv) · [threshold_bootstrap.csv](threshold_bootstrap.csv)

**Model:** Y = b₁x + b₂(x − c)₊ + other regressors + state FE + year FE + breaks, with c searched over the 10th–90th percentiles of x. Drivers use the FE drivers spec (other 7 drivers, lags, controls); indices use the main spec. b₂ is the change in slope above c.

**Rules, fixed before the run.** A threshold is kept only if all three hold:

- **a.** Across 500 bootstrap samples of whole states, re-searching c each time, the 95% interval of b₂ excludes zero.
- **b.** At least half of the bootstrap thresholds are within 0.25 SD of c.
- **c.** Re-estimated separately on 1997–2005 and 2006–2014, both halves give b₂ with the same sign and a threshold within 0.5 SD of c.

| Variable | c (SD) | Slope below → above | Bootstrap 95% CI of b₂ | Share of bootstrap c within 0.25 SD | c in 1997–2005 / 2006–2014 | Rules a / b / c | Kept |
|---|---|---|---|---|---|---|---|
| Residential electricity | −1.43 | 0.014 → 0.141 | [−0.086, 0.224] | 0.62 | −1.43 / +0.62 | ✗ / ✓ / ✗ | no |
| Fossil production | −0.88 | −0.236 → 0.131 | [−0.346, 2.076] | 0.30 | +1.06 / −0.57 | ✗ / ✗ / ✗ | no |
| Grid fossil share | +0.35 | 0.097 → −0.050 | [−0.398, 1.202] | 0.49 | +0.82 / +1.16 | ✗ / ✗ / ✗ | no |
| VMT | −0.42 | 0.011 → 0.033 | [−0.062, 0.098] | 0.32 | −1.28 / −0.36 | ✗ / ✗ / ✗ | no |
| D_corporate | −0.36 | 0.184 → 0.017 | **[−0.242, −0.111]** | **0.86** | +1.09 / −0.32 | ✓ / ✓ / ✗ | no |
| D_household | −1.49 | −0.012 → 0.041 | [−0.055, 0.092] | 0.64 | −1.49 / +0.76 | ✗ / ✓ / ✗ | no |

- **No driver has a stable tipping point.** For the four drivers, the bootstrap interval of the slope change includes zero, and the threshold location jumps between the two halves of the window.
- **The corporate index comes closest.** Its slope is 0.18 per SD below −0.36 SD and about zero above it, the slope change is clearly negative in the bootstrap, and 86% of bootstrap thresholds land within 0.25 SD. Both halves also show a flattening (slope change −0.07 and −0.17), but in 1997–2005 the break sits at +1.09 SD instead. So the flattening is robust, and the location is not. This is the same pattern as the forest's declining corporate effect with level: a gradual saturation rather than a sharp tipping point.
- The naive clustered p-values at the selected c (in thresholds.csv) are smaller than the bootstrap suggests, because they ignore the search over c. They are reported but not used.

## 4. Simulated emissions avoided

[visuals/het_avoided.png](../../visuals/het_avoided.png) · [avoided_national.csv](avoided_national.csv) · [avoided_by_state.csv](avoided_by_state.csv)

Each driver with a significant DML effect, and each index, is reduced by 1 SD in every state, one at a time. Avoided emissions per state = 2022 gross GHG × (1 − e^(−θ × size)), with θ and its 95% CI from Phase 5. Gross GHG is emissions per person × state population. 2022 is outside the estimation years, so this is a scenario on the latest emissions, not a forecast.

Two sizes:

- **One within-state SD:** a typical change in a state's own value over time. This is the realistic policy scale.
- **One pooled SD:** the unit θ is measured in. It mostly reflects differences between states, so it is a very large change for any single state (for fossil production it means eliminating production at the median state).

| Driver | 1 within-state SD is | Avoided, MMT CO₂e / yr (95% CI) | % of 48-state 2022 gross | 1 pooled SD is | Avoided, MMT (95% CI) |
|---|---|---|---|---|---|
| Residential electricity / person | 6% less per person | **154** (83–224) | 2.5% | 25% less | 707 (389–1,007) |
| Grid fossil share | 4.6 points lower | **103** (55–150) | 1.6% | 23.5 points lower | 503 (272–725) |
| Fossil production / person | 26% less at the median state | **56** (16–95) | 0.9% | elimination at the median state | 549 (163–912) |
| VMT / person | 4% less per person | **37** (6–67) | 0.6% | 15% less | 145 (24–264) |
| D_corporate | 0.24 SD of the index | 77 (9–145) | 1.2% | 1 SD | 319 (38–587) |
| D_household | 0.54 SD of the index | 142 (13–268) | 2.3% | 1 SD | 258 (23–483) |

48-state gross GHG in 2022: 6,250 MMT CO₂e.

- **Residential electricity is the largest lever at a realistic scale,** then grid fossil share. Both act through power-sector emissions, which are roughly electricity sold × the grid's carbon intensity, so their effects overlap: cutting both does not avoid 154 + 103 MMT. The same applies across all rows, since every driver was simulated with the others held fixed.
- **The household index's within-state SD is large relative to its pooled SD** (0.54, against 0.24 for corporate), so a typical within-state change in household activity is worth more than a typical change in corporate activity, even though the per-SD effects are similar.
- **With the forest's state-specific effects** in place of a single θ, a 1-SD reduction would avoid 341 MMT for corporate and 386 MMT for household (vs. 319 and 258 with one θ). The household number rises because the household effect is larger in high-emitting, high-use states.
- **By state,** avoided tons scale with each state's emissions (Texas, California, Florida, Pennsylvania and Ohio lead in total; Wyoming, North Dakota and West Virginia per person), since a constant θ is applied to each state's own base. The forest's state effects (section 1) are the place to look for genuinely state-specific differences.

## 5. Policy levers

[policy_levers.csv](policy_levers.csv)

| Driver | Within-state effect (Phase 5) | Policy levers | Note |
|---|---|---|---|
| Residential electricity / person | **0.120** per SD (p < 0.001) | Appliance and equipment efficiency standards; residential building energy codes; utility energy-efficiency resource standards and programs; rate design | Partly arithmetic: the payoff shrinks as the grid decarbonizes, so pair with grid policy |
| Grid fossil share | **0.084** (p < 0.001) | Renewable portfolio / clean energy standards (RPS, CES); carbon pricing on power (cap-and-trade); coal retirement and new-capacity siting | Acts on every kWh sold in the state, household and business |
| Fossil production / person | **0.092** (p = 0.007) | Oil and gas methane rules (leak detection and repair, venting and flaring limits); coal-mine methane capture; production policy (leasing, severance taxes) | Much of the effect is methane, which the EIA energy-CO₂ outcome leaves out (Phase 5) |
| VMT / person | **0.024** (p = 0.020) | Transit investment; land use and zoning (density, transit-oriented development); road pricing and fuel taxes | Vehicle efficiency and EV policy cut emissions per mile, not miles |
| Sand & gravel, industrial GDP share, air passenger-miles, real non-energy spending | not significant | None recommended on this evidence | |

**How this connects to the heterogeneity results:**

- **Efficiency standards and building codes** (residential electricity) target the strongest within-state effect. The household effect is largest in high-use and higher-income states (section 1), so the payoff per unit of policy should be largest there.
- **RPS / clean energy standards** (grid fossil share) are the main corporate-side lever outside the energy states, and they also shrink the payoff from household electricity measures.
- **In the energy states, the index-level effect per SD is small** (section 1), but fossil production carries large per-person emissions there (Wyoming, North Dakota). Methane rules target that directly, and their benefit would not show up in a CO₂-only accounting.
- **Transit and land use** (VMT) have a real but small within-state effect: about a quarter of the electricity effects per SD.

These links come from the estimated driver effects. The project does not estimate the effect of any specific policy; that would need data on policy adoption by state and year.

## 6. Secondary analysis: local temperature

[temperature.csv](temperature.csv)

Two-way FE of each state's temperature anomaly (°F, vs. its 1991–2020 mean) on the drivers, with log population density and log GDP per person, clustered by state, t(47), train years. Degree days are left out because they are computed from temperature.

| Specification | Result |
|---|---|
| Indices and spatial lags | No term significant (p = 0.10–0.52); joint F p = 0.17 |
| 8 drivers and spatial lags | Joint F p = 0.013. Real non-energy spending +0.92 °F per SD (p = 0.001); W_D_corporate −0.49 (p = 0.030) and W_D_household −0.40 (p = 0.047); the rest not significant |

- **Expected result for the indices:** GHGs mix globally, so one state's emissions cannot measurably warm that state. The null confirms there is no mechanical link between the indices and local temperature.
- **The driver-level rejection is a warning, not a finding.** Real consumer spending cannot change a state's temperature within a year. The more plausible reading is that regional weather and regional economic activity move together (a warm year in a region and a regional boom), and year effects only remove national shocks. That is the same regional-shock concern raised for the spatial lags in Phase 4. The main models include heating and cooling degree days, which absorb the direct weather-to-energy-demand channel.
- **Not done:** the urban-heat-island version of the secondary analysis needs the MODIS land-surface-temperature and NLCD land-cover loaders, which were never built (data_sources.md).

## 7. Addendum (exploratory): the gap in energy states vs. other states

**Exploratory.** This split was chosen after section 1 showed the corporate effect falling with the state's level, so its p-values are not confirmatory. One definition only; no alternative splits were run. Script: [code/models/energy_gap.py](../../code/models/energy_gap.py). Tables: [energy_states.csv](energy_states.csv), [energy_gap.csv](energy_gap.csv).

**Definition, fixed and saved before any model was estimated.** Energy states are the top quintile (10 of 48) by mean fossil fuel production per person (SEDS coal + crude + natural gas) over 1997–2014. In order: WY, WV, ND, NM, MT, KY, OK, LA, CO, TX. Utah is 11th, just below Texas (416 vs. 448 MMBtu per person).

**Models** (train years, 864 state-years):

- **FE:** the main spec plus D_corporate × energy and D_household × energy.
- **DML:** the main_best estimator with the two interactions as extra treatments, so six treatments, each fit with the other five and the controls in X.

Both use SEs clustered by state and t(47). One-sided p tests corporate > household.

| Comparison | FE | p (two / one-sided) | DML main_best | p (two / one-sided) |
|---|---|---|---|---|
| Corporate effect, other states | 0.124 | <0.001 / <0.001 | 0.111 | 0.001 / <0.001 |
| Household effect, other states | 0.045 | 0.019 / 0.010 | 0.049 | 0.021 / 0.011 |
| **Own gap, other states** | **0.080** [0.011, 0.149] | **0.025 / 0.012** | **0.062** [−0.017, 0.140] | 0.120 / 0.060 |
| Corporate effect, energy states | 0.031 | 0.21 / 0.10 | 0.045 | 0.22 / 0.11 |
| Household effect, energy states | 0.022 | 0.32 / 0.16 | 0.050 | 0.14 / 0.071 |
| **Own gap, energy states** | 0.010 [−0.074, 0.094] | 0.82 / 0.41 | −0.006 [−0.123, 0.111] | 0.92 / 0.54 |
| **Difference (energy − other)** | −0.070 [−0.158, 0.018] | 0.11 | −0.070 [−0.159, 0.019] | 0.12 |

**Reading:**

- **In the 38 other states the corporate effect is large (0.11–0.12 per SD) and the gap is positive.** It is significant in FE (0.080, p = 0.025) and borderline in DML (0.062, one-sided p = 0.060).
- **In the 10 energy states neither effect is significant and the gap is about zero.** This is the pattern section 1 found: a smaller corporate effect where corporate activity is already high.
- **The difference between the two groups (−0.07) is not significant** in either model (p = 0.11–0.12). The data are consistent with a gap in other states that is absent in energy states, but they cannot reject the gap being the same in both.
- **Why the pooled Phase 5 gap is small:** the energy states carry most of the corporate index's within-state variation (Wyoming and North Dakota alone 45%), so the pooled estimate leans toward their near-zero gap.

## Caveats

- **Train years only** (1997–2014), as in Phases 4 and 5; the simulation applies those effects to 2022 emissions.
- **Forest intervals are not clustered by state.** They are shown only as a guide; every test above uses the state-clustered best linear projection. Individual state effects are noisy, especially in states with little within-state variation.
- **"Own level" is the state's mean over 1997–2014.** It describes which kinds of states respond more, not how a state's response would change as its own level moves.
- **Thresholds were searched on a grid of the 10th–90th percentiles** of each variable. A tipping point outside that range, or one specific to a few states, could not be detected.
- **Simulated reductions are partial-equilibrium** and one driver at a time. They are not additive and do not account for leakage (e.g., reduced fossil production in one state replaced elsewhere).

## Files

| File | Contents |
|---|---|
| models/heterogeneity/forest_&lt;index&gt;/ | settings.json (learners, forest settings), tuning.csv (CV candidates), cate.csv (CATE and forest interval for every state-year) |
| models/heterogeneity/shap_nuisance/ | settings.json, tuning.csv, shap_values.csv |
| [blp_tests.csv](blp_tests.csv) | Clustered linear heterogeneity tests |
| [state_cate.csv](state_cate.csv), [cate_curves.csv](cate_curves.csv) | Mean CATE by state; CATE along each modifier with forest and linear-projection bands |
| [shap_within.csv](shap_within.csv) | Within-state SHAP importance, DML and Phase 4 SHAP importance and ranks |
| [thresholds.csv](thresholds.csv), [threshold_bootstrap.csv](threshold_bootstrap.csv) | Segmented-regression results and rules; every bootstrap draw's threshold and slope change |
| [avoided_national.csv](avoided_national.csv), [avoided_by_state.csv](avoided_by_state.csv) | Simulated avoided emissions, 48 states and by state |
| [policy_levers.csv](policy_levers.csv) | Driver-to-policy mapping |
| [temperature.csv](temperature.csv) | Secondary temperature regressions |
| [energy_states.csv](energy_states.csv), [energy_gap.csv](energy_gap.csv) | Addendum: energy-state definition (all 48 states ranked) and the FE / DML gap by state type |
| models/heterogeneity/energy_gap/ | Addendum: FE coefficients, DML nuisance learners and tuning |
| visuals/het_*.png | cate, state_cate, shap_dependence, thresholds, avoided |

Reproduce: `python code/models/heterogeneity.py data/split_data models findings/heterogeneity visuals` from the repository root, after dml.py (about 2 minutes).
