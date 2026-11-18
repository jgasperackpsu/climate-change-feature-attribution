# Post-checks

Two checks on the full-sample addendum ([findings/dml/full_sample.md](../findings/dml/full_sample.md)): on 1997–2014 the own-effect gap (D_corporate − D_household) is not significant; on 1997–2022 it is (DML 0.082, p = 0.031). Nothing in Phases 1–6 was changed. Both checks reuse the existing estimators unchanged:

- **FE:** the fe_baseline.py main spec, state-clustered SEs, t(47).
- **DML:** the main_best estimator (`dml_fit` in code/models/full_sample.py): state and year effects and break dummies partialled out, CV-chosen learners, 5 state folds × 20 repetitions, clustered t(47).

Sample: the corrected panel ([data/processed/panel.parquet](../data/processed/panel.parquet)), contiguous 48 states. One-sided p tests effect > 0 (coefficients, changes) or corporate > household (gaps).

| File | Contents |
|---|---|
| [period_interaction.py](period_interaction.py) → [period_interaction.csv](period_interaction.csv) | Check 1 |
| [drop_2020.py](drop_2020.py) → [drop_2020.csv](drop_2020.csv) | Check 2 |

Run from the repository root: `python post_checks/drop_2020.py data/processed/panel.parquet post_checks` (about 5 minutes) and `python post_checks/period_interaction.py data/processed/panel.parquet post_checks` (about 7 minutes).

## Check 1 (exploratory): did the effects change between 1997–2014 and 2015–2022?

**Exploratory:** the split was chosen after seeing that the result depends on the period.

**Model:** all 26 years (1,248 state-years), with D_corporate × late and D_household × late added, where late = 1 for 2015–2022. The period's own effect is absorbed by the year effects. The spatial lags are not interacted. DML treats the two interactions as extra treatments, six in all.

| | FE | p (two / one-sided) | DML main_best | p (two / one-sided) |
|---|---|---|---|---|
| Corporate effect, 1997–2014 | 0.101 | 0.002 / 0.001 | 0.107 | <0.001 / <0.001 |
| Corporate effect, 2015–2022 | 0.116 | 0.001 / 0.001 | 0.120 | 0.001 / <0.001 |
| **Change in corporate effect** | **+0.015** [−0.009, 0.038] | 0.21 / 0.11 | **+0.015** [−0.011, 0.041] | 0.24 / 0.12 |
| Household effect, 1997–2014 | 0.021 | 0.32 | 0.028 | 0.18 |
| Household effect, 2015–2022 | 0.016 | 0.38 | 0.027 | 0.24 |
| Change in household effect | −0.005 | 0.70 | −0.001 | 0.94 |
| Own gap, 1997–2014 | 0.081 | 0.068 / 0.034 | 0.079 | 0.047 / 0.024 |
| Own gap, 2015–2022 | 0.100 | 0.026 / 0.013 | 0.093 | 0.032 / 0.016 |
| **Change in own gap** | +0.019 [−0.016, 0.055] | 0.28 | +0.016 [−0.015, 0.047] | 0.31 |

**Reading:**

- **The effects did not change much between the periods.** The corporate effect is about 0.015 larger in 2015–2022, a small change that is not significant. The household effect is flat, and the gap changes by about 0.02 (p ≈ 0.3).
- **The period dependence comes from the estimation window, not from a later change in the effect.** Estimated with all 26 years, the 1997–2014 corporate effect is 0.10–0.11, against 0.052 (DML) and 0.061 (FE) when only 1997–2014 is used. The 1997–2014 data show the larger effect once state effects are taken over the full 26 years.
- **What a longer window changes:** state effects remove each state's mean. Over 26 years the within-state deviations include more slow, long-run movement (multi-decade shifts in production and the power mix), and emissions follow those movements more closely than the year-to-year changes that dominate an 18-year window. The full-sample gap therefore rests on longer-run within-state variation, and the train-years gap on shorter-run variation.
- **Caveat on that reading:** it is an interpretation of why the two windows differ, not a separately tested mechanism. The interaction model is also not identical to a 1997–2014-only fit, since the state effects, year effects and nuisance models all use 26 years.

## Check 2: the 1997–2022 main DML without 2020

**Model:** the full sample minus 2020 (25 years, 1,200 state-years), with everything else as in full_sample.md. FE on the same sample is shown for reference.

| | With 2020 (n = 1,248) | p (two / one-sided) | Without 2020 (n = 1,200) | p (two / one-sided) |
|---|---|---|---|---|
| **DML** D_corporate | 0.107 | 0.001 / <0.001 | 0.102 | 0.001 / <0.001 |
| **DML** D_household | 0.025 | 0.18 / 0.089 | 0.025 | 0.17 / 0.084 |
| **DML own gap** | **0.082** [0.008, 0.156] | **0.031 / 0.016** | **0.076** [0.006, 0.146] | **0.033 / 0.017** |
| **DML** total gap | 0.033 | 0.45 / 0.23 | 0.039 | 0.35 / 0.17 |
| FE D_corporate | 0.105 | 0.002 / 0.001 | 0.101 | 0.002 / 0.001 |
| FE D_household | 0.024 | 0.21 / 0.11 | 0.023 | 0.21 / 0.11 |
| FE own gap | 0.081 | 0.056 / 0.028 | 0.077 | 0.060 / 0.030 |
| FE total gap | 0.033 | 0.45 / 0.22 | 0.042 | 0.31 / 0.16 |

**Reading:** COVID is not driving the full-sample gap. Without 2020 the DML own gap is 0.076 (p = 0.033, one-sided 0.017), against 0.082 with it. The corporate effect, the household effect and the total gap barely move. The nuisance learners chosen are the same types as in the full-sample run.

## What the two checks say together

The full-sample own-effect gap is not a 2020 artifact, and it is not a new effect that appeared after 2014. It appears when the model uses each state's full 26-year history: with that window the corporate effect is about 0.10 per SD in both periods and the gap is about 0.08. On the 18-year train window the same estimator finds a corporate effect of about half that and no significant gap.

The pre-specified train-years test remains the primary result. The full-sample gap is a robustness result that depends on the length of the window.
