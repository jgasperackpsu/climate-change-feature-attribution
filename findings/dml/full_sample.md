# Phase 5 Addendum: Full-Sample Confirmation (1997–2022)

Phases 4 and 5 estimate on the train years (1997–2014) only. This addendum re-estimates the two headline models on every year of the corrected panel ([data/processed/panel.parquet](../../data/processed/panel.parquet): contiguous 48 states, 1997–2022, 1,248 state-years) with nothing else changed. Script: [code/models/full_sample.py](../../code/models/full_sample.py). Table: [full_sample.csv](full_sample.csv).

- **FE:** the fe_baseline.py main spec (indices, spatial lags, controls, the three break dummies, state and year effects), SEs clustered by state with the small-sample correction, t(47).
- **DML:** the dml.py main_best estimator. State and year effects and break dummies are partialled out. Each nuisance learner is chosen by 5-fold state-grouped CV MSE from OLS, Lasso, random forest and XGBoost. Cross-fitting uses 5 state folds × 20 repetitions with shared draws, clustered inference with G/(G − 1), t(47).
- **Same estimator check:** the script's DML helper, run on 1997–2014, reproduces the saved main_best coefficients and SEs exactly (e.g., D_corporate 0.052338, SE 0.022984).
- **Selection:** in the main 1997–2014 results, the validation (2015–2018) and test (2019–2022) years were never used to choose DML learners; they only report held-out nuisance R². In this full-sample run there is no held-out block: the CV that chooses the learners runs on all 26 years.

## Results

Effects are log points of emissions per person per SD of the index. One-sided p tests effect > 0 (coefficients) or corporate > household (gaps).

| Estimator | Term | 1997–2014 (train, n = 864) | p two / one-sided | 1997–2022 (full, n = 1,248) | p two / one-sided |
|---|---|---|---|---|---|
| FE | D_corporate | 0.061 [0.014, 0.109] | 0.013 / 0.006 | **0.105** [0.042, 0.168] | 0.002 / 0.001 |
| FE | D_household | 0.028 [−0.006, 0.063] | 0.108 / 0.054 | 0.024 [−0.014, 0.063] | 0.210 / 0.105 |
| FE | Own gap | 0.033 [−0.032, 0.098] | 0.313 / 0.156 | **0.081** [−0.002, 0.164] | 0.056 / **0.028** |
| FE | Total gap | 0.074 [−0.017, 0.166] | 0.110 / 0.055 | 0.033 [−0.053, 0.118] | 0.447 / 0.224 |
| DML main_best | D_corporate | 0.052 [0.006, 0.099] | 0.027 / 0.014 | **0.107** [0.047, 0.166] | 0.001 / <0.001 |
| DML main_best | D_household | 0.042 [0.004, 0.080] | 0.032 / 0.016 | 0.025 [−0.012, 0.062] | 0.179 / 0.089 |
| DML main_best | Own gap | 0.012 [−0.052, 0.075] | 0.714 / 0.357 | **0.082** [0.008, 0.156] | **0.031** / **0.016** |
| DML main_best | Total gap | 0.032 [−0.058, 0.122] | 0.479 / 0.240 | 0.033 [−0.055, 0.121] | 0.451 / 0.225 |

95% CIs in brackets.

## Reading

- **On all 26 years the corporate effect is about twice as large** (0.105–0.107 vs. 0.052–0.061), and FE and DML agree closely.
- **The household effect is smaller and not significant** on the full sample (0.024–0.025), against 0.028 (FE) and 0.042 (DML) on the train years.
- **The own-effect gap becomes significant.** DML: 0.082, p = 0.031 two-sided and 0.016 one-sided. FE: 0.081, p = 0.056 two-sided and 0.028 one-sided. On 1997–2014 neither was close (p = 0.31 and 0.71).
- **The total-effect gap is not significant on either sample** (0.033, p ≈ 0.45 on the full sample). The spillover terms offset the own-effect gap on the full sample, as in the earliest full-sample run.
- **So the primary hypothesis test depends on the sample period.** It is supported on 1997–2022 and not on 1997–2014, with the same estimator and specification. The difference comes from adding 2015–2022, the years held out of every Phase 4–6 model. It is not that those years add unusually much variation: they hold 23% of the corporate index's within-state variation on the full sample, less than their 31% share of rows. Rather, the relationship differs in the later years, and every state's mean is now taken over 26 years instead of 18, which changes the within-state deviations of all years. That is consistent with Phase 6's finding that the corporate effect depends on which states and which variation identify it.
- **The full-sample DML nuisance models are weak for the corporate treatment:** the chosen learner explains about 7% of D_corporate's within-state variation (CV R²; 25% on the train years). The DML and FE estimates nonetheless agree, which suggests the result does not depend on the nuisance fit.

## How to report it

The pre-specified analysis is the train-years estimate: no significant own-effect gap. The full sample is a confirmation run, added after the main results. It finds a significant own-effect gap of about 0.08 (one-sided p = 0.016 for DML), which is the direction stated in the midterm. Both should be reported together: the corporate > household result is not robust to the sample period, and the total-effect gap is not significant in either.

## Files

| File | Contents |
|---|---|
| [full_sample.csv](full_sample.csv) | FE and DML, own and total gaps and both indices, 1997–2014 (saved Phase 4/5 results) and 1997–2022 |
| models/full_sample/ | fe_main_coefficients.csv (every term, full sample), dml_main_best_nuisance.csv (learner chosen per nuisance and its CV R²), dml_main_best_tuning.csv |

Reproduce: `python code/models/full_sample.py data/processed/panel.parquet data/split_data models findings/dml` (about 3 minutes; `--check` also re-runs main_best on 1997–2014 and compares it with the saved results).
