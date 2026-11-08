# Parent paper 3 replication (18 countries, 2000-2024, 450 obs.)

## Table 2. Descriptive statistics

| index | CO2E | CE | EC | EO | EUPE | IC | IEI | UP |
|---|---|---|---|---|---|---|---|---|
| Mean | 5.675 | 0.435 | 9576.069 | 109.939 | 0.463 | 0.199 | 3812.647 | 60.071 |
| Median | 4.918 | 0.487 | 8756.700 | 102.951 | 0.456 | 0.317 | 3605.455 | 60.199 |
| Maximum | 17.934 | 1.354 | 26134.627 | 204.056 | 1.668 | 1.355 | 7354.893 | 74.004 |
| Minimum | 0.903 | -0.852 | 1229.215 | 48.521 | -0.847 | -1.561 | 1172.261 | 39.218 |
| Std. Dev. | 3.267 | 0.506 | 6145.445 | 32.497 | 0.587 | 0.629 | 1522.156 | 9.280 |
| Skewness | 1.282 | -0.363 | 0.460 | 0.452 | -0.244 | -0.387 | 0.520 | -0.440 |
| Kurtosis | 5.092 | 2.207 | 2.092 | 2.291 | 2.222 | 2.385 | 2.434 | 2.316 |

## Table 3. Correlation matrix (r; p-values in CSV)

| index | LNCO2E | LNEC | LNIEI | CE | EO | EUPE | IC | UP |
|---|---|---|---|---|---|---|---|---|
| LNCO2E | 1.000 | nan | nan | nan | nan | nan | nan | nan |
| LNEC | 0.638 | 1.000 | nan | nan | nan | nan | nan | nan |
| LNIEI | 0.861 | 0.806 | 1.000 | nan | nan | nan | nan | nan |
| CE | 0.626 | 0.855 | 0.699 | 1.000 | nan | nan | nan | nan |
| EO | 0.473 | 0.602 | 0.686 | 0.512 | 1.000 | nan | nan | nan |
| EUPE | 0.508 | 0.804 | 0.630 | 0.825 | 0.537 | 1.000 | nan | nan |
| IC | 0.512 | 0.886 | 0.724 | 0.827 | 0.603 | 0.901 | 1.000 | nan |
| UP | 0.363 | 0.332 | 0.402 | 0.312 | 0.286 | 0.489 | 0.414 | 1.000 |

## Table 4. Panel unit root (Choi Z with p; Fisher-ADF chi2 with p)

| Variable | Level - No Trend | Level - No Trend Fisher | Level - With Trend | Level - With Trend Fisher | 1st Diff - No Trend | 1st Diff - No Trend Fisher | 1st Diff - With Trend | 1st Diff - With Trend Fisher | Decision |
|---|---|---|---|---|---|---|---|---|---|
| lnco2e | 0.818 (0.793) | 41.82 (0.233) | -1.339 (0.090) | 52.46 (0.037) | -17.633 (0.000) | 409.68 (0.000) | -15.846 (0.000) | 344.98 (0.000) | I(1) |
| lnec | -0.441 (0.330) | 40.19 (0.290) | -1.013 (0.156) | 40.72 (0.270) | -14.405 (0.000) | 307.23 (0.000) | -12.770 (0.000) | 260.49 (0.000) | I(1) |
| lniei | -2.170 (0.015) | 58.98 (0.009) | 0.259 (0.602) | 37.36 (0.406) | -15.836 (0.000) | 351.10 (0.000) | -14.660 (0.000) | 321.53 (0.000) | I(0) |
| ce | -0.537 (0.296) | 52.43 (0.038) | -1.046 (0.148) | 44.31 (0.161) | -13.643 (0.000) | 299.72 (0.000) | -12.044 (0.000) | 256.04 (0.000) | I(1) |
| eo | 0.794 (0.787) | 33.77 (0.575) | -3.830 (0.000) | 129.78 (0.000) | -16.559 (0.000) | 379.97 (0.000) | -14.213 (0.000) | 294.94 (0.000) | I(1) |
| eupe | -9.397 (0.000) | 246.30 (0.000) | -4.935 (0.000) | 138.99 (0.000) | -14.445 (0.000) | 315.91 (0.000) | -13.705 (0.000) | 291.31 (0.000) | I(0) |
| ic | -3.056 (0.001) | 75.11 (0.000) | -4.678 (0.000) | 123.70 (0.000) | -19.245 (0.000) | 483.80 (0.000) | -16.997 (0.000) | 400.08 (0.000) | I(0) |
| up | 0.311 (0.622) | 130.31 (0.000) | 5.316 (1.000) | 67.38 (0.001) | 0.829 (0.796) | 64.51 (0.002) | -1.360 (0.087) | 97.93 (0.000) | I(1) |

## Table 5. VAR lag order selection

| Lag | LogL | LR | LR_p | FPE | AIC | SC | HQ | selected_by |
|---|---|---|---|---|---|---|---|---|
| 0 | -2737.221 | nan | nan | 0.009 | 17.943 | 18.040 | 17.982 |  |
| 1 | 1265.132 | 7769.273 | 0.000 | 0.000 | -7.798 | -6.922 | -7.448 | SC |
| 2 | 1367.649 | 193.642 | 0.000 | 0.000 | -8.050 | -6.395 | -7.388 |  |
| 3 | 1508.055 | 257.871 | 0.000 | 0.000 | -8.549 | -6.116 | -7.576 | LR FPE AIC HQ |
| 4 | 1549.363 | 73.707 | 0.190 | 0.000 | -8.401 | -5.189 | -7.116 |  |
| 5 | 1605.703 | 97.581 | 0.004 | 0.000 | -8.351 | -4.360 | -6.755 |  |
| 6 | 1659.305 | 90.038 | 0.018 | 0.000 | -8.283 | -3.513 | -6.375 |  |
| 7 | 1717.194 | 94.211 | 0.008 | 0.000 | -8.243 | -2.694 | -6.024 |  |
| 8 | 1787.673 | 111.016 | 0.000 | 0.000 | -8.285 | -1.958 | -5.755 |  |

## Table 6. Kao residual cointegration test

| index | value |
|---|---|
| adf_t_stat | -0.898 |
| p | 0.184 |
| residual_variance | 0.015 |
| hac_variance | 0.033 |
| adf_lag | 2.000 |
| bandwidth | 2.000 |
| raw_t | -4.496 |

## Table 7. Panel ARDL (PMG), ARDL(2,1); converged in 76 iterations

| index | part | coef | se | t | p |
|---|---|---|---|---|---|
| LNEC | Long run | 0.204 | 0.041 | 4.924 | 0.000 |
| LNIEI | Long run | 0.833 | 0.049 | 17.017 | 0.000 |
| CE | Long run | 0.108 | 0.021 | 5.057 | 0.000 |
| EO | Long run | -0.002 | 0.000 | -5.993 | 0.000 |
| EUPE | Long run | -0.019 | 0.042 | -0.456 | 0.648 |
| IC | Long run | -0.075 | 0.027 | -2.808 | 0.005 |
| UP | Long run | 0.011 | 0.005 | 2.183 | 0.029 |
| COINTEQ01 | Short run | -0.318 | 0.129 | -2.467 | 0.014 |
| D(LNCO2E(-1)) | Short run | 0.036 | 0.065 | 0.553 | 0.581 |
| D(LNEC) | Short run | 0.095 | 0.154 | 0.619 | 0.536 |
| D(LNIEI) | Short run | 0.503 | 0.201 | 2.506 | 0.012 |
| D(CE) | Short run | -0.033 | 0.074 | -0.445 | 0.656 |
| D(EO) | Short run | 0.001 | 0.001 | 1.853 | 0.064 |
| D(EUPE) | Short run | -0.042 | 0.037 | -1.135 | 0.256 |
| D(IC) | Short run | 0.042 | 0.029 | 1.446 | 0.148 |
| D(UP) | Short run | -0.184 | 0.080 | -2.306 | 0.021 |
| C | Short run | -2.412 | 0.975 | -2.475 | 0.013 |

## Table 8. Bounds test (Wald, long-run = 0)

| test | value | df | p |
|---|---|---|---|
| F-statistic | 125.353 | (7, 227) | 0.000 |
| Chi-square | 877.470 | 7 | 0.000 |

## Table 9. Dumitrescu-Hurlin panel causality (lags = 2)

| Null Hypothesis | W-Stat. | Zbar-Stat. | Prob. |
|---|---|---|---|
| LNCO2E does not homogeneously cause LNEC | 2.386 | 0.226 | 0.821 |
| LNEC does not homogeneously cause LNCO2E | 4.421 | 3.611 | 0.000 |
| LNCO2E does not homogeneously cause LNIEI | 3.339 | 1.811 | 0.070 |
| LNIEI does not homogeneously cause LNCO2E | 4.280 | 3.376 | 0.001 |
| LNCO2E does not homogeneously cause CE | 3.479 | 2.044 | 0.041 |
| CE does not homogeneously cause LNCO2E | 1.693 | -0.926 | 0.354 |
| LNCO2E does not homogeneously cause EO | 3.994 | 2.900 | 0.004 |
| EO does not homogeneously cause LNCO2E | 3.948 | 2.823 | 0.005 |
| LNCO2E does not homogeneously cause EUPE | 5.048 | 4.653 | 0.000 |
| EUPE does not homogeneously cause LNCO2E | 2.779 | 0.879 | 0.379 |
| LNCO2E does not homogeneously cause IC | 3.266 | 1.690 | 0.091 |
| IC does not homogeneously cause LNCO2E | 3.024 | 1.288 | 0.198 |
| LNCO2E does not homogeneously cause UP | 3.395 | 1.905 | 0.057 |
| UP does not homogeneously cause LNCO2E | 6.025 | 6.278 | 0.000 |
| LNEC does not homogeneously cause LNIEI | 4.136 | 3.137 | 0.002 |
| LNIEI does not homogeneously cause LNEC | 2.187 | -0.105 | 0.917 |
| LNEC does not homogeneously cause CE | 3.676 | 2.371 | 0.018 |
| CE does not homogeneously cause LNEC | 2.076 | -0.289 | 0.772 |
| LNEC does not homogeneously cause EO | 5.609 | 5.586 | 0.000 |
| EO does not homogeneously cause LNEC | 1.539 | -1.182 | 0.237 |
| LNEC does not homogeneously cause EUPE | 4.490 | 3.725 | 0.000 |
| EUPE does not homogeneously cause LNEC | 1.719 | -0.883 | 0.377 |
| LNEC does not homogeneously cause IC | 4.671 | 4.026 | 0.000 |
| IC does not homogeneously cause LNEC | 3.240 | 1.647 | 0.100 |
| LNEC does not homogeneously cause UP | 5.089 | 4.721 | 0.000 |
| UP does not homogeneously cause LNEC | 3.651 | 2.330 | 0.020 |
| LNIEI does not homogeneously cause CE | 4.051 | 2.995 | 0.003 |
| CE does not homogeneously cause LNIEI | 2.067 | -0.304 | 0.761 |
| LNIEI does not homogeneously cause EO | 4.213 | 3.264 | 0.001 |
| EO does not homogeneously cause LNIEI | 3.109 | 1.428 | 0.153 |
| LNIEI does not homogeneously cause EUPE | 3.252 | 1.667 | 0.096 |
| EUPE does not homogeneously cause LNIEI | 1.854 | -0.658 | 0.510 |
| LNIEI does not homogeneously cause IC | 3.685 | 2.387 | 0.017 |
| IC does not homogeneously cause LNIEI | 2.365 | 0.191 | 0.849 |
| LNIEI does not homogeneously cause UP | 5.397 | 5.233 | 0.000 |
| UP does not homogeneously cause LNIEI | 5.931 | 6.121 | 0.000 |
| CE does not homogeneously cause EO | 2.623 | 0.620 | 0.535 |
| EO does not homogeneously cause CE | 2.480 | 0.382 | 0.702 |
| CE does not homogeneously cause EUPE | 4.207 | 3.255 | 0.001 |
| EUPE does not homogeneously cause CE | 2.481 | 0.384 | 0.701 |
| CE does not homogeneously cause IC | 3.178 | 1.543 | 0.123 |
| IC does not homogeneously cause CE | 3.669 | 2.361 | 0.018 |
| CE does not homogeneously cause UP | 5.482 | 5.375 | 0.000 |
| UP does not homogeneously cause CE | 6.805 | 7.575 | 0.000 |
| EO does not homogeneously cause EUPE | 4.489 | 3.723 | 0.000 |
| EUPE does not homogeneously cause EO | 2.680 | 0.715 | 0.475 |
| EO does not homogeneously cause IC | 2.944 | 1.154 | 0.249 |
| IC does not homogeneously cause EO | 4.068 | 3.024 | 0.002 |
| EO does not homogeneously cause UP | 4.254 | 3.332 | 0.001 |
| UP does not homogeneously cause EO | 10.075 | 13.013 | 0.000 |
| EUPE does not homogeneously cause IC | 4.985 | 4.549 | 0.000 |
| IC does not homogeneously cause EUPE | 3.211 | 1.599 | 0.110 |
| EUPE does not homogeneously cause UP | 3.638 | 2.308 | 0.021 |
| UP does not homogeneously cause EUPE | 7.719 | 9.095 | 0.000 |
| IC does not homogeneously cause UP | 3.831 | 2.629 | 0.009 |
| UP does not homogeneously cause IC | 5.745 | 5.812 | 0.000 |

## Table 10. Response of LNCO2E to Cholesky one S.D. innovations

| Period | LNCO2E | LNEC | LNIEI | CE | EO | EUPE | IC | UP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.074 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.069 | -0.006 | 0.003 | 0.004 | -0.009 | 0.008 | -0.008 | -0.001 |
| 3 | 0.067 | -0.004 | 0.001 | 0.005 | -0.011 | 0.011 | -0.006 | -0.002 |
| 4 | 0.065 | -0.004 | -0.000 | 0.005 | -0.012 | 0.012 | -0.006 | -0.002 |
| 5 | 0.064 | -0.004 | -0.002 | 0.006 | -0.013 | 0.014 | -0.005 | -0.002 |
| 6 | 0.062 | -0.004 | -0.003 | 0.007 | -0.013 | 0.015 | -0.005 | -0.003 |
| 7 | 0.061 | -0.005 | -0.004 | 0.007 | -0.014 | 0.016 | -0.004 | -0.003 |
| 8 | 0.059 | -0.005 | -0.005 | 0.008 | -0.014 | 0.016 | -0.003 | -0.003 |
| 9 | 0.057 | -0.005 | -0.006 | 0.009 | -0.014 | 0.017 | -0.003 | -0.004 |
| 10 | 0.056 | -0.006 | -0.007 | 0.010 | -0.014 | 0.017 | -0.003 | -0.004 |

## Table 11. Variance decomposition of LNCO2E

| Period | S.E. | LNCO2E | LNEC | LNIEI | CE | EO | EUPE | IC | UP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.074 | 100.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.103 | 97.512 | 0.298 | 0.099 | 0.132 | 0.746 | 0.627 | 0.567 | 0.019 |
| 3 | 0.124 | 96.119 | 0.328 | 0.080 | 0.236 | 1.365 | 1.178 | 0.665 | 0.029 |
| 4 | 0.141 | 95.109 | 0.331 | 0.062 | 0.319 | 1.781 | 1.679 | 0.678 | 0.040 |
| 5 | 0.156 | 94.214 | 0.341 | 0.060 | 0.404 | 2.110 | 2.154 | 0.663 | 0.055 |
| 6 | 0.170 | 93.367 | 0.358 | 0.076 | 0.497 | 2.392 | 2.603 | 0.635 | 0.072 |
| 7 | 0.182 | 92.545 | 0.380 | 0.110 | 0.603 | 2.640 | 3.028 | 0.602 | 0.092 |
| 8 | 0.193 | 91.741 | 0.407 | 0.162 | 0.720 | 2.862 | 3.426 | 0.568 | 0.113 |
| 9 | 0.203 | 90.951 | 0.437 | 0.228 | 0.849 | 3.062 | 3.798 | 0.536 | 0.137 |
| 10 | 0.212 | 90.175 | 0.470 | 0.310 | 0.990 | 3.243 | 4.143 | 0.506 | 0.163 |

## Table 12. Hausman test

| test | chi2 | df | p |
|---|---|---|---|
| Cross-section random | 16.425 | 7 | 0.022 |

## Table 13. Panel least squares (fixed effects), SE: unadjusted

| index | coef | se | t | p |
|---|---|---|---|---|
| LNEC | -0.099 | 0.049 | -2.021 | 0.044 |
| LNIEI | 1.086 | 0.072 | 15.125 | 0.000 |
| CE | 0.001 | 0.035 | 0.037 | 0.971 |
| EO | -0.002 | 0.000 | -5.453 | 0.000 |
| EUPE | 0.336 | 0.041 | 8.244 | 0.000 |
| IC | -0.234 | 0.042 | -5.641 | 0.000 |
| UP | -0.003 | 0.004 | -0.749 | 0.454 |
| C | -6.060 | 0.452 | -13.397 | 0.000 |
| r2_lsdv | 0.955 | nan | nan | nan |
| adj_r2_lsdv | 0.953 | nan | nan | nan |
| r2_within | 0.554 | nan | nan | nan |
| durbin_watson | 0.413 | nan | nan | nan |
| n_obs | 450 | nan | nan | nan |
| cov | unadjusted | nan | nan | nan |

## Table 13. Panel least squares (fixed effects), SE: kernel

| index | coef | se | t | p |
|---|---|---|---|---|
| LNEC | -0.099 | 0.067 | -1.493 | 0.136 |
| LNIEI | 1.086 | 0.145 | 7.479 | 0.000 |
| CE | 0.001 | 0.032 | 0.041 | 0.967 |
| EO | -0.002 | 0.001 | -3.672 | 0.000 |
| EUPE | 0.336 | 0.066 | 5.061 | 0.000 |
| IC | -0.234 | 0.069 | -3.405 | 0.001 |
| UP | -0.003 | 0.004 | -0.766 | 0.444 |
| C | -6.060 | 0.733 | -8.266 | 0.000 |
| r2_lsdv | 0.955 | nan | nan | nan |
| adj_r2_lsdv | 0.953 | nan | nan | nan |
| r2_within | 0.554 | nan | nan | nan |
| durbin_watson | 0.413 | nan | nan | nan |
| n_obs | 450 | nan | nan | nan |
| cov | kernel | nan | nan | nan |

## Table 13. Panel least squares (fixed effects), SE: clustered

| index | coef | se | t | p |
|---|---|---|---|---|
| LNEC | -0.099 | 0.158 | -0.629 | 0.530 |
| LNIEI | 1.086 | 0.249 | 4.362 | 0.000 |
| CE | 0.001 | 0.100 | 0.013 | 0.990 |
| EO | -0.002 | 0.001 | -1.990 | 0.047 |
| EUPE | 0.336 | 0.103 | 3.263 | 0.001 |
| IC | -0.234 | 0.076 | -3.075 | 0.002 |
| UP | -0.003 | 0.008 | -0.392 | 0.695 |
| C | -6.060 | 1.189 | -5.096 | 0.000 |
| r2_lsdv | 0.955 | nan | nan | nan |
| adj_r2_lsdv | 0.953 | nan | nan | nan |
| r2_within | 0.554 | nan | nan | nan |
| durbin_watson | 0.413 | nan | nan | nan |
| n_obs | 450 | nan | nan | nan |
| cov | clustered | nan | nan | nan |

## Table 14. VIF

| index | VIF |
|---|---|
| LNEC | 7.830 |
| LNIEI | 3.912 |
| CE | 4.858 |
| EO | 2.017 |
| EUPE | 7.204 |
| IC | 8.979 |
| UP | 1.503 |

## Table 15. Wooldridge serial correlation (rho = 0.056)

| test | value | df | p |
|---|---|---|---|
| t-statistic | 18.743 | 17 | 0.000 |
| F-statistic | 351.284 | (1, 17) | 0.000 |
| Chi-square | 351.284 | 1 | 0.000 |

## Table 16. DML effect estimates (GB nuisance, per 1 SD of D)

| variable | estimate | se | t | p | ci_low | ci_high | sd_of_driver |
|---|---|---|---|---|---|---|---|
| ln_ec | 0.024 | 0.039 | 0.611 | 0.541 | -0.053 | 0.101 | 0.768 |
| ln_iei | 0.564 | 0.025 | 23.010 | 0.000 | 0.516 | 0.613 | 0.417 |
| ce | 0.077 | 0.029 | 2.705 | 0.007 | 0.021 | 0.134 | 0.506 |
| eo | -0.086 | 0.014 | -6.099 | 0.000 | -0.113 | -0.058 | 32.497 |
| eupe | 0.095 | 0.044 | 2.166 | 0.030 | 0.009 | 0.181 | 0.587 |
| ic | -0.265 | 0.044 | -6.036 | 0.000 | -0.351 | -0.179 | 0.629 |
| up | 0.097 | 0.011 | 8.473 | 0.000 | 0.074 | 0.119 | 9.280 |

## Table 17. DML stability across folds

| variable | mean | std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 |
|---|---|---|---|---|---|---|---|
| ln_ec | -0.131 | 0.428 | 0.405 | -0.222 | -0.708 | 0.155 | -0.287 |
| ln_iei | 0.556 | 0.244 | 0.724 | 0.527 | 0.790 | 0.574 | 0.164 |
| ce | 0.013 | 0.235 | 0.058 | -0.025 | -0.305 | -0.015 | 0.354 |
| eo | -0.093 | 0.091 | -0.121 | -0.047 | -0.057 | -0.004 | -0.237 |
| eupe | 0.175 | 0.380 | -0.385 | 0.428 | 0.288 | 0.561 | -0.017 |
| ic | -0.224 | 0.392 | -0.410 | 0.052 | -0.610 | -0.471 | 0.319 |
| up | 0.077 | 0.062 | 0.153 | 0.072 | 0.120 | -0.006 | 0.048 |

## Table 18. Counterfactual: +1 SD in each driver

| variable | delta_ln_co2e_dml | pct_change_plus_1sd | pct_change_minus_1sd | delta_ln_co2e_model |
|---|---|---|---|---|
| ln_ec | 0.024 | 2.432 | -2.375 | -0.049 |
| ln_iei | 0.564 | 75.844 | -43.131 | 0.399 |
| ce | 0.077 | 8.051 | -7.451 | 0.076 |
| eo | -0.086 | -8.211 | 8.946 | -0.030 |
| eupe | 0.095 | 9.985 | -9.078 | 0.072 |
| ic | -0.265 | -23.251 | 30.294 | -0.043 |
| up | 0.097 | 10.140 | -9.206 | 0.059 |

## Table 19. Random forest: test R2 = 0.943, RMSE = 0.147 (train R2 0.992)

| index | mean_abs_shap | perm_mean | perm_std |
|---|---|---|---|
| ln_ec | 0.029 | 0.041 | 0.006 |
| ln_iei | 0.432 | 1.460 | 0.114 |
| ce | 0.040 | 0.042 | 0.008 |
| eo | 0.036 | 0.044 | 0.006 |
| eupe | 0.057 | 0.090 | 0.014 |
| ic | 0.024 | 0.022 | 0.006 |
| up | 0.031 | 0.043 | 0.003 |

## Table 20. Gradient boosting: test R2 = 0.947, RMSE = 0.141 (train R2 0.989)

| index | mean_abs_shap | perm_mean | perm_std |
|---|---|---|---|
| ln_ec | 0.040 | 0.034 | 0.006 |
| ln_iei | 0.411 | 1.450 | 0.107 |
| ce | 0.099 | 0.122 | 0.020 |
| eo | 0.043 | 0.034 | 0.005 |
| eupe | 0.068 | 0.094 | 0.015 |
| ic | 0.042 | 0.028 | 0.004 |
| up | 0.039 | 0.036 | 0.006 |
