# Phase 4: two-way fixed-effects baseline and comparison specifications

| | main | no_gdp | no_wy | no_ippu | pooled | no_lags | main_dk | drivers |
|---|---|---|---|---|---|---|---|---|
| D_corporate | 0.1048***<br>(0.0313) | 0.1205***<br>(0.0295) | 0.1246***<br>(0.0321) | 0.1065***<br>(0.0334) | 0.4258***<br>(0.0532) | 0.1090***<br>(0.0317) | 0.1048***<br>(0.0200) |  |
| D_household | 0.0282<br>(0.0215) | 0.0362<br>(0.0217) | 0.0311<br>(0.0224) | 0.0379*<br>(0.0222) | 0.0285<br>(0.0514) | 0.0501**<br>(0.0210) | 0.0282***<br>(0.0085) |  |
| W_D_corporate | 0.0278<br>(0.0353) | 0.0335<br>(0.0389) | 0.0226<br>(0.0366) | 0.0395<br>(0.0377) | 0.0573<br>(0.0789) |  | 0.0278<br>(0.0261) | 0.0382<br>(0.0300) |
| W_D_household | 0.0948**<br>(0.0359) | 0.1026***<br>(0.0372) | 0.0904**<br>(0.0368) | 0.0844**<br>(0.0339) | 0.0733<br>(0.0582) |  | 0.0948***<br>(0.0258) | 0.1000***<br>(0.0329) |
| log_pop | -0.3333***<br>(0.1186) | -0.3320**<br>(0.1250) | -0.3323***<br>(0.1176) | -0.3790***<br>(0.1239) | -0.1177***<br>(0.0433) | -0.3377***<br>(0.1247) | -0.3333***<br>(0.0482) | -0.3010**<br>(0.1153) |
| log_gdp_real_pc | 0.1652*<br>(0.0908) |  | 0.1568*<br>(0.0917) | 0.1405<br>(0.1018) | -0.2063<br>(0.1752) | 0.1972**<br>(0.0962) | 0.1652***<br>(0.0478) | 0.3486***<br>(0.1190) |
| hdd_k | 0.0363***<br>(0.0072) | 0.0364***<br>(0.0070) | 0.0380***<br>(0.0072) | 0.0402***<br>(0.0076) | 0.0595*<br>(0.0330) | 0.0448***<br>(0.0074) | 0.0363***<br>(0.0113) | 0.0231***<br>(0.0084) |
| cdd_k | -0.0206<br>(0.0142) | -0.0293*<br>(0.0160) | -0.0214<br>(0.0151) | -0.0265*<br>(0.0148) | 0.1087<br>(0.0800) | -0.0095<br>(0.0120) | -0.0206<br>(0.0208) | -0.0687***<br>(0.0233) |
| brk_mn_2010 | 0.0829***<br>(0.0251) | 0.0793***<br>(0.0256) | 0.0842***<br>(0.0254) |  | -0.0370<br>(0.1128) | 0.1333***<br>(0.0118) | 0.0829***<br>(0.0126) | 0.1003***<br>(0.0269) |
| brk_mi_2010 | 0.0566***<br>(0.0192) | 0.0500**<br>(0.0195) | 0.0624***<br>(0.0196) |  | -0.2763***<br>(0.0834) | 0.0616***<br>(0.0200) | 0.0566***<br>(0.0128) | 0.0739***<br>(0.0197) |
| brk_md_2011 | 0.0035<br>(0.0083) | 0.0114<br>(0.0074) | 0.0053<br>(0.0083) |  | 0.0140<br>(0.0912) | 0.0023<br>(0.0088) | 0.0035<br>(0.0313) | 0.0174**<br>(0.0078) |
| z_d_industrial_share |  |  |  |  |  |  |  | 0.0134<br>(0.0107) |
| z_d_fossil_prod_pc |  |  |  |  |  |  |  | 0.0645*<br>(0.0336) |
| z_d_grid_fossil_share |  |  |  |  |  |  |  | 0.1103***<br>(0.0221) |
| z_d_sand_gravel_pc |  |  |  |  |  |  |  | 0.0021<br>(0.0184) |
| z_d_vmt_pc |  |  |  |  |  |  |  | 0.0093<br>(0.0133) |
| z_d_air_pax_pc |  |  |  |  |  |  |  | 0.0180<br>(0.0227) |
| z_d_pce_nonenergy_pc |  |  |  |  |  |  |  | -0.0561<br>(0.0698) |
| z_d_res_elec_pc |  |  |  |  |  |  |  | 0.1089***<br>(0.0326) |
| const |  |  |  |  | 6.8257***<br>(1.9725) |  |  |  |
| **Own-effect gap (primary): D_corporate − D_household** | 0.0766*<br>(0.0427) | 0.0842**<br>(0.0413) | 0.0935**<br>(0.0456) | 0.0685<br>(0.0446) | 0.3973***<br>(0.0842) | 0.0588<br>(0.0444) | 0.0766***<br>(0.0249) |  |
| 95% CI | [-0.0092, 0.1625] | [0.0012, 0.1673] | [0.0017, 0.1852] | [-0.0212, 0.1583] | [0.2279, 0.5667] | [-0.0305, 0.1481] | [0.0254, 0.1279] |  |
| p (two-sided / one-sided) | 0.079 / 0.039 | 0.047 / 0.023 | 0.046 / 0.023 | 0.131 / 0.066 | 0.000 / 0.000 | 0.192 / 0.096 | 0.005 / 0.002 |  |
| **Total-effect gap: (D + W·D)_corporate − (D + W·D)_household** | 0.0097<br>(0.0458) | 0.0152<br>(0.0455) | 0.0256<br>(0.0478) | 0.0237<br>(0.0477) | 0.3813***<br>(0.0519) |  | 0.0097<br>(0.0483) |  |
| 95% CI | [-0.0824, 0.1017] | [-0.0763, 0.1067] | [-0.0705, 0.1218] | [-0.0723, 0.1196] | [0.2769, 0.4856] |  | [-0.0898, 0.1091] |  |
| p (two-sided / one-sided) | 0.834 / 0.417 | 0.740 / 0.370 | 0.594 / 0.297 | 0.622 / 0.311 | 0.000 / 0.000 |  | 0.843 / 0.422 |  |
| Outcome | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_noippu_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc |
| Fixed effects | state + year | state + year | state + year | state + year | none | state + year | state + year | state + year |
| Standard errors | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | Driscoll-Kraay (Bartlett, bw 3) | clustered by state |
| Dropped states | none | none | WY | none | none | none | none | none |
| Observations | 1,248 | 1,248 | 1,222 | 1,248 | 1,248 | 1,248 | 1,248 | 1,248 |
| Inference df | 47 | 47 | 46 | 47 | 47 | 47 | 25 | 47 |
| Within R² / model R² | 0.393 / 0.393 | 0.380 / 0.380 | 0.404 / 0.404 | 0.378 / 0.378 | — / 0.812 | 0.365 / 0.365 | 0.393 / 0.393 | 0.489 / 0.489 |
| RMSE | 0.0537 | 0.0542 | 0.0535 | 0.0556 | 0.2688 | 0.0549 | 0.0537 | 0.0494 |
| Joint F: driver terms = 0 (p) | 8.75 (2.3e-05) | 12.79 (4e-07) | 10.89 (2.7e-06) | 8.90 (1.9e-05) | 42.49 (4.8e-15) | 14.06 (1.6e-05) | 33.67 (9.9e-10) | 11.41 (1.4e-09) |

Standard errors in parentheses: clustered by state (with small-sample correction), except main_dk (Driscoll-Kraay, Bartlett kernel, bandwidth 3). p-values and CIs use a t-distribution with G − 1 degrees of freedom for clustered SEs and T − 1 for Driscoll-Kraay. * p<0.1, ** p<0.05, *** p<0.01. One-sided gap p tests D_corporate > D_household.

z_ terms in the drivers column are the individual drivers standardized to pooled SD 1 over the main sample, so each coefficient is the change in Y per SD of that driver, within a state.

Each break dummy is nonzero in a single state, so its clustered standard error is identified from one cluster and should not be interpreted; the dummies are nuisance controls. no_ippu leaves them out: all three breaks are in iron and steel, which that outcome already excludes.
