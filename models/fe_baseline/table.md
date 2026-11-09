# Phase 4: two-way fixed-effects baseline and comparison specifications

Estimated on the train years (1997–2014) of the 70/15/15 split (data/split_data); validation and test RMSE are out-of-sample predictions.

| | main | no_gdp | no_wy | no_ippu | pooled | no_lags | main_dk | drivers |
|---|---|---|---|---|---|---|---|---|
| D_corporate | 0.0614**<br>(0.0233) | 0.0816***<br>(0.0216) | 0.0777***<br>(0.0269) | 0.0661**<br>(0.0250) | 0.4290***<br>(0.0512) | 0.0726***<br>(0.0251) | 0.0614***<br>(0.0162) |  |
| D_household | 0.0342*<br>(0.0200) | 0.0446**<br>(0.0211) | 0.0339<br>(0.0203) | 0.0413**<br>(0.0198) | 0.0204<br>(0.0495) | 0.0409**<br>(0.0201) | 0.0342**<br>(0.0119) |  |
| W_D_corporate | 0.0613**<br>(0.0275) | 0.0660**<br>(0.0305) | 0.0539*<br>(0.0273) | 0.0827***<br>(0.0272) | 0.0220<br>(0.0751) |  | 0.0613**<br>(0.0239) | 0.0577***<br>(0.0210) |
| W_D_household | 0.0359<br>(0.0331) | 0.0438<br>(0.0325) | 0.0319<br>(0.0329) | 0.0122<br>(0.0276) | 0.0883<br>(0.0573) |  | 0.0359**<br>(0.0155) | 0.0184<br>(0.0218) |
| log_pop | -0.3260*<br>(0.1708) | -0.3557*<br>(0.1886) | -0.3323*<br>(0.1744) | -0.4173**<br>(0.1799) | -0.1151***<br>(0.0420) | -0.3105*<br>(0.1666) | -0.3260***<br>(0.0462) | -0.2459**<br>(0.1182) |
| log_gdp_real_pc | 0.1866**<br>(0.0908) |  | 0.1816*<br>(0.0953) | 0.1458<br>(0.0960) | -0.2251<br>(0.1799) | 0.2197**<br>(0.1052) | 0.1866***<br>(0.0349) | 0.3474***<br>(0.1016) |
| hdd_k | 0.0369***<br>(0.0083) | 0.0377***<br>(0.0082) | 0.0380***<br>(0.0084) | 0.0443***<br>(0.0086) | 0.0633*<br>(0.0340) | 0.0399***<br>(0.0081) | 0.0369***<br>(0.0051) | 0.0223***<br>(0.0083) |
| cdd_k | 0.0118<br>(0.0104) | 0.0073<br>(0.0109) | 0.0120<br>(0.0105) | 0.0134<br>(0.0110) | 0.1257<br>(0.0828) | 0.0150<br>(0.0105) | 0.0118<br>(0.0182) | -0.0423***<br>(0.0152) |
| brk_mn_2010 | 0.1067***<br>(0.0197) | 0.1029***<br>(0.0203) | 0.1091***<br>(0.0198) |  | -0.0203<br>(0.1127) | 0.1299***<br>(0.0099) | 0.1067***<br>(0.0099) | 0.1168***<br>(0.0195) |
| brk_mi_2010 | 0.0480***<br>(0.0161) | 0.0395**<br>(0.0178) | 0.0543***<br>(0.0160) |  | -0.2563***<br>(0.0824) | 0.0557***<br>(0.0163) | 0.0480***<br>(0.0095) | 0.0805***<br>(0.0128) |
| brk_md_2011 | -0.0470***<br>(0.0115) | -0.0346***<br>(0.0092) | -0.0436***<br>(0.0118) |  | -0.0010<br>(0.0973) | -0.0488***<br>(0.0123) | -0.0470*<br>(0.0230) | -0.0137<br>(0.0125) |
| z_d_industrial_share |  |  |  |  |  |  |  | -0.0047<br>(0.0086) |
| z_d_fossil_prod_pc |  |  |  |  |  |  |  | 0.0888**<br>(0.0376) |
| z_d_grid_fossil_share |  |  |  |  |  |  |  | 0.0855***<br>(0.0198) |
| z_d_sand_gravel_pc |  |  |  |  |  |  |  | 0.0245<br>(0.0158) |
| z_d_vmt_pc |  |  |  |  |  |  |  | 0.0264***<br>(0.0089) |
| z_d_air_pax_pc |  |  |  |  |  |  |  | -0.0040<br>(0.0114) |
| z_d_pce_nonenergy_pc |  |  |  |  |  |  |  | -0.0613<br>(0.0475) |
| z_d_res_elec_pc |  |  |  |  |  |  |  | 0.1204***<br>(0.0269) |
| const |  |  |  |  | 6.9731***<br>(2.0393) |  |  |  |
| **Own-effect gap (primary): D_corporate − D_household** | 0.0272<br>(0.0339) | 0.0370<br>(0.0361) | 0.0438<br>(0.0368) | 0.0248<br>(0.0348) | 0.4086***<br>(0.0828) | 0.0318<br>(0.0358) | 0.0272<br>(0.0257) |  |
| 95% CI | [-0.0409, 0.0953] | [-0.0357, 0.1097] | [-0.0302, 0.1178] | [-0.0453, 0.0948] | [0.2420, 0.5752] | [-0.0402, 0.1037] | [-0.0271, 0.0815] |  |
| p (two-sided / one-sided) | 0.426 / 0.213 | 0.311 / 0.155 | 0.240 / 0.120 | 0.480 / 0.240 | 0.000 / 0.000 | 0.379 / 0.189 | 0.305 / 0.152 |  |
| **Total-effect gap: (D + W·D)_corporate − (D + W·D)_household** | 0.0527<br>(0.0478) | 0.0593<br>(0.0505) | 0.0658<br>(0.0492) | 0.0952**<br>(0.0448) | 0.3423***<br>(0.0561) |  | 0.0527<br>(0.0490) |  |
| 95% CI | [-0.0434, 0.1488] | [-0.0423, 0.1608] | [-0.0332, 0.1649] | [0.0051, 0.1854] | [0.2295, 0.4551] |  | [-0.0507, 0.1561] |  |
| p (two-sided / one-sided) | 0.276 / 0.138 | 0.247 / 0.123 | 0.187 / 0.094 | 0.039 / 0.019 | 0.000 / 0.000 |  | 0.298 / 0.149 |  |
| Outcome | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_noippu_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc |
| Fixed effects | state + year | state + year | state + year | state + year | none | state + year | state + year | state + year |
| Standard errors | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | Driscoll-Kraay (Bartlett, bw 3) | clustered by state |
| Dropped states | none | none | WY | none | none | none | none | none |
| Observations | 864 | 864 | 846 | 864 | 864 | 864 | 864 | 864 |
| Inference df | 47 | 47 | 46 | 47 | 47 | 47 | 17 | 47 |
| Within R² / model R² | 0.409 / 0.409 | 0.395 / 0.395 | 0.410 / 0.410 | 0.395 / 0.395 | — / 0.808 | 0.388 / 0.388 | 0.409 / 0.409 | 0.537 / 0.537 |
| RMSE (train) | 0.0441 | 0.0446 | 0.0441 | 0.0457 | 0.2649 | 0.0448 | 0.0441 | 0.0392 |
| RMSE (validation / test, out of sample) | 0.0820 / 0.1504 | 0.0808 / 0.1396 | 0.0805 / 0.1462 | 0.0806 / 0.1414 | 0.2765 / 0.2901 | 0.0810 / 0.1507 | 0.0820 / 0.1504 | 0.0681 / 0.1002 |
| Joint F: driver terms = 0 (p) | 7.48 (9.6e-05) | 17.53 (7.2e-09) | 6.94 (0.00018) | 8.69 (2.4e-05) | 43.25 (3.4e-15) | 8.16 (0.00091) | 26.51 (4.1e-07) | 16.02 (5.1e-12) |

Standard errors in parentheses: clustered by state (with small-sample correction), except main_dk (Driscoll-Kraay, Bartlett kernel, bandwidth 3). p-values and CIs use a t-distribution with G − 1 degrees of freedom for clustered SEs and T − 1 for Driscoll-Kraay. * p<0.1, ** p<0.05, *** p<0.01. One-sided gap p tests D_corporate > D_household.

z_ terms in the drivers column are the individual drivers standardized to pooled SD 1 over the main sample, so each coefficient is the change in Y per SD of that driver, within a state.

Each break dummy is nonzero in a single state, so its clustered standard error is identified from one cluster and should not be interpreted; the dummies are nuisance controls. no_ippu leaves them out: all three breaks are in iron and steel, which that outcome already excludes.
