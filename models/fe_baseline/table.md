# Phase 4: two-way fixed-effects baseline and comparison specifications

Estimated on the train years (1997–2014) of the 70/15/15 split (data/split_data); validation and test RMSE are out-of-sample predictions.

| | main | no_gdp | no_wy | no_ippu | pooled | no_lags | main_dk | drivers |
|---|---|---|---|---|---|---|---|---|
| D_corporate | 0.0615**<br>(0.0237) | 0.0804***<br>(0.0221) | 0.0781***<br>(0.0274) | 0.0660**<br>(0.0254) | 0.4425***<br>(0.0523) | 0.0724***<br>(0.0253) | 0.0615***<br>(0.0161) |  |
| D_household | 0.0284<br>(0.0173) | 0.0386**<br>(0.0177) | 0.0277<br>(0.0176) | 0.0348**<br>(0.0168) | 0.0068<br>(0.0426) | 0.0335*<br>(0.0172) | 0.0284**<br>(0.0101) |  |
| W_D_corporate | 0.0613**<br>(0.0282) | 0.0652**<br>(0.0310) | 0.0540*<br>(0.0280) | 0.0835***<br>(0.0276) | -0.0738<br>(0.0744) |  | 0.0613**<br>(0.0247) | 0.0585***<br>(0.0214) |
| W_D_household | 0.0204<br>(0.0277) | 0.0262<br>(0.0271) | 0.0174<br>(0.0277) | 0.0010<br>(0.0230) | 0.0665<br>(0.0622) |  | 0.0204<br>(0.0135) | 0.0069<br>(0.0181) |
| log_pop_density | -0.3321*<br>(0.1742) | -0.3543*<br>(0.1915) | -0.3395*<br>(0.1778) | -0.4237**<br>(0.1829) | -0.0915*<br>(0.0522) | -0.3070*<br>(0.1685) | -0.3321***<br>(0.0491) | -0.2483**<br>(0.1192) |
| log_gdp_real_pc | 0.1847*<br>(0.0935) |  | 0.1805*<br>(0.0977) | 0.1421<br>(0.0981) | -0.1943<br>(0.2461) | 0.2109*<br>(0.1067) | 0.1847***<br>(0.0379) | 0.3481***<br>(0.1015) |
| hdd_k | 0.0387***<br>(0.0082) | 0.0394***<br>(0.0081) | 0.0396***<br>(0.0082) | 0.0459***<br>(0.0085) | 0.0782*<br>(0.0391) | 0.0405***<br>(0.0081) | 0.0387***<br>(0.0052) | 0.0230***<br>(0.0083) |
| cdd_k | 0.0149<br>(0.0103) | 0.0104<br>(0.0107) | 0.0152<br>(0.0103) | 0.0163<br>(0.0108) | 0.1516<br>(0.0986) | 0.0157<br>(0.0106) | 0.0149<br>(0.0179) | -0.0411**<br>(0.0153) |
| brk_mn_2010 | 0.1123***<br>(0.0206) | 0.1095***<br>(0.0210) | 0.1145***<br>(0.0208) |  | -0.0469<br>(0.1064) | 0.1310***<br>(0.0102) | 0.1123***<br>(0.0105) | 0.1215***<br>(0.0195) |
| brk_mi_2010 | 0.0486***<br>(0.0164) | 0.0405**<br>(0.0180) | 0.0548***<br>(0.0163) |  | -0.2891***<br>(0.0929) | 0.0552***<br>(0.0163) | 0.0486***<br>(0.0099) | 0.0813***<br>(0.0128) |
| brk_md_2011 | -0.0479***<br>(0.0115) | -0.0360***<br>(0.0093) | -0.0445***<br>(0.0119) |  | 0.1146<br>(0.0946) | -0.0487***<br>(0.0123) | -0.0479*<br>(0.0228) | -0.0136<br>(0.0125) |
| z_d_industrial_share |  |  |  |  |  |  |  | -0.0049<br>(0.0087) |
| z_d_fossil_prod_pc |  |  |  |  |  |  |  | 0.0914**<br>(0.0372) |
| z_d_grid_fossil_share |  |  |  |  |  |  |  | 0.0848***<br>(0.0200) |
| z_d_sand_gravel_pc |  |  |  |  |  |  |  | 0.0246<br>(0.0159) |
| z_d_vmt_pc |  |  |  |  |  |  |  | 0.0266***<br>(0.0089) |
| z_d_air_pax_pc |  |  |  |  |  |  |  | -0.0040<br>(0.0114) |
| z_d_pce_nonenergy_pc |  |  |  |  |  |  |  | -0.0393<br>(0.0319) |
| z_d_res_elec_pc |  |  |  |  |  |  |  | 0.1224***<br>(0.0270) |
| const |  |  |  |  | 5.1984**<br>(2.5205) |  |  |  |
| **Own-effect gap (primary): D_corporate − D_household** | 0.0331<br>(0.0325) | 0.0418<br>(0.0342) | 0.0504<br>(0.0354) | 0.0312<br>(0.0333) | 0.4357***<br>(0.0798) | 0.0390<br>(0.0342) | 0.0331<br>(0.0237) |  |
| 95% CI | [-0.0322, 0.0985] | [-0.0269, 0.1106] | [-0.0209, 0.1217] | [-0.0359, 0.0983] | [0.2750, 0.5963] | [-0.0298, 0.1077] | [-0.0169, 0.0831] |  |
| p (two-sided / one-sided) | 0.313 / 0.156 | 0.227 / 0.114 | 0.161 / 0.081 | 0.354 / 0.177 | 0.000 / 0.000 | 0.260 / 0.130 | 0.180 / 0.090 |  |
| **Total-effect gap: (D + W·D)_corporate − (D + W·D)_household** | 0.0741<br>(0.0455) | 0.0808<br>(0.0484) | 0.0871*<br>(0.0470) | 0.1137**<br>(0.0431) | 0.2954***<br>(0.0614) |  | 0.0741<br>(0.0469) |  |
| 95% CI | [-0.0174, 0.1656] | [-0.0166, 0.1782] | [-0.0076, 0.1817] | [0.0270, 0.2004] | [0.1718, 0.4190] |  | [-0.0248, 0.1730] |  |
| p (two-sided / one-sided) | 0.110 / 0.055 | 0.102 / 0.051 | 0.071 / 0.035 | 0.011 / 0.006 | 0.000 / 0.000 |  | 0.132 / 0.066 |  |
| Outcome | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_noippu_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc | y_log_gross_pc |
| Fixed effects | state + year | state + year | state + year | state + year | none | state + year | state + year | state + year |
| Standard errors | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | clustered by state | Driscoll-Kraay (Bartlett, bw 3) | clustered by state |
| Dropped states | none | none | WY | none | none | none | none | none |
| Observations | 864 | 864 | 846 | 864 | 864 | 864 | 864 | 864 |
| Inference df | 47 | 47 | 46 | 47 | 47 | 47 | 17 | 47 |
| Within R² / model R² | 0.404 / 0.404 | 0.391 / 0.391 | 0.406 / 0.406 | 0.392 / 0.392 | — / 0.801 | 0.387 / 0.387 | 0.404 / 0.404 | 0.536 / 0.536 |
| RMSE (train) | 0.0443 | 0.0447 | 0.0443 | 0.0458 | 0.2693 | 0.0449 | 0.0443 | 0.0392 |
| RMSE (validation / test, out of sample) | 0.0802 / 0.1413 | 0.0792 / 0.1298 | 0.0788 / 0.1377 | 0.0797 / 0.1347 | 0.2791 / 0.2884 | 0.0802 / 0.1462 | 0.0802 / 0.1413 | 0.0693 / 0.1167 |
| Joint F: driver terms = 0 (p) | 7.28 (0.00012) | 17.82 (5.8e-09) | 6.70 (0.00025) | 8.69 (2.5e-05) | 29.68 (2.5e-12) | 8.11 (0.00094) | 20.15 (2.8e-06) | 16.04 (5e-12) |

Standard errors in parentheses: clustered by state (with small-sample correction), except main_dk (Driscoll-Kraay, Bartlett kernel, bandwidth 3). p-values and CIs use a t-distribution with G − 1 degrees of freedom for clustered SEs and T − 1 for Driscoll-Kraay. * p<0.1, ** p<0.05, *** p<0.01. One-sided gap p tests D_corporate > D_household.

z_ terms in the drivers column are the individual drivers standardized to pooled SD 1 over the main sample, so each coefficient is the change in Y per SD of that driver, within a state.

Each break dummy is nonzero in a single state, so its clustered standard error is identified from one cluster and should not be interpreted; the dummies are nuisance controls. no_ippu leaves them out: all three breaks are in iron and steel, which that outcome already excludes.
