# Static reference comparison

Four matched M2 variants, two OBS sizes and two RCT training sizes; 32 independent seeds per condition. Fixed recipes; no unseen-mechanism claim.

Coverage: 3072/3072 successful; 0 failed; 0 missing.

Available RCT labels are 160/544: 128/512 training and 32 reserved tuning labels. All fixed recipes ignore the reserved tuning labels. Evaluator-only factual test labels are separate.

All intervals below are pointwise 95% Student-t intervals over independent seed blocks. MCSE describes precision conditional on these development-inspected mechanisms. There is no cross-mechanism raw-error ranking.

## Condition scores

| Condition | Method | Mean MSE | MCSE | Pointwise 95% interval | Successful / expected |
| --- | --- | ---: | ---: | --- | ---: |
| `structural:M2@clean_o512_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@clean_o512_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@clean_o512_r128` | `obs_t_extra_trees` | 0.206419 | 0.0111 | [0.183699, 0.229139] | 32 / 32 |
| `structural:M2@clean_o512_r128` | `naive_pooled_t_extra_trees` | 0.180512 | 0.00938 | [0.161381, 0.199643] | 32 / 32 |
| `structural:M2@clean_o512_r128` | `cornet_official` | 0.210706 | 0.0232 | [0.163299, 0.258112] | 32 / 32 |
| `structural:M2@clean_o512_r128` | `datafusionppi` | 0.668991 | 0.0724 | [0.521264, 0.816718] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `obs_t_extra_trees` | 0.206419 | 0.0111 | [0.183699, 0.229139] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `naive_pooled_t_extra_trees` | 0.162651 | 0.00598 | [0.150454, 0.174849] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `cornet_official` | 0.0978955 | 0.00781 | [0.0819568, 0.113834] | 32 / 32 |
| `structural:M2@clean_o512_r512` | `datafusionppi` | 0.159271 | 0.0154 | [0.127836, 0.190705] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `obs_t_extra_trees` | 0.153344 | 0.00433 | [0.144513, 0.162175] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `naive_pooled_t_extra_trees` | 0.151468 | 0.00418 | [0.142942, 0.159994] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `cornet_official` | 0.201924 | 0.0209 | [0.159294, 0.244554] | 32 / 32 |
| `structural:M2@clean_o2048_r128` | `datafusionppi` | 0.652666 | 0.082 | [0.485518, 0.819813] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `obs_t_extra_trees` | 0.153344 | 0.00433 | [0.144513, 0.162175] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `naive_pooled_t_extra_trees` | 0.14634 | 0.00351 | [0.139178, 0.153502] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `cornet_official` | 0.0969947 | 0.00793 | [0.0808129, 0.113176] | 32 / 32 |
| `structural:M2@clean_o2048_r512` | `datafusionppi` | 0.138671 | 0.0117 | [0.114815, 0.162526] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `obs_t_extra_trees` | 0.912593 | 0.0441 | [0.822715, 1.00247] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `naive_pooled_t_extra_trees` | 0.652436 | 0.0328 | [0.585484, 0.719388] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `cornet_official` | 0.228392 | 0.0243 | [0.178756, 0.278028] | 32 / 32 |
| `structural:M2@confounded_o512_r128` | `datafusionppi` | 0.665837 | 0.0674 | [0.528316, 0.803357] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `obs_t_extra_trees` | 0.912593 | 0.0441 | [0.822715, 1.00247] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `naive_pooled_t_extra_trees` | 0.334218 | 0.0138 | [0.306136, 0.3623] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `cornet_official` | 0.120291 | 0.00903 | [0.101871, 0.138712] | 32 / 32 |
| `structural:M2@confounded_o512_r512` | `datafusionppi` | 0.168297 | 0.0155 | [0.136586, 0.200007] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `obs_t_extra_trees` | 0.893373 | 0.021 | [0.850637, 0.93611] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `naive_pooled_t_extra_trees` | 0.806445 | 0.0187 | [0.768389, 0.844502] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `cornet_official` | 0.216165 | 0.0209 | [0.173456, 0.258873] | 32 / 32 |
| `structural:M2@confounded_o2048_r128` | `datafusionppi` | 0.652696 | 0.0912 | [0.466696, 0.838695] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `obs_t_extra_trees` | 0.893373 | 0.021 | [0.850637, 0.93611] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `naive_pooled_t_extra_trees` | 0.606566 | 0.0162 | [0.573509, 0.639622] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `cornet_official` | 0.112072 | 0.00753 | [0.0967073, 0.127437] | 32 / 32 |
| `structural:M2@confounded_o2048_r512` | `datafusionppi` | 0.157263 | 0.0152 | [0.126336, 0.188191] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `obs_t_extra_trees` | 0.616141 | 0.0286 | [0.557778, 0.674504] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `naive_pooled_t_extra_trees` | 0.44482 | 0.0192 | [0.405701, 0.483939] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `cornet_official` | 0.240597 | 0.025 | [0.189629, 0.291564] | 32 / 32 |
| `structural:M2@shift_o512_r128` | `datafusionppi` | 0.699534 | 0.0697 | [0.557375, 0.841693] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `obs_t_extra_trees` | 0.616141 | 0.0286 | [0.557778, 0.674504] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `naive_pooled_t_extra_trees` | 0.276415 | 0.0107 | [0.254526, 0.298304] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `cornet_official` | 0.125644 | 0.00898 | [0.107321, 0.143966] | 32 / 32 |
| `structural:M2@shift_o512_r512` | `datafusionppi` | 0.186624 | 0.0195 | [0.146861, 0.226386] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `obs_t_extra_trees` | 0.540152 | 0.0134 | [0.512905, 0.567399] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `naive_pooled_t_extra_trees` | 0.492513 | 0.011 | [0.470076, 0.51495] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `cornet_official` | 0.233049 | 0.0239 | [0.184238, 0.28186] | 32 / 32 |
| `structural:M2@shift_o2048_r128` | `datafusionppi` | 0.708908 | 0.0967 | [0.511753, 0.906063] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `obs_t_extra_trees` | 0.540152 | 0.0134 | [0.512905, 0.567399] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `naive_pooled_t_extra_trees` | 0.392796 | 0.0083 | [0.375876, 0.409717] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `cornet_official` | 0.123242 | 0.00951 | [0.103839, 0.142645] | 32 / 32 |
| `structural:M2@shift_o2048_r512` | `datafusionppi` | 0.15763 | 0.013 | [0.131041, 0.184219] | 32 / 32 |
| `structural:M2@both_o512_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@both_o512_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@both_o512_r128` | `obs_t_extra_trees` | 1.88686 | 0.0665 | [1.75113, 2.02258] | 32 / 32 |
| `structural:M2@both_o512_r128` | `naive_pooled_t_extra_trees` | 1.2966 | 0.0482 | [1.19823, 1.39498] | 32 / 32 |
| `structural:M2@both_o512_r128` | `cornet_official` | 0.261921 | 0.0267 | [0.207512, 0.31633] | 32 / 32 |
| `structural:M2@both_o512_r128` | `datafusionppi` | 0.743528 | 0.0888 | [0.56238, 0.924677] | 32 / 32 |
| `structural:M2@both_o512_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@both_o512_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@both_o512_r512` | `obs_t_extra_trees` | 1.88686 | 0.0665 | [1.75113, 2.02258] | 32 / 32 |
| `structural:M2@both_o512_r512` | `naive_pooled_t_extra_trees` | 0.596465 | 0.0213 | [0.553099, 0.639832] | 32 / 32 |
| `structural:M2@both_o512_r512` | `cornet_official` | 0.146306 | 0.0106 | [0.124714, 0.167899] | 32 / 32 |
| `structural:M2@both_o512_r512` | `datafusionppi` | 0.181699 | 0.0172 | [0.146575, 0.216823] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `rct_t_extra_trees` | 0.313421 | 0.033 | [0.246075, 0.380768] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `rct_dr_extra_trees` | 0.482445 | 0.0368 | [0.407429, 0.557461] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `obs_t_extra_trees` | 1.91015 | 0.0319 | [1.84503, 1.97528] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `naive_pooled_t_extra_trees` | 1.71206 | 0.0275 | [1.65597, 1.76815] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `cornet_official` | 0.244531 | 0.0234 | [0.196835, 0.292226] | 32 / 32 |
| `structural:M2@both_o2048_r128` | `datafusionppi` | 0.723454 | 0.103 | [0.514164, 0.932744] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `rct_t_extra_trees` | 0.207101 | 0.00897 | [0.188796, 0.225405] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `rct_dr_extra_trees` | 0.315001 | 0.0112 | [0.292195, 0.337806] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `obs_t_extra_trees` | 1.91015 | 0.0319 | [1.84503, 1.97528] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `naive_pooled_t_extra_trees` | 1.26602 | 0.0248 | [1.21546, 1.31658] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `cornet_official` | 0.137534 | 0.00938 | [0.118412, 0.156656] | 32 / 32 |
| `structural:M2@both_o2048_r512` | `datafusionppi` | 0.172464 | 0.0158 | [0.140178, 0.204751] | 32 / 32 |

## All prespecified primary contrasts

Differences are fusion minus RCT reference; negative favors fusion. Two-sided p-values are adjusted together over 64 tests. Pointwise intervals do not have simultaneous family coverage.

| Condition | Fusion | RCT reference | Mean difference | MCSE | Pointwise 95% interval | Holm p |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `structural:M2@clean_o512_r128` | `cornet_official` | `rct_t_extra_trees` | -0.102716 | 0.0196 | [-0.142746, -0.0626853] | 0.0002971 |
| `structural:M2@clean_o512_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.271739 | 0.0279 | [-0.328722, -0.214756] | 2.727e-09 |
| `structural:M2@clean_o512_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.35557 | 0.0689 | [0.215136, 0.496004] | 0.000349 |
| `structural:M2@clean_o512_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.186546 | 0.0661 | [0.0517053, 0.321388] | 0.08013 |
| `structural:M2@clean_o512_r512` | `cornet_official` | `rct_t_extra_trees` | -0.109205 | 0.00769 | [-0.124888, -0.0935225] | 2.213e-13 |
| `structural:M2@clean_o512_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.217105 | 0.0102 | [-0.237838, -0.196372] | 2.687e-18 |
| `structural:M2@clean_o512_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0478298 | 0.0163 | [-0.0810244, -0.0146351] | 0.06791 |
| `structural:M2@clean_o512_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.15573 | 0.0148 | [-0.185851, -0.125609] | 4.27e-10 |
| `structural:M2@clean_o2048_r128` | `cornet_official` | `rct_t_extra_trees` | -0.111498 | 0.0202 | [-0.152633, -0.0703621] | 0.0001392 |
| `structural:M2@clean_o2048_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.280521 | 0.0286 | [-0.338854, -0.222188] | 2.287e-09 |
| `structural:M2@clean_o2048_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.339244 | 0.0702 | [0.195988, 0.4825] | 0.0008392 |
| `structural:M2@clean_o2048_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.170221 | 0.0728 | [0.0218388, 0.318603] | 0.1246 |
| `structural:M2@clean_o2048_r512` | `cornet_official` | `rct_t_extra_trees` | -0.110106 | 0.00864 | [-0.127729, -0.0924829] | 3.812e-12 |
| `structural:M2@clean_o2048_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.218006 | 0.0115 | [-0.24148, -0.194532] | 8.331e-17 |
| `structural:M2@clean_o2048_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0684298 | 0.00873 | [-0.0862388, -0.0506209] | 2.669e-07 |
| `structural:M2@clean_o2048_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.17633 | 0.00826 | [-0.193168, -0.159492] | 2.687e-18 |
| `structural:M2@confounded_o512_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0850291 | 0.0198 | [-0.125395, -0.044663] | 0.003178 |
| `structural:M2@confounded_o512_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.254053 | 0.0289 | [-0.312979, -0.195126] | 2.518e-08 |
| `structural:M2@confounded_o512_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.352415 | 0.0545 | [0.24136, 0.463471] | 1e-05 |
| `structural:M2@confounded_o512_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.183392 | 0.0595 | [0.062069, 0.304715] | 0.05137 |
| `structural:M2@confounded_o512_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0868093 | 0.00849 | [-0.10413, -0.0694887] | 8.895e-10 |
| `structural:M2@confounded_o512_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.19471 | 0.011 | [-0.217234, -0.172185] | 6.202e-16 |
| `structural:M2@confounded_o512_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0388039 | 0.0137 | [-0.0668202, -0.0107876] | 0.08013 |
| `structural:M2@confounded_o512_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.146704 | 0.0127 | [-0.172529, -0.12088] | 4.344e-11 |
| `structural:M2@confounded_o2048_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0972568 | 0.0199 | [-0.137821, -0.0566923] | 0.000736 |
| `structural:M2@confounded_o2048_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.26628 | 0.0285 | [-0.324444, -0.208116] | 6.908e-09 |
| `structural:M2@confounded_o2048_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.339274 | 0.079 | [0.178253, 0.500296] | 0.003178 |
| `structural:M2@confounded_o2048_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.170251 | 0.0839 | [-0.000869228, 0.341371] | 0.1533 |
| `structural:M2@confounded_o2048_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0950282 | 0.0083 | [-0.111965, -0.0780913] | 5.83e-11 |
| `structural:M2@confounded_o2048_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.202928 | 0.0111 | [-0.225567, -0.18029] | 2.252e-16 |
| `structural:M2@confounded_o2048_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0498371 | 0.0127 | [-0.0757507, -0.0239234] | 0.007249 |
| `structural:M2@confounded_o2048_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.157737 | 0.0106 | [-0.179303, -0.136172] | 6.04e-14 |
| `structural:M2@shift_o512_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0728248 | 0.018 | [-0.109465, -0.0361843] | 0.005349 |
| `structural:M2@shift_o512_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.241848 | 0.0284 | [-0.299744, -0.183952] | 4.95e-08 |
| `structural:M2@shift_o512_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.386113 | 0.0698 | [0.243824, 0.528401] | 0.0001392 |
| `structural:M2@shift_o512_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.217089 | 0.0682 | [0.077997, 0.356181] | 0.04628 |
| `structural:M2@shift_o512_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0814568 | 0.00821 | [-0.0982033, -0.0647102] | 1.784e-09 |
| `structural:M2@shift_o512_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.189357 | 0.0112 | [-0.212233, -0.166481] | 2.068e-15 |
| `structural:M2@shift_o512_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0204769 | 0.0207 | [-0.062692, 0.0217381] | 0.3302 |
| `structural:M2@shift_o512_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.128377 | 0.0179 | [-0.164927, -0.091827] | 1.557e-06 |
| `structural:M2@shift_o2048_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0803719 | 0.0189 | [-0.118909, -0.0418353] | 0.003234 |
| `structural:M2@shift_o2048_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.249395 | 0.0296 | [-0.309704, -0.189087] | 6.02e-08 |
| `structural:M2@shift_o2048_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.395486 | 0.0862 | [0.219765, 0.571208] | 0.001455 |
| `structural:M2@shift_o2048_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.226463 | 0.0888 | [0.0452732, 0.407653] | 0.1012 |
| `structural:M2@shift_o2048_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0838586 | 0.00915 | [-0.102515, -0.0652027] | 1.028e-08 |
| `structural:M2@shift_o2048_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.191759 | 0.0126 | [-0.217523, -0.165995] | 3.813e-14 |
| `structural:M2@shift_o2048_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0494703 | 0.0107 | [-0.0713502, -0.0275905] | 0.001435 |
| `structural:M2@shift_o2048_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.157371 | 0.00939 | [-0.176516, -0.138225] | 2.473e-15 |
| `structural:M2@both_o512_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0515 | 0.0182 | [-0.0885601, -0.0144399] | 0.08013 |
| `structural:M2@both_o512_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.220523 | 0.0293 | [-0.280265, -0.160782] | 5.935e-07 |
| `structural:M2@both_o512_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.430107 | 0.0796 | [0.267801, 0.592413] | 0.0001884 |
| `structural:M2@both_o512_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.261083 | 0.0833 | [0.0911185, 0.431048] | 0.04895 |
| `structural:M2@both_o512_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0607942 | 0.00914 | [-0.0794283, -0.04216] | 6.2e-06 |
| `structural:M2@both_o512_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.168694 | 0.0126 | [-0.194326, -0.143063] | 9.849e-13 |
| `structural:M2@both_o512_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0254016 | 0.0168 | [-0.0597674, 0.00896416] | 0.2836 |
| `structural:M2@both_o512_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.133302 | 0.0147 | [-0.163226, -0.103377] | 1.234e-08 |
| `structural:M2@both_o2048_r128` | `cornet_official` | `rct_t_extra_trees` | -0.0688904 | 0.0181 | [-0.10584, -0.0319404] | 0.009457 |
| `structural:M2@both_o2048_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.237914 | 0.0294 | [-0.29782, -0.178008] | 1.404e-07 |
| `structural:M2@both_o2048_r128` | `datafusionppi` | `rct_t_extra_trees` | 0.410033 | 0.0877 | [0.231115, 0.588951] | 0.001255 |
| `structural:M2@both_o2048_r128` | `datafusionppi` | `rct_dr_extra_trees` | 0.241009 | 0.093 | [0.0512968, 0.430721] | 0.1012 |
| `structural:M2@both_o2048_r512` | `cornet_official` | `rct_t_extra_trees` | -0.0695664 | 0.00869 | [-0.0872862, -0.0518466] | 1.746e-07 |
| `structural:M2@both_o2048_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.177467 | 0.0123 | [-0.202542, -0.152392] | 1.453e-13 |
| `structural:M2@both_o2048_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.0346361 | 0.0147 | [-0.0646088, -0.00466344] | 0.1246 |
| `structural:M2@both_o2048_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.142536 | 0.0125 | [-0.168079, -0.116993] | 6.555e-11 |

The JSON/CSV files preserve every requested condition and comparison. This analysis reconstructs terminal metric summaries; it does not independently replay fitted predictions. See `docs/REFERENCE_STUDY.md` for the frozen design and limitations.
