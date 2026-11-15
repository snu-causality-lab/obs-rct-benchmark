| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | Extrapolation: better / worse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `native_r/integrative_r_linear` | primary | 64 | 46 | 1 | 17 | 0 | 9 / 1 |
| `native_r/integrative_r_quadratic` | primary | 64 | 32 | 4 | 28 | 0 | 6 / 2 |
| `gp/causal_icm_fixed` | primary | 64 | 27 | 10 | 27 | 0 | 8 / 2 |
| `tabpfn/tsr_tabpfn_v2` | primary | 64 | 3 | 42 | 19 | 0 | 3 / 9 |
| `tabpfn/tsr_tabpfn_v25` | primary | 64 | 3 | 43 | 18 | 0 | 3 / 10 |
| `tabpfn/tsr_tabpfn_v3` | primary | 64 | 3 | 42 | 19 | 0 | 3 / 10 |
| `tabpfn/tabpfn_v2_obs_only` | primary | 64 | 29 | 26 | 9 | 0 | 5 / 9 |
| `tabpfn/tabpfn_v25_obs_only` | primary | 64 | 30 | 24 | 10 | 0 | 6 / 7 |
| `tabpfn/tabpfn_v3_obs_only` | primary | 64 | 29 | 28 | 7 | 0 | 6 / 8 |
| `tsr_legacy/tsr_tabpfn_lmcgp` | primary | 64 | 3 | 41 | 20 | 0 | 3 / 10 |
| `tsr_legacy/tabpfn_obs_only` | primary | 64 | 29 | 26 | 9 | 0 | 5 / 9 |
| `gp/causal_icm_fixed vs causal_icm_rct_only` | matched_obs | 64 | 57 | 0 | 7 | 0 | 14 / 0 |
| `tabpfn/tsr_tabpfn_v2 vs lmcgp_rct_only` | matched_obs | 64 | 3 | 12 | 49 | 0 | 2 / 1 |
| `tabpfn/tsr_tabpfn_v25 vs lmcgp_rct_only` | matched_obs | 64 | 3 | 11 | 50 | 0 | 3 / 0 |
| `tabpfn/tsr_tabpfn_v3 vs lmcgp_rct_only` | matched_obs | 64 | 3 | 12 | 49 | 0 | 3 / 1 |
| `tsr_legacy/tsr_tabpfn_lmcgp vs lmcgp_rct_only` | matched_obs | 64 | 3 | 10 | 51 | 0 | 2 / 1 |
| `tabpfn/tsr_tabpfn_v25 vs tsr_tabpfn_v2` | tabpfn_release | 64 | 4 | 0 | 60 | 0 | 4 / 0 |
| `tabpfn/tsr_tabpfn_v3 vs tsr_tabpfn_v2` | tabpfn_release | 64 | 3 | 2 | 59 | 0 | 2 / 0 |
| `tabpfn/tabpfn_v25_obs_only vs tabpfn_v2_obs_only` | tabpfn_release | 64 | 38 | 4 | 22 | 0 | 16 / 0 |
| `tabpfn/tabpfn_v3_obs_only vs tabpfn_v2_obs_only` | tabpfn_release | 64 | 44 | 10 | 10 | 0 | 16 / 0 |

Each family pairs recipes fitted in one group's environment. Each group has its own `rct_selected`; ratios set side by side across groups, or against stress study 1, are descriptive unless that group's `rct_selected` reproduces stress study 1.

| Group | `rct_selected` equal to stress study 1 | Largest difference | `rct_t_ridge` within 1e-9 of stress study 1 |
| --- | ---: | ---: | --- |
| `native_r` | 4096 of 4096 | 0.0 | yes |
| `gp` | 4096 of 4096 | 0.0 | yes |
| `tabpfn` | 4096 of 4096 | 0.0 | yes |
| `tsr_legacy` | 4096 of 4096 | 0.0 | yes |
