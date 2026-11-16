| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | Extrapolation: better / worse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gp/causal_icm_selected` | primary | 64 | 32 | 4 | 28 | 0 | 7 / 0 |
| `gp/causal_icm_selected vs causal_icm_rct_only` | matched_obs | 64 | 57 | 0 | 7 | 0 | 14 / 0 |
| `gp/causal_icm_selected vs causal_icm_fixed` | selection | 64 | 35 | 2 | 27 | 0 | 5 / 1 |

| Recipe refitted from wave 2 | Equal to wave 2 | Largest difference | Within 1e-9, same statuses |
| --- | ---: | ---: | --- |
| `rct_selected` | 4096 of 4096 | 0.0 | yes |
| `rct_t_ridge` | 4096 of 4096 | 0.0 | yes |
| `causal_icm_fixed` | 4096 of 4096 | 0.0 | yes |
| `causal_icm_rct_only` | 4096 of 4096 | 0.0 | yes |

Successful selected fits that chose 0.5: 17; differing from `causal_icm_fixed` or without a successful `causal_icm_fixed` twin: 0.

| Selected rho | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Successful fits | 11 | 1 | 3 | 9 | 9 | 17 | 38 | 70 | 161 | 2162 | 1615 |
