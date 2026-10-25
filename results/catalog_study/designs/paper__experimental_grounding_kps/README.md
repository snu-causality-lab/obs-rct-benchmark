# paper:experimental_grounding_kps

All declared OBS/RCT budgets and methods are retained. Means describe successful runs with their coverage; N/A is a prospectively unsupported combination. For incomplete rows, the reported mean, MCSE and interval describe successful fits only; they are not full-grid performance estimates and assign no score to failed fits. Intervals are pointwise and Holm decisions require the complete eligible cross-catalog family. The optional budget plot focuses on the two nonlinear RCT controls and official fusion recipes; all recipes appear in the table.

**All 472 eligible family-level Holm decisions are withheld:** 4 eligible contrasts lack a successful pair at every requested seed. Complete individual pointwise intervals do not restore the withheld family-level inference.

| Condition | Method | Mean MSE | MCSE | Pointwise 95% interval | Success / expected | State |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `paper:experimental_grounding_kps@o512_r128` | `rct_t_ridge` | 0.37248 | 0.029635 | [0.31325, 0.4317] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `obs_t_ridge` | 1.369 | 0.054972 | [1.2592, 1.4789] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `naive_pooled_t_ridge` | 1.0974 | 0.047039 | [1.0034, 1.1914] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `rct_t_extra_trees` | 0.60592 | 0.033611 | [0.53876, 0.67309] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `obs_t_extra_trees` | 0.45308 | 0.021515 | [0.41008, 0.49607] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `naive_pooled_t_extra_trees` | 0.31205 | 0.016517 | [0.27904, 0.34506] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `rct_dr_ridge` | 0.37467 | 0.029339 | [0.31605, 0.4333] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `rct_dr_extra_trees` | 0.64334 | 0.040183 | [0.56304, 0.72364] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `datafusionppi` | 0.3062 | 0.032952 | [0.24035, 0.37205] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r128` | `cornet_official` | 0.409 | 0.027813 | [0.35342, 0.46458] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `rct_t_ridge` | 0.18516 | 0.0051125 | [0.17495, 0.19538] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `obs_t_ridge` | 1.369 | 0.054972 | [1.2592, 1.4789] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `naive_pooled_t_ridge` | 0.70144 | 0.031371 | [0.63875, 0.76413] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `rct_t_extra_trees` | 0.39735 | 0.015965 | [0.36545, 0.42926] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `obs_t_extra_trees` | 0.45308 | 0.021515 | [0.41008, 0.49607] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `naive_pooled_t_extra_trees` | 0.19536 | 0.012053 | [0.17127, 0.21945] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `rct_dr_ridge` | 0.18502 | 0.0051255 | [0.17478, 0.19526] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `rct_dr_extra_trees` | 0.4397 | 0.023337 | [0.39306, 0.48633] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `datafusionppi` | 0.11464 | 0.0084203 | [0.097817, 0.13147] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o512_r512` | `cornet_official` | 0.28037 | 0.012841 | [0.25471, 0.30603] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `rct_t_ridge` | 0.37248 | 0.029635 | [0.31325, 0.4317] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `obs_t_ridge` | 1.3608 | 0.023623 | [1.3136, 1.408] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `naive_pooled_t_ridge` | 1.2799 | 0.023078 | [1.2338, 1.326] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `rct_t_extra_trees` | 0.60592 | 0.033611 | [0.53876, 0.67309] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `obs_t_extra_trees` | 0.50879 | 0.019211 | [0.4704, 0.54718] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `naive_pooled_t_extra_trees` | 0.46285 | 0.02001 | [0.42287, 0.50284] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `rct_dr_ridge` | 0.37467 | 0.029339 | [0.31605, 0.4333] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `rct_dr_extra_trees` | 0.64334 | 0.040183 | [0.56304, 0.72364] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `datafusionppi` | 0.27785 | 0.028972 | [0.21995, 0.33574] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r128` | `cornet_official` | 0.40916 | 0.028961 | [0.35128, 0.46703] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `rct_t_ridge` | 0.18516 | 0.0051125 | [0.17495, 0.19538] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `obs_t_ridge` | 1.3608 | 0.023623 | [1.3136, 1.408] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `naive_pooled_t_ridge` | 1.0964 | 0.021063 | [1.0544, 1.1385] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `rct_t_extra_trees` | 0.39735 | 0.015965 | [0.36545, 0.42926] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `obs_t_extra_trees` | 0.50879 | 0.019211 | [0.4704, 0.54718] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `naive_pooled_t_extra_trees` | 0.37749 | 0.01902 | [0.33948, 0.4155] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `rct_dr_ridge` | 0.18502 | 0.0051255 | [0.17478, 0.19526] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `rct_dr_extra_trees` | 0.4397 | 0.023337 | [0.39306, 0.48633] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `datafusionppi` | 0.091636 | 0.0060393 | [0.079568, 0.1037] | 64 / 64 | complete |
| `paper:experimental_grounding_kps@o2048_r512` | `cornet_official` | 0.28274 | 0.012813 | [0.25714, 0.30835] | 64 / 64 | complete |

| Condition | Fusion | RCT reference | Paired mean MSE difference | Pointwise 95% interval | Holm p | Decision |
| --- | --- | --- | ---: | --- | ---: | --- |
| `paper:experimental_grounding_kps@o512_r128` | `cornet_official` | `rct_t_extra_trees` | -0.19692 | [-0.24645, -0.14739] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.23434 | [-0.30144, -0.16723] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r128` | `datafusionppi` | `rct_t_extra_trees` | -0.29972 | [-0.36194, -0.2375] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r128` | `datafusionppi` | `rct_dr_extra_trees` | -0.33714 | [-0.41318, -0.2611] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r512` | `cornet_official` | `rct_t_extra_trees` | -0.11698 | [-0.14739, -0.08657] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.15933 | [-0.20454, -0.11411] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.28271 | [-0.31217, -0.25325] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o512_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.32505 | [-0.37049, -0.27961] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r128` | `cornet_official` | `rct_t_extra_trees` | -0.19676 | [-0.24496, -0.14856] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r128` | `cornet_official` | `rct_dr_extra_trees` | -0.23418 | [-0.29917, -0.16919] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r128` | `datafusionppi` | `rct_t_extra_trees` | -0.32807 | [-0.3902, -0.26595] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r128` | `datafusionppi` | `rct_dr_extra_trees` | -0.36549 | [-0.44135, -0.28964] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r512` | `cornet_official` | `rct_t_extra_trees` | -0.11461 | [-0.14613, -0.083086] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r512` | `cornet_official` | `rct_dr_extra_trees` | -0.15695 | [-0.20372, -0.11018] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r512` | `datafusionppi` | `rct_t_extra_trees` | -0.30572 | [-0.33588, -0.27555] | unavailable | withheld_incomplete_family |
| `paper:experimental_grounding_kps@o2048_r512` | `datafusionppi` | `rct_dr_extra_trees` | -0.34806 | [-0.39378, -0.30235] | unavailable | withheld_incomplete_family |

Negative differences favor fusion. Unresolved is not equivalence. Source assumptions, overlap and transport stresses differ across presets; no raw-MSE global ranking is reported.
