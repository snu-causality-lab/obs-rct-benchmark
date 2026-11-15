# What stress study 1 shows

The [plan](../../docs/V1_STRESS_CORNET.md) was frozen at commit `c4bd9a1` before any fit on its seeds. It built eight generator dials from the assumptions CorNet's paper states. It then ran the twelve wave-1 recipes on 8 dials × 4 levels × 2 trial sizes × 64 seeds.

## Coverage and checks

- **Fits.** All 49,152 fits succeeded; none failed or are missing.
- **Data check.** A separate script regenerated every role's covariates, assignments and outcomes from the plan's formulas, without calling the generator. It matched all 64 conditions at 1e-12.
- **Reproduction.** Two refits of each of the 768 first-seed cells matched the records: 1,536 refits.
- **Arithmetic.** An independent recomputation of every contrast, Holm adjustment, count and readout state found no mismatch.

## CorNet's weak point is the shared representation

The `hidden_modifier` dial removes an effect modifier from the OBS outcomes while holding the amount of confounding fixed. CorNet's error relative to `rct_selected`, the RCT-only reference, rises steadily:

| Level | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: |
| CorNet raw, RCT 128 | 0.54 | 0.73 | 0.84 | 0.91 |
| CorNet raw, RCT 512 | 0.99 | 1.60 | 1.92 | 2.10 |

Entries are ratios of mean CATE MSE to `rct_selected`.

- **RCT 128.** CorNet's advantage disappears: it is significantly better at levels 0 and 1 and unresolved at levels 2 and 3.
- **RCT 512.** It becomes significantly *worse* from level 1 onward, with twice the reference's error at level 3.
- **Standardized inputs.** CorNet with standardized inputs follows the same pattern at RCT 128. At RCT 512 it is significantly worse at every level, including level 0.

This is the assumption the paper's theory leans on most (§5.1, condition 1): what the trial correction needs must be carried by a representation learned from OBS outcomes. When the OBS outcomes hide an effect modifier, the correction cannot recover it, however much trial data there is.

## The other stated assumptions barely mattered

With a 128-row trial, CorNet was significantly better than the reference at level 0 of every dial. It stayed better at level 3 in seven of the eight dials ("kept"); the eighth is `hidden_modifier` ("stopped").

| Dial | CorNet raw, RCT 128, levels 0 → 3 | CorNet raw, RCT 512, levels 0 → 3 |
| --- | --- | --- |
| `bias_complexity` | 0.61 → 0.46 | 1.28 → 0.80 |
| `common_shift` | 0.54 → 0.50 | 0.86 → 0.79 |
| `obs_selection` | 0.48 → 0.46 | 0.76 → 0.74 |
| `obs_size` | 0.58 → 0.63 | 1.04 → 1.24 |
| `noise_shape` | 0.53 → 0.51 | 1.10 → 0.75 |
| `trial_shrink` (extrapolation) | 0.54 → 0.35 | 0.99 → 0.44 |
| `target_shift` (extrapolation) | 0.55 → 0.60 | 1.08 → 1.12 |

- **RCT 512.** With a 512-row trial, CorNet was not significantly better than the reference at level 0 of any dial, consistent with the [v1 core result](../v1_core/SCIENTIFIC_RESULT.md). The readout state is therefore "no level-0 win" for all eight dials.
  - At level 0 it was unresolved, or significantly worse in a few dials: `bias_complexity` in both input forms, and with standardized inputs also `hidden_modifier` and `noise_shape`.
  - At later levels, raw-input CorNet became significantly better under `trial_shrink` (levels 2–3) and `noise_shape` (level 3). With standardized inputs it became worse under `obs_size` (level 3).
- **The paper's predictions, checked against the data:**
  - **Confirmed.** CorNet's advantage grows as the trial covers a narrower part of the target population (`trial_shrink`). The predicted loss under a hidden effect modifier was also confirmed.
  - **Not observed.** The predicted loss of advantage with a more complex bias, a per-arm shift or a smaller OBS sample was not observed at RCT 128. At RCT 512, a smaller OBS sample cost CorNet a little.
  - **Consistent.** OBS treatment selection and unequal noise had no material effect, as predicted.

## A limitation of the bias-frequency dial

Holding the bias variance fixed while raising its frequency made the bias easier, not harder, for smoothing learners. A high-frequency bias of fixed size averages out under smoothing. The OBS-only ridge learner shows this: its error falls from 1.54 to 0.08 times the reference's at RCT 128. So `bias_complexity` and `common_shift` do not test "a bias too complex for the correction" as intended.

A dial that tests that assumption needs a bias that is complex yet not averaged away, such as a non-smooth function of a low-dimensional projection. That needs a new, separately frozen construction.

## Other recipes on the same data

- **DataFusionPPI** stays close to the reference everywhere. It is never significantly better, and significantly worse in 3 of 64 conditions. Its error rises when the trial is narrow (2.03 times the reference at `trial_shrink` level 3, RCT 128) or the OBS sample is small (1.57 at `obs_size` level 3).
- **OBS-only and naive pooled learners** are excellent when the OBS CATE is unbiased (`common_shift`, `obs_selection`): about 0.08 times the reference at RCT 128. They are poor under a shifted target or a small OBS sample. CorNet does not exploit an unbiased OBS CATE nearly as well; it stays near 0.5 in those dials.

## Counts

Holm's correction runs within each recipe's family of 64 conditions. The 16 extrapolation conditions come from `trial_shrink` and `target_shift`.

| Recipe | Better | Worse | Unresolved | Extrapolation: better / worse / unresolved |
| --- | ---: | ---: | ---: | --- |
| CorNet, raw inputs | 23 | 4 | 21 | 10 / 0 / 6 |
| CorNet, standardized inputs | 22 | 7 | 19 | 10 / 0 / 6 |
| DataFusionPPI | 0 | 1 | 47 | 0 / 2 / 14 |
| OBS-only T ridge | 22 | 16 | 10 | 5 / 9 / 2 |
| OBS-only T trees | 9 | 28 | 11 | 2 / 9 / 5 |
| Naive pooled T ridge | 30 | 13 | 5 | 6 / 8 / 2 |
| Naive pooled T trees | 10 | 20 | 18 | 2 / 9 / 5 |

## What this study does not show

- **Synthetic constructions.** These are synthetic constructions built to test stated assumptions, not real data. CorNet's loss under a hidden effect modifier shows one way its central assumption can fail. It does not estimate how often that happens in practice.
- **One construction per assumption.** Each dial is one violation of its assumption, and the bias-frequency dial turned out not to stress what it was meant to.
- **The reference.** `rct_selected` is an observable reference with about 1.3 times the error of the best RCT-only candidate in the v1 study; ratios against it are not ratios against the best possible RCT-only learner.
- **Recipes not included.** The wave-2 recipes ran later on the same seeds, in the [stress addendum](../stress_wave2/SCIENTIFIC_RESULT.md).
- **No human check.** One AI agent designed, ran and analyzed this study. A fresh-context AI review shaped its dials before the freeze.
