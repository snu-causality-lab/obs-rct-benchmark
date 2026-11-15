# What the stress addendum shows

The [addendum](../../docs/V1_STRESS_WAVE2.md) was frozen at commit `61cb584` before any fit on its seeds. It ran the wave-2 recipes on [stress study 1](../stress_cornet/SCIENTIFIC_RESULT.md)'s eight dials: each dial moves one of CorNet's stated assumptions over four levels, at two trial sizes, on the same 64 seeds. That gives 64 conditions per recipe, paired within each shard against `rct_selected`, the RCT-only reference.

## Coverage and checks

- **Fits.** All 94,208 requested cells have a successful record, after the slow-node rerun below.
- **Same data as stress study 1.** In all four environments, every sample role matched stress study 1 at the first seed. All 4,096 `rct_t_ridge` and all 4,096 `rct_selected` scores per group equal stress study 1's exactly. So ratios against `rct_selected` here share their denominator with stress study 1's; comparisons across the two are descriptive.
- **RCT-only identity.** In the five dials that change only OBS, every RCT-only recipe scores identically at all four levels: no mismatch.
- **Reproduction.** On the first seed at RCT 128, levels 0 and 3 of every dial, each recipe was refitted twice in its group's environment: 368 cases, 736 refits, no mismatch.
- **Arithmetic.** An independent recomputation of every contrast, Holm adjustment, count and readout state found no mismatch.

## Two corrections after the fits

Both are recorded in the plan's [revisions and errata](../../docs/V1_STRESS_WAVE2.md#revisions-and-errata).
- **Source check.** Before any metric was read, the analyzer's first run refused the study. Each environment had recorded the version of a stale package install, or none, and these differed from the analysis package's, although every run used byte-identical code. The analysis runs the frozen analyzer unchanged through a [runner](evidence/jobs/analyze_erratum.py) that checks the code hashes instead.
- **Slow-storage node.** The first analysis had 15 failed fits, all from four shards that ran on one slow-storage node, which the CPU array had not excluded. Rerun whole on a normal node, those shards had no failures, and the cells that succeeded both times scored identically. The rerun is used. This rule was set after the first analysis was read, so that analysis is kept in [`as_frozen/`](as_frozen/headline.md). Its only differences are the 15 cells in [`as_frozen/changed_cells.csv`](as_frozen/changed_cells.csv), which made 12 Integrative R conditions untested, all at level 3. Some findings below rest on those conditions and did not exist in the first analysis, where they read "stopped" with level 3 untested:
  - the linear learner's one loss and the "broke" state of both bases under `target_shift` with RCT 128;
  - the "kept" state of the linear learner under `bias_complexity` (RCT 512) and `obs_selection` (both trial sizes), and of the quadratic one under `obs_selection`.

## Integrative R holds up best

The linear Integrative R-learner is the recipe that uses OBS most reliably here. It is significantly better than `rct_selected` in 46 of 64 conditions and worse in one:

| Recipe | Other 48: better / worse / unresolved | Extrapolation 16: better / worse / unresolved |
| --- | --- | --- |
| Integrative R, linear | 37 / 0 / 11 | 9 / 1 / 6 |
| Integrative R, quadratic | 26 / 2 / 20 | 6 / 2 / 8 |
| Causal-ICM with OBS | 19 / 8 / 21 | 8 / 2 / 6 |
| OBS-only TabPFN v2.5 | 24 / 17 / 7 | 6 / 7 / 3 |
| TSR, TabPFN v2.5 | 0 / 33 / 15 | 3 / 10 / 3 |
| *Stress study 1: CorNet, raw inputs* | *23 / 4 / 21* | *10 / 0 / 6* |

Holm's correction runs within each recipe's 64 conditions. No condition is untested. The v2 and v3 releases and the legacy TSR recipes give nearly the same counts as v2.5 ([headline](headline.md)).

- **With the larger trial.** At RCT 512 the linear learner wins at level 0 in seven of eight dials and keeps the win at level 3 in six. CorNet won at level 0 in none.
- **CorNet's weak point.** The `hidden_modifier` dial removes an effect modifier from the OBS outcomes. The linear Integrative R-learner stays significantly better at level 3 at both trial sizes, although its advantage narrows in between: it is unresolved at level 2, and also at level 1 with RCT 512:

  | Ratio to `rct_selected`, level 0 → 3 | RCT 128 | RCT 512 |
  | --- | --- | --- |
  | Integrative R, linear | 0.63 → 0.78 | 0.49 → 0.44 |
  | CorNet, raw inputs (stress study 1) | 0.54 → 0.91 | 0.99 → 2.10 |

  The quadratic basis fares worse under this dial: with RCT 128 its advantage is gone from level 2 on (0.53 at level 0, 1.00 at level 2, 0.89 at level 3), and with RCT 512 it is significantly worse at level 2. Why the linear learner resists the dial is not tested here.
- **Where it breaks.** With the smaller trial, a shifted target population breaks it: significantly better at level 0, significantly worse at level 3 (0.71 → 1.51). The OBS-only learners break in the same way. With the smaller trial, its level-3 verdict is unresolved under a narrow trial and under unequal noise. Under the narrow trial its error ratio keeps falling (0.71 → 0.52) and only the Holm-adjusted significance is lost; under unequal noise the advantage does shrink (0.65 → 0.78). At level 0 of `obs_size`, with 8,192 OBS rows, it is not significantly better at either trial size.
- **The quadratic basis.** Descriptively, averaged over levels, its error is higher than the linear basis's in seven of eight dials with RCT 512, and in four of eight with RCT 128. At RCT 512 it wins at level 0 only where the OBS CATE is unbiased (`common_shift`, `obs_selection`).

## Causal-ICM: OBS helps, but the backbone is weak

Adding OBS to Causal-ICM's GP never significantly hurts it: better than its own RCT-only fit in 57 of 64 conditions, worse in none. Against `rct_selected`, though, it wins at level 0 in seven dials with the smaller trial, keeping the win at level 3 in four, and in none with the larger trial. At level 0 its RCT-only fit has 1.1 to 1.5 times the reference's error. At RCT 512 it is significantly worse in 10 conditions: every level of `obs_size`, levels 2–3 under unequal noise (`noise_shape`) and a shifted target, and level 1 of `bias_complexity` and `common_shift`.

## TSR never beats the reference at level 0

TSR is never significantly better than `rct_selected` at level 0, in any dial, trial size or TabPFN release. Its only wins are under a narrow trial (`trial_shrink` levels 2–3), three conditions in each release. Against its own RCT-only GP it is better in 3 of 64 conditions and worse in 10 to 12.
- **Unequal noise** is its worst case: at level 3 with RCT 512 its error is 4.4 times the reference's under v2.5 and 4.6 times under the legacy recipe. Its RCT-only GP shows almost the same rise (3.9 times), and TSR is not significantly different from it there, which is consistent with the rise coming from the GP stage, which assumes constant noise, rather than from the OBS stage. No heteroscedastic control was run to test that.
- **Release.** TabPFN v2.5 and v3 change TSR's error in at most 5 of 64 conditions. For the OBS-only TabPFN model, newer releases help: v2.5 is better than v2 in 38 conditions and worse in 4, and v3 better in 44 and worse in 10.

## OBS-only learners: excellent when OBS is unbiased, poor with a biased OBS and a larger trial

At level 0 with the smaller trial, OBS-only TabPFN v2.5 reaches about 0.1 times the reference's error when the OBS CATE is unbiased (`common_shift`, `obs_selection`). With a biased OBS CATE and the smaller trial it is mixed: better in 12 of the 24 conditions of the six biased dials and worse in 4. With the larger trial, at level 0 of those six dials, it has 2.3 to 4.6 times the reference's error (2.4 to 5.1 in the other releases and the legacy recipe). This repeats stress study 1's finding for OBS-only ridge and pooled learners.

## The RCT-only forest

Descriptively, grf's causal forest fitted to trial data has 0.89 to 0.97 times `rct_selected`'s error at level 0 with RCT 128, and 1.37 to 1.55 times with RCT 512. The reference is an observable RCT-only selector, not the best possible RCT-only learner, as the [v1 core study](../v1_core/SCIENTIFIC_RESULT.md) notes.

## What this study does not show

- **CorNet's assumptions, not every method's.** The dials were built from the assumptions CorNet's paper states. They do not target Integrative R's, Causal-ICM's or TSR's own assumptions. A recipe that holds up here has not been stressed where its theory is weakest.
- **The bias-frequency dial.** As in stress study 1, raising the bias frequency at fixed variance made the bias easier for smoothing learners: Integrative R's error falls from 0.88 to 0.18 times the reference at RCT 128, and OBS-only TabPFN's from 1.38 to 0.10. That dial does not test "a bias too complex for the correction".
- **Synthetic constructions.** These are synthetic constructions, not real data. They show how each method responds to one violation at a time, not how often a violation occurs in practice.
- **The reference.** Ratios are against `rct_selected`, an observable reference, not the best possible RCT-only learner.
- **No human check.** One AI agent designed, ran and analyzed this addendum. Fresh-context AI reviews checked the plan before the freeze and this write-up against the data. No human has reproduced it.
