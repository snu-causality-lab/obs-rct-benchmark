# Stress study 1, addendum: the wave-2 recipes on the same seeds

**Status: addendum, not yet run.** This addendum to [stress study 1](V1_STRESS_CORNET.md) fixes the recipes, contrasts and resources before any wave-2 fit on its seeds, 80000–80063. The freeze is the commit that fills in [Recorded before launch](#recorded-before-launch); earlier commits carry drafts. The job record holds the freeze commit's hash before the first fit.

The stress plan said the wave-2 recipes would join its block later, on the same seeds, paired within each shard as in [wave 2](V1_CORE_WAVE2.md). This addendum does that. It keeps from the stress plan:
- the eight dials, four levels and two trial sizes: 64 conditions;
- the generator, `stress_dgp.py` version 2.0.0, and the seeds;
- the reference, `rct_selected`;
- the decision rule: Holm's correction within each family of 64 at 0.05, the failure rule with p = 1, and extrapolation conditions counted separately.

Adding recipes cannot change a stress-study-1 decision, because each recipe is its own family.

## Recipes and pairing

The recipes, options, environments and groups are those of [wave 2](V1_CORE_WAVE2.md#recipes-by-environment), unchanged:
- `native_r`: Integrative R, linear and quadratic, and the grf causal forest;
- `gp`: Causal-ICM with OBS and its RCT-only control;
- `tabpfn`: TSR and OBS-only TabPFN under releases v2, v2.5 and v3, and the RCT-only GP;
- `tsr_legacy`: the registered TSR recipes under `tabpfn==2.0.9`.

Every shard also refits `rct_selected` and `rct_t_ridge`. Each contrast pairs two recipes fitted in the same shard and environment. The configurations are [`v1_stress_wave2_native_r.toml`](../configs/v1_stress_wave2_native_r.toml), [`v1_stress_wave2_gp.toml`](../configs/v1_stress_wave2_gp.toml), [`v1_stress_wave2_tabpfn.toml`](../configs/v1_stress_wave2_tabpfn.toml) and [`v1_stress_wave2_tsr_legacy.toml`](../configs/v1_stress_wave2_tsr_legacy.toml). The [shard writer](../scripts/make_stress_wave2_configs.py) validates them. It writes one shard per seed and dial, holding that dial's eight conditions: 512 shards per group.

**Same data.** For each group, the analysis compares its `rct_selected` and `rct_t_ridge` scores with stress study 1's.
- A group ran on the stress-study-1 data only if all of its `rct_t_ridge` scores reproduce stress study 1 within 1e-9. Otherwise no claim pairs its results with stress study 1.
- Differences in `rct_selected` are reported separately; its ExtraTrees candidates can differ across library versions.
- Neither check blocks a verdict, since verdicts never cross environments.

## Contrasts

Every contrast is a paired t-test over 64 seeds. There are 20 families of 64 conditions each, named as in wave 2.

- **Primary: each recipe that uses OBS against `rct_selected`.** Eleven families:
  - Integrative R, linear and quadratic;
  - Causal-ICM with OBS;
  - TSR under v2, v2.5 and v3, and OBS-only TabPFN under v2, v2.5 and v3;
  - the legacy TSR and OBS-only TabPFN recipes.
- **Matched: adding OBS to a fixed backbone.** Five families:
  - Causal-ICM against its RCT-only control;
  - TSR under v2, v2.5 and v3, each against the RCT-only GP;
  - legacy TSR against the RCT-only GP.
- **TabPFN release.** Four families: TSR and OBS-only TabPFN under v2.5 and v3, each against v2 of the same recipe.
- **RCT-only recipes against `rct_selected` are descriptive**, as in the stress plan: the causal forest, the RCT-only Causal-ICM and the RCT-only GP.

**Readout.** For every primary family, each dial and each trial size gets the stress plan's four states, from its verdicts at levels 0 and 3: no level-0 win, kept, stopped or broke.

**Counts.** Each family's verdicts are counted separately for the 16 extrapolation conditions and the other 48.

**Descriptive.** Every recipe's error relative to `rct_selected` by dial and level, as dose-response curves. As in the stress plan, the curves show the ratio of mean errors and the geometric mean of the seedwise ratios, with a pointwise 95% interval.

## Execution

- **Before results are read.** No metric is read until every shard has finished and coverage is authenticated. Monitoring reads only status, runtime, memory and failure types.
- **Driver and failures.** The wave-2 rules apply, with the same shard driver and environment check:
  - the four kinds of infrastructure failure;
  - reruns in fresh directories with the old output kept;
  - GPU-memory reruns alone on a GPU;
  - no deadline beyond the recipes' own.
- **Preflight.** One seed outside the grid, 79999, runs every group's eight shards before the freeze. Its job deletes all records after writing status files.
- **Limits.** The wave-2 rule turns the preflight into limits: CPU time and memory per shard, and shards per GPU.
- **Expected cost.** The `obs_size` shard at level 0 has 8,192 OBS rows, four times wave 2's largest. The GP recipes that fit OBS rows, Causal-ICM in particular, may be slow there; the preflight shows whether they finish.

## Checks

1. **Coverage.** Every declared shard and every requested cell has a terminal record. A group's shards share one source, environment and external-method provenance. Every successful TabPFN fit ran stage 1 on CUDA with the pinned checkpoint, package and release.
2. **Same data.** The identity check above, recomputed by the independent checker.
3. **Reproducibility.** On the first seed at RCT 128, levels 0 and 3 of every dial, each recipe is refitted twice in its group's environment. CPU recipes must agree within 1e-12, TabPFN recipes on a GPU within a relative 1e-6.
4. **Independent arithmetic.** A [separate script](../scripts/verify_stress_wave2.py) recomputes every contrast, Holm adjustment, count and readout state from the per-seed records and the declared families, without importing the [analyzer](../scripts/analyze_stress_wave2.py).

The stress plan's data check is not repeated: the generator and seeds are unchanged, and the identity check covers the data.

## What this addendum will not show

- **Dials built for CorNet.** The dials violate CorNet's stated assumptions, not those of the wave-2 recipes. A wave-2 recipe that keeps winning here may still fail where its own assumptions break; that needs its own stress study.
- **The stress plan's limits apply**, including the bias-frequency dial, which made the bias easier for smoothing learners.
- **No human check.** One AI agent designed, implemented and will analyze this addendum.

## Recorded before launch

Preflight results (seed 79999, eight shards per group). Each group's limit follows from the wave-2 rule.

| Group | Longest shard | Median shard | Peak memory | Preflight failures | Main-run limit per shard |
| --- | ---: | ---: | ---: | --- | --- |
| `native_r` | *pending* | | | | |
| `gp` | *pending* | | | | |
| `tabpfn` | *pending* | | | | |
| `tsr_legacy` | *pending* | | | | |

## Revisions and errata

None yet.
