# Stress study 1, addendum: the wave-2 recipes on the same seeds

**Status: addendum, fixed before its fits.** This addendum to [stress study 1](V1_STRESS_CORNET.md) fixes the recipes, contrasts and resources before any wave-2 fit on its seeds, 80000–80063. The freeze is the commit that fills in [Recorded before launch](#recorded-before-launch); earlier commits carry drafts. The job record holds the freeze commit's hash before the first fit.

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

**Same data.** `rct_t_ridge` uses only trial rows, and five of the eight dials act only on OBS, so its scores cannot show that OBS rows agree.
- Before launch, the [data check](../scripts/check_data_identity.py) regenerates all 64 conditions at the first seed, 80000, in each group's environment. It compares every sample role with stress study 1's environment: OBS, trial training, tuning and test rows, evaluation covariates and effects.
- A group ran on the stress-study-1 data only if every array agrees within 1e-12 there **and** all of its `rct_t_ridge` scores reproduce stress study 1 within 1e-9. Otherwise no claim pairs its results with stress study 1.
- **Reference drift.** Each group's `rct_selected` is compared with stress study 1's, and the analyzer prints the exact count and largest difference next to the headline. Its ExtraTrees candidates can differ across library versions. Ratios set side by side across groups, or against stress study 1's recipes, therefore have different denominators and are descriptive unless that group's `rct_selected` reproduces.
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
  - TSR under v2, v2.5 and v3, each against the `tabpfn` group's RCT-only GP;
  - legacy TSR against the `tsr_legacy` group's RCT-only GP.
- **TabPFN release.** Four families: TSR and OBS-only TabPFN under v2.5 and v3, each against v2 of the same recipe.
- **RCT-only recipes against `rct_selected` are descriptive**, as in the stress plan: the causal forest, the RCT-only Causal-ICM and the RCT-only GP.

**Readout.** For every primary family, each dial and each trial size gets the stress plan's four states, from its verdicts at levels 0 and 3: no level-0 win, kept, stopped or broke. As in the stress plan, "untested" counts as not better. A state read from an untested level reflects failed fits, not a loss, so the readout flags the untested levels.

**Counts.** Each family's verdicts are counted separately for the 16 extrapolation conditions and the other 48.

**Descriptive.** Every recipe's error relative to `rct_selected` by dial and level, as dose-response curves. As in the stress plan, the curves show the ratio of mean errors and the geometric mean of the seedwise ratios, with a pointwise 95% interval.

## Execution

- **Before results are read.** No metric is read until every shard has finished and coverage is authenticated. Monitoring reads only status, runtime, memory and failure types.
- **Driver and failures.** The execution rules of [wave 2](V1_CORE_WAVE2.md#resources-and-execution), as frozen at the commit recorded below, apply with the same shard driver and environment check:
  - four kinds of infrastructure failure: a job limit (time or memory), an interruption, a failed environment check, and a CUDA out-of-memory record;
  - reruns in fresh directories with the old output kept;
  - a job-limit rerun with double the limit that was reached, and a GPU-memory rerun alone on a GPU;
  - one rerun only for a job limit: if the rerun also reaches a limit, its recorded cells stand and its unrecorded cells become `JobLimit` method failures, listed for the analyzer and checker;
  - no deadline beyond the recipes' own. Causal-ICM and the GPs have none, so the final-attempt rule bounds them.
- **Preflight.** One seed outside the grid, 79999, runs every group's eight shards before the freeze. Its job deletes all records after writing status files.
- **Limits.** The wave-2 rule turns the preflight into limits: CPU time and memory per shard, and shards per GPU.
- **Expected cost.** The `obs_size` condition at level 0 has 8,192 OBS rows, four times wave 2's largest. Causal-ICM, which fits a GP to OBS rows, may be slow there; the preflight shows whether it finishes.

## Checks

1. **Coverage.** Every declared shard and every requested cell has a terminal record. A group's shards share one source, environment and external-method provenance. Every successful TabPFN fit ran stage 1 on CUDA with the pinned checkpoint, package and release.
2. **Identities.**
   - The data check before launch, and the `rct_t_ridge` and `rct_selected` comparisons above, recomputed by the independent checker.
   - In the five dials that change only OBS (`hidden_modifier`, `bias_complexity`, `common_shift`, `obs_selection`, `obs_size`), trial rows, targets and method seeds are the same at every level. Every RCT-only recipe must therefore score identically across levels; mismatches are counted.
3. **Reproducibility.** On the first seed at RCT 128, levels 0 and 3 of every dial, each recipe is refitted twice in its group's environment. CPU recipes must agree within 1e-12, TabPFN recipes on a GPU within a relative 1e-6.
4. **Independent arithmetic.** A [separate script](../scripts/verify_stress_wave2.py) recomputes every contrast, Holm adjustment, count, headline row, readout state and identity count from the per-seed records and the declared families, without importing the [analyzer](../scripts/analyze_stress_wave2.py).

The stress plan's check of the generator against its formulas is not repeated, since the generator and seeds are unchanged; the data check compares environments.

## What this addendum will not show

- **Dials built for CorNet.** The dials violate CorNet's stated assumptions, not those of the wave-2 recipes. A wave-2 recipe that keeps winning here may still fail where its own assumptions break; that needs its own stress study.
- **The stress plan's limits apply**, including the bias-frequency dial, which made the bias easier for smoothing learners.
- **No human check.** One AI agent designed, implemented and will analyze this addendum.

## Recorded before launch

Preflight results (seed 79999, eight shards per group). Each group's limit follows from the wave-2 rule: per shard for the CPU groups, per GPU job for the GPU groups.

| Group | Longest shard | Median shard | Peak memory | Preflight failures | Main-run limit |
| --- | ---: | ---: | ---: | --- | --- |
| `native_r` | 1,228 s (`trial_shrink`) | 696 s | 466 MiB | none | 1 CPU, 4 GiB, 2 h |
| `gp` | 700 s (`obs_size`) | 173 s | 5,052 MiB | none | 1 CPU, 10 GiB, 2 h |
| `tabpfn` | 211 s (`obs_selection`) | 195 s | 2,614 MiB | none | 2 shards per GPU, 4 CPUs, 16 GiB, 2 h (longest preflight GPU job 218 s) |
| `tsr_legacy` | 120 s (`target_shift`) | 103 s | 1,751 MiB | none | 2 shards per GPU, 4 CPUs, 16 GiB, 2 h (longest preflight GPU job 127 s) |

**Execution.** The shard driver and job scripts are published with the results; their hashes are fixed here.

| Item | Value |
| --- | --- |
| Wave-2 freeze commit (execution rules) | `80c9129d0d6046a9e6845c9f0a3580c67ca024d9` |
| Code | The freeze commit, run from a `git archive` snapshot |
| Shard driver and job scripts | Those of the wave-2 main run: `run_shard.py` `1a299a69c5132804ad619df9300861df6fa44559b47a836bf27023e536c0a23b`; `common.sh` `e19ad30219e8b26e474aa5bc7e8ab5143c1ea9b2c57900a51698fce88796dbf0`; `cpu_array.sbatch` `05cd6fda42accfc470c841e8bfb52115c30eb4a9769fa61a2d9b311a6f24182f`; `gpu_array.sbatch` `d6cb20f028180a0103548f21cd824be65643d96df3f0a0596f28d0e380a96a16`. Shards are written by `write_stress_shards.sbatch` `28961c7979ee98a134224736ece68ed3f6c30f74402687583c4a3cea940f954b` |
| GPU jobs | As in wave 2: two shards per GPU, one RTX 3090, 4 CPUs and 16 GiB of host memory per job; nodes b03, b14, b15, b24–b28, b30 and b31 excluded |
| Preflight runs | Draft code whose fitting code, configurations and shard writer equal the freeze's. Jobs 822060 (`native_r`) and 822061 (`gp`), driver `8bf51696…`; jobs 822062 (`tabpfn`) and 822063 (`tsr_legacy`), driver `1a299a69…`, two shards per GPU |
| Data check (seed 80000, against stress study 1's environment) | Passed. In all 64 conditions, every array of every sample role is identical in the `gp`, `tabpfn` and `tsr_legacy` environments; the largest difference is 0. `native_r` runs in stress study 1's interpreter itself |

## Revisions and errata

**2026-11-15, before the freeze.** A fresh-context AI review of the first draft led to these changes:
- the execution rules cite the frozen wave-2 addendum and add a final attempt for job limits;
- the same-data rule adds a data check of every sample role;
- the launch record fixes the driver, job scripts and GPU settings;
- the RCT-only identity across levels was added;
- the readout flags untested levels;
- cross-group ratios are stated as descriptive;
- the checker also verifies the headline, identity counts and duplicate rows.
