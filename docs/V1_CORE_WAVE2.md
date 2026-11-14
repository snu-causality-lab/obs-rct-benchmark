# v1 core wave 2: native recipes and TabPFN releases

**Status: addendum, fixed before its fits.** This addendum to the [v1 core plan](V1_CORE_STUDY.md) fixes the recipes, environments, contrasts and resources of wave 2 before any wave-2 fit on seeds 70000–70063. The freeze is the commit that fills in [Recorded before launch](#recorded-before-launch); earlier commits carry drafts. The job record holds the freeze commit's hash before the first fit.

Wave 2 keeps everything from the plan that it does not change here:
- the 124 conditions;
- the seeds;
- the reference rule, `rct_selected`;
- the decision rule: per-recipe Holm families at 0.05, the failure rule with p = 1, M11 descriptive, transport conditions counted separately, and RCT-only recipes counted per distinct (preset, RCT size).

## Recipes, by environment

The recipes need four different environments, so they run in four groups. Every shard of every group also refits `rct_selected` and `rct_t_ridge`.

| Group | Recipes | Settings | Environment |
| --- | --- | --- | --- |
| `native_r` | `integrative_r_linear`, `integrative_r_quadratic`, `rct_causal_forest` | As admitted ([V03_ADAPTERS](V03_ADAPTERS.md)): Integrative R with five folds, its native SCAD search and a 1,200-second deadline; grf with 2,000 honest trees, one thread and a 600-second deadline | The wave-1 Python environment, R from the GP pilot environment, pinned intRlearner and grf builds |
| `gp` | `causal_icm_fixed`, `causal_icm_rct_only` | Fixed ρ = 0.5, all covariates, native optimizer defaults | The GPy environment of the GP pilot (Python 3.11, GPy 1.13.2) |
| `tabpfn` | `lmcgp_rct_only`; `tsr_tabpfn_v2`, `tsr_tabpfn_v25`, `tsr_tabpfn_v3`; `tabpfn_v2_obs_only`, `tabpfn_v25_obs_only`, `tabpfn_v3_obs_only` | TSR's GP as in the [TSR study](TSR_STUDY.md). Stage 1 is the TabPFN regressor of each release with that release's default settings, `random_state=0`, and a checkpoint pinned by SHA-256 | `tabpfn==10.0.0`, torch 2.15, RTX 3090 GPUs shared by several shards |
| `tsr_legacy` | `tsr_tabpfn_lmcgp`, `lmcgp_rct_only`, `tabpfn_obs_only` | Exactly as in the [TSR study](TSR_STUDY.md): `tabpfn==2.0.9`, the pinned v2 checkpoint | The TSR study's environment, RTX 3090 GPUs shared by several shards |

The configurations are [`v1_wave2_native_r.toml`](../configs/v1_wave2_native_r.toml), [`v1_wave2_gp.toml`](../configs/v1_wave2_gp.toml), [`v1_wave2_tabpfn.toml`](../configs/v1_wave2_tabpfn.toml) and [`v1_wave2_tsr_legacy.toml`](../configs/v1_wave2_tsr_legacy.toml). The [shard writer](../scripts/make_v1_wave2_configs.py) validates them and writes one shard per seed and preset: four budget conditions each, 1,984 shards per group.

**The TSR replication.** The v1 plan promised that wave 2 would replicate the TSR study's comparison on fresh seeds. The `tsr_legacy` group does that, with the registered recipes and package version unchanged. The `tabpfn` group adds the release comparison separately.

**TabPFN releases.** The `tabpfn` group runs releases v2, v2.5 and v3 under one package version, so differences come from the weights and each release's default settings.
- **Not the TSR study's recipe.** Its v2 recipe differs from the `tsr_legacy` recipe, which runs `tabpfn==2.0.9` with the same v2 weights.
- **v2.5 is the paper's version.** The R-Design paper used TabPFN-2.5.
- **License.** The v2.5 and v3 weights carry Prior Labs' non-commercial license, which the project lead accepted.

| Release | Checkpoint | SHA-256 |
| --- | --- | --- |
| v2 | `tabpfn-v2-regressor.ckpt` | `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` |
| v2.5 | `tabpfn-v2.5-regressor-v2.5_default.ckpt` | `6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8` |
| v3 | `tabpfn-v3-regressor-v3_default.ckpt` | `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301` |

## Pairing with the reference

Each wave-2 recipe is paired with the `rct_selected` fit from **the same shard and environment**. The plan required every addendum shard to rerun `rct_t_ridge` and reproduce wave 1 exactly, because pairing with wave-1 records needs identical data. Wave 2 does not pair with wave-1 records, so that requirement becomes a reported diagnostic.

- **Same data.** `rct_t_ridge` uses only trial rows, so its scores cannot show that OBS rows agree. Before launch, a [data check](../scripts/check_data_identity.py) regenerates every condition at the first seed, 70000, in each group's environment. It compares every sample role with the wave-1 environment: OBS, trial training, tuning and test rows, evaluation covariates and effects. A group ran on the wave-1 data only if every array agrees within 1e-12 there **and** all of its `rct_t_ridge` scores reproduce wave 1 within 1e-9, the TSR study's tolerance. Otherwise its results are not described as being on the same data, and no claim pairs them with wave 1. The data check's result is recorded below.
- **Reference drift.** For each group, the analysis reports how many of its `rct_selected` scores equal wave 1's and the largest difference. Its ExtraTrees candidates can differ across library versions, so ratios set side by side across groups have different denominators and are descriptive.
- **Verdicts unaffected.** Neither check blocks wave-2 verdicts, which never cross environments.

## Contrasts

Every contrast is a paired t-test over 64 seeds. Each family has its own Holm correction.

Families are named `group/recipe`, with ` vs reference` added when the reference is not `rct_selected`. There are 24 families: 15 primary, 5 matched and 4 release.

**Primary: each recipe against `rct_selected`.** These answer the v1 question: is the recipe competitive with an analyst who has only the trial?
- Recipes that use OBS have 120 conditions.
- RCT-only recipes (`rct_causal_forest`, `causal_icm_rct_only`, `lmcgp_rct_only`) have 60.

**Matched: adding OBS to a fixed backbone.** These are the only contrasts that isolate the effect of OBS data. Each has 120 conditions; the reference is the backbone fitted to trial data alone.
- `causal_icm_fixed` against `causal_icm_rct_only`;
- `tsr_tabpfn_v2`, `tsr_tabpfn_v25` and `tsr_tabpfn_v3`, each against `lmcgp_rct_only`;
- in `tsr_legacy`, `tsr_tabpfn_lmcgp` against `lmcgp_rct_only`: the TSR study's family A on fresh seeds.

**TabPFN release: does the version matter?** Each has 120 conditions; the reference is release v2.
- `tsr_tabpfn_v25` and `tsr_tabpfn_v3`, each against `tsr_tabpfn_v2`;
- `tabpfn_v25_obs_only` and `tabpfn_v3_obs_only`, each against `tabpfn_v2_obs_only`.

These four families answer whether TSR's or the OBS model's error changes with the TabPFN release. That question arose because the R-Design paper used TabPFN-2.5 without comparing versions.

When a family's reference is RCT-only, its OBS 512 record serves both OBS sizes, as in the plan.

## Resources and execution

**Preflight.** A preflight at seed 69999, outside the grid, ran every group's 31 shards before the freeze. Its jobs deleted all records after writing status files. Those files hold coverage counts, failure stages and types, runtime, peak memory and the GPU used, and no metric.
- **CPU groups.** `native_r` and `gp` ran on the code before the review fixes. Those fixes do not touch their fitting code.
- **GPU groups.** `tabpfn` and `tsr_legacy` ran on the fixed code, which changed TabPFN's cache and provenance. Their first attempts failed for infrastructure reasons:
  - with four shards per GPU, TabPFN fits ran out of GPU memory;
  - on one node, some jobs saw no GPU, and every TabPFN fit in them failed with "No CUDA GPUs are available". The same fit ran normally on another node;
  - preempted jobs were requeued into their old, non-empty output directories and stopped.

  Both groups were rerun with two shards per GPU, that node and the slow-storage nodes excluded, and a driver that checks the GPU before fitting and restarts an interrupted shard in a fresh directory. The values recorded below come from these reruns.

**Environment check.** Every shard checks its environment before fitting:
- external sources and runtimes report as available and verified;
- for TabPFN groups, the required package version is installed and every pinned checkpoint file has its hash;
- for TabPFN groups, CUDA computes a test product correctly and at least 8 GiB of GPU memory is free.

A failed check stops the shard before any record. It is an infrastructure failure, so a bad setup is never recorded as method failures. Each environment's package list is saved in the job record.

**Limits.** The main run's limits follow from the preflight by this rule:
- **CPU groups** get one CPU per shard. The time limit is the larger of 2 hours and 3 times the longest preflight shard, capped at 12 hours. Memory is the larger of 4 GiB and twice the largest preflight peak.
- **The GPU groups** (`tabpfn`, `tsr_legacy`) pack several shards onto one GPU. The number per GPU is the largest tried that produced no CUDA out-of-memory error in the preflight, at most four. The job's time limit is the larger of 2 hours and 3 times the longest preflight GPU job, capped at 12 hours. Each GPU job gets 4 CPUs and host memory of at least twice its shards' summed preflight peaks.

The values that result are recorded below before launch.

No fit gets a deadline beyond the recipes' own deadlines (Integrative R 1,200 seconds, forest 600 seconds). A fit that reaches its own deadline is a method failure.

Four kinds of event are infrastructure failures:
- **Job limit.** A shard that reaches the job's time or memory limit.
- **Interruption.** A shard stopped by preemption or a node fault before writing its status. A requeued job restarts it automatically.
- **Failed check.** A shard that fails the environment check, which stops it before any record.
- **GPU memory.** A shard with any CUDA out-of-memory record. Whether a fit runs out of GPU memory depends on how many processes share the GPU, not on the method.

Before any metric is read, such a shard is rerun whole in a fresh output directory, with the old output kept. A job-limit rerun gets double the limit that was reached, beyond the 12-hour cap if needed; a GPU-memory rerun runs alone on its GPU. A CUDA out-of-memory error that recurs with the shard alone on a GPU is a method failure.

**Final attempt.** A shard is rerun for a job limit only once. If the rerun also reaches a limit, its recorded cells stand, and every cell it did not record is a method failure with error type `JobLimit`. Such shards are listed, with their configuration hashes, in a file that the analyzer and the checker both read. The affected conditions are untested for the affected recipes, by the failure rule; if `rct_selected` is among them, those conditions are untested for every family of the group.

**Expected failures.** In the native panel, quadratic Integrative R reached its deadline on the CorNet design at RCT 512 at all 64 seeds. The same is expected here. Those conditions are untested for that recipe, under the plan's failure rule.

## Checks

1. **Coverage.** Every declared shard of every group, and every requested cell, has a terminal record.
   - A group's shards share one source and environment, and agree on external-method provenance.
   - Every successful TabPFN fit ran its stage 1 on CUDA with the pinned checkpoint.
   - The analyzer refuses anything else.
2. **Identities.**
   - The data check before launch, above.
   - Wave-1 reproduction of `rct_selected` and `rct_t_ridge`, per group.
   - The OBS-size identity of RCT-only records.
3. **Reproducibility.** On the first seed at OBS 512 · RCT 128, each recipe is refitted twice in its group's environment, and each refit's score is compared with the record. Each refit fits its own TabPFN stage.
   - CPU recipes must agree within 1e-12.
   - TabPFN recipes run on an RTX 3090, whose kernels need not be bit-reproducible, so they must agree within a relative 1e-6. A failure is reported and investigated; no recorded score is replaced.
4. **Independent arithmetic.** A separate script recomputes every contrast, Holm adjustment and headline count, including the headline table, from the per-seed records and the declared family sizes, without importing the analyzer.

## What wave 2 will not show

- **Inspected designs.** All 31 presets were inspected during development. Fresh seeds replicate sampling within these designs, not new mechanisms or real data.
- **Not OBS contribution in the primary families.** Beating `rct_selected` is competitiveness. Only the matched families isolate the effect of OBS data.
- **Release effect within one package version.** The release families compare weights and default settings under `tabpfn==10.0.0`. They are not a comparison with older package versions.
- **No human check.** One AI agent designed, implemented and will analyze wave 2.

## Recorded before launch

Preflight results (seed 69999, 31 shards per group). Each group's limit follows from the rule above: per shard for the CPU groups, per GPU job for the GPU groups.

| Group | Longest shard | Median shard | Peak memory | Preflight failures | Main-run limit |
| --- | ---: | ---: | ---: | --- | --- |
| `native_r` | 4,077 s (CorNet design) | 549 s | 579 MiB | `integrative_r_quadratic`, 3 deadline errors in the CorNet design | 1 CPU, 4 GiB, 3 h 24 min |
| `gp` | 102 s | 72 s | 740 MiB | none | 1 CPU, 4 GiB, 2 h |
| `tabpfn` | 339 s (CorNet design) | 84 s | 2,508 MiB | none | 2 shards per GPU, 4 CPUs, 16 GiB, 2 h (longest preflight GPU job 346 s) |
| `tsr_legacy` | 181 s (CorNet design) | 47 s | 1,661 MiB | none | 2 shards per GPU, 4 CPUs, 16 GiB, 2 h (longest preflight GPU job 186 s) |

**Execution.** The shard driver and job scripts are published with the results; their hashes are fixed here.

| Item | Value |
| --- | --- |
| Code | The freeze commit, run from a `git archive` snapshot |
| Shard driver `run_shard.py` | SHA-256 `1a299a69c5132804ad619df9300861df6fa44559b47a836bf27023e536c0a23b` |
| Job scripts | `common.sh` `e19ad30219e8b26e474aa5bc7e8ab5143c1ea9b2c57900a51698fce88796dbf0`; `cpu_array.sbatch` `05cd6fda42accfc470c841e8bfb52115c30eb4a9769fa61a2d9b311a6f24182f`; `gpu_array.sbatch` `d6cb20f028180a0103548f21cd824be65643d96df3f0a0596f28d0e380a96a16` |
| GPU jobs | Two shards per GPU: four caused out-of-memory errors, and three was not tried. One RTX 3090, 4 CPUs and 16 GiB of host memory per job. Nodes b03, b14, b15, b24–b28, b30 and b31 excluded |
| Preflight runs | CPU groups: jobs 820474 (`native_r`) and 820475 (`gp`), on the code before the review fixes, drivers `432e8cf4…` and `fe6183f4…`. GPU groups: jobs 822028 (`tabpfn`) and 822029 (`tsr_legacy`), on the fixed code, driver `8bf51696…`, two shards per GPU, 16 GiB. The main-run driver differs from `8bf51696…` only in exiting cleanly when a requeued job finds a shard already finished |
| Data check (seed 70000, against the wave-1 environment) | Passed. In all 124 conditions, every array of every sample role is identical in the `gp`, `tabpfn` and `tsr_legacy` environments; the largest difference is 0. `native_r` runs in the wave-1 interpreter itself |

## Revisions and errata

**2026-11-15, before the freeze.** A fresh-context AI review of the [stress addendum](V1_STRESS_WAVE2.md) found gaps that apply here too:
- the same-data rule now also requires a data check of every sample role, since `rct_t_ridge` never uses OBS rows;
- a shard that reaches a job limit gets one rerun, and its unrecorded cells then become `JobLimit` failures;
- a memory limit counts as a job limit;
- the driver, job scripts and GPU settings are fixed in the launch record;
- the checker also verifies the headline table.

**2026-11-14, before the freeze.** A fresh-context AI review led to these changes:
- the `tsr_legacy` group restores the promised TSR replication;
- families carry group names;
- the wave-1 data identity uses a 1e-9 gate;
- TabPFN provenance records the computed checkpoint hash, package and release, and the analyzer checks all three;
- the shard driver checks the environment before fitting;
- the TabPFN cache holds three releases;
- the checker compares headline counts.
