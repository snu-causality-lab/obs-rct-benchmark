# Addendum: Causal-ICM with the paper's borrowing selection

**Status: both blocks fixed before their main runs; the core block's run has finished.** This addendum to the [v1 core plan](V1_CORE_STUDY.md) and [stress study 1](V1_STRESS_CORNET.md) fixes the recipe, contrasts and resources for `causal_icm_selected` before its main run. One development check already fitted the recipe at seed 70000 in four core conditions; it is disclosed [below](#what-this-addendum-will-not-show). Each block is fixed by the commit that fills in its part of [Recorded before launch](#recorded-before-launch). The stress block's freeze may come later. It may change the stress block's record, sharding and limits, and this status line, but no fitting code and nothing the core block uses. Each block's job record holds its freeze commit's hash before that block's first main-run fit, and each block is analyzed with the code of its own freeze.

[Wave 2](V1_CORE_WAVE2.md) and the [stress addendum](V1_STRESS_WAVE2.md) ran Causal-ICM with ρ fixed at 0.5. The paper chooses ρ from data (Dimitriou et al., CLeaR 2026, Section 3.4.2). Version 0.3.0a8 adds that procedure as `causal_icm_selected` ([V03_ADAPTERS](V03_ADAPTERS.md)). This addendum runs it wherever `causal_icm_fixed` ran.

It keeps from those plans:
- **Conditions and seeds.** The v1 core block has 124 conditions on seeds 70000–70063. The stress block has 64 conditions on seeds 80000–80063 with the generator `stress_dgp.py` 2.0.0.
- **Reference.** `rct_selected`.
- **Decision rule.**
  - Holm's correction within each family at 0.05.
  - The failure rule: an incomplete condition gets p = 1 and is untested.
  - In the core block, M11 is descriptive, transport conditions are counted separately, and an RCT-only reference's OBS 512 record serves both OBS sizes.
  - In the stress block, extrapolation conditions are counted separately.

## Recipe and environment

`causal_icm_selected` runs the authors' GP from the fixed recipe and chooses ρ before the final fit:
- **Grid.** ρ ∈ {0, 0.1, …, 1}.
- **Folds.** Five folds over the pooled RCT and OBS rows, shuffled with the fit's method seed.
- **Score.** Each fold's held-out RCT rows are scored by squared error, weighted by 1 / P(trial | x). That probability comes from a default logistic regression on all observed covariates.
- **Choice.** The lowest mean score over folds wins, one ρ for both arms. Ties go to the smaller ρ.
- **Final fit.** The fixed recipe refits all rows at the chosen ρ with the same seed.

The runner gives `causal_icm_selected` the method seed of `causal_icm_fixed`, so a fit that selects 0.5 equals the fixed fit exactly. The selection's departures from the paper (fold-mean criterion, all covariates in the weights) are listed in [V03_ADAPTERS](V03_ADAPTERS.md). The procedure takes no options.

Every shard fits five recipes in wave 2's `gp` environment (Python 3.11, GPy 1.13.2, scikit-learn 1.5.2, numpy 1.26.4, the authors' code at `0f0c6ca`):
- `rct_selected` and `rct_t_ridge`;
- `causal_icm_fixed` and `causal_icm_rct_only`, both at ρ = 0.5, as in wave 2;
- `causal_icm_selected`.

The configurations are [`v1_icm_selected_core.toml`](../configs/v1_icm_selected_core.toml) and [`v1_icm_selected_stress.toml`](../configs/v1_icm_selected_stress.toml). The [shard writer](../scripts/make_icm_selected_configs.py) validates them and writes:
- **Core.** One shard per seed and preset, holding four budget conditions: 1,984 shards.
- **Stress.** One shard per seed, dial and trial size, holding four levels: 1,024 shards. The stress addendum held both trial sizes in one shard. Splitting them halves the longest shard, where 8,192 OBS rows make each selection slow.

## Pairing and identities

Every contrast pairs two recipes fitted in the same shard.

- **Same data and fits as wave 2.** Two checks, both required:
  - **Data check, before launch.** The [data check](../scripts/check_data_identity.py) regenerates every condition at each block's first seed (70000, 80000) with this addendum's code in the `gp` environment. It compares every sample role with wave 2's `gp` dump: OBS, trial training, tuning and test rows, evaluation covariates and effects. Every array must agree within 1e-12.
  - **Score reproduction, after the fits.** The four recipes refitted from wave 2 run with the same seeds and method seeds. Their scores must reproduce wave 2's `gp` scores within 1e-9, with the same success or failure in every cell. Equal scores do not prove equal predictions, so this is a reproduction diagnostic on top of the data check.

  Only if both pass are wave 2's `gp` verdicts and this addendum's described as being on the same data. Either way, verdicts here never cross shards.
- **Selection at 0.5.** Every successful selected fit that chose 0.5 must equal `causal_icm_fixed` exactly. A difference is a defect, reported and investigated; no recorded score is replaced.
- **RCT-only recipes.**
  - Core block: their OBS 512 and OBS 2048 records must be equal.
  - Stress block: in the five dials that change only OBS, their records must be equal at every level.

  Mismatches are counted.

## Contrasts

Every contrast is a paired t-test over 64 seeds. Each block has three families, each with its own Holm correction. A family has 120 conditions in the core block and 64 in the stress block.

| Family | Kind | Question |
| --- | --- | --- |
| `gp/causal_icm_selected` | primary | Is the recipe competitive with an analyst who has only the trial (`rct_selected`)? |
| `gp/causal_icm_selected vs causal_icm_rct_only` | matched | Does adding OBS to this backbone help, with ρ chosen from data? |
| `gp/causal_icm_selected vs causal_icm_fixed` | selection | Does the paper's selection beat ρ = 0.5? Same data and method seed; only the choice of ρ differs. |

**Stress readout.** For the primary family, each dial and trial size gets the stress plan's four states, read from the verdicts at levels 0 and 3:
- no level-0 win;
- kept;
- stopped;
- broke.

"Untested" counts as not better, and the readout lists every untested level among the four.

**Descriptive.**
- **Selected ρ.** For each condition, how often each grid value was selected and the mean selected ρ, over successful fits. Diagnostics count every attempt, failed ones included: successful fits with fewer than five usable folds, failures during selection or the final fit, and unconverged participation models among the attempts that recorded one.
- **Condition means.** The mean error of every recipe in every condition.
- **Dose-response (stress block).** Every recipe's error relative to `rct_selected` by dial and level, as in the stress addendum. Where a pair has zero error, the log ratio is undefined and only the ratio of means is reported.

## Execution

- **Before results are read.** No metric is read until every shard of a block has finished and coverage is authenticated. Monitoring reads only status, runtime, memory and failure types.
- **Rules.** The execution rules of wave 2, as frozen at `80c9129d0d6046a9e6845c9f0a3580c67ca024d9`, apply with the same shard driver and job scripts, including its [2026-11-15 erratum](V1_CORE_WAVE2.md#revisions-and-errata) on slow-storage nodes:
  - infrastructure failures: a job limit (time or memory), an interruption, a failed environment check, or a final attempt on a slow-storage node that recorded a method failure;
  - such a shard is rerun whole in a fresh directory, and its old output is kept.

  This addendum excludes the slow-storage nodes (b03, b14, b15, b24–b28, b30, b31) from every job.
- **Limits.** The preflight sets one limit per block:
  - time: the larger of 2 hours and 3 times the longest preflight shard, capped at 12 hours;
  - memory: the larger of 4 GiB and twice the largest preflight peak.
- **Job-limit reruns.** The partition allows at most 12 hours. A shard that reaches its time limit is rerun once with the smaller of twice that limit and 12 hours. A shard that reaches its memory limit is rerun once with twice that memory. If the rerun also reaches a limit, its recorded cells stand and its unrecorded cells become `JobLimit` method failures, listed for the analyzer and checker. A listed shard may hold every record, since a job can stop after its last record.
- **Deadlines.** No fit has a deadline; the final-attempt rule bounds them.
- **Preflight.** Before a block's freeze, one seed outside its grid runs every shard of the block: seed 69999 for the core block (31 shards, 8 GiB and 6 hours each) and seed 79999 for the stress block (16 shards, 16 GiB and 12 hours each). The jobs delete all records after writing status files. If any preflight shard stops before writing its status, its time is unknown and the rule above cannot be applied. The block is then not frozen until its sharding or limits are revised and its preflight is rerun.
- **Expected cost.** In a check at one seed, a selected fit took 20 to 27 times as long as a fixed fit: about 19 seconds at OBS 512 · RCT 128 and 350 seconds at OBS 2048 · RCT 512. The preflight times the 8,192-row stress conditions.

## Checks

1. **Coverage.** Every declared shard and every requested cell has one terminal record with its configuration hash, record ID and method seed. All shards of a block share one source, environment and external-method provenance. Every successful record has a finite, nonnegative error and its square root. Every successful selected fit records a completed selection with a ρ on the grid and its own fold seed. The analyzer refuses anything else.
2. **Identities.** The wave-2 reproduction, the selection at 0.5, and the RCT-only identities above.
3. **Reproducibility.** On each block's first seed, every recipe is refitted twice in the `gp` environment and compared with the record within 1e-12:
   - core block: OBS 512 · RCT 128 in all 31 presets;
   - stress block: RCT 128 at levels 0 and 3 of every dial.

   A cell without a record, such as an unrecorded `JobLimit` cell, cannot be refitted against it; the check then fails as incomplete. A failure is reported and investigated; no recorded score is replaced.
4. **Independent arithmetic.** A [separate script](../scripts/verify_icm_selected.py) imports no analysis script. It rebuilds each block's shards from the frozen configuration and authenticates the runs itself: shard hashes, record IDs, method seeds, metrics, selections and shared provenance. It then recomputes every field of every file the [analyzer](../scripts/analyze_icm_selected.py) writes and compares them:
   - per-seed scores, condition means, selection diagnostics and dose-response;
   - every contrast with its pairing, interval, p-value, Holm adjustment and verdict;
   - the readout, the identities, the counts, and the whole headline text.

## What this addendum will not show

- **Selection per dataset, not per setting.** The authors tuned ρ once per simulation setting and reused it across replications. Here every fit selects its own ρ from its own data, as an analyst with one dataset would. This costs more and makes the result depend on the selection's own noise.
- **Against ρ = 0.5, not the best fixed ρ.** The selection family compares the paper's procedure with the default that wave 2 ran. It does not compare it with the best ρ in hindsight.
- **Not a fresh replication.** The addendum reuses the seeds of wave 2 and the stress addendum so that it can pair with their records. Their fixed-ρ results, including Causal-ICM's, were read before this plan was written. A check of the new recipe on the cluster also read its errors and the fixed recipe's at seed 70000 in four core conditions: M0 and M2 at OBS 512 · RCT 128 and OBS 2048 · RCT 512. The selection follows the paper and the authors' notebooks; its code has not changed since that check. Still, verdicts on reused, already-read seeds carry less weight than a confirmation on untouched seeds would.
- **Inspected designs.** All presets and dials were inspected during development. The seeds replicate sampling within these designs, not new mechanisms or real data.
- **AI verification only.** One AI agent designed, implemented and will analyze this addendum. A second AI model reviewed the selection code. No person has reproduced it.

## Recorded before launch

### Core block

**Preflight history.** The first preflight (draft `1e864f1`, job 834000) stopped one core shard, `paper:cvci_official_linear_bias`, when it ran out of its 8 GiB. The block was therefore not frozen.
- **Cause.** Memory probes found that the GPy models of each selection fit stay in reference cycles. Over one selection's 56 native fits, resident memory grew to 4.2 GB at 2,560 rows. With a collection after every fit it stayed near 1.2 GB, at the same speed.
- **Change.** Draft `effea13` collects after every selection fit. A rerun of the development check at seed 70000 gave exactly the same selected ρ, fold scores, participation weights and errors in all four cases, so the change leaves results unchanged.
- **Second preflight.** Job 834235 ran all 31 core shards on `effea13`. The values below come from it. The freeze commit differs from `effea13` only in this document.

| Block | Longest shard | Median shard | Peak memory | Preflight failures | Main-run limit |
| --- | ---: | ---: | ---: | --- | --- |
| core | 3,997 s (CorNet design) | 3,164 s | 984 MiB | none | 1 CPU, 4 GiB, 3 h 20 min |

All 31 preflight shards shared one node with other jobs, so these times are slower than on a quiet node.

**Execution.**

| Item | Value |
| --- | --- |
| Wave-2 freeze commit (execution rules) | `80c9129d0d6046a9e6845c9f0a3580c67ca024d9` |
| Code | The core freeze commit, run from a `git archive` snapshot |
| Shard driver and CPU job script | Those of the wave-2 main run, unchanged: `run_shard.py` `1a299a69c5132804ad619df9300861df6fa44559b47a836bf27023e536c0a23b`; `common.sh` `e19ad30219e8b26e474aa5bc7e8ab5143c1ea9b2c57900a51698fce88796dbf0`; `cpu_array.sbatch` `05cd6fda42accfc470c841e8bfb52115c30eb4a9769fa61a2d9b311a6f24182f`, submitted with this addendum's run folder |
| This addendum's job scripts | `write_shards.sbatch` `4ab035d1c2c09ce52004366e30e1e12c9439e804a5d9eeb8bb0ed4f64c00b659`; `data_check.sbatch` `358d0026892d6578d72385ceddc1797f6001f859b76eb4a0f0c98fca5052e07e`; `analyze.sbatch` `207f13930706eeb3a188cfcd72a5e331eba911ae47563edb9208bd6f230ac512`; `reproduce.sbatch` `c56640a0b0e811aba5f65afab50b5fce49901980f4a38f17a063a8e699a0cd16` |
| Nodes | b03, b14, b15, b24–b28, b30 and b31 excluded from every job |
| Data check (seed 70000, against wave 2's `gp` dump) | Passed (job 834151). Every array of every sample role in all 124 conditions is identical; the two files are byte-identical |

### Stress block

**Preflight.** Job 834236 ran all 16 stress shards on `effea13` at seed 79999, 14 on a node that was also running core main-run shards. None failed or stopped. The two `obs_size` shards took 4.5 and 4.9 hours. Almost all of that was one selection fit at level 0, with 8,192 OBS rows: about 4.2 and 4.5 hours. Levels 1–3 together took minutes.

| Block | Longest shard | Median shard | Peak memory | Preflight failures | Main-run limit |
| --- | ---: | ---: | ---: | --- | --- |
| stress | 17,519 s (`obs_size`, RCT 512) | 3,844 s | 5,053 MiB | none | 1 CPU, 10 GiB, 12 h |

Three times the longest shard is 14.6 hours, so the 12-hour cap sets the time limit; a job-limit rerun keeps 12 hours. Twice the peak is 9.9 GiB, rounded up to 10 GiB.

**Load.** In the core main run, completed shards on fully loaded nodes took up to four times the median time of the same preset. 14 of 1,984 reached their 3 h 20 min limit, and all finished within an hour on rerun. An `obs_size` shard could reach 12 hours twice. Its unrecorded cells would then become `JobLimit` failures, as the rule above states. To lower that risk, the 128 `obs_size` shards are submitted as their own array before the other 896.

**Execution.**

| Item | Value |
| --- | --- |
| Code | The stress freeze commit, run from a `git archive` snapshot. It differs from the core freeze only in this document, so both blocks fit and analyze with the same code |
| Shard driver, CPU job script, rules and nodes | As for the core block |
| This addendum's job scripts | `write_shards.sbatch` `4ab035d1c2c09ce52004366e30e1e12c9439e804a5d9eeb8bb0ed4f64c00b659`; `analyze.sbatch` `acc4903d23aa2196ef170c987474408e30a37ed08c3d2f9871a9f27cf0bf6c2b` and `reproduce.sbatch` `0022c70fe82055dfca325609931824a4ecf7fd99bca3e25b45bc791ba0f5fad3` (see [Revisions and errata](#revisions-and-errata)); `joblimit_reruns.py` `9683cc73cff2a08be272ce72bdb3dfeed85947d4fc3b78588b45cc293525e4fd`, which lists job-limit reruns from scheduler states and status files and submits them |
| Arrays | Two, each mapping its tasks to shards through an index file: the 128 `obs_size` shards, then the other 896 |
| Data check (seed 80000, against wave 2's `gp` dump) | Passed (job 834151). Every array of every sample role in all 64 conditions is identical; the two files are byte-identical |

## Revisions and errata

- **Analysis job scripts (2026-11-16, during the core main run, before any analysis).** The recorded `analyze.sbatch` passed `--block` to the checker's arithmetic step, which the frozen checker does not accept; it would have stopped there with a usage error. It now passes `--config`, as the checker requires. `reproduce.sbatch` now also passes a block's job-limit list when one exists, as `analyze.sbatch` already did (`0022c70fe82055dfca325609931824a4ecf7fd99bca3e25b45bc791ba0f5fad3`). The recorded versions are kept beside them. No analysis or checking code changed.
- **Wave-2 path in `analyze.sbatch` (2026-11-16, first core analysis).** The script stored the wave-2 packet path in a shell variable that the shared environment script it sources then overwrote, so the first core analysis (job 838098) read no wave-2 scores. Its identity check caught this: the wave-2 reproduction compared no cell and failed. The checker read the same path and agreed. That analysis is kept but void. The variable was renamed, and the script now stops if the packet has no `gp` score files (`analyze.sbatch` `acc4903d23aa2196ef170c987474408e30a37ed08c3d2f9871a9f27cf0bf6c2b`). The core analysis was rerun with the same code.
- **Stress reproducibility check in parts (2026-11-16, before its main run finished).** The check refits every recipe twice, one condition after another, in one job. In the stress block the two selection refits at 8,192 OBS rows alone take most of the 12-hour limit, so the 16 conditions cannot finish in one job. A [wrapper](../results/v1_icm_selected/evidence/jobs/reproduce_part.py) (`25766a99a95de50fa571e795f283841a4d0a2e5402185d3ffa685ac93ff8cdad`, with its [job script](../results/v1_icm_selected/evidence/jobs/reproduce_part.sbatch) `23f742f971bec6967ac5b4bba771bb544ecda54c233656e0c4df9989a60cb025`) runs the same frozen function on one condition per job, authenticating records against the whole block, and merges the parts. On a synthetic stress block with a job-limit cell and mismatches, the merged output equals one run exactly. On the core block's first condition, its output equals the single core run's output for that condition.
