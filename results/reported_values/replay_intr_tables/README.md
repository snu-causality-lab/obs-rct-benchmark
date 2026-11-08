# Integrative R Tables 1 and 2 replay: execution record

**Status: complete.** Run on 2026-11-08. The fixed question, arms and readouts are in [PROTOCOL.md](PROTOCOL.md). The [results](#results) and [interpretation](#interpretation) are below.

## Where it runs

All paths are on GSDS under `/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01`, with permissions `0700`.

| Path | Content |
| --- | --- |
| `code/obs-rct-benchmark-173af2a/` | `git archive` of the public benchmark at `173af2a`: `src/` and `pyproject.toml` only |
| `drivers/`, `jobs/` | Copies of this folder's scripts and Slurm files |
| `inputs/` | `PROTOCOL.md`, `reported_values.csv` and the native panel's per-seed records for this DGP, as staged |
| `SHA256SUMS` | Hashes of the 30 staged files, verified on the server after transfer |
| `preflight/` | Execution checks and the README example run for R0 |
| `results/author/CELL/seeds-AA-BB/`, `results/benchmark/CELL/seeds-AA-BB/`, `results/analysis/` | Arm A, Arm B and aggregated outputs |
| `logs/` | One log per job or array task |

Two existing assets in `obsrct-v03-adapters-20261026-r01` are reused read-only.
- `environment-v2` supplies both runtimes. R is 4.3.3 with glmnet 4.1-10, mvtnorm 1.3-3, stringr 1.5.2, caret 6.0-94 and ncvreg 3.15.0. Python is 3.11 with numpy 1.26.4. That environment also has an older installed copy of the benchmark, so every Python driver checks that it imported the staged snapshot instead.
- `upstreams/intRlearner` is the author checkout at `a46ce2c`, with no tracked or untracked changes.

The run folder was first created by the upload with permissions `0755`, because the step that creates it with `0700` did not run. Its parent work root is `0700`. The folder was set to `0700` before any job was submitted.

## Source identities

| File | SHA-256 |
| --- | --- |
| intRlearner `R/intRlearner.R` | `98563e2423f41153fcbb2548f3b54ac779c72bcb8d154608bf51f778086fdce8` |
| intRlearner `R/utils.R` | `2b85e2e56db83e33427f6f43aa1b59d995bafb6d46bfc1aec8e5f2e6a215525b` |
| Benchmark `src/obs_rct_benchmark/integrative_r.py` at `173af2a` | `593a883eaf88002a5ce3746deb61dc70fff7b8985a62be6f7aa5bc551dacbc5f` |
| Benchmark `src/obs_rct_benchmark/paper_dgp.py` at `173af2a` | `77bb899367e6ce8d12ee127b4ff6deb833215e90a707bb0f35c9aacf55f6a3f2` |
| `drivers/author_replay.R` | `778fc556865edc40aa4dd76d319ca06f41f747ec51498570fa1c91ea01fd7d24` |
| `drivers/benchmark_grid.py` | `05ab87bc63dac641504735b9fbe9126f4aa7f0b991f5a24abcc31cff38b449f4` |
| `drivers/preflight.py` | `e91a85bb46ab1a3222c796aac002bfef48ff3e65827cb17a2abf6d7b779aea58` |
| `drivers/aggregate.py` | `919ddd69a33f3cdaec9e4cf29867011cc805dbe403f6a88246ab73ce113dfee0` |
| `PROTOCOL.md` as staged | `9378f4c64779e76dae469e91b98ee8027d814a69d9861eeb301b3c11b173eb97` |

## Jobs

| Job | Slurm ID | Request | Runs after |
| --- | --- | --- | --- |
| Preflight | 770635 | 1 CPU, 4 GiB, 45 min | none |
| Arm A, released code, 7 cells × 4 seed blocks | 770636, array 0-27, at most 12 at once | 1 CPU, 4 GiB, 4 h per task | preflight succeeds |
| Arm B, benchmark, 9 cells × 4 seed blocks | 770637, array 0-35, at most 12 at once | 1 CPU, 4 GiB, 4 h per task | preflight succeeds |
| Aggregation | 770638 | 1 CPU, 4 GiB, 15 min | both arrays end, whatever their outcome |

Aggregation waits for both arrays to end rather than to succeed, so that it can count missing and failed replicates. All jobs run on partition `P2` without a GPU.

## Preflight

Preflight 770635 completed with exit code 0 in 2 min 59 s on b22. Every check passed. The staged benchmark snapshot was the imported copy, the intRlearner source and R runtime were verified, and mvtnorm 1.3.3 was present. One Arm A replicate and one Arm B fit of each simulation succeeded.

**R0, environment check.** The README example, run once at seed 0 with J = 8 and b = 2.5, reproduces the README's printed RMSEs to all seven printed decimals.

| Learner | README | This environment |
| --- | ---: | ---: |
| int rlearner | 0.8664778 | 0.86647782 |
| rct rlearner | 1.5589467 | 1.55894674 |
| os rlearner | 1.4570438 | 1.45704377 |
| naive rlearner | 1.3032480 | 1.30324800 |

Single runs took 7.6 s for an Arm A Simulation I replicate at n = 1000 and 66 s for the Simulation II README replicate. Arm B fits took 10.3 s and 73.2 s for the same two simulations.

## Results

All 66 allocations completed with exit code 0: the preflight, 28 Arm A tasks, 36 Arm B tasks and the aggregation. Arm A tasks took 3 min 6 s to 28 min 20 s, and Arm B tasks took 2 min 30 s to 31 min 22 s. Batch-step MaxRSS was at most 864 MiB. All 700 Arm A replicates and all 900 Arm B fits succeeded. Every Arm A task's source hashes match the pinned files, and every Arm B task verified its source and imported the staged snapshot. The key readouts were recomputed from the raw outputs without the aggregation script, and they agree.

R emitted one warning per Table 2 cell, all at seed 79 and all the same. It is a glmnet notice that the path's smallest penalty did not converge; glmnet returned the larger-penalty solutions, and the fits completed.

Compact outputs are in [`results/`](results/). They are the aggregated tables, `readouts.json`, the preflight record, and per-task records with every replicate's RMSE. The job IDs are in `results/SUBMITTED.txt`. All values are mean RMSE over 100 seeds, which is mean root PEHE; standard errors are in parentheses.

**R1, reproduction by the released code.** 20 of the 24 published R-learner values agree with the replay, including all seven integrative R-learner values. A dagger marks the four that do not.

Table 1, Simulation I, shown as paper / replay:

| n | int rlearner | rct rlearner | os rlearner | naive rlearner |
| ---: | --- | --- | --- | --- |
| 200 | 0.23 / 0.224 | 0.26 / 0.246 | | 0.56 / 0.556 |
| 500 | 0.11 / 0.123 | 0.15 / 0.148 | 0.58 / 0.579 | 0.53 / 0.533 |
| 1000 | 0.09 / 0.094 | 0.11 / 0.113 | | 0.50 / 0.490 † |

The paper prints the OBS-only value once, in the n = 500 row. The replay's OBS-only values at n = 200 and 1000 are 0.575 and 0.573.

Table 2, Simulation II, shown as paper / replay:

| Cell | int rlearner | rct rlearner | os rlearner | naive rlearner |
| --- | --- | --- | --- | --- |
| J = 1, b = 0 | 0.28 / 0.291 | 0.88 / 0.893 | 0.49 / 0.500 | 0.41 / 0.407 |
| J = 1, b = 2.5 | 0.48 / 0.430 | as b = 0 | 1.33 / 1.323 | 1.01 / 0.999 |
| J = 9, b = 0 | 0.55 / 0.612 | 1.64 / 1.759 † | 0.97 / 1.065 † | 0.77 / 0.819 † |
| J = 9, b = 2.5 | 1.12 / 1.134 | as b = 0 | 1.54 / 1.548 | 1.31 / 1.318 |

The paper prints the RCT-only value once per J. In the replay it is identical for both b at every seed: b changes only the OBS treatment probabilities, not the RCT draw or the random numbers that the RCT-only fit uses.

**R2, translation and adapter.** For the integrative R-learner, the benchmark agrees with the released code in all seven cells.

| Cell | Paper | Released code | Benchmark | Benchmark minus released code (tolerance) |
| --- | ---: | --- | --- | --- |
| Table 1, n = 200 | 0.23 | 0.224 (0.015) | 0.234 (0.013) | +0.010 (0.039) |
| Table 1, n = 500 | 0.11 | 0.123 (0.007) | 0.134 (0.008) | +0.010 (0.021) |
| Table 1, n = 1000 | 0.09 | 0.094 (0.006) | 0.088 (0.005) | -0.006 (0.016) |
| Table 2, J = 1, b = 0 | 0.28 | 0.291 (0.025) | 0.292 (0.026) | +0.001 (0.073) |
| Table 2, J = 1, b = 2.5 | 0.48 | 0.430 (0.039) | 0.524 (0.040) | +0.094 (0.113) |
| Table 2, J = 9, b = 0 | 0.55 | 0.612 (0.028) | 0.612 (0.031) | 0.000 (0.083) |
| Table 2, J = 9, b = 2.5 | 1.12 | 1.134 (0.040) | 1.110 (0.041) | -0.024 (0.115) |

**Bridge to the native panel.**

| Cell | Native panel, 64 seeds | Replay, Arm B, 100 seeds |
| --- | --- | --- |
| OBS 512, RCT 128 | 0.306 (0.023) | 0.323 (0.018) |
| OBS 512, RCT 512 | 0.138 (0.010) | 0.146 (0.008) |

**RCT size.** Over the 100 seeds, Arm A's Simulation II RCT has a mean of 242.8 units, SD 14.0 and range 203 to 276. This matches the expected 243.7 with SD 15.6, not the text's "around 200". The size is the same in all four cells at a given seed, because selection depends only on the covariates.

## Interpretation

- **The released code reproduces Table 1 and most of Table 2.** The one Table 1 flag is marginal: the naive R-learner at n = 1000 gives 0.490 against a printed 0.50, a gap of 0.010 against a tolerance of 0.009.
- **The other three flags are all in the J = 9 rows, and the replay is higher in each.** The RCT-only learner, which does not depend on b, gives 1.76 against 1.64. At b = 0 the OBS-only and pooled learners are also higher, and so is the integrative learner, though within its tolerance. At b = 2.5 the OBS-based learners agree. One explanation would be that the paper's J = 9 rows used the README's p0 = 1, which gives J = 8. Fewer effect modifiers would lower the estimation error that dominates at b = 0 more than the confounding bias that dominates at b = 2.5. This replay did not test that explanation.
- **The benchmark's DGP translation and adapter are faithful at the paper's settings.** They agree with the released code in every cell. The benchmark stratifies its folds by source and arm in both tables. In Table 2 it also fixes the RCT size and evaluates on a fresh sample instead of the finite population.
- **The native-panel values differ from the paper because of sample size.** At the native-panel sizes the replay reproduces the native panel. At the paper's sizes the benchmark gives 0.234, 0.134 and 0.088, against the paper's 0.23, 0.11 and 0.09. The earlier gap between 0.306 and 0.23 therefore comes from fewer OBS and RCT rows, not from the method or the DGP translation.
