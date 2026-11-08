# Integrative R Tables 1 and 2 replay protocol

Fixed on 2026-11-08 before any replay output was produced. Preflight outputs check execution only and do not change this protocol.

## Question

On the benchmark's translation of Simulation I, the Integrative R recipe has mean root PEHE 0.306 in the native panel at OBS 512 and RCT 128. Table 1 of Wu and Yang (2022) reports 0.23 at OBS 2000 and RCT 200. The two settings differ, so these numbers are not yet a comparison. This replay asks three questions:

1. Does the released code reproduce the published R-learner columns of Tables 1 and 2?
2. At the paper's settings, does the benchmark's DGP translation and adapter agree with the released code?
3. Does the replay reproduce the native-panel values at the native-panel sizes?

## Arms

**A. Released code.** Use `lwu9/intRlearner` at `a46ce2c1cc814865898b863d6a8e130bbc2c802b`. For each cell and seed s = 0, ..., 99, call `set.seed(s)`, then `gen_data` with the README's arguments, then the README's four fits in the README's order: the RCT-only, OBS-only and pooled R-learners (`rlasso`), then the integrative R-learner (`intrlearner`). Each fit is scored on `newx_rlearner`. The deviations from the README are these:
- The two source files are sourced into R, not installed as a package. The package has no compiled code.
- In Simulation I, `gen_data` returns tau = 0, so the truth is computed as 1 + 2x + 0.75x^2 from the test covariate.
- For J = 9, Simulation II uses p0 = 0. The README offers only p0 = 1 or p - 2, which give J = 8 and J = 1.
- Warnings are counted and suppressed in the log.
- The paper's seeds and its Table driver are not released, so seeds are 0 to 99.

The cells are these:
- Table 1, Simulation I: `KPS_data=TRUE`, m = 2000, N = 1e5, cf = 0.75, `multidim=FALSE`, p = 2, with n = 200, 500 and 1000.
- Table 2, Simulation II: `KPS_data=FALSE`, m = 500, N = 1e5, n0 = 7, p = 10, with b = 0 or 2.5 and p0 = 8 (J = 1) or 0 (J = 9).

Only the four R-learner columns can be produced by released code. The paper's oa, rf and KPS columns are out of scope.

**B. Benchmark.** Use the public package at `173af2a` with the same intRlearner checkout and R environment.
- Build `make_paper_config(paper_id=..., n_obs=..., n_rct_train=..., n_rct_tune=4, n_rct_test=4, n_eval=100000, ...)` and call `generate_paper_problem(cfg, seed=s)` for s = 0, ..., 99. Four is the generator's minimum role size, and the adapter uses neither role.
- Fit `make_method("integrative_r_quadratic", k_folds=5, timeout_seconds=1200).fit_predict(view, eval_X, seed=s)`. These are the native panel's options.
- Table 1 cells use `integrative_r_learner_kps_code` with OBS 2000 and RCT 200, 500 and 1000.
- Table 2 cells use `integrative_r_learner_multivariate_paper` with OBS 500 and RCT 244, the expected RCT size of 243.7 rounded. J = 1 uses `int_r_zero_cate_coefficients=8` and J = 9 uses 0. The parameter b is `int_r_confounding_strength`, 0 or 2.5.
- Two bridge cells repeat the native-panel sizes on `integrative_r_learner_kps_code`: OBS 512 with RCT 128, and OBS 512 with RCT 512.

## Metric and summaries

The metric is the paper's RMSE, which is root PEHE on each replicate's test set. Each cell is summarized over its successful replicates by the mean, the SD with ddof = 1, the standard error SD/sqrt(n), the minimum and the maximum.

## Readouts

All readouts are descriptive; no tests.

- **R0, environment check.** The preflight runs the README example once: seed 0, J = 8 and b = 2.5. Report its four RMSEs next to the values printed in the README. A difference does not block the run.
- **R1, reproduction.** Compare each of the 24 published R-learner values with the Arm A mean in the matching cell. Table 1 has 10 such values and Table 2 has 14. A value printed once for several cells is compared with one of them: Table 1's OBS-only value with n = 500, the row where it is printed, and Table 2's RCT-only values with b = 0. The agreement flag is |difference| <= 2 sqrt(SE_paper^2 + SE_rerun^2) + 0.005, where the last term covers two-decimal rounding. Report the count and the int rlearner values.
- **R2, translation and adapter.** In the seven paper cells, compare Arm B with Arm A for the integrative R-learner. The agreement flag is |difference| <= 2 sqrt(SE_A^2 + SE_B^2). Also report Arm B minus the paper value.
- **Bridge.** Report Arm B at the two bridge cells next to the native-panel values, 0.3063 with MCSE 0.0225 and 0.1378 with MCSE 0.0101, each over 64 seeds.
- **RCT size.** Report the realized RCT sizes of Arm A's Table 2 cells (mean, SD and range) next to the expected 243.7 with SD 15.6.

Failures are recorded and never retried. No setting changes after outputs are seen. Any further run is a new, separately labeled step.

## Resources

Every job runs on GSDS `P2` with no GPU, one CPU and 4 GiB of memory. Each array task runs 25 seeds of one cell: Arm A has 28 tasks, Arm B has 36, and each task has a four-hour limit. The preflight has 45 minutes and the aggregation 15 minutes. At most 12 tasks of each array run at once.

The time limits rest on the native panel, where this recipe took a median of 7 to 8 seconds per fit at OBS 512. Fits at the paper's sizes are not timed beforehand. Table 1 has up to six times more rows, and Table 2 has 54 basis columns instead of 2, so the limits leave wide margins.

All paths are under `/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01`. The R and Python environment `obsrct-v03-adapters-20261026-r01/environment-v2` and its intRlearner checkout are reused read-only. The environment has R 4.3.3 with glmnet 4.1-10, mvtnorm 1.3-3, stringr 1.5.2, caret 6.0-94 and ncvreg 3.15.0.
