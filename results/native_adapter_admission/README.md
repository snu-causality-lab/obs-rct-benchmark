# Native adapter checks and CPU profiles

**Recorded 2026-10-26; package 0.3.0a1.** The optional Integrative R-learner and Causal-ICM adapters passed fixed native-call, repeated-fit and independently specified numerical checks on GSDS. This establishes the three recipes below in the recorded environment. The full suite passed **304 tests** with no failures or skips. Six bounded CPU profiles provide execution measurements. A new multi-DGP comparison has not been run; the frozen v0.2 results are unchanged.

## Recipes and native agreement

| Recipe | Checked native component | Direct-call and fresh-repeat maximum absolute prediction difference |
| --- | --- | ---: |
| `causal_icm_fixed` | Author two-arm GP with fixed `rho=0.5`; treated task 0 minus control task 0 | 0 |
| `integrative_r_linear` | Author joint SCAD effect/confounding fit on the benchmark's restricted raw-X basis | 0 |
| `integrative_r_quadratic` | Same fit with the author's degree-two polynomial/interaction feature map | 0 |

Each comparison uses one direct author-function fit and two fresh adapter fits on fixed float64 observations. The second adapter fit changes tuning outcomes by `+1e6` and changes outcome-free target-reference covariates, keeping fitting observations, prediction covariates, settings and seed fixed. Predictions remain unchanged. The information contract exposes neither evaluation truth nor RCT test outcomes.

The Causal-ICM fixture has 12 RCT rows, 16 OBS rows, two covariates and four prediction rows. Integrative R uses 80 RCT rows, 120 OBS rows, two covariates and five explicit source/arm-stratified folds; the linear check predicts seven rows, and the quadratic check predicts one. Both Integrative R recipes retain the complete native 4-by-11 SCAD search and internally estimated treatment propensities. The quadratic one-row check retains the native feature map by expanding two duplicate prediction rows and keeping the first, because the upstream feature routine drops dimensions on a singleton row.

Fidelity tolerances were fixed before the outputs: absolute **1e-10**, relative **1e-8**. Independent analytic checks use absolute **1e-12**, relative zero:

| Independent check | Maximum absolute difference |
| --- | ---: |
| Explicit GP covariance and Gaussian posterior mean/latent variance, fixed `rho=0.6` | 1.1102230246251565e-15 |
| Hand-specified native R coefficient map, including confounding-coefficient perturbation | 0 |
| Fitted two-covariate feature order, pooled sample standardization and coefficient-map checks | Passed; prediction-map maximum 2.220446049250313e-16 |

Direct/wrapper agreement checks source-call fidelity. The independent calculations supply separate mathematical checks; agreement with the same author routine alone would not establish scientific correctness. These are bounded component/recipe checks, with no paper-table, adaptive-rho, CATE-accuracy or superiority claim. The retained [Causal-ICM comparison record](native-v2/icm-compare/20261026T095717.556388Z-2376139/record.json), [Integrative R record](integrative_native_admission.json) and [summary](summary.json) state their scopes.

## Measured execution costs

All six cases use five raw covariates, 2,048 prediction rows and seed 2642, with one fixed observed-data fixture at each size. They request one CPU and 4 GiB RAM on GSDS P2/b08, use one numerical thread and enforce a 600-second outer worker limit. They retain native algorithms without approximation or subsampling.

| Recipe | OBS / RCT fitting rows | Fit + predict (s) | End to end (s) | Worker self RSS (MiB) | Worker child RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Causal-ICM, fixed rho | 512 / 128 | 2.71 | 3.63 | 169.8 | 119.0 |
| Causal-ICM, fixed rho | 2048 / 512 | 16.44 | 17.36 | 574.3 | 119.0 |
| Integrative R, linear | 512 / 128 | 6.04 | 10.39 | 44.0 | 352.6 |
| Integrative R, linear | 2048 / 512 | 11.20 | 15.20 | 43.0 | 341.3 |
| Integrative R, quadratic | 512 / 128 | 12.48 | 16.40 | 43.0 | 346.1 |
| Integrative R, quadratic | 2048 / 512 | 29.62 | 33.79 | 43.0 | 344.2 |

End-to-end time is the profiling launcher's elapsed time, including worker startup, separate prefit provenance inspection, fitting, prediction and artifact handling. It excludes queue delay and Slurm startup. `fit_predict` times exclude that separate prefit inspection; the R inspection itself takes about 3.5–3.7 seconds in these cases. Exact values and separate launcher measurements are in the six [profile records](summary.json).

RSS columns convert the worker's Linux `ru_maxrss` values from KiB to MiB. Self and child values are separate high-water marks; child usage includes reaped subprocesses, such as R or provenance commands. They are neither additive nor simultaneous process-group/whole-job peaks. The JSON retains both worker and launcher self/child marks. These single-fixture measurements are not sampling distributions, performance rankings, medians or projections of a future campaign's runtime or memory. Array tasks could run concurrently; no runtime ratio is a controlled speed comparison.

## Identity and retained evidence

The [producer identity](producer_identity.json) binds all **16 Python package source files** to the checked source snapshot; the records bind actual imports and driver hashes. The [runtime fingerprint](runtime_fingerprint.json) contains Python distribution identities, verified import resolution, and R version/namespace metadata with path hashes. [Portable Linux environment specifications](environment/README.md) retain the installed package versions/builds; fresh-prefix reconstruction passed on GSDS with exactly matching predictions and retained Integrative R parameters. Complete raw diagnostics remain private server evidence. These metadata do not archive every dependency's implementation or guarantee bitwise replay on another platform.

The pinned author revisions are Causal-ICM `0f0c6ca424b7ca44b84649ca072384b9933b2afe` and Integrative R `a46ce2c1cc814865898b863d6a8e130bbc2c802b`; their file hashes and actual loaded identities are in the runtime fingerprint and fit records. Causal-ICM executes only the intact `ICM` definition in zero-based cell 3 of `rebuttals/extrapolation.ipynb`. Its nominal helper's merge markers are preserved. The wrapper maps both arms to trial task 0. This is the specified native component, rather than the paper's complete borrowing-selection pipeline.

- [Full-test summary](tests_summary.json): GSDS job 720450, Python 3.12.3, pytest 8.3.5; 304 passed, zero failed/skipped, three console warnings. Native fits and profiles use the separately fingerprinted Python 3.11.17 / R 4.3.3 environment.
- [Preparation attempts and fixes](attempts.json): b30 cancellations, four dependency/source/provenance prerequisite failures, subsequent checks and successful reruns. Partial environments, failure logs and unsuccessful outputs are retained. Numerical criteria and native fitting rules were preserved.
- Reproduction drivers: [native analytic/Causal-ICM checks](../../scripts/native_admission.py), [Integrative R checks](../../scripts/integrative_native_admission.py), [CPU profiling](../../scripts/profile_native_adapters.py). Public stage JSON omits numerical vectors; its original-record hashes and prospective-plan hash bind the retained originals. Profile JSON contains array hashes/shapes, with raw arrays and logs retained privately.

Before a borrowing-benefit comparison, Integrative R still needs a trial-only control with the same effect basis and matched SCAD/nuisance treatment; the author's L1/OLS `rlasso` is not automatically that control. Causal-ICM needs an actual trial-only GP fit: `rho=0` still allows OBS labels to estimate shared GP hyperparameters. Trial-target evaluation draws, tuning rules and campaign resource limits also remain to be fixed under the [v0.3 design](../../docs/V03_DESIGN.md).
