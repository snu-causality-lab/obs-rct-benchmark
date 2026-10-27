# v0.3 optional native estimators

Version **0.3.0a4** provides the four native recipes below and two source-restricted GP controls. The RCT-only causal forest has separate default-recipe admission; it has not entered a comparative accuracy study. The separately frozen **0.3.0a2** [controlled pilot](../results/gp_source_pilot/SCIENTIFIC_RESULT.md) completes 576 fits across six conditions and 32 paired seeds: OBS addition helps the fixed GP at discrepancy levels 0/1 and hurts it at level 2 relative to the matched RCT-only GP at both trial budgets. The broader v0.3 campaign remains pending.

The original **0.3.0a1 admission** remains tied to its own source snapshot: 304 GSDS tests, direct-source agreement and fresh repeats for all three native recipes, independent analytic checks, and six bounded cost profiles. Those admission checks do not establish comparative accuracy. Exact evidence is in [native_adapter_admission](../results/native_adapter_admission/README.md); the frozen v0.2 catalog is unchanged. The [method inventory](v03_methods.toml) records scientific scope; this document records executable recipes.

| Method ID | Recipe | Native target and qualification |
| --- | --- | --- |
| `integrative_r_linear` | Author `intrlearner` with a raw-X linear basis, five source/arm-stratified cross-fitting folds and its full native SCAD search | Shared conditional effect across studies, evaluated using the trial effect component. This restricted basis is a benchmark recipe. |
| `integrative_r_quadratic` | Same estimator with the author's `poly_interaction(d=2, interact=TRUE)` feature map | Quadratic expansion used by the author generator. Matching this feature map does not reproduce a paper table. |
| `causal_icm_fixed` | Author two-arm multi-task GP, rank 2, fixed `rho=0.5`, all observed X by default | Trial task 0 in both arms. This is a fixed borrowing recipe; the paper's borrowing selection is not implemented. |
| `rct_causal_forest` | Official `grf` 2.6.1, 2,000 honest trees, known trial propensity, native OOB outcome nuisance, one thread and no tuning | RCT full-X CATE; only RCT fitting rows enter nuisances and trees. Admission covers the default recipe, without native variance intervals or accuracy claims. |

All four ignore `rct_tune`, target-reference rows and evaluation outcomes. They receive observed fitting samples and prediction covariates through `TrainingView`; the causal forest also accepts an RCT-only `SourceSample`. Integrative R and Causal-ICM do not consume the supplied known randomization probabilities; the causal forest supplies them as `W.hat`. Internal native optimization remains part of the fusion recipes; this is distinct from the proposed common selection rule in [V03_DESIGN.md](V03_DESIGN.md).

For new generic runs, these native recipes and source controls require an explicit per-scenario declaration for each metric before seed-level uncertainty is reported. Source fidelity and deterministic replay do not establish finite variance of fitted-method risk. Omission retains descriptive scores and complete paired means. See the [inference policy](INFERENCE.md); the pilot's frozen analysis retains its own rules.

Run the installation commands below from the `obs-rct-benchmark` repository root. Sibling checkout directories are examples; the environment variables may point to any verified checkout. After installing dependencies and setting the source variable, this bounded example runs on an allocated compute node:

```python
from obs_rct_benchmark import generate_structural_problem, make_structural_config
from obs_rct_benchmark.methods import make_method

config = make_structural_config(
    name="native-example", mechanism="M2", n_obs=64, n_rct_train=32, n_rct_tune=8, n_rct_test=8, n_eval=8,
)
problem = generate_structural_problem(config, seed=2601026)
method = make_method("causal_icm_fixed")
prediction = method.fit_predict(problem.training_view(), problem.eval_X, seed=2601026)
assert prediction.shape == (8,)
```

`TrainingView` is the observed-data contract in [data.py](../src/obs_rct_benchmark/data.py); constructing it from a simulation excludes truth and RCT test outcomes. Seeds control input generation, deterministic folds and estimator initialization. Exact replay also requires the recorded source, numerical environment and thread settings.

## RCT-only honest causal forest

The [recipe and installation instructions](CAUSAL_FOREST.md) specify the pinned official source, separate build copy, mandatory build manifest and actual loaded namespace/native-library checks. The adapter uses only `rct_train.X/T/Y/known_propensity`, with `Y.hat=NULL` and native OOB outcome regression. OBS, tuning and target-reference fields are not accessed; evaluation covariates enter prediction only.

The separate [0.3.0a4 admission](../results/causal_forest_admission/README.md) passes six native fits: a direct call, two fresh adapter processes, query extension and two cost profiles. Direct/repeated predictions and the original query prefix agree exactly; an independent local-intercept calculation differs by at most 2.220446049250313e-15. Its 60 focused and 475 full-suite tests pass with no skips. This checks the default source call and prediction arithmetic conditional on native neighborhoods/nuisances, not comparative CATE performance or source-paper tables. R-learner and Bayesian causal forest admission remain planned.

## Integrative R-learner

Obtain the [author repository](https://github.com/lwu9/intRlearner/tree/a46ce2c1cc814865898b863d6a8e130bbc2c802b) separately and retain its Git metadata:

```bash
git clone https://github.com/lwu9/intRlearner.git ../intRlearner
git -C ../intRlearner checkout --detach a46ce2c1cc814865898b863d6a8e130bbc2c802b
export INTEGRATIVE_R_SOURCE="$(cd ../intRlearner && pwd)"
export INTEGRATIVE_R_RSCRIPT="$(command -v Rscript)"
```

R must provide `caret`, `glmnet`, `ncvreg` and `stringr`. The adapter verifies source revisions and hashes, invokes `Rscript --vanilla` as a subprocess, and records the actual R and namespace versions. It does not install these packages. [sources.lock.toml](../sources.lock.toml) pins source identity; a separate environment lock is required for exact replay.

RCT rows precede OBS rows. The source indicator is 1 for RCT and 0 for OBS. The source algorithm estimates nuisances and uses pooled fitting-data standardization. Prediction retains the effect intercept/slopes and zeros the observational confounding columns. Five cross-fitting folds are explicitly stratified by source and treatment; that deterministic fold construction is a disclosed adapter choice. The default effect search retains all 4 scale values and 11 gamma values, with the native internal CV defaults. Failed processes, timeouts or invalid results raise errors; full R diagnostics remain available on the estimator for server-side retention.

The native trial-only `rlasso` uses a different effect regularizer and is not automatically a capacity/regularizer-matched control for the SCAD integrative fit. Such a control must be specified before attributing a comparison to OBS borrowing.

## Causal-ICM

Install the narrowly pinned optional Python environment and obtain the [author repository](https://github.com/EvanDimitriou/CausalICM/tree/0f0c6ca424b7ca44b84649ca072384b9933b2afe) separately:

```bash
python -m pip install -e '.[causal-icm]'
git clone https://github.com/EvanDimitriou/CausalICM.git ../CausalICM
git -C ../CausalICM checkout --detach 0f0c6ca424b7ca44b84649ca072384b9933b2afe
export CAUSAL_ICM_SOURCE="$(cd ../CausalICM && pwd)"
```

The numerical lock is NumPy 1.26.4, SciPy 1.12.0, GPy 1.13.2 and paramz 0.9.6. GPy also imports Matplotlib; the optional extra includes it. Source verification includes the full notebook, exact definition cell, Git state and import resolution. The pinned `functions/model_training.py` contains unresolved merge markers. The adapter therefore executes only the intact, unchanged `ICM` definition in cell 3 (zero-based) of `rebuttals/extrapolation.ipynb`, in a private module. It does not execute the notebook or patch upstream source.

Both treatment models are queried with source task 0 (RCT); task 1 denotes OBS. Selected effect-modifier columns are an explicit model restriction. A reduced-X fit is rejected when known RCT assignment probabilities vary across rows, because dropping an assignment covariate can change the conditional arm contrast. The all-X default has no such reduction. No simulation truth chooses columns or borrowing values.

Even `rho=0` still allows OBS labels to fit shared GP hyperparameters. It is not an RCT-only control. Version 0.3.0a2 implements `causal_icm_rct_only`, which excludes OBS rows from both fitting and shared-hyperparameter optimization, and `causal_icm_obs_start`, which excludes RCT rows and queries the trial task 0. Each control can receive only its included `SourceSample`; the pilot uses that source-restricted API. The OBS starting prediction is prior-shrunken by `rho/(1+kappa)` relative to the same fitted GP's OBS-task mean (`kappa=1e-9`), not the unshrunk OBS contrast or an optimal OBS-only estimate. Both source controls use every observed covariate and retain the same native rank 2 GP and optimization defaults. See the [frozen source-access specification](GP_SOURCE_PILOT.md). Native GP matrix operations grow with source sample size; the pilot's resource measurements do not establish feasibility of a full catalog run.

## Reproduction status

The original 0.3.0a1 checks required deterministic direct-source/wrapper agreement at absolute tolerance 1e-10 and relative tolerance 1e-8, repeated fresh fits, and independently derived analytic identities at 1e-12. Tiny fitted examples test API fidelity and reproducibility; they do not establish CATE accuracy or a method ranking. All direct-source/wrapper and repeated-fit prediction discrepancies were zero on those declared fixtures, including changes to unused tuning labels and target-reference rows. Independent GP covariance/posterior checks had maximum absolute error 1.11e-15; the native R hand-coefficient map was exact. The separate fitted d2 polynomial/standardization checks had maximum prediction-map error 2.22e-16. These checks establish the stated source-call and target-map scope, not paper-table reproduction.

The original admission used package `0.3.0a1`; the source-control pilot uses `0.3.0a2`. The [pilot checks](../results/gp_source_pilot/checks_summary.json) pass 371 tests in the fresh numerical environment plus one separately configured CorNet integration test, 13 native/data preflight fits, independent reconstruction of all 576 risks and 160 source-exclusion groups, 18 prespecified fresh replays with identical predictions, and a byte-identical completed-run resume with no refits. See the [pilot evidence](../results/gp_source_pilot/README.md) for checks, failures retained from an out-of-panel helper attempt, and replay scope. These checks do not establish unseen-mechanism performance or source-paper table reproduction. Replay v0.2 results with their recorded producer source and environment. Adding optional methods changes package source fingerprints, so old result trees must not be resumed under this source tree.


## Measured execution costs

The original admission's six single-thread CPU profiles use one fixed observed-data fixture per size, five covariates, 2,048 prediction rows and seed 2642. The following seconds measure `fit_predict`; they are not medians across DGPs or projected campaign runtimes.

| Recipe | OBS 512 / RCT 128 | OBS 2048 / RCT 512 |
| --- | ---: | ---: |
| Causal-ICM fixed rho | 2.71 s | 16.44 s |
| Integrative R linear | 6.04 s | 11.20 s |
| Integrative R quadratic | 12.48 s | 29.62 s |

All profiles completed within their 600-second limit under one CPU and a 4 GiB allocation. The largest observed GP worker high-water mark was 588,036 KiB; R child marks reached 361,056 KiB. These separate process marks do not establish simultaneous whole-job peak RAM. Full process/launcher times and identities are retained in the evidence. No approximation or subsampling changes the native algorithms.

The causal forest has two separately recorded RCT-only cost fixtures under its [own default settings and admission](CAUSAL_FOREST.md#recorded-admission). Those full-adapter measurements include provenance checks and process startup; their child RSS marks are not simultaneous whole-job peaks. Neither set of fixtures forecasts a full campaign or establishes a performance ranking.

The pilot supplies a genuinely source-excluded GP control with frozen trial-target samples and budgets. The broader campaign still needs other matched RCT-only backbones and finalized tuning/resource limits. In particular, the native L1 `rlasso` fit does not automatically match the integrative SCAD regularizer. The planned broader method inventory remains incomplete.


## Running the recorded checks

Use a fresh output directory under `GSDS_WORK_ROOT` in a Slurm CPU job. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1, and configure both author source variables and `INTEGRATIVE_R_RSCRIPT` as above. From the package root:

```bash
python scripts/native_admission.py --repo . --output-dir "$GSDS_WORK_ROOT/new-native-checks/gp" \
  --stage analytic-gp --stage analytic-intr --stage icm-direct \
  --stage icm-adapter-1 --stage icm-adapter-2 --stage icm-compare
python scripts/integrative_native_admission.py \
  --output "$GSDS_WORK_ROOT/new-native-checks/integrative" --basis both
python scripts/profile_native_adapters.py --method causal_icm_fixed \
  --n-obs 2048 --n-rct 512 --d 5 --n-eval 2048 --seed 2642 \
  --timeout-seconds 600 --output-dir "$GSDS_WORK_ROOT/new-native-checks/profile"
```

Successful stages report `passed`; the profile reports `completed`. Full logs and arrays stay at these server destinations. Use the published compact evidence for the original hashes and versions; installing newer dependencies does not reproduce that environment.
