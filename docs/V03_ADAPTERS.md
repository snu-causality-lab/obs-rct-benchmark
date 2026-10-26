# v0.3 optional native estimators

The 0.3.0a1 admission added three fixed recipes. Version0.3.0a2 adds the source-restricted GP controls specified by the [controlled pilot](GP_SOURCE_PILOT.md); the original admission evidence below remains tied to its own source snapshot. Native execution checks passed on GSDS: 304 tests, direct-source agreement and fresh repeats for all three recipes, independent analytic checks, and six bounded cost profiles. These adapters have not been included in the frozen v0.2 catalog and add no comparative performance result. Exact evidence is in [native_adapter_admission](../results/native_adapter_admission/README.md). The [method inventory](v03_methods.toml) records scientific scope; this document records executable recipes.

| Method ID | Recipe | Native target and qualification |
| --- | --- | --- |
| `integrative_r_linear` | Author `intrlearner` with a raw-X linear basis, five source/arm-stratified cross-fitting folds and its full native SCAD search | Shared conditional effect across studies, evaluated using the trial effect component. This restricted basis is a benchmark recipe. |
| `integrative_r_quadratic` | Same estimator with the author's `poly_interaction(d=2, interact=TRUE)` feature map | Quadratic expansion used by the author generator. Matching this feature map does not reproduce a paper table. |
| `causal_icm_fixed` | Author two-arm multi-task GP, rank 2, fixed `rho=0.5`, all observed X by default | Trial task 0 in both arms. This is a fixed borrowing recipe; the paper's borrowing selection is not implemented. |

All three ignore `rct_tune`, target-reference rows and evaluation outcomes. They receive observed fitting samples and evaluation covariates through `TrainingView`. Their source algorithms do not consume the supplied known randomization probabilities. Internal native optimization remains part of each recipe; this is distinct from the proposed common selection rule in [V03_DESIGN.md](V03_DESIGN.md).

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

Even `rho=0` still allows OBS labels to fit shared GP hyperparameters. It is not an RCT-only control. A matched trial-only GP must actually exclude OBS fitting rows. Native GP matrix operations grow with source sample size; bounded cost measurements are required before a full catalog run.

## Reproduction status

The prospective checks require deterministic direct-source/wrapper agreement at absolute tolerance 1e-10 and relative tolerance 1e-8, repeated fresh fits, and independently derived analytic identities at 1e-12. Tiny fitted examples test API fidelity and reproducibility; they do not establish CATE accuracy or a method ranking. All direct-source/wrapper and repeated-fit prediction discrepancies were zero on the declared fixtures, including changes to unused tuning labels and target-reference rows. Independent GP covariance/posterior checks had maximum absolute error 1.11e-15; the native R hand-coefficient map was exact. The separate fitted d2 polynomial/standardization checks had maximum prediction-map error 2.22e-16. These checks establish the stated source-call and target-map scope, not paper-table reproduction.

The original admission used package `0.3.0a1`; the source-control pilot uses `0.3.0a2`. Replay v0.2 results with their recorded producer source and environment. Adding optional methods changes package source fingerprints, so old result trees must not be resumed under this source tree.


## Measured execution costs

Six single-thread CPU profiles use one fixed observed-data fixture per size, five covariates, 2,048 prediction rows and seed 2642. The following seconds measure `fit_predict`; they are not medians across DGPs or projected campaign runtimes.

| Recipe | OBS 512 / RCT 128 | OBS 2048 / RCT 512 |
| --- | ---: | ---: |
| Causal-ICM fixed rho | 2.71 s | 16.44 s |
| Integrative R linear | 6.04 s | 11.20 s |
| Integrative R quadratic | 12.48 s | 29.62 s |

All profiles completed within their 600-second limit under one CPU and a 4 GiB allocation. The largest observed GP worker high-water mark was 588,036 KiB; R child marks reached 361,056 KiB. These separate process marks do not establish simultaneous whole-job peak RAM. Full process/launcher times and identities are retained in the evidence. No approximation or subsampling changes the native algorithms.

Campaign execution still needs genuinely matched RCT-only backbones, finalized tuning/resource limits and trial-target evaluation samples. In particular, neither the native L1 `rlasso` fit nor `rho=0` supplies the required matched control automatically. The planned broader method inventory remains incomplete.


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
