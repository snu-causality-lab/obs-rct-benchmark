# OBS–RCT Benchmark

A benchmark for estimating conditional average treatment effects (CATE) from a large observational sample and a smaller randomized trial. It provides source-documented data generators, a common observed-data interface, public-method adapters and reproducible evaluation.

This is a research alpha. The initial scope is binary treatment and tabular covariates. The catalog covers multiple borrowing assumptions and failure mechanisms; it does not claim to cover every OBS–RCT setting.

## Choose an evaluation track

- **Static estimation:** compare fixed observed-data recipes across named conditions and OBS/RCT sample budgets. Optional condition labels let one DGP appear at several budgets without changing its sampling seed. Reports show each condition's scores, paired differences, intervals and failures. See the [protocol](docs/PROTOCOL.md) and [budget-grid example](docs/EXTENDING.md).
- **Sequential recruitment:** reveal a common pilot, choose additional trial subjects from an outcome-hidden candidate pool, then fit at fixed outcome budgets. Every acquired outcome is charged. Uniform, geometry and outcome-assisted reference policies are included. See the [adaptive protocol](docs/ADAPTIVE.md).

The [full catalog study](results/catalog_study/SCIENTIFIC_RESULT.md) evaluates all 31 native presets, four sample-budget pairs, 64 seeds and ten recipes: **79,083 successful fits, 21 terminal failures, zero missing applicable cells**, plus 256 prospectively unsupported cells. All failures are DataFusionPPI calibration failures in two Elastic conditions. The frozen rule withholds all 472 eligible Holm decisions; all condition-level results remain reported. Prespecified refits and independent score reaggregation pass. The earlier [matched-M2](results/reference_study/SCIENTIFIC_RESULT.md) and [recruitment](results/adaptive_reference/SCIENTIFIC_RESULT.md) studies remain separate evidence. These inspected synthetic mechanisms do not establish unseen-mechanism generalization.

## What is included

| Track | Available designs | Purpose |
| --- | --- | --- |
| Paper-related designs | 8 translations and 1 explicit extension from 6 research sources | Make source assumptions and adaptations explicit |
| Structural mechanisms | M0–M11 | Vary hidden treatment selection, bias geometry, effect complexity, study participation and source shift |
| Paired structural comparisons | C01–C10 | Change a stated mechanism while retaining the comparison's shared components |
| Empirical covariates | User-supplied IHDP NPZ/CSV and ACIC2016 CSV files | Hide selected covariates and generate new OBS/RCT samples with exact finite-population targets |
| Marginal sensitivity model | Constant Γ and user-defined Γ(X) | Control hidden treatment selection separately from outcomes and source shifts |

The six paper sources are CorNet, Experimental Grounding, Integrative R-learner, Elastic Integrative, power likelihood and CVCI. The Integrative R-learner normal-target extension deliberately evaluates outside the Uniform[-1,1] trial support; its original paper and code use a Uniform[-1,1] test population. Variants of one source are not counted as independent papers. See the [paper cards](docs/PAPER_DGPS.md), [coverage](docs/COVERAGE.md) and [extraction notes](docs/DGP_EXTRACTION.md).

Generic baselines include RCT-only, OBS-only and naive pooled T-learners, plus an RCT DR-learner. Optional adapters call pinned official DataFusionPPI and CorNet implementations. Baseline settings and adaptations are explicit; the included recipes are not tuned reproductions of every source paper's results.

The [empirical-data interface](docs/DATASETS.md#local-file-adapters) reads separately obtained files and defines new semi-synthetic OBS/RCT problems; it does not preserve the source experiment's treatment assignments or noisy outcomes. The [Γ/Γ(X) factory](docs/SENSITIVITY.md) preserves a specified marginal propensity and checks the sensitivity bound at evaluated covariates. Configuration files support constant and logistic Γ profiles; the Python API accepts pointwise callbacks. The factory implements one configurable construction within the marginal sensitivity model.

## References

These seven research sources have different roles in the benchmark. A translated DGP does not imply that the paper's estimator is implemented.

| Paper | DGP translation | Method adapter |
| --- | --- | --- |
| Hatt et al. (2022), [Combining Observational and Randomized Data for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/2202.12891v1) | CorNet simulation | Optional official CorNet |
| Kallus, Puli and Shalit (2018), [Removing Hidden Confounding by Experimental Grounding](https://arxiv.org/abs/1810.11646v1) | Experimental Grounding simulation | Not included |
| Wu and Yang (2022), [Integrative R-learner of heterogeneous treatment effects combining experimental and observational studies](https://proceedings.mlr.press/v177/wu22a.html) | Two translations and an explicit target extension | Not included |
| Yang et al. (2022), [Elastic Integrative Analysis of Randomized Trial and Real-World Data for Treatment Heterogeneity Estimation](https://arxiv.org/abs/2005.10579v3) | Elastic Integrative simulation | Not included |
| Lin, Tarp and Evans (2025), [Combining experimental and observational data through a power likelihood](https://academic.oup.com/biometrics/article/81/1/ujaf008/8016472) | Paper and supplement variants | Not included |
| Yang et al. (2025), [Cross-Validated Causal Inference: a Modern Method to Combine Experimental and Observational Data](https://arxiv.org/abs/2511.00727v1) | CVCI linear simulation | Not included |
| Jung and Yang (2026), [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332v1) | Not included | Optional official DataFusionPPI |

For empirical benchmark provenance, see Hill (2011), [Bayesian Nonparametric Modeling for Causal Inference](https://doi.org/10.1198/jcgs.2010.08162), and Dorie et al. (2019), [Automated versus Do-It-Yourself Methods for Causal Inference: Lessons Learned from a Data Analysis Competition](https://doi.org/10.1214/18-STS667). [Dataset sources and terms](docs/DATASETS.md) distinguish the original IHDP archive, semi-synthetic IHDP releases and the ACIC2016 package. No dataset is bundled with this repository.

For benchmark-design background, Curth and van der Schaar (2021), [Doing Great at Estimating CATE? On the Neglected Assumptions in Benchmark Comparisons of Treatment Effect Estimators](https://arxiv.org/abs/2107.13346v1), examines how IHDP/ACIC2016 data-generation assumptions interact with estimator comparisons.

## Learn with notebooks

Start with the [executed notebook tutorials](notebooks/README.md). They combine editable controls, explanations, plots and small checks:

1. [First comparison](notebooks/01_first_comparison.ipynb): generate OBS/RCT samples, compare fixed baselines and interpret CATE/region errors.
2. [Gamma and assumptions](notebooks/02_gamma_and_assumptions.ipynb): separate hidden treatment selection, outcome loadings and source shifts.
3. [Empirical covariates](notebooks/03_empirical_covariates.ipynb): understand masking, coarsening and displayed-X targets; optionally connect your IHDP/ACIC files.
4. [Custom methods and reproducibility](notebooks/04_custom_method_and_reproducibility.ipynb): add an estimator, read paired comparisons, preserve failures and resume runs.
5. [Multi-source identification](notebooks/05_multisource_identification.ipynb): enumerate a counterexample where complementary RCT covariates do not identify joint CATE. This is a mathematical tutorial; the package's current source contract is unchanged.

Install `python -m pip install -e '.[baselines,notebooks]'` for JupyterLab and execution dependencies. The default notebooks use synthetic data and include stored outputs; no external dataset or official method checkout is needed. Their small runs are teaching examples, not method-ranking evidence.

## Quick start

Python 3.11 or newer is required. The core generators use NumPy; the baseline extra adds scikit-learn.

```bash
python -m pip install -e '.[baselines,test]'
obs-rct-benchmark list-methods
obs-rct-benchmark run --config configs/quickstart.toml --output outputs/quickstart
```

The runner writes individual records, a machine-readable summary and a Markdown report. Resume the same configuration with `--resume`; changed configurations and duplicate IDs are rejected. A run with failures or incomplete coverage exits unsuccessfully and reports those cases.

The quickstart uses two designs and two seeds; the catalog smoke uses all 31 designs and the same two seeds. Both use OBS 120, RCT training 60, tuning 30, test 30 and evaluation 128 rows per design/seed. The initial single-worker CPU verification completed in under a minute; see the [measured scope and limits](docs/VERIFICATION.md).

For execution coverage across the full initial catalog:

```bash
obs-rct-benchmark run --config configs/smoke.toml --output outputs/catalog-smoke
python -m pytest
python scripts/check_release.py
```

The two-seed smoke configuration checks that the paths run. It is too small to support a method ranking. See [example results](results/README.md) for the executed configuration, coverage and limitations.

## Public upstream methods

DataFusionPPI is optional and installed from the revision recorded in [sources.lock.toml](sources.lock.toml):

```bash
python -m pip install -e '.[datafusionppi]'
```

The CorNet adapter uses a separate, clean checkout because upstream code is not bundled:

```bash
python -m pip install -e '.[cornet]'
git clone https://github.com/tobhatt/CorNet.git _upstreams/CorNet
git -C _upstreams/CorNet checkout 38b9c37d5a86bc87950945d6ddb1ceb5354a45ba
export CORNET_SOURCE="$PWD/_upstreams/CorNet"
obs-rct-benchmark run --config configs/official_smoke.toml --output outputs/official-smoke
```

The official smoke configuration uses reduced CorNet training epochs to verify integration. Default adapter settings use 500 source epochs and 200 correction epochs, a three-layer width-25 representation, full-batch SGD and optional OBS-fitted standardization. CorNet fits the representation on OBS and the released armwise correction on RCT training data. DataFusionPPI uses its own three-way rotated splits within the RCT training sample. These fixed recipes do not consume the separate RCT tuning sample. The [protocol](docs/PROTOCOL.md) describes sample use and optimization.

The benchmark's MIT license covers its original code. External sources retain their own terms; public availability alone is not a redistribution license. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Evaluation contract

A method receives `TrainingView`: OBS training data, RCT training data, RCT tuning data and any permitted outcome-free target covariates. Query `eval_X` is for prediction only: fit preprocessing, density models and adaptation using training data or the separate permitted `target_reference`. Latent confounders, evaluation outcomes and effect truth stay with the evaluator. This interface helps prevent accidental leakage; it is not a security sandbox.

Supported generators provide an independent target-covariate sample when `n_target_reference>0`; the default is no such sample. The separate `rct_test` is evaluator-owned and reserved for factual diagnostics. The current runner scores predictions against `eval_tau` at `eval_X`, without using RCT test outcomes. All methods within one scenario/seed receive the same realized samples. The [protocol](docs/PROTOCOL.md) and [extension guide](docs/EXTENDING.md) specify prediction shape, registration and failure handling.

The primary metric is CATE MSE, `mean((prediction - target_cate)**2)`. Root-PEHE is its square root and is reported separately. Paired comparisons use the same independent replicate seeds, with failed and missing cases counted explicitly. Student t intervals from very few seeds should not be interpreted as reliable inferential evidence. The [protocol](docs/PROTOCOL.md) defines targets, sample roles, budgets and aggregation.

Paper translations, constructed stress mechanisms and empirical-covariate benchmarks answer different questions. Empirical targets condition on the covariates actually displayed after any explicit coarsening; dropping a covariate alone need not create residual confounding under a finite empirical distribution. The same Γ treatment-selection bound can coexist with different outcome loadings, source shifts and outcome bias. See the [dataset contract](docs/DATASETS.md#local-file-adapters) and [sensitivity-model definition](docs/SENSITIVITY.md).

## Add a method or DGP

Use the public package from a separate research repository and implement `fit_predict(training, eval_X, *, seed)`. The [custom-method example](examples/run_custom_method.py) and [extension guide](docs/EXTENDING.md) show the interface. The benchmark does not need access to that research repository.

Contributions should include source attribution, explicit adaptations and checks of the target and information contract. See [CONTRIBUTING.md](CONTRIBUTING.md). Cite this software using [CITATION.cff](CITATION.cff), and cite the original papers for the designs or methods used.
