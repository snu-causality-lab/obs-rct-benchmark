# OBS–RCT Benchmark

A research benchmark for estimating conditional average treatment effects (CATE) from observational data and a randomized trial. It provides source-documented generators, an observed-data method interface, public-method adapters and reproducible evaluation for binary treatment and tabular covariates.

Start with the **[AI-generated technical report](docs/REPORT.md)** ([PDF](https://github.com/snu-causality-lab/obs-rct-benchmark/releases/download/report-2026-10-30-r2/obsrct_benchmark_icml.pdf)) for the scientific account, or the [results index](results/README.md) for complete study packets. The report is an AI-generated research draft, not peer reviewed. This package is a research alpha; its inspected synthetic mechanisms do not establish empirical or unseen-mechanism performance.

## Quick start

Python 3.11 or newer is required. The core uses NumPy; the baseline extra adds scikit-learn.

```bash
python -m pip install -e '.[baselines,test]'
obs-rct-benchmark list-methods
obs-rct-benchmark run --config configs/quickstart.toml --output outputs/quickstart
```

The runner writes per-replicate records, a machine-readable summary and a Markdown report. `--resume` authenticates the same configuration and skips terminal records; changed configurations and duplicate IDs are rejected. Failures or incomplete coverage produce a nonzero exit and remain visible.

For execution coverage across all 31 presets:

```bash
obs-rct-benchmark run --config configs/smoke.toml --output outputs/catalog-smoke
```

The complete source/test checks are separate from these small examples. Run substantial tests and fitting in an allocated compute job:

```bash
python -m pytest
python scripts/check_release.py
```

Quickstart uses two designs and two seeds; catalog smoke uses all 31 designs and the same two seeds. Both use OBS 120, RCT training 60, tuning 30, test 30 and evaluation 128 rows per design/seed. These bounded examples check execution, not method rankings. See [verification scopes](docs/VERIFICATION.md) and [stored examples](results/README.md#original-alpha-execution-examples).

## Read the results

Five [study packets](results/README.md) cover the 31-preset catalog, a controlled GP source-access experiment, a native/default trial-CATE comparison, matched-M2 conditions and charged recruitment. Each has its own frozen producer, target, budget and inference rules.

The studies distinguish improving an OBS starting prediction, benefiting from OBS within one learner, and outperforming another available procedure. In the [controlled GP experiment](results/gp_source_pilot/SCIENTIFIC_RESULT.md), OBS lowers mean trial-CATE MSE at discrepancy levels 0/1 and raises it at 2 relative to the matched RCT-only GP at both trial budgets. Its prior-shrunken OBS start is not an optimal OBS-only estimator. In the complete native-panel M2/RCT128 condition, adding OBS reduces the GP's mean MSE from 0.53675 to 0.35128, while RCT-only forest has mean MSE 0.18386. These are descriptive comparisons.

The [native panel](results/native_panel/SCIENTIFIC_RESULT.md) retains 8,350 successes, 64 failures, 31 timeouts and three missing attempts from 8,448 requests. All 72 joint decisions and simultaneous precision remain withheld; mean-risk inference was never established. Successful-only means are conditional, and incomplete paired summaries are withheld. Only GP Full versus its matched RCT-only version isolates OBS access within an admitted common backbone. Other fusion-versus-forest contrasts compare complete procedures.

The [comparison design](docs/V03_DESIGN.md) separates the remaining method, matched-control, target and observable-tuning questions. Its [method inventory](docs/V03_METHODS.md) has 22 method or method-group entries; the [scenario map](docs/V03_SCENARIOS.md) classifies all 31 presets by target, support and transport assumptions.

## What is included

| Track | Available designs | Purpose |
| --- | --- | --- |
| Paper-related designs | 8 translations and 1 explicit extension from 6 research sources | Make source assumptions and adaptations explicit |
| Structural mechanisms | M0–M11 | Vary hidden treatment selection, bias geometry, effect complexity, study participation and source shift |
| Paired structural comparisons | C01–C10 | Change a stated mechanism while retaining the comparison's shared components |
| Empirical covariates | User-supplied IHDP NPZ/CSV and ACIC2016 CSV files | Hide selected covariates and generate new OBS/RCT samples with exact finite-population targets |
| Marginal sensitivity model | Constant Γ and user-defined Γ(X) | Control hidden treatment selection separately from outcomes and source shifts |

The six paper sources are CorNet, Experimental Grounding, Integrative R-learner, Elastic Integrative, power likelihood and CVCI. The Integrative R-learner normal-target extension deliberately evaluates outside the Uniform[-1,1] trial support; its original paper and code use a Uniform[-1,1] test population. Variants of one source are not counted as independent papers. See the [paper cards](docs/PAPER_DGPS.md), [coverage](docs/COVERAGE.md) and [extraction notes](docs/DGP_EXTRACTION.md).

Version **0.3.0a4** includes RCT-only, OBS-only and naive pooled T-learners, plus an RCT DR-learner, each with Ridge or ExtraTrees. Optional native recipes include DataFusionPPI, CorNet, Integrative R with two bases, fixed-rho Causal-ICM with two source controls, and an RCT-only honest causal forest. These are 16 registered recipes spanning variants and controls, including four fusion approaches; they are not 16 distinct algorithms. [Native settings and admission](docs/V03_ADAPTERS.md) are explicit; the included recipes are not tuned reproductions of every source paper's results.

The [empirical-data interface](docs/DATASETS.md#local-file-adapters) reads separately obtained files and defines new semi-synthetic OBS/RCT problems; it does not preserve the source experiment's treatment assignments or noisy outcomes. The [Γ/Γ(X) factory](docs/SENSITIVITY.md) preserves a specified marginal propensity and checks the sensitivity bound at evaluated covariates. Configuration files support constant and logistic Γ profiles; the Python API accepts pointwise callbacks. The factory implements one configurable construction within the marginal sensitivity model.

## Choose an evaluation track

- **Static estimation:** compare fixed observed-data recipes across named conditions and OBS/RCT sample budgets. Optional condition labels let one DGP appear at several budgets without changing its sampling seed. Reports show each condition's scores, paired differences, eligible intervals and failures. See the [protocol](docs/PROTOCOL.md) and [budget-grid example](docs/EXTENDING.md).
- **Sequential recruitment:** reveal a common pilot, choose additional trial subjects from an outcome-hidden candidate pool, then fit at fixed outcome budgets. Every acquired outcome is charged. Uniform, geometry and outcome-assisted reference policies are included. See the [adaptive protocol](docs/ADAPTIVE.md).

## Learn with notebooks

Start with the [executed notebook tutorials](notebooks/README.md). They combine editable controls, explanations, plots and small checks:

1. [First comparison](notebooks/01_first_comparison.ipynb): generate OBS/RCT samples, compare fixed baselines and interpret CATE/region errors.
2. [Gamma and assumptions](notebooks/02_gamma_and_assumptions.ipynb): separate hidden treatment selection, outcome loadings and source shifts.
3. [Empirical covariates](notebooks/03_empirical_covariates.ipynb): understand masking, coarsening and displayed-X targets; optionally connect your IHDP/ACIC files.
4. [Custom methods and reproducibility](notebooks/04_custom_method_and_reproducibility.ipynb): add an estimator, read paired comparisons, preserve failures and resume runs.
5. [Multi-source identification](notebooks/05_multisource_identification.ipynb): enumerate a counterexample where complementary RCT covariates do not identify joint CATE. This is a mathematical tutorial; the package's current source contract is unchanged.

Install `python -m pip install -e '.[baselines,notebooks]'` for JupyterLab and execution dependencies. The default notebooks use synthetic data and include stored outputs; no external dataset or official method checkout is needed. Their small runs are teaching examples, not method-ranking evidence.

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

The RCT-only forest uses a separately built, pinned `grf` installation, known trial assignment probabilities and native outcome nuisances. Its [installation and build-record instructions](docs/CAUSAL_FOREST.md#source-installation-and-runtime-binding) preserve a pristine upstream checkout and bind the selected source, installed package and runtime evidence. Upstream code is not bundled.

The benchmark's MIT license covers its original code. External sources retain their own terms; public availability alone is not a redistribution license. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Evaluation contract

`TrainingView` separates OBS training, RCT training, RCT tuning and permitted outcome-free target covariates. The [GP source controls](docs/GP_SOURCE_PILOT.md) and [RCT-only forest](docs/CAUSAL_FOREST.md) use narrower included-sample-only entries. `eval_X` is for prediction only; preprocessing, density estimation and adaptation use permitted training data or a separate `target_reference`. Latent confounders, evaluation outcomes and effect truth remain evaluator-owned. The interface helps prevent accidental leakage; it is not a security sandbox.

Supported generators supply an independent target-covariate sample when `n_target_reference>0`; the default is none. `rct_test` is reserved for evaluator factual diagnostics. CATE scoring uses `eval_tau` at `eval_X`, without RCT test outcomes. Methods within a scenario/seed receive identical permitted samples. See the [protocol](docs/PROTOCOL.md) and [extension guide](docs/EXTENDING.md) for role, prediction and failure contracts.

CATE MSE is `mean((prediction - target_cate)**2)`; per-replicate root-PEHE is its square root. Paired comparisons use shared replicate seeds and retain failed/missing cases. Generic Student-t uncertainty requires complete pairs, at least two blocks and [metric-specific permission](docs/INFERENCE.md) in every contributing condition. Student-t(3) noise blocks mean-MSE uncertainty; root-PEHE has a separate policy. Native, external and custom recipes need explicit assumptions. Permission does not establish finite variance or reliable small-sample coverage; frozen studies retain their own analysis rules.

Paper translations, constructed stresses and empirical-covariate designs answer different questions. Empirical targets condition on displayed covariates; dropping a covariate need not create residual confounding in a finite empirical distribution. A common Γ treatment-selection bound can coexist with different outcome loadings and source shifts. The [dataset contract](docs/DATASETS.md#local-file-adapters) and [sensitivity model](docs/SENSITIVITY.md) state these distinctions.

## Add a method or DGP

Use the public package from a separate research repository and implement `fit_predict(training, eval_X, *, seed)`. The [custom-method example](examples/run_custom_method.py) and [extension guide](docs/EXTENDING.md) show the interface. The benchmark does not need access to that research repository.

Contributions should include source attribution, explicit adaptations and checks of the target and information contract. See [CONTRIBUTING.md](CONTRIBUTING.md). Cite this software using [CITATION.cff](CITATION.cff), and cite the original papers for the designs or methods used.

## References

These research sources have different roles in the benchmark. A translated DGP does not imply that the paper's estimator is implemented.

| Paper | DGP translation | Method adapter |
| --- | --- | --- |
| Hatt et al. (2022), [Combining Observational and Randomized Data for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/2202.12891v1) | CorNet simulation | Optional official CorNet |
| Kallus, Puli and Shalit (2018), [Removing Hidden Confounding by Experimental Grounding](https://arxiv.org/abs/1810.11646v1) | Experimental Grounding simulation | Not included |
| Wu and Yang (2022), [Integrative R-learner of heterogeneous treatment effects combining experimental and observational studies](https://proceedings.mlr.press/v177/wu22a.html) | Two translations and an explicit target extension | Optional native linear/quadratic recipes |
| Yang et al. (2022), [Elastic Integrative Analysis of Randomized Trial and Real-World Data for Treatment Heterogeneity Estimation](https://arxiv.org/abs/2005.10579v3) | Elastic Integrative simulation | Not included |
| Lin, Tarp and Evans (2025), [Combining experimental and observational data through a power likelihood](https://academic.oup.com/biometrics/article/81/1/ujaf008/8016472) | Paper and supplement variants | Not included |
| Yang et al. (2025), [Cross-Validated Causal Inference: a Modern Method to Combine Experimental and Observational Data](https://arxiv.org/abs/2511.00727v1) | CVCI linear simulation | Not included |
| Jung and Yang (2026), [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332v2) | Not included | Optional official DataFusionPPI |
| Dimitriou et al. (2026), [Causal-ICM](https://proceedings.mlr.press/v323/dimitriou26a.html) | No source-paper DGP translation; a separate [controlled GP experiment](results/gp_source_pilot/README.md) is available | Optional fixed-rho native GP recipe and source controls |
| Athey, Tibshirani and Wager (2019), [Generalized Random Forests](https://arxiv.org/abs/1610.01271) | Not included | Optional RCT-only honest `grf` recipe |

For empirical benchmark provenance, see Hill (2011), [Bayesian Nonparametric Modeling for Causal Inference](https://doi.org/10.1198/jcgs.2010.08162), and Dorie et al. (2019), [Automated versus Do-It-Yourself Methods for Causal Inference: Lessons Learned from a Data Analysis Competition](https://doi.org/10.1214/18-STS667). [Dataset sources and terms](docs/DATASETS.md) distinguish the original IHDP archive, semi-synthetic IHDP releases and the ACIC2016 package. No dataset is bundled with this repository.

For benchmark-design background, Curth, Svensson, Weatherall and van der Schaar (2021), [Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2a79ea27c279e471f4d180b08d62b00a-Abstract-round2.html), examines how semi-synthetic assumptions and evaluation metrics affect CATE comparisons.
