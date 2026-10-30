# OBS–RCT Benchmark

A benchmark for estimating conditional average treatment effects (CATE) from a large observational sample and a smaller randomized trial. It provides source-documented data generators, a common observed-data interface, public-method adapters and reproducible evaluation.

This is a research alpha. The initial scope is binary treatment and tabular covariates. The catalog covers multiple borrowing assumptions and failure mechanisms; it does not claim to cover every OBS–RCT setting.

Read the **[AI-generated technical report](docs/REPORT.md)** ([PDF](https://github.com/snu-causality-lab/obs-rct-benchmark/releases/download/report-2026-10-30/obsrct_benchmark_icml.pdf)): one integrated manuscript, ten main-text pages and 58 total, covering the framework, native checks and five studies. It is an AI-generated research draft; the report page binds its document hash and study-specific evidence.

## Choose an evaluation track

- **Static estimation:** compare fixed observed-data recipes across named conditions and OBS/RCT sample budgets. Optional condition labels let one DGP appear at several budgets without changing its sampling seed. Reports show each condition's scores, paired differences, eligible intervals and failures. See the [protocol](docs/PROTOCOL.md) and [budget-grid example](docs/EXTENDING.md).
- **Sequential recruitment:** reveal a common pilot, choose additional trial subjects from an outcome-hidden candidate pool, then fit at fixed outcome budgets. Every acquired outcome is charged. Uniform, geometry and outcome-assisted reference policies are included. See the [adaptive protocol](docs/ADAPTIVE.md).

The [full catalog study](results/catalog_study/SCIENTIFIC_RESULT.md) evaluates all 31 native presets, four sample-budget pairs, 64 seeds and ten recipes: **79,083 successful fits, 21 terminal failures, zero missing applicable cells**, plus 256 prospectively unsupported cells. All failures are DataFusionPPI calibration failures in two Elastic conditions. The frozen rule withholds all 472 eligible Holm decisions; all condition-level results remain reported. Prespecified refits and independent score reaggregation pass. The earlier [matched-M2](results/reference_study/SCIENTIFIC_RESULT.md) and [recruitment](results/adaptive_reference/SCIENTIFIC_RESULT.md) studies remain separate evidence. These inspected synthetic mechanisms do not establish unseen-mechanism generalization.

## Current comparison status

The [controlled GP source-access pilot](results/gp_source_pilot/SCIENTIFIC_RESULT.md) compares the same native Causal-ICM GP using RCT only, OBS-conditioned trial starting predictions, or OBS+RCT. All **576 fits succeeded**: three procedures × three OBS discrepancy levels (0/1/2) × two trial budgets (128/512) × 32 paired sample seeds. At fixed `rho=0.5`, OBS data reduced mean trial-CATE MSE at discrepancy levels 0 and 1, but increased it at level 2 relative to the matched RCT-only GP at both trial budgets. OBS+RCT improved over the prior-shrunken OBS-conditioned trial prediction in all six conditions; that reference uses no RCT outcomes and is not an optimal OBS-only estimator. These are descriptive development results with unadjusted pointwise intervals. See the [prospective protocol](docs/GP_SOURCE_PILOT.md), [all results and figures](results/gp_source_pilot/README.md), and [independent checks](results/gp_source_pilot/verification/verification.json).

Version **0.3.0a4** registers 16 recipes, including an [RCT-only honest causal forest](docs/CAUSAL_FOREST.md) from official `grf`. Its default recipe passes direct-native agreement, two fresh adapter calls, an independent weighted-regression prediction calculation and two bounded cost profiles. The [admission record](results/causal_forest_admission/README.md) has 60 focused and 475 full-suite tests passing without skips; these establish implementation fidelity. Its subsequent comparative results are in the [native/default panel](results/native_panel/SCIENTIFIC_RESULT.md). Each study retains its original producer and [verification scope](docs/VERIFICATION.md).

New generic runs retain [condition- and metric-specific inference permissions](docs/INFERENCE.md). Complete comparisons retain descriptive means when uncertainty is withheld. Student-t(3) noise blocks mean-MSE uncertainty; root-PEHE has a separate policy. Native, external and custom recipes require an explicit declaration.

The completed [native/default trial-CATE collection](results/native_panel/SCIENTIFIC_RESULT.md) requests **8,448 fits** across six inspected mechanisms, RCT budgets 128/512, eleven fixed recipes and 64 seeds: **8,350 successes, 64 failures, 31 timeouts and three preserved missing attempts**. It uses the explicit C08 trial target and no extra tuning labels. Separate incomplete-collection verification reconstructs successful prediction scores and trial targets; the original complete-union contract remains unmet. All 72 joint strict-win decisions and the simultaneous precision statement are withheld. Mean risks are descriptive, and successful-only means are conditional on success. See the [complete coverage and evidence](results/native_panel/README.md), [frozen protocol](docs/NATIVE_PANEL.md), [configuration](configs/native_panel.toml) and [preflight](results/native_panel_preflight/README.md).

The new panel separates the benefit of OBS access from competitiveness against another learner. In M2 at RCT128, the GP's mean MSE decreases from 0.53675 to 0.35128 with OBS, while the RCT-only forest has mean MSE 0.18386; all three have 64 successful seeds. These descriptive values concern that condition. Only GP Full versus its matched RCT-only version isolates OBS access within an admitted common backbone; other method-versus-forest comparisons assess complete procedures.

The broader [v0.3 design](docs/V03_DESIGN.md) records 22 method or method-group entries and classifies all 31 presets by target, support, transport and implementation assumptions. Remaining native methods, matched controls, external targets and observable-data tuning extend beyond this bounded panel. The preserved v0.2 performance results remain separate.

## What is included

| Track | Available designs | Purpose |
| --- | --- | --- |
| Paper-related designs | 8 translations and 1 explicit extension from 6 research sources | Make source assumptions and adaptations explicit |
| Structural mechanisms | M0–M11 | Vary hidden treatment selection, bias geometry, effect complexity, study participation and source shift |
| Paired structural comparisons | C01–C10 | Change a stated mechanism while retaining the comparison's shared components |
| Empirical covariates | User-supplied IHDP NPZ/CSV and ACIC2016 CSV files | Hide selected covariates and generate new OBS/RCT samples with exact finite-population targets |
| Marginal sensitivity model | Constant Γ and user-defined Γ(X) | Control hidden treatment selection separately from outcomes and source shifts |

The six paper sources are CorNet, Experimental Grounding, Integrative R-learner, Elastic Integrative, power likelihood and CVCI. The Integrative R-learner normal-target extension deliberately evaluates outside the Uniform[-1,1] trial support; its original paper and code use a Uniform[-1,1] test population. Variants of one source are not counted as independent papers. See the [paper cards](docs/PAPER_DGPS.md), [coverage](docs/COVERAGE.md) and [extraction notes](docs/DGP_EXTRACTION.md).

Generic baselines include RCT-only, OBS-only and naive pooled T-learners, plus an RCT DR-learner, each with Ridge or ExtraTrees. Optional native recipes include DataFusionPPI, CorNet, Integrative R with two bases, fixed-rho Causal-ICM with two source controls, and an RCT-only honest causal forest. These are 16 registered recipes spanning variants and controls, including four fusion approaches; they are not 16 distinct algorithms. [Native settings and admission](docs/V03_ADAPTERS.md) are explicit; the included recipes are not tuned reproductions of every source paper's results.

The [empirical-data interface](docs/DATASETS.md#local-file-adapters) reads separately obtained files and defines new semi-synthetic OBS/RCT problems; it does not preserve the source experiment's treatment assignments or noisy outcomes. The [Γ/Γ(X) factory](docs/SENSITIVITY.md) preserves a specified marginal propensity and checks the sensitivity bound at evaluated covariates. Configuration files support constant and logistic Γ profiles; the Python API accepts pointwise callbacks. The factory implements one configurable construction within the marginal sensitivity model.

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

The RCT-only forest uses a separately built, pinned `grf` installation, known trial assignment probabilities and native outcome nuisances. Its [installation and build-record instructions](docs/CAUSAL_FOREST.md#source-installation-and-runtime-binding) preserve a pristine upstream checkout and bind the selected source, installed package and runtime evidence. Upstream code is not bundled.

The benchmark's MIT license covers its original code. External sources retain their own terms; public availability alone is not a redistribution license. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Evaluation contract

The common method entry uses `TrainingView` to separate OBS training, RCT training, RCT tuning and permitted outcome-free target covariates. The [GP source-access controls](docs/GP_SOURCE_PILOT.md) and [RCT-only causal forest](docs/CAUSAL_FOREST.md) provide narrower included-sample-only entries; excluded-source outcomes never enter their method arguments. Query `eval_X` is for prediction only: fit preprocessing, density models and adaptation using training data or the separate permitted `target_reference`. Latent confounders, evaluation outcomes and effect truth stay with the evaluator. This interface helps prevent accidental leakage; it is not a security sandbox.

Supported generators provide an independent target-covariate sample when `n_target_reference>0`; the default is no such sample. The separate `rct_test` is evaluator-owned and reserved for factual diagnostics. The current runner scores predictions against `eval_tau` at `eval_X`, without using RCT test outcomes. All methods within one scenario/seed receive the same realized samples. The [protocol](docs/PROTOCOL.md) and [extension guide](docs/EXTENDING.md) specify prediction shape, registration and failure handling.

The primary metric is CATE MSE, `mean((prediction - target_cate)**2)`. Root-PEHE is its square root and is reported separately. Paired comparisons use the same independent replicate seeds, with failed and missing cases counted explicitly. Approximate Student-t uncertainty requires complete pairs, at least two seed blocks and permission for that metric in every contributing condition. Permissions declare assumptions; they do not prove finite variance or reliable coverage with few seeds. See the [inference policy and configuration example](docs/INFERENCE.md) and the [protocol](docs/PROTOCOL.md) for targets, sample roles, budgets and aggregation.

Paper translations, constructed stress mechanisms and empirical-covariate benchmarks answer different questions. Empirical targets condition on the covariates actually displayed after any explicit coarsening; dropping a covariate alone need not create residual confounding under a finite empirical distribution. The same Γ treatment-selection bound can coexist with different outcome loadings, source shifts and outcome bias. See the [dataset contract](docs/DATASETS.md#local-file-adapters) and [sensitivity-model definition](docs/SENSITIVITY.md).

## Add a method or DGP

Use the public package from a separate research repository and implement `fit_predict(training, eval_X, *, seed)`. The [custom-method example](examples/run_custom_method.py) and [extension guide](docs/EXTENDING.md) show the interface. The benchmark does not need access to that research repository.

Contributions should include source attribution, explicit adaptations and checks of the target and information contract. See [CONTRIBUTING.md](CONTRIBUTING.md). Cite this software using [CITATION.cff](CITATION.cff), and cite the original papers for the designs or methods used.
