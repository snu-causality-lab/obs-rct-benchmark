# OBS–RCT Benchmark

A benchmark for estimating conditional average treatment effects (CATE) from a large observational sample and a smaller randomized trial. It provides source-documented data generators, a common observed-data interface, public-method adapters and reproducible evaluation.

This is an unreleased alpha. The initial scope is binary treatment and tabular covariates. The catalog covers multiple borrowing assumptions and failure mechanisms; it does not claim to cover every OBS–RCT setting.

## What is included

| Track | Available designs | Purpose |
| --- | --- | --- |
| Paper-related designs | 8 translations and 1 explicit extension from 6 research sources | Make source assumptions and adaptations explicit |
| Structural mechanisms | M0–M11 | Vary hidden treatment selection, bias geometry, effect complexity, study participation and source shift |
| Paired structural comparisons | C01–C10 | Change a stated mechanism while retaining the comparison's shared components |

The six paper sources are CorNet, Experimental Grounding, Integrative R-learner, Elastic Integrative, power likelihood and CVCI. The Integrative R-learner normal-target extension deliberately evaluates outside the Uniform[-1,1] trial support; its original paper and code use a Uniform[-1,1] test population. Variants of one source are not counted as independent papers. See the [paper cards](docs/PAPER_DGPS.md), [coverage](docs/COVERAGE.md) and [extraction notes](docs/DGP_EXTRACTION.md).

Generic baselines include RCT-only, OBS-only and naive pooled T-learners, plus an RCT DR-learner. Optional adapters call pinned official DataFusionPPI and CorNet implementations. Baseline settings and adaptations are explicit; the included recipes are not tuned reproductions of every source paper's results.

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

The primary metric is CATE MSE, `mean((prediction - target_cate)**2)`. Root-PEHE is its square root and is reported separately. Paired comparisons use the same independent replicate seeds, with failed and missing cases counted explicitly. Student t intervals from very few seeds should not be interpreted as reliable inferential evidence. The [protocol](docs/PROTOCOL.md) defines targets, sample roles, budgets and aggregation.

Paper translations, constructed stress mechanisms and empirical-covariate benchmarks answer different questions. The initial release contains synthetic generators. A common IHDP/ACIC variable-hiding interface and an arbitrary Γ/Γ(X) sensitivity-model factory are planned extensions. Existing hidden-confounding parameters do not constitute that general factory, and the same treatment-selection bound need not imply the same outcome bias.

## Add a method or DGP

Use the public package from a separate research repository and implement `fit_predict(training, eval_X, *, seed)`. The [custom-method example](examples/run_custom_method.py) and [extension guide](docs/EXTENDING.md) show the interface. The benchmark does not need access to that research repository.

Contributions should include source attribution, explicit adaptations and checks of the target and information contract. See [CONTRIBUTING.md](CONTRIBUTING.md). Cite this software using [CITATION.cff](CITATION.cff), and cite the original papers for the designs or methods used.
