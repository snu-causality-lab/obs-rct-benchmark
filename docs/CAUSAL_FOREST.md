# RCT-only honest causal forest

Version **0.3.0a4** admits the default RCT-only honest causal-forest recipe after direct-native agreement, fresh-process repeats and an independent prediction calculation. The [execution record](../results/causal_forest_admission/README.md) retains the six fits, source/environment identity and bounded cost measurements. This establishes implementation fidelity, without a comparative accuracy ranking or source-paper table reproduction. The registered name is `rct_causal_forest`; the implementation is [`RCTCausalForest`](../src/obs_rct_benchmark/causal_forest.py).

## Native recipe and information use

The adapter calls official `grf::causal_forest` and its standard CATE prediction method at revision [`5bee99b51471f76cb2d63acbc8a9b0ffec408ba0`](https://github.com/grf-labs/grf/tree/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0), package version **2.6.1**. Binary randomized treatment identifies the fitting population's CATE under the usual consistency and overlap assumptions; an external target additionally requires conditional effect agreement and covariate support. Admission checks implementation fidelity, not these assumptions.

`fit_predict(training, eval_X, *, seed)` reads only `training.rct_train`. The narrower `fit_predict_sample(sample, eval_X, *, seed)` accepts a `SourceSample` with source `rct` and role `rct_train`. Both use its `X`, `T`, `Y` and `known_propensity`; both treatment arms must be present. OBS data, RCT tuning outcomes and target-reference covariates are not accessed. Every fitted RCT outcome remains charged to the existing budget. `eval_X` enters prediction only; it does not fit preprocessing, nuisances or trees.

| Setting | Fixed default recipe |
| --- | --- |
| Treatment nuisance | `W.hat=sample.known_propensity`; no propensity forest |
| Outcome nuisance | `Y.hat=NULL`; upstream honest regression forest with OOB training predictions, 500 trees at the default causal-tree count |
| Causal trees and threads | 2,000 trees; one thread for fitting and prediction |
| Sampling and honesty | `sample.fraction=0.5`, `honesty=TRUE`, `honesty.fraction=0.5`, `honesty.prune.leaves=TRUE` |
| Splitting | `mtry=min(ceiling(sqrt(p)+20),p)`, `min.node.size=5`, `alpha=0.05`, `imbalance.penalty=0`, `stabilize.splits=TRUE` |
| Other native options | No sample weights or clusters; `equalize.cluster.weights=FALSE`, `ci.group.size=2`, `compute.oob.predictions=TRUE` |
| Tuning and prediction | `tune.parameters="none"`; standard `predict(..., newdata=eval_X, linear.correction.variables=NULL, estimate.variance=FALSE)` |
| Reproducibility | Accept a uint32 benchmark seed; native seed is `seed % 2147483647`. Set both R's seed and the native seed, with `grf.legacy.seed=FALSE` and `grf.verbose=FALSE`. |

The outcome nuisance receives the same native seed and uses `max(50, num.trees/4)` trees. The class exposes `num_trees`, `num_threads` and `timeout_seconds`, with defaults 2,000, one and 600. Tree counts must be positive multiples of four, an adapter restriction that avoids paired-tree rounding and a fractional nuisance-tree count. This admission covers the defaults; changed options require separate checks. The pinned [fitting source](https://github.com/grf-labs/grf/blob/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0/r-package/grf/R/causal_forest.R) and [seed options](https://github.com/grf-labs/grf/blob/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0/r-package/grf/R/grf_options.R) specify the upstream behavior.

## Source installation and runtime binding

The official [package requirements](https://github.com/grf-labs/grf/blob/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0/r-package/grf/DESCRIPTION) are R >=3.5.0, C++17 and GNU make, with Rcpp/RcppEigen and the declared R imports. Preserve a complete pinned checkout for verification and build from a separate full archive with its symlinks intact. Copying only `r-package/grf` loses the linked core and bindings; compiling in the pristine checkout can leave intermediates that fail its clean-source check.

Environment selection:

| Variable | Meaning |
| --- | --- |
| `GRF_SOURCE` | Required complete, clean Git checkout at the pinned revision |
| `GRF_LIBRARY` | Required dedicated R library containing the recorded `grf` installation |
| `GRF_BUILD_MANIFEST` | Required JSON build record binding the selected source, installed files and recorded runtime evidence |
| `GRF_RSCRIPT` | Recommended explicit `Rscript` executable from that environment; otherwise resolved from `PATH` |
| `GRF_ARTIFACT_ROOT` | Optional existing private directory retaining each bridge's inputs, script, output and logs |

First obtain an allocated CPU job and provision the selected R/compiler environment with the declared R dependencies. The commands below build and record `grf`; they do not provision that environment. Activate it, set and export the required variables to absolute server paths, and set `GRF_RSCRIPT` explicitly to its executable for the recorder. From the benchmark repository root, use new source, build and manifest destinations:

```sh
set -eu
export GRF_SOURCE GRF_LIBRARY GRF_BUILD_MANIFEST GRF_RSCRIPT
GRF_REVISION=5bee99b51471f76cb2d63acbc8a9b0ffec408ba0
test ! -e "$GRF_SOURCE"
test ! -e "$GRF_BUILD_MANIFEST"
test ! -e build/grf-source.tar
git clone https://github.com/grf-labs/grf.git "$GRF_SOURCE"
git -C "$GRF_SOURCE" checkout --detach "$GRF_REVISION"
mkdir -p build "$GRF_LIBRARY"
mkdir build/grf-source
git -C "$GRF_SOURCE" archive "$GRF_REVISION" --output="$PWD/build/grf-source.tar"
tar -xf build/grf-source.tar -C build/grf-source
cat > build/grf-build-command.sh <<'SH'
R CMD INSTALL --preclean --clean --library="$GRF_LIBRARY" build/grf-source/r-package/grf
SH
sh build/grf-build-command.sh
python scripts/record_causal_forest_build.py \
  --source "$GRF_SOURCE" --library "$GRF_LIBRARY" --rscript "$GRF_RSCRIPT" \
  --build-command-file build/grf-build-command.sh --output "$GRF_BUILD_MANIFEST"
```

[`record_causal_forest_build.py`](../scripts/record_causal_forest_build.py) records a completed build; it does not compile or install. It requires an allocated job and a new, nonexistent manifest path. Schema version 1 records revision/tree, source roots, complete tracked-file hashes and symlink targets, the completed build command, all installed `grf` files, Rscript hash/R version, canonical dependency versions, library roots and selected dependency R-code/data/native-library hashes. The adapter independently compares source files to raw Git blobs, verifies installation/runtime identities and repeats checks after execution; missing or changed identities fail closed.

This binds observed bytes and the supplied completed-build assertion. It does not prove compiler correspondence or authenticate the whole R runtime and transitively loaded system libraries. The manifest contains private operational paths and remains with the build. The recorded environment uses R 4.3.3, conda-forge GCC 16.2.0, DiceKriging 1.6.1, zoo 1.9-1, lmtest 0.9-40 and sandwich 3.1-3, with the remaining identities retained in its build record.

## Admission protocol

[`scripts/admit_causal_forest.py`](../scripts/admit_causal_forest.py) runs only in an allocated server job. Set `GRF_ARTIFACT_ROOT` to retain the complete bridge evidence; otherwise temporary bridge directories are removed. The script writes the observed input fixture, input hash, driver/source identity, seeds and tolerances before fitting. It uses a fixed numerical fixture with known, nonconstant assignment probabilities and no evaluator truth. The default plan contains six fits:

1. One independently assembled direct native call with 128 RCT rows, five covariates and eight prediction queries.
2. Two fresh adapter processes on the identical inputs and seed: one through `fit_predict_sample`, one through a typed `TrainingView` that raises if an excluded field is read.
3. One fresh adapter fit after appending two fixed distant query rows; the original eight predictions must agree.
4. Two cost profiles using 128 and 512 RCT rows, five covariates and 2,048 prediction queries each.

Direct-call, fresh-process and query-extension agreement use absolute tolerance `1e-10` and relative tolerance `1e-8`. The independent prediction calculation below uses absolute tolerance `1e-12`. Provenance must remain unchanged. Failures retain a terminal failure record and traceback; observed arrays, native fitted forest and complete process logs stay on the server.

After selecting the recorded environment and an existing private artifact directory, run inside the allocated server job with a new output directory:

```sh
python scripts/admit_causal_forest.py --output "$GRF_ARTIFACT_ROOT/admission-v1"
```

The profiles record elapsed time for the full adapter call, including provenance checks, serialization and process startup. They also record child CPU time and the largest observed child-process RSS. That RSS is not simultaneous whole-job peak memory. The run requests one CPU, 4 GiB RAM and 20 minutes; native calls have a 600-second timeout and adapter workers a 660-second timeout. Profiles are descriptive fixtures, without a prespecified statistical cost-acceptance threshold or a forecast of campaign costs.

## Recorded admission

The [0.3.0a4 record](../results/causal_forest_admission/README.md) passes 60 focused and 475 full-suite tests without skips, with CorNet and the R namespace probe configured. All six native fits complete. Direct-native versus each fresh adapter has maximum absolute prediction difference zero; appending distant queries also leaves the original eight predictions unchanged. Independent weighted-intercept reconstruction has maximum absolute error **2.220446049250313e-15**, below the prespecified `1e-12` limit.

The two default-recipe profiles use five covariates and 2,048 queries:

| RCT training rows | Full adapter wall time | Child CPU time | Largest child RSS |
| --- | ---: | ---: | ---: |
| 128 | 7.101 s | 2.601 s | 267,928 KiB |
| 512 | 8.305 s | 3.586 s | 295,756 KiB |

These are individual fixture measurements. Full precision and producer identity remain in the [machine-readable summary](../results/causal_forest_admission/summary.json). They do not establish an accuracy ranking, universal runtime bound or simultaneous whole-job memory peak. The frozen catalog, GP pilot, other scientific results and integrated report retain their original producers and conclusions.

## Independent prediction calculation

For each query, the direct call exports native forest weights and fitted `Y.hat`/`W.hat`. Let normalized weights be `a_i`, outcome residuals `y_i=Y_i-Y.hat_i` and treatment residuals `w_i=T_i-W.hat_i`. Independently implemented scalar arithmetic reconstructs the slope of weighted regression with an intercept:

```text
tau = [sum(a*y*w) - sum(a*y)*sum(a*w)]
      / [sum(a*w*w) - sum(a*w)^2]
```

The checker requires finite inputs, nonnegative weights summing to one, and positive local treatment variance. The native [causal binding](https://github.com/grf-labs/grf/blob/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0/r-package/grf/bindings/CausalForestBindings.cpp) uses treatment as instrument; its [prediction strategy](https://github.com/grf-labs/grf/blob/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0/core/src/prediction/InstrumentalPredictionStrategy.cpp) implies this formula. A hand-computable fixture protects the checker:

```text
a     = [0.1, 0.2, 0.3, 0.4]
T     = [0,   1,   0,   1]
W.hat = [0.5, 0.5, 0.5, 0.5]
Y     = [0,   3,   1,   5]
Y.hat = [0,   1,   1,   2]
```

The slope is `8/3`. Omitting the intercept gives `3.2`; raw weighted arm means give `43/12`. This check is independent arithmetic conditional on upstream neighborhoods and nuisances. It does not independently verify tree training, establish CATE accuracy or require a noisy fit to recover a chosen truth.

Native confidence intervals are not requested. Benchmark uncertainty remains subject to the separate [metric-specific inference policy](INFERENCE.md): native recipes require explicit assumptions, and source fidelity does not establish score moments, unseen-mechanism performance or interval coverage.

With the recorded native environment selected, `python -m obs_rct_benchmark.cli run --config configs/causal_forest_smoke.toml --output outputs/grf-smoke` checks the common runner on four cells. The smoke and byte-preserving resume passed; its two seeds check execution rather than method ranking. The native recipe remains descriptive without a justified metric declaration.
