# Static reference comparison

This study compares fixed public recipes under controlled observational bias and sample budgets. Its protocol and [complete configuration](../configs/reference_study.toml) are declared before inspecting its results. Earlier two-seed smoke outputs remain execution examples. This study does not use private methods or private comparison results.

## Question and fixed design

Does the value of observational data change with hidden treatment selection, source-specific treatment effects and the sizes of the observational sample and trial? Four matched variants of the existing M2 generator vary two controls:

| Label | Hidden-treatment coefficient | OBS CATE shift |
| --- | ---: | ---: |
| `clean` | 0 | 0 |
| `confounded` | 1.5 | 0 |
| `shift` | 0 | 1 |
| `both` | 1.5 | 1 |

All remaining M2 equations and parameters are fixed, including five observed covariates, the smooth CATE and confounding geometry, outcome loadings, noise and signal seed 2025. No trial or target covariate shift is introduced. Changing the hidden-treatment coefficient changes treatment assignment, not the target effect equation. Changing `obs_cate_shift` changes OBS outcomes, not trial or target outcomes. The four cases are controlled variants of one development-inspected mechanism, not four external replications or an unseen-mechanism challenge. They do not isolate arbitrary differences among the catalog's M IDs.

Each case is crossed with OBS sizes 512 and 2,048 and RCT training sizes 128 and 512: **16 conditions**. Each condition has 32 independent sample seeds, 10000–10031, and 2,048 independent target evaluation rows per seed. The mechanism seed is fixed. Within a condition/seed, methods receive identical observed samples and evaluation covariates. Across budget conditions the same sample seed provides a pairing convention; differently sized samples are not claimed to be nested.

## Data and computation budgets

| Role | Rows per condition/seed | Use |
| --- | ---: | --- |
| OBS training | 512 or 2,048 | Available to OBS/fusion recipes |
| RCT training | 128 or 512 | Available to all trial/fusion recipes |
| RCT tuning | 32 | Reserved, accessible, but unused by every fixed recipe in this study |
| RCT factual test | 64 | Evaluator-only; unused by fitting and the CATE score |
| Target evaluation | 2,048 | Prediction and evaluator-only CATE scoring |
| Target reference | 0 | No adaptation on target-reference or evaluation covariates |

The declared available randomized-label budgets are therefore **160 and 544**, including the reserved tuning labels. Actual fitting uses 128 or 512 labels for every trial/fusion recipe; no method gains extra tuning outcomes. The evaluator's 64 unused factual test labels are recorded separately and excluded from the learning budget. This is a static comparison, not a recruitment policy. Its budget cannot be equated to a sequential recruitment budget without declaring that track's selection and fitting roles.

Six recipes give **3,072 requested fits**:

- RCT-only T-learner and two-fold DR-learner with ExtraTrees.
- OBS-only and naive pooled T-learners with the same ExtraTrees defaults: 100 trees, minimum leaf size five, all features, one worker.
- Official CorNet adapter: 500 source epochs, 200 correction epochs, width 25, learning rate .1, source outcome penalty .1, correction penalty zero, OBS-fitted standardization. There is no early stopping or search. These are the disclosed full adapter defaults, not a claim to reproduce a source paper's table.
- Official DataFusionPPI adapter: fixed DR learner and covariate-shift handling enabled, including the pinned upstream implementation's internal RCT splits. The external tuning sample is not passed.

There are 512 CorNet fits. A single numerical thread and CPU-only execution are sufficient; parallelism should be across disjoint seed shards. Before the final campaign, measure one fixed seed with the same full recipes to estimate time and memory. That timing seed may belong to the declared grid and must be retained, not selected by accuracy. A provisional request of one CPU worker, 4–8 GiB RAM and a two-hour limit per four-seed shard is a planning bound, not a measured guarantee. Record actual environment, source/config hashes, job resources, observed runtime/memory and failures with the final results. Do not reduce epochs in response to accuracy or to label a smoke run as this study.

## Analysis fixed before results

The primary score is per-replicate target CATE MSE. Tables report each condition separately, with mean MSE, its Monte Carlo standard error, an approximate two-sided 95% Student-t interval over the 32 independent seeds, and successful/expected counts. Root-PEHE is a separate descriptive score. Raw errors from different mechanisms are not collapsed into a universal ranking.

The **64 prespecified primary contrasts** are both official fusion methods minus each of the two nonlinear RCT-only controls in all 16 conditions. Negative differences favor fusion. Each difference is paired within a condition/sample seed. Two-sided paired t tests are adjusted jointly with Holm's procedure over all 64 contrasts at .05. Pointwise 95% intervals remain explicitly pointwise; a star in the figure denotes the Holm decision, not simultaneous interval coverage. All favorable, adverse and unresolved contrasts are reported. No contrast is chosen after looking at results.

Any failure or missing cell is retained. Condition-level means may describe successes with their coverage stated. A primary contrast needs every requested pair; if any primary contrast is incomplete, family-level Holm decisions are withheld rather than silently testing a smaller family. Neither evaluation rows nor conditions are counted as independent seed replicates.

Thirty-two seeds are a fixed, bounded reference campaign, not a promise of narrow intervals. Report achieved precision: for paired seed differences with sample standard deviation `s`, Monte Carlo standard error is `s/sqrt(32)` and the pointwise half-width is `t_31,.975*s/sqrt(32)`. No accuracy-driven extension or stopping is performed. Wide intervals are an outcome, not permission to add selected replications. Subsequent higher-precision or external-challenge studies need a separate declared protocol.

The [analysis script](../scripts/analyze_reference_study.py) independently reconstructs scores' seed summaries and contrast statistics from terminal JSONL records, verifies the complete declared grid and shard/config/source identity, and writes condition tables, all 64 contrasts and two figures. It does not replay fitted predictions: the static runner does not retain prediction arrays. Generator truth has existing independent analytic checks, while the study's aggregation has separate hand-calculated regression tests. Recomputing JSON summaries alone is not independent validation of every fitted learner or DGP.

## Run and reproduce

Run computation on an appropriate server. The commands below are project-relative; select actual CPU/RAM/time allocations and installed optional dependencies before submission.

```sh
python scripts/make_reference_configs.py --config configs/reference_study.toml --output outputs/reference-configs --shards 8
obs-rct-benchmark run --config outputs/reference-configs/shard-00.toml --output outputs/reference/shard-00
# Run each of shard-01 through shard-07 once, then aggregate the complete union.
python scripts/analyze_reference_study.py --config configs/reference_study.toml --runs outputs/reference/shard-* --output outputs/reference-analysis
```

Install `python -m pip install -e '.[baselines,cornet,datafusionppi,analysis]'` and obtain the documented locked clean CorNet checkout. The `analysis` extra provides SciPy and Matplotlib. Failed records are terminal under `--resume`. Preserve unsuccessful attempts and do not overwrite results with favorable reruns. Full environments, logs and models stay on the execution server; publish compact records, tables, figures and sanitized provenance only.
