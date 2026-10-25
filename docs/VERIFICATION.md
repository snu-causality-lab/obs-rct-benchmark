# Verification

## Current release: 0.2.0a1

Latest catalog checks completed on 2026-10-26 on Linux x86_64, Python 3.12.3, with CPU-only execution on GSDS. Scientific fitting, tests, notebooks and aggregation ran on allocated compute nodes. Worker numerical libraries used one thread; the full studies used disjoint seed shards. Configurations, package hashes, optional-method revisions and numerical-library versions are retained with the [results](../results/README.md).

- **188 tests passed** after the catalog protocol/checkers and reporting regressions were added. The scientific producer remains byte-identical to the earlier published package.
- The [full catalog study](../results/catalog_study/SCIENTIFIC_RESULT.md) records all **79,104 applicable cells**: 79,083 successes, 21 terminal DataFusionPPI calibration failures, zero missing cells; 256 nominal cells were prospectively unsupported. M11 retains descriptive summaries without mean-t inference. Four incomplete eligible contrasts withhold all 472 Holm decisions.
- All 31 separate target calculations pass (maximum difference 3.552713678800501e-15). The 309 prespecified reconstruction cases each have two fresh refits: **618/618 succeed**, with maximum direct-dot MSE difference 8.881784197001252e-16 from the saved score. Two regenerated datasets match for every preset. Original arrays were not retained, so this is not original-array byte replay or full-study prediction replay.
- Every one of the **128 completed-run resumes** leaves terminal records byte-identical. Independent code checks all 1,240 score rows and 488 supported contrast rows from compact seed metrics. Post-hoc diagnosis repeats all 21 failed fits twice and reproduces their stage/type without changing terminal records. All 34 scientific CSVs remain byte-identical after reporting/identity improvements.

The catalog execution uses one CPU/4 GiB/2h per shard and up to 16 concurrent single-thread workers; its first 1,200-fit shard took 3m47s with 358,824 KiB maximum resident size. Exact environment and source identities are published with the results. Deterministic reconstruction and arithmetic consistency are distinct from independent estimator validation.

## Earlier static and recruitment studies

These separate studies completed on 2026-10-25; their seeds and inference are not pooled with the full catalog.

- **159 tests passed**, including the original generator/adapter checks and new budget-grid, legacy-config-hash, condition-report, sequential information/cost, failure/replay and independent aggregation regressions.
- The distributable source check, wheel and source archive build passed. Five existing notebooks executed in separate kernels; their original alpha stored outputs remain historical teaching examples.
- The [static reference study](../results/reference_study/SCIENTIFIC_RESULT.md) completed **3,072/3,072 cells**, including 512 full CorNet fits at 500+200 epochs. All 64 prespecified primary contrasts are retained, with one joint Holm family. Aggregation checks exact configuration, disjoint seed coverage, source/environment fingerprints and terminal metrics.
- All 96 fits in the first prespecified static seed were separately refitted and scored using a direct dot product. Maximum absolute difference from the saved MSE was **2.220446049250313e-16**, below the fixed 1e-12 absolute/relative replay tolerance. The other 31 seeds' fitted predictions were not replayed; their saved metrics were authenticated and reaggregated.
- The [adaptive reference study](../results/adaptive_reference/SCIENTIFIC_RESULT.md) completed **1,152/1,152 cells**. All acquired indices, costs, policy orders, final predictions and scores replayed. Policy comparisons hold the final estimator fixed. The separate two-seed CI smoke completed and replayed 24/24 cells.
- No failed or missing cells occurred in these two earlier studies. Failure paths remain covered by explicit tests; terminal failures are never silently retried to obtain a favorable score.

Full-study workers requested one CPU and 2 GiB RAM per shard; Slurm allocated two logical CPUs per task. Static one-seed shards took about 34–41 seconds and adaptive four-seed shards about 58–60 seconds including resume and replay. The first static fit process recorded 344,492 KiB maximum resident size with GNU time. These are measurements of bounded configurations, not runtime or memory guarantees for larger studies. Environments, complete operational logs and upstream checkouts stay on the execution server; only compact scientific records and selected assets are published.

## What the checks establish

Truth is derived from declared equations rather than fitted methods. Independent checks include binary-state Bayes calculations, rational odds examples, finite-population conditional means and hand-calculated score/aggregation tests. Sequential tests vary purchased outcomes while holding covariates fixed, check hidden-label access boundaries, reject repeated or excessive queries, preserve costs after policy failure, and reject tampered failed-record costs and incomplete shards.

Official DataFusionPPI is bound to its locked installed revision, complete source hashes and import resolution. CorNet uses real training routines from a separate clean locked checkout; modified source and added package initializers are rejected. Changing unused tuning labels leaves the fixed CorNet recipe unchanged. External code retains its own terms and is not bundled.

Fresh sample seeds provide sampling replications conditional on development-inspected mechanisms. Deterministic replay is not an independent scientific replication. No actual IHDP or ACIC file was loaded in these checks, and no unseen-mechanism challenge was performed. Empirical adapters' invented file fixtures are schema/target tests, not empirical-dataset performance results. Extreme custom quadrature, mixture, selection or callback settings still need independent numerical checks beyond input validation.

## Reproduce

Install the extras and official checkout described in the main README; analysis uses `.[analysis]`. Run `python -m pytest`, `python scripts/check_release.py`, the [static study commands](REFERENCE_STUDY.md#run-and-reproduce), and the [adaptive reproduction and replay commands](../results/adaptive_reference/SCIENTIFIC_RESULT.md#reproduce-and-verify). Completed-run resume requires identical source, configuration and numerical environment. One source archive includes the configurations, analysis scripts, compact records and text-based figure assets; vector manuscript PDFs can be regenerated with `--paper-assets`.

GitHub CI also independently verifies the published full-catalog CSV arithmetic. It runs core tests, catalog/Gamma/file-fixture examples, the 24-cell adaptive smoke and replay, and all default notebooks. It does not rerun the full reference campaigns on every push. Optional official integration tests require the documented dependencies and checkout.

## Preserved 0.1.0a1 execution evidence

The 2026-10-18 reference run passed **126 tests**, built a wheel and completed installation plus examples in 41 seconds with one CPU worker/thread, no GPU and 4 GiB requested RAM. The [308 preserved example cells](../results/README.md#original-alpha-execution-examples) are unchanged:

| Configuration or recipe | Successful cells | Completed-run resume |
| --- | ---: | --- |
| `configs/quickstart.toml` | 16 / 16 | Unchanged records |
| `configs/smoke.toml` | 248 / 248 | Unchanged records |
| `configs/official_smoke.toml` | 12 / 12 | Unchanged records |
| `configs/gamma_smoke.toml` | 16 / 16 | Unchanged records |
| `examples/run_synthetic_file_smoke.py` | 16 / 16 | Unchanged records |

Those two-seed examples establish execution, not method ranking. CorNet used only 5+5 epochs. The custom-method example completed, and the five notebooks additionally executed 47 code cells and 11 figures in 15 seconds after dependency installation. Their optional real-data branches stayed disabled. File fixtures were entirely invented IHDP/ACIC-compatible records. The original manifests and [notebook execution evidence](../notebooks/execution.json) preserve their own source and environment bindings.
