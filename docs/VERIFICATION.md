# Verification

Version `0.1.0a1`, checked 2026-10-18 on Linux x86_64 with one CPU worker, one numerical-library thread and no GPU. Installation, tests and all example executions completed in 41 seconds with a 4 GiB requested memory budget. This is a bounded reference run, not a runtime or memory guarantee for larger configurations.

## Executed checks

- A fresh virtual environment installed the package and a wheel was built.
- All **126 tests passed**, covering the fixed catalog, Gamma/Gamma(X) factory, empirical-file schemas, independent analytic truth checks, estimator information boundaries, metrics, failed/missing cells, interrupted runs and resume identity.
- Gamma checks include independent rational odds/Bayes examples, pointwise callbacks, zero-confounding limits and explicit rejection of unrepresentable bounds.
- Empirical checks use invented NPZ/CSV fixtures and independent four-row and unequal-mass conditional-mean calculations. Regressions cover subnormal propensity underflow and near-one rare-arm calibration. Input-file changes are bound to resume identity and rejected before fitting subsequent cells.
- DataFusionPPI ran from its locked official revision. Tests reject changed installed source and shadowed imports.
- CorNet ran its real training routines from the locked official checkout. Source files and added package initializers are checked; changing unused tuning labels leaves its fixed recipe unchanged.
- The source-export checker passed. Independent mathematical, source and fresh-reader reviews prompted fixes to rare-arm numerical handling, file-byte provenance, conditional targets and documentation.

| Executed configuration or recipe | Successful cells | Resume |
| --- | ---: | --- |
| `configs/quickstart.toml` | 16 / 16 | Unchanged records |
| `configs/smoke.toml` | 248 / 248 | Unchanged records |
| `configs/official_smoke.toml` | 12 / 12 | Unchanged records |
| `configs/gamma_smoke.toml` | 16 / 16 | Unchanged records |
| `examples/run_synthetic_file_smoke.py` | 16 / 16 | Unchanged records |

The custom-method example also completed. Manifests record resolved generator controls, package versions, benchmark source hashes, upstream fingerprints and empirical input hashes where applicable. The [results](../results/README.md) contain **308 successful cells**, with no failed or missing cells. Their two seeds verify execution, not scientific superiority. CorNet's official smoke uses only five source and five correction epochs.

The file-adapter recipe uses entirely invented records in IHDP/ACIC-compatible formats. **No actual IHDP or ACIC file was loaded in this verification**, and these scores are not empirical-dataset performance results. Real distributions may require explicit schema mappings or preprocessing under [the dataset contract](DATASETS.md).

## Reproduce and interpret

Install the extras and external checkout described in the main README. Run `python -m pytest`, `python scripts/check_release.py`, and each configuration or example above into a fresh output directory. Add `--resume` to a completed configured run to check record reuse. The synthetic-file example checks its own completed-run resume. Optional CorNet execution is skipped by its integration test if the external checkout is absent; the reference run included it.

The GitHub Actions workflow covers core tests, the fixed catalog, Gamma profiles and synthetic-file integration. Its external actions are pinned to commits. Optional official integration additionally requires the dependencies and checkout documented in the README.

Tests combine broad consistency checks and independent analytic examples. They do not independently verify every custom quadrature, mixture, selection or user callback. Extreme settings can fail explicit numerical or sampling constraints. These designs provide controlled mechanisms; they do not empirically validate assumptions in a real population.
