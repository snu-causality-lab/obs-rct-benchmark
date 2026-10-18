# Verification of the initial candidate

Version `0.1.0a0`, checked 2026-10-18 on Linux x86_64 with one CPU worker, one numerical-library thread and no GPU. The complete verification job took 35 seconds, including installation, tests and example execution. The requested memory budget was 4 GiB. This is a reference run, not a runtime or memory guarantee for larger configurations.

## Executed checks

- The package installed into a fresh virtual environment using existing numerical dependencies, and a wheel was built.
- All **67 tests passed**, including every DGP preset, independent analytic truth checks, information boundaries, metrics, missing/failure coverage, interrupted runs and resume identity.
- DataFusionPPI ran from the locked official revision. Tests reject changed installed source and shadowed imports even when installation metadata still names the expected version.
- CorNet ran from the locked official checkout through its real training routines. Its source files are hashed, added package initializers are rejected, and unused tuning-label changes do not affect its fixed recipe.
- The source-export checker passed. Independent source and scientific reviews found and corrected the Integrative R-learner target attribution, endpoint handling, rejection-sampling limits, an M11 axis label and provenance gaps. A separate reader prompted an explicit prediction-only rule for query covariates.

| Executed configuration | Successful cells | Resume |
| --- | ---: | --- |
| `configs/quickstart.toml` | 16 / 16 | Unchanged records |
| `configs/smoke.toml` | 248 / 248 | Unchanged records |
| `configs/official_smoke.toml` | 12 / 12 | Unchanged records |

The custom-method example also completed. Example manifests record resolved DGP controls, package versions, benchmark source hashes and requested official implementation fingerprints. The result files are [here](../results/README.md). Their two sample seeds verify execution coverage, not scientific superiority. The official smoke uses only five source and five correction epochs for CorNet.

## Reproduce and interpret

Install the extras and external checkout described in the main README. Run `python -m pytest`, `python scripts/check_release.py`, and each configuration into a fresh output directory. Add `--resume` to a completed run to check its identity and record reuse. Optional CorNet execution is skipped by its integration test if the external checkout is not configured; the run described above included it.

The prepared GitHub Actions workflow covers core tests and the full catalog smoke. It has not run on GitHub yet. Its external actions are pinned to commits rather than floating tags.

The tests combine broad consistency checks with narrower independent analytic examples. They do not independently verify every custom quadrature, mixture or selection configuration. Extreme custom settings can fail the stated sampling budget. These synthetic designs provide controlled mechanisms, not empirical validation of assumptions in a real population.
