# Adding a method or mechanism

Keep additions reproducible from public sources and explicit about the target population, information available to the estimator, and numerical truth. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the repository workflow and [the protocol](PROTOCOL.md) for data roles and scoring.

## A method adapter

Implement the `CATEMethod` protocol in [`methods.py`](../src/obs_rct_benchmark/methods.py):

```python
class MyMethod:
    def fit_predict(self, training, eval_X, *, seed):
        # Fit using observed TrainingView data; return one CATE per eval row.
        ...
```

Register a factory through `register_method("my_method", MyMethod, provenance="public source URL and exact version")` before calling the runner. Supply method options through the run configuration. Registration is an in-process Python API; the CLI does not execute a module merely because it appears in a configuration file.

The adapter may inspect `obs_train`, `rct_train`, `rct_tune`, and an explicitly supplied covariate-only `target_reference`. It receives evaluation covariates for prediction, never evaluation CATE, hidden variables, OBS oracle propensities, test outcomes, or a `SimulationTruth` object. Do not close over these objects. Declare which roles are fitted, tuned, or unused. Preserve the supplied trial probabilities; reject unsupported nonconstant probabilities rather than silently replacing them by 0.5.

Use fixed options or a disclosed observed-data tuning rule. Seed every stochastic component under your control and state limitations of an upstream dependency's RNG handling. Return finite one-dimensional predictions with exactly one row per evaluation covariate. Raise an informative error when assumptions or dependency requirements fail so the runner can preserve the failure record.

An official implementation should be a separately installed dependency pinned to a public revision, with provenance checked before import where feasible. A translation must be labeled as a translation and must not claim official implementation equivalence without evidence. Add tests of the data boundary, malformed predictions, unsupported settings, deterministic behavior within the supported environment, and the adapter's call to the intended public API.

## A structural configuration

For existing equations, instantiate `StructuralDGPConfig(name="custom_setting", mechanism="custom", ...)` or use a preset factory with explicit overrides. This already permits configurable hidden confounding. Save the full resolved configuration and give the setting its own descriptive name. Do not present an overridden M/C preset as the unchanged canonical contrast.

For new equations, provide sampling and target-truth calculations together. Define the target law, OBS law, trial-enrollment law, treatment assignment, both potential outcomes, and all hidden-variable distributions. State whether the source conditional causal effect transports. Keep hidden-selection bias distinct from source-specific potential-outcome shift and trial-participation shift. Add dimension and parameter validation that rejects undefined settings.

Check zero-confounding and zero-shift limits, analytic special cases, arm-swap behavior where applicable, and identities connecting arm discrepancies to CATE discrepancy. If truth uses quadrature or another approximation, compare against an independent calculation and document convergence over supported parameter ranges. Use a separate sampling or integration calculation for verification, rather than deriving expected values by calling the same truth helper under test. Substantive simulations and verification should run on appropriately provisioned compute; local examples should remain tiny.

## A published DGP or dataset adapter

Add a primary paper citation and a card in [PAPER_DGPS.md](PAPER_DGPS.md), including the exact paper/code version, supported dimensions, target versus source populations, confounding versus outcome shift, truth derivation, and every translation convention. If paper and code disagree in a way that changes the estimand or DGP, give them distinct variant IDs. Update [sources.lock.toml](../sources.lock.toml), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), [CITATION.cff](../CITATION.cff), and [COVERAGE.md](COVERAGE.md) as applicable.

Public availability alone does not grant code or dataset redistribution rights. Check the pinned license and required notices. Prefer an original equation translation or an external installation/download step when bundling rights are absent or unclear. A new dataset adapter must specify access terms, preprocessing, treatment/outcome definitions, target population, split independence, and whether CATE truth exists. Semi-synthetic truth and factual outcomes must remain distinct.

IHDP/ACIC adapters and a general `Gamma(x)` factory are not implemented by merely accepting a file path or a callback. Each needs its own definition, validation, role split, provenance, and independent checks before coverage claims are expanded.
