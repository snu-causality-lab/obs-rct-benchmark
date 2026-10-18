# Public DGP extraction

The public catalog contains nine source-related variants (eight translations and one labeled normal-target extension) and the M0–M11/C01–C10 structural presets, with a common training/evaluation data contract.

The package implements the mathematical mechanisms in NumPy. It does not load a private checkout, download executable code, or require PyTorch to generate data. Public source provenance and upstream licensing are recorded separately in the source documentation.

## Data contract

`SimulationProblem` is the evaluator's object. It contains the four observed samples, evaluation covariates and CATE, metadata, and optional `SimulationTruth`. Call `problem.training_view()` to obtain the estimator's input:

| Field | Contents |
| --- | --- |
| `obs_train` | Observational `X`, binary `T`, and `Y` |
| `rct_train` | Randomized training sample and known assignment probability |
| `rct_tune` | Independent randomized tuning sample and known assignment probability |
| `target_reference` | Optional outcome-free target covariates |

The view omits randomized test outcomes, evaluation labels, metadata, seeds, hidden variables, true observational propensities, and structural truth. Observational assignment probabilities are rejected by `SourceSample`; known RCT propensities may be nonconstant. Arrays are copied into finite float64 buffers and marked read-only. This protects against accidental mutation; it is not a security boundary against a malicious Python extension.

`SimulationTruth` stores target arm means and CATE, observational conditional bias `c_t`, source potential-outcome shift `d_t`, and total observational regression discrepancy `r_t = c_t + d_t`. Hidden RCT participation presets also store the enrolled trial's arm means and CATE. These quantities are for evaluation and explicitly declared diagnostics.

## Public entry points

```python
from obs_rct_benchmark import (
    generate_paper_problem,
    generate_structural_problem,
    make_paper_config,
    make_structural_addendum_config,
    make_structural_config,
)

cfg = make_structural_config(name="example", mechanism="M2", n_obs=1000)
problem = generate_structural_problem(cfg, seed=11)
training = problem.training_view()

paper_cfg = make_paper_config(
    name="grounding", paper_id="experimental_grounding_kps"
)
paper_problem = generate_paper_problem(paper_cfg, seed=11)

contrast_cfg = make_structural_addendum_config(name="contrast", cell="C01")
```

`StructuralDGPConfig(name="custom", ...)` exposes the generic mechanisms directly. The controls cover hidden treatment selection, arm-specific hidden outcome loading, latent severity, observable severity, source outcome shifts, covariate and study selection, hidden RCT participation, CATE shape, covariate marginals, and noise. `structural_truth_at(X, cfg)` and `paper_truth_at(X, cfg)` return evaluation-only truth at supplied covariates. Each required sample role has at least four rows; structural generators require `d >= 3`, while paper-specific dimensions are checked by `PaperDGPConfig`.

## Numerical preservation and disclosed changes

For M0–M11 and C01–C10, the extraction retains the preset coefficients, sampling equations, five independent `SeedSequence.spawn(5)` role streams, quadrature rules, and evaluation CATE definition. The eight non-CorNet source-related variants retain their equations and role-specific streams. Fixed-size conditional trial sampling, distinct power-likelihood paper/supplement variants, and full-covariate power-likelihood CATE remain benchmark translation conventions; consult each paper card before interpreting a result.

`integrative_r_learner_kps_code` uses the original paper and released code's `Uniform[-1,1]` evaluation target. `integrative_r_learner_kps_normal_target_extension` deliberately uses a Gaussian evaluation target while keeping the same outcome equations. Its normal target is a benchmark extension, not a paper/code disagreement.

The CorNet variant retains the simulation architecture and scaling rules: three square representation layers, tanh after the first two layers, linear base and correction heads, normalized base-head weights, correction-weight sparsification excluding coordinate zero, randomized treatment in both sources, and observational outcomes that omit the target correction. It therefore represents source outcome shift, with zero hidden treatment-selection bias.

CorNet parameter values deliberately differ from the official implementation. A local NumPy PCG64 generator replaces the mixed PyTorch/legacy NumPy initialization and upstream setup-call order. Parameters use uniform initialization bounds `[-1/sqrt(d), 1/sqrt(d)]`; base head weights are then normalized. Sparsification follows initialization. Correction CATE RMS is set using 10,000 independent standard-normal covariate rows from `SeedSequence(mechanism_seed, spawn_key=(0x434F524E,))`. This is a deterministic architecture translation, not a bitwise replay of official PyTorch parameters. The original public scripts did not set their RNG seeds.

The generic structural generator optionally draws an independent outcome-free target-reference sample with `n_target_reference > 0`. Its RNG uses `SeedSequence(sample_seed, spawn_key=(0x54415247,))`; changing that row count does not advance the five existing streams. The public structural presets default to zero reference rows.

Generator metadata uses version `2.0.0` and serializes the public configuration fields. Rejection samplers stop with an explicit failure if they exceed `max(1,000,000, 20,000*n)` proposals; extreme custom selection settings can trigger this limit. The `clean_fraction` controls support both endpoints zero and one.

## Verification and limits

Run `python -m pytest tests/test_dgps.py` after installing the test extra described in the README. The tests cover every public preset, reproducibility, data immutability, omitted evaluation roles, an independent two-point hidden-variable calculation, closed-form KPS and power-likelihood CATE values, and matched structural controls. CorNet norm, sparsity, and reference-RMS tests check its declared numerical contract; they do not establish bitwise agreement with upstream software.

The structural density ratios use evaluation-sample normalization, as in the source translation. Gaussian hidden-variable expectations use deterministic quadrature; finite quadrature error remains possible, especially for extreme custom treatment loadings. A paper-source translation is not independent empirical evidence for its associated estimator. Training-view separation prevents labels from entering through the supplied data object; extension authors remain responsible for their own external data access.
