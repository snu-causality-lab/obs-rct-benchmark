# Two-stage residual (TSR) static study

This protocol and its [configuration](../configs/tsr_study.toml) were fixed on 2026-11-13, before any TSR recipe was fitted to a catalog preset. The study adds one estimator, not an acquisition rule.

## Question

[Gao et al. (2026)](https://arxiv.org/abs/2603.03785v1) pair an acquisition rule (R-EPIG) with a two-stage residual estimator (TSR). An observational learner is frozen as an offset, and a Gaussian process (GP) learns the residual from trial outcomes. Their comparisons change the estimator and the acquisition rule together. This study asks about the estimator alone, under uniform trial sampling, across the 31-preset catalog:

1. **Within one learner.** Does adding the OBS offset lower CATE error relative to the same GP fitted to trial outcomes only?
2. **Against the existing control.** Does TSR lower CATE error relative to the catalog's RCT-only ExtraTrees T-learner?

## Recipes

| Recipe | OBS | RCT | Prediction |
| --- | --- | --- | --- |
| `tsr_tabpfn_lmcgp` | Stage 1 | residual GP | `tau_obs(x) + E[tau_residual(x)]` |
| `lmcgp_rct_only` | — | GP on raw outcomes | `E[tau(x)]` |
| `tabpfn_obs_only` | Stage 1 | — | `tau_obs(x)` |
| `rct_t_extra_trees` | — | T-learner | Catalog settings: 100 trees, minimum leaf 5 |

- **Stage 1.** A TabPFN v2 regressor (`tabpfn==2.0.9`, default settings, `random_state=0`) is fitted as an S-learner on raw OBS covariates and treatment, then frozen.
- **Stage 2.** A two-output GP over (covariates, arm) uses a linear model of coregionalization with two RBF components with ARD. Each component has an arm covariance `w w' + diag(kappa)`, and the noise is shared. Covariates are standardized with the fitted trial rows' means and scales, and outcomes within the fit. Hyperparameters maximize the marginal likelihood (L-BFGS-B, at most 200 iterations, one fixed start, fixed bounds).
- **Matched controls.** Each control differs from TSR in exactly one component. All recipes use the same Stage 1 (fixed random state, shared within a process) and the same GP code and settings. Predictions do not depend on the runner's method seed.
- **Unused inputs.** RCT tuning rows and target-reference covariates are not used. Nothing is tuned on evaluation truth.

**Deviations from the paper, fixed in advance.**
- This is our reimplementation; the authors released no code.
- TabPFN v2 replaces TabPFN-2.5, whose weights require an account-bound license token.
- Our GP replaces the original CMGP code.
- The paper's acquisition rule is not run.

## Design

The 124 conditions are those of the [catalog study](CATALOG_STUDY.md): 31 presets, OBS 512 or 2,048, RCT training 128 or 512, RCT tuning 32, RCT test 64 and target evaluation 2,048 rows. Every preset parameter is unchanged. Seeds 20000–20063 give 64 seed blocks. Recipes share the realized samples within a condition and seed. The grid has `124 × 64 × 4 = 31,744` fits. There is no applicability exemption: every unexpected error is a terminal failed cell.

## Independence and leakage checks

These checks were designed against the common ways a validation confirms itself.

1. **Controls isolate one component.** A unit test checks that TSR equals the OBS-only prediction plus a GP fitted to the Stage 1 residuals, and that the RCT-only GP is the same GP fitted to raw outcomes. A gain over `lmcgp_rct_only` is therefore attributable to the OBS offset, not to a different learner.
2. **No weak comparator alone.** TSR is also compared with the catalog's ExtraTrees control. The results also report, descriptively, how often TSR's mean is lower than all four catalog RCT-only recipes on the same samples. The README overview shows why: at RCT 512, an RCT-only ridge learner also beats the ExtraTrees control in 29 of 31 presets.
3. **Data identity is checked, not assumed.** The rerun of `rct_t_extra_trees` must reproduce the catalog's published per-seed MSE (`replicate_scores.csv`) within `1e-9` in every cell. If it does not, the cross-study descriptive comparisons are dropped and the mismatch is reported.
4. **The designer did not choose the data.** Presets, seeds, targets and sizes come unchanged from the catalog, which was fixed before TSR was implemented. The recipe code comes from an earlier check on a different data-generating process. Before this protocol it was not run on any catalog preset. Development runs use only the smoke configuration's seeds and sizes and are inspected for execution status, not accuracy. A timing preflight at the study's sizes uses seed 99999, outside the study grid; only runtime, memory and status are read from it.
5. **An independent implementation checks the GP.** On a fixed subset, the first seed of every preset at OBS 512 · RCT 128, the GP's posterior effect mean is recomputed with GPy's coregionalized GP at the fitted hyperparameters. The maximum absolute difference must be below `1e-6` times the outcome scale.
6. **A fresh reviewer checks the code before the main run.** The reviewer receives only this protocol and the code, and writes findings before any main-run result exists.
7. **Disclosure.** The same AI agent designed, implemented and analyses this study. The protocol was fixed first and the checks above were added for that reason. No external human review has taken place.

## Analysis

All differences are first recipe minus second recipe in CATE MSE, paired by seed within a condition. Negative values favor the first recipe.

- **Family A (OBS offset within one learner):** `tsr_tabpfn_lmcgp − lmcgp_rct_only` in each condition.
- **Family B (complete procedure):** `tsr_tabpfn_lmcgp − rct_t_extra_trees` in each condition.

Each family uses two-sided paired Student-t tests with one Holm adjustment at .05. As in the catalog study, M11's four conditions (Student-t(3) noise) are excluded from mean-t inference and reported descriptively, which leaves 120 tests per family. If any required pair is missing or failed, that family's decisions are withheld and its intervals are reported unadjusted.

The following are descriptive:
- `tsr_tabpfn_lmcgp − tabpfn_obs_only`;
- per-condition means of all four recipes;
- counts of conditions where TSR's mean MSE is below (i) the ExtraTrees control and (ii) all four catalog RCT-only recipes, by budget;
- a TSR column added to the README overview figure.

**Expected directions, recorded before results.**
- Family A: TSR is better than the matched GP in most RCT-128 conditions and in fewer RCT-512 conditions.
- The advantage is smaller at OBS 512 than at OBS 2,048.
- Reversals occur where OBS-only recipes are far worse than RCT-only recipes in the catalog (strong hidden confounding).
- Family B: no direction is asserted.

## Execution

Stage 1 needs a GPU, so the main run uses GSDS `P2` RTX 3090 nodes with `device="cuda"`. A missing GPU fails the cell; there is no CPU fallback. Runs are sharded by seed with [`make_reference_configs.py`](../scripts/make_reference_configs.py). Every shard runs the same code snapshot and environment, and the analysis refuses mixed source or environment identities. The environment matches the catalog's numerical stack (Python 3.12, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.2) and adds `torch` with CUDA and `tabpfn==2.0.9`. Job IDs, allocations, runtimes and any failures are recorded with the results.
