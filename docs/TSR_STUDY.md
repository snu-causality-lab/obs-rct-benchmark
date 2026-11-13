# Two-stage residual (TSR) static study

This protocol and its [configuration](../configs/tsr_study.toml) were fixed on 2026-11-13, before any TSR recipe was fitted to a catalog preset. The study adds one estimator, not an acquisition rule.

The same day, still before any main-run result, an independent code review led to five revisions: a second catalog comparator, a pinned TabPFN checkpoint, per-record Stage 1 device checks, shard-setting checks in the analysis, and a non-catalog smoke generator. The [revision record](#revisions-before-results) lists them.

## Question

[Gao et al. (2026)](https://arxiv.org/abs/2603.03785v1) pair an acquisition rule (R-EPIG) with a two-stage residual estimator (TSR). An observational learner is frozen as an offset, and a Gaussian process (GP) learns the residual from trial outcomes. Their comparisons change the estimator and the acquisition rule together. This study asks about the estimator alone, under uniform trial sampling, across the 31-preset catalog:

1. **Within one learner.** Does adding the OBS offset lower CATE error relative to the same GP fitted to trial outcomes only?
2. **Against existing RCT-only learners.** Does TSR lower CATE error relative to the catalog's RCT-only ExtraTrees and ridge T-learners?

## Recipes

| Recipe | OBS | RCT | Prediction |
| --- | --- | --- | --- |
| `tsr_tabpfn_lmcgp` | Stage 1 | residual GP | `tau_obs(x) + E[tau_residual(x)]` |
| `lmcgp_rct_only` | — | GP on raw outcomes | `E[tau(x)]` |
| `tabpfn_obs_only` | Stage 1 | — | `tau_obs(x)` |
| `rct_t_extra_trees` | — | T-learner | Catalog settings: 100 trees, minimum leaf 5 |
| `rct_t_ridge` | — | T-learner | Catalog settings: penalty 1, standardized |

- **Stage 1.** A TabPFN v2 regressor (`tabpfn==2.0.9`, default settings, `random_state=0`) is fitted as an S-learner on raw OBS covariates and treatment, then frozen. The checkpoint is pinned by SHA-256 in the configuration; a file with another hash fails the cell.
- **Stage 2.** A two-output GP over (covariates, arm) uses a linear model of coregionalization with two RBF components with ARD. Each component has an arm covariance `w w' + diag(kappa)`, and the noise is shared. Covariates are standardized with the fitted trial rows' means and scales, and outcomes within the fit. Hyperparameters maximize the marginal likelihood (L-BFGS-B, at most 200 iterations, one fixed start, fixed bounds).
- **Matched controls.** Each control differs from TSR in exactly one component. All recipes use the same Stage 1 (fixed random state, shared within a process) and the same GP code and settings. Predictions do not depend on the runner's method seed.
- **Unused inputs.** RCT tuning rows and target-reference covariates are not used. Nothing is tuned on evaluation truth.

**Deviations from the paper, fixed in advance.**
- This is our reimplementation; the authors released no code.
- TabPFN v2 replaces TabPFN-2.5, whose weights require an account-bound license token.
- Our GP replaces the original CMGP code.
- The paper's acquisition rule is not run.

## Design

The 124 conditions are those of the [catalog study](CATALOG_STUDY.md): 31 presets, OBS 512 or 2,048, RCT training 128 or 512, RCT tuning 32, RCT test 64 and target evaluation 2,048 rows. Every preset parameter is unchanged. Seeds 20000–20063 give 64 seed blocks. Recipes share the realized samples within a condition and seed. The grid has `124 × 64 × 5 = 39,680` fits. There is no applicability exemption: every unexpected error is a terminal failed cell.

## Independence and leakage checks

These checks were designed against the common ways a validation confirms itself.

1. **Controls isolate one component.** A unit test checks that TSR equals the OBS-only prediction plus a GP fitted to the Stage 1 residuals, and that the RCT-only GP is the same GP fitted to raw outcomes. A gain over `lmcgp_rct_only` is therefore attributable to the OBS offset, not to a different learner.
2. **No weak comparator alone.** TSR is also compared with two catalog RCT-only learners. ExtraTrees is the stronger of them at RCT 128 and ridge at RCT 512, where ridge beats ExtraTrees in 29 of 31 presets. The results also report, descriptively, how often TSR's mean is lower than all four catalog RCT-only recipes on the same samples.
3. **Data identity is checked, not assumed.** The reruns of `rct_t_extra_trees` and `rct_t_ridge` must reproduce the catalog's published per-seed MSE (`replicate_scores.csv`) within `1e-9` in every cell. If it does not, the cross-study descriptive comparisons are dropped and the mismatch is reported.
4. **The designer did not choose the data.** Presets, seeds, targets and sizes come unchanged from the catalog, which was fixed before TSR was implemented. The recipe code comes from an earlier check on a different data-generating process. Before this protocol it was not run on any catalog preset. Development runs use [`configs/gamma_smoke.toml`](../configs/gamma_smoke.toml), a sensitivity-model generator outside the catalog. A timing preflight at the study's sizes uses seed 99999, outside the study grid. Its job removes every metric before the records are read, leaving runtime, memory and status.
5. **An independent implementation checks the GP.** The subset is the first seed of every preset at OBS 512 · RCT 128. On it, GPy's coregionalized GP recomputes two quantities, each within a relative tolerance of `1e-6`:
   - the negative log marginal likelihood, at the default and at the fitted hyperparameters, which checks the objective the optimizer used;
   - the posterior effect mean at the fitted hyperparameters.

   The check runs in GPy's own environment and records versions and the TSR source hash.
6. **A fresh reviewer checks the code before the main run.** The reviewer received only this protocol and the code, and wrote findings before any main-run result existed.
7. **The run is checked, not trusted.** Every shard's settings must equal this configuration apart from its seeds, and seeds must partition the grid. Record identities must match the runner's derivation. Every successful Stage 1 record must show CUDA and the pinned checkpoint. Otherwise the analysis stops without reporting.
8. **Disclosure.** The same AI agent designed, implemented and analyses this study. The protocol was fixed first and the checks above were added for that reason. No external human review has taken place.

## Analysis

All differences are first recipe minus second recipe in CATE MSE, paired by seed within a condition. Negative values favor the first recipe.

- **Family A (OBS offset within one learner):** `tsr_tabpfn_lmcgp − lmcgp_rct_only` in each condition.
- **Family B (complete procedure):** `tsr_tabpfn_lmcgp − rct_t_extra_trees` and `tsr_tabpfn_lmcgp − rct_t_ridge` in each condition.

Each family uses two-sided paired Student-t tests with one Holm adjustment at .05. As in the catalog study, M11's four conditions (Student-t(3) noise) are excluded from mean-t inference and reported descriptively. That leaves 120 tests in Family A and 240 in Family B. The required pairs are all 64 seeds of every inferential condition. If any is missing or failed, that family's decisions are withheld and its intervals are reported unadjusted.

The following are descriptive:
- `tsr_tabpfn_lmcgp − tabpfn_obs_only`;
- per-condition means of all four recipes;
- counts of conditions where TSR's mean MSE is below (i) the matched GP, (ii) each catalog control and (iii) all four catalog RCT-only recipes, by budget. Only complete conditions are counted;
- a TSR column added to the README overview figure.

**Expected directions, recorded before results.**
- Family A: TSR is better than the matched GP in most RCT-128 conditions and in fewer RCT-512 conditions.
- The advantage is smaller at OBS 512 than at OBS 2,048.
- Reversals occur where OBS-only recipes are far worse than RCT-only recipes in the catalog (strong hidden confounding).
- Family B: no direction is asserted.

## Revisions before results

Made on 2026-11-13 after the independent pre-run review and before any main-run cell was fitted:

1. Added `rct_t_ridge` as a second catalog comparator in Family B. ExtraTrees alone is a weak control at RCT 512.
2. Pinned the TabPFN checkpoint by SHA-256, and recorded the Stage 1 device, GPU and checkpoint hash in every record.
3. Made the analysis check shard settings, seed partition, record identities and the Stage 1 device.
4. Added the marginal-likelihood comparison to the GPy check.
5. Moved development runs off catalog presets to the gamma smoke generator, and redacted metrics from the timing preflight.

## Execution

Stage 1 needs a GPU, so the main run uses GSDS `P2` RTX 3090 nodes with `device="cuda"`. A missing GPU fails the cell; there is no CPU fallback. Runs are sharded by seed with [`make_reference_configs.py`](../scripts/make_reference_configs.py). Every shard runs the same code snapshot and environment, and the analysis refuses mixed source or environment identities. The environment matches the catalog's numerical stack (Python 3.12, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.2) and adds `torch` with CUDA and `tabpfn==2.0.9`. Job IDs, allocations, runtimes and any failures are recorded with the results.
