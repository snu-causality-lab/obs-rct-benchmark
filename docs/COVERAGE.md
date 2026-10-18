# Scope and coverage

The candidate includes 31 named scenarios: eight published-mechanism translations and one target-population extension associated with six research sources, twelve structural mechanisms (`M0`–`M11`), and ten paired structural variants (`C01`–`C10`). This is a bounded simulation collection. It does not claim comprehensive coverage of observational–trial fusion, all kinds of hidden confounding, or every published benchmark. [Paper cards](PAPER_DGPS.md) document the external sources; structural configurations and truth equations live in [`structural_dgp.py`](../src/obs_rct_benchmark/structural_dgp.py).

## Structural mechanisms

| ID | Main feature | Interpretation |
| --- | --- | --- |
| M0 | No hidden-selection discrepancy | Clean reference with smooth CATE |
| M1 | Global constant confounding strength with unequal arm loadings | Broad confounding with linear CATE |
| M2 | Smooth covariate-dependent confounding | Smooth reference with separate effect/discrepancy directions |
| M3 | Localized confounding with an exactly clean complement | Observable regions differ in reliability |
| M4 | Localized magnitude with nonzero confounding throughout | Low-confounding regions need not be clean |
| M5 | Hidden reliability variation within observed `X` | Unit-level reliability is not recoverable from `X` alone |
| M6 | Sign-changing confounding | Direction and magnitude both vary |
| M7 | Interaction confounding with opposite arm signs | Arm-specific outcome distortion matters |
| M8 | Confounding aligned with a rare treatment-effect subgroup | Large effects and unreliable OBS overlap |
| M9 | M8-like effects with weaker trial coverage in the effect region | Observed trial overlap limits correction |
| M10 | Hidden treatment-selection confounding and source CATE shift | Two sources of regression discrepancy coexist |
| M11 | Multiple confounding regions, Gaussian-mixture hidden variable, discontinuous CATE | Several modeling stresses coexist |

M mechanisms span several stresses at once. Differences between arbitrary M IDs therefore do not isolate a single causal design choice.

## Paired structural variants

| Variant | Reference | Intended contrast |
| --- | --- | --- |
| C01 | M6 | Fixed-sign versus sign-changing confounding magnitude |
| C02 | M7 | RMS-matched symmetric versus asymmetric hidden-outcome arm loadings |
| C03 | M11 | Variance-matched Gaussian versus heavy-tailed outcome noise |
| C04 | M0 | Add an arm-common observational baseline shift |
| C05 | M0 | Set the CATE to zero |
| C06 | M0 | Center linear heterogeneity to yield zero target ATE |
| C07 / C08 | Each other | Hidden trial-participation variable is prognostic only or also modifies treatment effects |
| C09 / C10 | Each other | Effect and confounding use shared or disjoint coordinate blocks |

These are matched contrasts at their canonical presets. Arbitrary overrides can break their matching, including marginal-distribution matching after introducing correlated covariates. Consult the exported cards' `matched_quantities` and configuration differences when designing a new contrast. C09 and C10 use six dimensions by default; C10 requires room for the disjoint three-coordinate blocks.

## Configurable mechanisms and limits

`StructuralDGPConfig` already supports custom hidden-confounding mechanisms and validated overrides. Controls include treatment and arm-specific outcome loadings, hidden normal/binary/normal-mixture distributions, observed predictability of hidden variables, confounding geometry, clean-region size, treatment overlap, CATE geometry, effect/discrepancy alignment, source outcome shift, observed or hidden trial participation, covariate correlation/marginals, and noise shape/heteroskedasticity. Structural mechanisms require at least three covariates, with stricter coordinate requirements for some settings. Not every possible combination is scientifically comparable or numerically accurate merely because it passes input validation.

The released API does not yet provide an arbitrary user-specified `Gamma(x)` sensitivity-model factory, general IHDP or ACIC loading/conversion, survival or longitudinal outcomes, multi-valued treatments, or a general missing-data interface. None of those datasets are bundled. Additional published DGPs and dataset adapters can be contributed through the [extension workflow](EXTENDING.md); they are future scope until implemented and checked.

The built-in estimators are generic reference baselines and optional adapters to separately installed official DataFusionPPI and CorNet. The presence of a paper's DGP does not imply that paper's estimator is implemented, installed, or independently reproduced. Read [the protocol](PROTOCOL.md) before interpreting pooled rankings or uncertainty intervals.
