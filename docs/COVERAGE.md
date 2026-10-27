# Scope and coverage

The canonical catalog contains 31 named scenarios: eight published-mechanism translations and one target-population extension associated with six research sources, twelve structural mechanisms (`M0`–`M11`), and ten paired structural variants (`C01`–`C10`). This is a bounded simulation collection. It does not claim comprehensive coverage of observational–trial fusion, all kinds of hidden confounding, or every published benchmark. [Paper cards](PAPER_DGPS.md) document the external sources; structural configurations and truth equations live in [`structural_dgp.py`](../src/obs_rct_benchmark/structural_dgp.py).

The [scenario annotations](V03_SCENARIOS.md) distinguish trial/OBS effect transport, support and method assumptions for every canonical preset; the [comparison design](V03_DESIGN.md) states how those annotations constrain scientific claims. These annotations do not change the preset equations. The separately frozen [GP source-access experiment](GP_SOURCE_PILOT.md) uses one additional controlled two-covariate mechanism through its dedicated driver. Its six discrepancy/budget conditions are not six new canonical presets, and it does not change the 31-preset catalog count.

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

The [MSM factory](SENSITIVITY.md) implements scalar Gamma and arbitrary callable Gamma(x), with constant/logistic profiles in TOML. The [empirical adapters](DATASETS.md) load separately obtained IHDP NPZ/CSV and ACIC2016 CSV files, with explicit masking/coarsening and exact finite-population CATE. Dataset files are not bundled. Canonical CLI loading uses the documented standard columns; custom ACIC column mappings, categorical declarations and row-ID validation are available through the Python loader API.

The current contract contains one OBS source and one RCT per replicate. Multiple sources with differing variable sets, survival or longitudinal outcomes, multivalued treatments and a general missing-data interface remain separate extensions. A multi-source extension would need to state the target conditioning variables, each source's observed variables, residual confounding, trial support and effect transportability. Sampling and masking a common full-data population alone does not ensure identification of joint-covariate CATE.

Relevant precedents include [Hussain et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html), who combine one RCT and multiple OBS sources for group-effect confidence sets with at least one valid OBS estimator; [Li et al. (2023)](https://proceedings.mlr.press/v189/li23c.html), who estimate common-covariate CATE with source-specific adjustment variables under within-source unconfoundedness; and [Steingrimsson et al. (2024)](https://arxiv.org/abs/2205.00610), who transport average effects from trials with systematically missing covariates under pattern-specific transportability conditions. These methods are not implemented here. Additional designs can follow the [extension workflow](EXTENDING.md).

Version 0.3.0a3 registers 15 recipes: eight generic references, official CorNet and DataFusionPPI adapters, native Integrative R linear/quadratic recipes, a fixed-rho native Causal-ICM GP and its two source controls. Recipe variants and controls are not distinct fusion algorithm families. The catalog executes its original ten recipes; the GP experiment executes the full GP and its source controls; Integrative R has native checks and cost profiles without a new 31-preset accuracy campaign. The [method inventory](V03_METHODS.md) distinguishes available, checked, executed and pending methods. A paper's DGP does not imply that its estimator has been implemented or its paper tables reproduced. Read [the protocol](PROTOCOL.md) before interpreting pooled scores or uncertainty intervals.
