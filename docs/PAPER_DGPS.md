# Published simulation mechanisms

This package implements nine source-related DGP variants from six public research sources: eight mathematical translations and one explicitly labeled target-population extension. These use a common OBS/RCT contract; they are not bitwise reruns of the authors' experiments or evidence that any method performs well. A paper's own simulation can favor its modeling assumptions. Report these variants separately from the benchmark's [structural mechanisms](COVERAGE.md).

`make_paper_config(name=..., paper_id=..., **overrides)` constructs a preset; `generate_paper_problem(config, seed=...)` draws its role-separated samples. The paper DGP contract version is `2.0.0`. The source of truth for accepted options and dimensions is [`paper_dgp.py`](../src/obs_rct_benchmark/paper_dgp.py). Every variant accepts separate OBS, RCT training, tuning, test, and target-evaluation sample sizes. All variants use binary treatment and continuous outcomes. A change to a preset is a new experimental setting and must be reported.

The target CATE is always the conditional contrast for the declared target population. Observational conditional regressions can differ from target potential-outcome means through hidden treatment selection, source-specific outcome mechanisms, or both; [the protocol](PROTOCOL.md) defines the decomposition. Oracle truth is supplied to the evaluator, outside `TrainingView`. Full provenance, including file hashes where recorded, is in [sources.lock.toml](../sources.lock.toml).

## CorNet: `cornet_official_n_conf`

**Primary source.** Tobias Hatt, Jeroen Berrevoets, Alicia Curth, Stefan Feuerriegel, and Mihaela van der Schaar (2022), [Combining Observational and Randomized Data for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/2202.12891v1). Equation and architecture reference: [`tobhatt/CorNet` at `38b9c37d5a86bc87950945d6ddb1ceb5354a45ba`](https://github.com/tobhatt/CorNet/tree/38b9c37d5a86bc87950945d6ddb1ceb5354a45ba), especially `simulation_studies/simdata.py` and `sim_n_conf.py`.

**Mechanism and populations.** OBS, RCT, and evaluation covariates follow `N(0,I_d)`. A three-layer shared representation feeds linear base and sparse correction heads. RCT and target outcomes include corrections omitted from OBS outcomes. Both sources assign treatment with probability 0.5. This creates source-specific outcome and CATE shift; it does not create hidden treatment-selection confounding. Sharing the representation between target effects and corrections makes this an especially relevant check for representation-based methods.

**Dimensions and controls.** Preset `d=25`, outcome noise standard deviation 0.5, base-head norm 10, correction RMS 2, and sparse ratio 0.666. `d` is configurable, subject to the correction sparsity constraint; it must be at least one and at least one head coordinate must remain. Noise and correction scales are configurable.

**Translation caveat.** The package uses its own NumPy implementation of the equation architecture, with `tanh` after the first two layers and uniform initialization bounds corresponding to the upstream linear layers. NumPy PCG64 draws, draw order, and an independent 10,000-row Gaussian reference for correction RMS normalization differ from upstream PyTorch initialization. A matching integer seed therefore does not reproduce the upstream network or dataset bit for bit. The original scripts do not set all simulation seeds and ignore their `noise_scale` argument in favor of 0.5; the benchmark exposes an explicit noise setting. No upstream class is imported or vendored.

**Evaluation truth.** The target CATE is the corrected head difference. Arm-specific confounding terms are zero; source shifts and total regression discrepancies come from the omitted corrections.

## Experimental Grounding: `experimental_grounding_kps`

**Primary source.** Nathan Kallus, Aahlad Manas Puli, and Uri Shalit (2018), [Removing Hidden Confounding by Experimental Grounding](https://arxiv.org/abs/1810.11646v1), NeurIPS 2018, simulation study. Equation reference: [`CausalML/RemovingHiddenConfounding` at `a25cf3d2524a01c8a8d821f25cdeeb312c34a9cd`](https://github.com/CausalML/RemovingHiddenConfounding/tree/a25cf3d2524a01c8a8d821f25cdeeb312c34a9cd), `simulation.ipynb` and `lin_eta_gaussianU.py`.

**Mechanism and populations.** OBS and target `X` are standard normal; trial `X` is uniform on `[-1,1]`, so trial support omits part of the target population. The OBS joint distribution of observed `X` and hidden `U` depends on treatment. Treatment has marginal probability 0.5, but the treatment-dependent association with `U` creates conditional confounding. Arm outcome surfaces are quadratic. Under the preset, the target CATE is `1+x+0.25*x²` and the OBS CATE discrepancy is linear.

**Dimensions and controls.** Requires `d=1`. Preset noise standard deviation 0.5, `kps_kappa=0.5`, `kps_nu0=-0.25`, `kps_nu1=0.25`. Hidden association and arm means are configurable within validated bounds.

**Translation caveat.** Coefficients are fixed while independent, fixed-size OBS/RCT/evaluation roles are drawn for each sample seed. The common role split replaces the notebook's original fitting workflow; stochastic learner search in the notebook is not reproduced.

**Evaluation truth.** Analytic quadratic target CATE, arm-specific confounding and source-shift terms, OBS propensity, and trial-to-target covariate density are available outside the training contract.

## Integrative R-learner: three variants

**Primary source.** Lili Wu and Shu Yang (2022), [Integrative R-learner of heterogeneous treatment effects combining experimental and observational studies](https://proceedings.mlr.press/v177/wu22a.html), CLeaR, PMLR 177:904–926. Equation reference: [`lwu9/intRlearner` at `a46ce2c1cc814865898b863d6a8e130bbc2c802b`](https://github.com/lwu9/intRlearner/tree/a46ce2c1cc814865898b863d6a8e130bbc2c802b), `R/intRlearner.R` and `R/utils.R`.

| Variant | Populations and supported dimension | Target and discrepancy |
| --- | --- | --- |
| `integrative_r_learner_kps_code` | Original Simulation I in §6.1.1 and released `gen_data(KPS_data=TRUE)`. `d=1`; OBS standard normal; trial and target uniform on `[-1,1]` | Quadratic target CATE `1+2*x+int_r_cf*x²`; treatment-dependent hidden association creates linear conditional confounding |
| `integrative_r_learner_kps_normal_target_extension` | Benchmark extension of Simulation I. `d=1`; OBS and target standard normal; trial uniform on `[-1,1]` | Same conditional target CATE and hidden-association mechanism, evaluated partly outside trial support |
| `integrative_r_learner_multivariate_paper` | Simulation II in §6.1.2. OBS and target `N(0,I_d)`; trial participation depends on observed `X`; requires `d>=2`, preset `d=9` | Sparse linear CATE, Gaussian hidden variable affecting OBS treatment assignment, and known covariate-dependent trial treatment probability |

**Controls.** The two one-dimensional presets use noise standard deviation 0.5, `kps_kappa=0.5`, zero hidden arm means, and `int_r_cf=0.75`. The multivariate preset uses noise standard deviation 1, confounding strength 2.5, trial-selection intercept 7, and nine nonzero covariate modifiers. The number of zero modifiers and selection/confounding strengths are configurable.

**Translation caveats.** Both [the paper, §6.1.1, p. 10](https://proceedings.mlr.press/v177/wu22a/wu22a.pdf) and the pinned released code specify uniform test covariates on `[-1,1]`. The normal-target extension changes that evaluation law deliberately; it must not be described as a paper/code discrepancy or an original paper design. The released README reverses the paper's Simulation I/II labels; this package follows the paper. For the multivariate setting, the example code's `p0=1` would leave eight nonzero covariate modifiers, so the paper preset uses zero zeroed modifiers. Trial-participant covariates are drawn directly from the selected-population law to achieve requested fixed role sizes, rather than producing a random trial count from a finite population.

**Evaluation truth.** Analytic quadratic or linear target CATE is scored under each variant's declared target law. The normal-target extension measures extrapolation beyond trial support; identification there requires additional modeling assumptions. Conditional hidden moments for the multivariate logistic assignment use deterministic Gaussian quadrature. These numerical truth quantities must be checked for quadrature stability when increasing confounding strength. Known trial randomization probabilities are provided per row; an adapter restricted to constant trial propensity cannot run the multivariate variant without changing its assumptions.

## Elastic Integrative: `elastic_integrative_gen_data_overlap`

**Primary source.** Shu Yang, Chenyin Gao, Donglin Zeng, and Xiaofei Wang, [Elastic Integrative Analysis of Randomized Trial and Real-World Data for Treatment Heterogeneity Estimation](https://arxiv.org/abs/2005.10579v3), cited preprint version 3 (2022). Equation reference: [`Gaochenyin/ElasticIntegrative` at `a8c9a06371485c25caca5cc8396ea5ba47f381b6`](https://github.com/Gaochenyin/ElasticIntegrative/tree/a8c9a06371485c25caca5cc8396ea5ba47f381b6), `R/GenData.R`, overlap setting.

**Mechanism and populations.** OBS and target observed covariates follow `N(1,I_2)` in the preset. A hidden Gaussian baseline variable also affects OBS treatment assignment. Trial participation depends on observed covariates, and trial treatment is randomized with probability 0.5. The source and target conditional causal effect agree; covariate selection and hidden OBS confounding are separate features.

**Dimensions and controls.** Requires `d=2`. Preset noise standard deviation 1, hidden-treatment loading `elastic_tlocalpar=2`, observed mean 1, and trial-selection intercept -4.5. These scales and selection controls are configurable.

**Translation caveat.** This variant covers the upstream overlap setting. It replaces finite-sample empirical treatment-propensity centering with its population value and draws fixed-size trial roles from the participant conditional law. It does not reproduce the finite-population random trial count, original estimation procedure, or the separate `overlap=FALSE` setting.

**Evaluation truth.** Target CATE is `x1+x2`; a hidden baseline covariate creates arm-specific OBS conditional-regression discrepancies. Deterministic Gaussian quadrature supplies conditional hidden moments and OBS propensity.

## Power-likelihood: two variants

**Primary source.** Xi Lin, Jens Magelund Tarp, and Robin J. Evans (2025), [Combining experimental and observational data through a power likelihood](https://academic.oup.com/biometrics/article/81/1/ujaf008/8016472), *Biometrics* 81(1), ujaf008, [DOI 10.1093/biomtc/ujaf008](https://doi.org/10.1093/biomtc/ujaf008), Section 5. The publisher's supplemental archive is pinned by SHA-256 in [the source lock](../sources.lock.toml); its `sec_5_code/run_simulations.R` and `functions.R` provide the code-equation variant.

| Variant | Hidden variable and copula preset | Source discrepancy |
| --- | --- | --- |
| `power_likelihood_paper_text` | Binary `U`, treatment coefficient 2.5 in the Gaussian-copula correlation | Hidden treatment selection plus an arm-common OBS baseline shift |
| `power_likelihood_supplement_code` | Gaussian `U`, treatment coefficient 1.0 in the Gaussian-copula correlation | Hidden treatment selection; the centered hidden variable has no marginal baseline shift |

**Mechanism and populations.** Observed `W=(C,Z)` has `C~Bernoulli(0.5)` and `Z|C~N(1+C,1)` in OBS, trial, and target. Treatment-dependent Gaussian copulas connect outcome and `Z` residuals; hidden `U` affects OBS treatment and outcome. Trial treatment is randomized and the direct hidden-outcome term is removed from trial outcomes.

**Dimensions and controls.** Requires `d>=2`, preset `d=2`. Coordinates after `(C,Z)` are independent Gaussian covariates with no effect in the translated mechanism. Preset noise standard deviation 1, `power_omega=0.5`, copula intercept 1. Confounding and copula scales are configurable; hidden distribution must agree with the selected variant.

**Translation caveats.** The published text and supplement differ in hidden distribution and copula coefficient, so they remain separate. The benchmark translates the copula equations into NumPy draws; it does not execute the original R code. The supplement does not pin its `causl` revision, all R package versions, or parallel RNG streams. Its archived source is identifiable, but its original runtime environment is not fully reproducible from that archive alone.

**Evaluation truth.** The paper's subgroup-marginal effect is `tau(C)=0.1+0.1*C`. Conditioning on all observed `W` adds the copula-induced `Z` term, so the benchmark scores the full-`W` CATE and preserves `paper_subgroup_tau_c` as auxiliary truth. Using the subgroup effect as full-`W` truth would score a different estimand. Binary hidden moments are summed exactly; Gaussian hidden moments use quadrature.

## CVCI: `cvci_official_linear_bias`

**Primary source.** Xuelin Yang, Licong Lin, Susan Athey, Michael I. Jordan, and Guido W. Imbens (2025), [Cross-Validated Causal Inference: a Modern Method to Combine Experimental and Observational Data](https://arxiv.org/abs/2511.00727v1). Equation reference: [`xyang23/cross_validated_causal` at `d74cb7c928938fd1a2978481f9aab6ac654c7d08`](https://github.com/xyang23/cross_validated_causal/tree/d74cb7c928938fd1a2978481f9aab6ac654c7d08), `linear_eps.py` and `causal_sim.py`.

**Mechanism and populations.** OBS, RCT, and target covariates are standard multivariate normal. Source-specific linear baseline coefficients and a constant OBS treatment-effect offset produce regression discrepancy. Treatment probabilities are fixed by source and do not depend on hidden variables. This is source-specific outcome/CATE shift, not hidden treatment-selection confounding. The upstream study concerns a constant treatment effect; the common CATE score is valid here because its target CATE is constant.

**Dimensions and controls.** Requires `d>=1`, preset `d=5`. Preset target effect 0.5, OBS effect offset 0.3, OBS treatment probability 0.2, RCT treatment probability 0.5, and noise standard deviation 1. Dimension, effects, OBS assignment probability, and noise are configurable.

**Translation caveat.** Benchmark mechanism coefficients remain fixed across sample seeds; independent RCT training, tuning, and test roles replace the original single RCT sample. Original estimators and experimental sweeps are not reproduced by this DGP translation.

**Evaluation truth.** The target CATE is exactly the configured constant. Hidden-confounding terms are zero; source baseline and CATE shifts remain separately visible in evaluation truth.
