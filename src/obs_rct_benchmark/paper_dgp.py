"""Source-based DGP translations and labeled benchmark extensions.

These assets are implementation checks, not independent evidence for a method.
Each card states whether the source fully specifies its mechanism and which
benchmark conventions were needed to map it onto separate OBS/RCT roles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray

from .data import SimulationProblem, SimulationTruth, SourceSample


FloatArray = NDArray[np.float64]
PaperDGP = Literal[
    "cornet_official_n_conf",
    "experimental_grounding_kps",
    "integrative_r_learner_kps_normal_target_extension",
    "integrative_r_learner_kps_code",
    "integrative_r_learner_multivariate_paper",
    "elastic_integrative_gen_data_overlap",
    "power_likelihood_paper_text",
    "power_likelihood_supplement_code",
    "cvci_official_linear_bias",
]

PAPER_DGP_VERSION = "2.0.0"
PAPER_DGPS: tuple[str, ...] = (
    "cornet_official_n_conf",
    "experimental_grounding_kps",
    "integrative_r_learner_kps_normal_target_extension",
    "integrative_r_learner_kps_code",
    "integrative_r_learner_multivariate_paper",
    "elastic_integrative_gen_data_overlap",
    "power_likelihood_paper_text",
    "power_likelihood_supplement_code",
    "cvci_official_linear_bias",
)

PAPER_TARGET_POPULATIONS: Mapping[str, str] = MappingProxyType(
    {
        "cornet_official_n_conf": "X ~ N(0, I_d)",
        "experimental_grounding_kps": "X ~ N(0, 1)",
        "integrative_r_learner_kps_normal_target_extension": "benchmark extension X ~ N(0, 1)",
        "integrative_r_learner_kps_code": (
            "paper and released-code test law X ~ Uniform[-1, 1]"
        ),
        "integrative_r_learner_multivariate_paper": "X ~ N(0, I_d)",
        "elastic_integrative_gen_data_overlap": "X ~ N(1, I_2)",
        "power_likelihood_paper_text": (
            "C ~ Bernoulli(0.5), Z | C ~ N(1+C, 1)"
        ),
        "power_likelihood_supplement_code": (
            "C ~ Bernoulli(0.5), Z | C ~ N(1+C, 1)"
        ),
        "cvci_official_linear_bias": "X ~ N(0, I_d)",
    }
)

PAPER_DGP_AXIS_LEVELS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "cornet_official_n_conf": {
            "discrepancy_origin": "source_outcome_shift",
            "confounding_geometry": "none",
            "clean_region": "none",
            "reliability_resolution": "x_resolvable",
            "arm_structure": "asymmetric_source_shift",
            "cate_geometry": "neural_shared_representation",
            "bias_cate_alignment": "shared_representation_by_construction",
            "rct_support": "common",
            "transportability": "source_cate_differs",
            "hidden_distribution": "not_applicable",
        },
        "experimental_grounding_kps": {
            "discrepancy_origin": "treatment_conditioned_hidden_association",
            "confounding_geometry": "global_linear",
            "clean_region": "point_only",
            "reliability_resolution": "x_resolvable",
            "arm_structure": "asymmetric_signed",
            "cate_geometry": "quadratic",
            "bias_cate_alignment": "partly_aligned",
            "rct_support": "limited_interval",
            "transportability": "holds",
            "hidden_distribution": "gaussian",
        },
        "integrative_r_learner_kps_normal_target_extension": {
            "discrepancy_origin": "treatment_conditioned_hidden_association",
            "confounding_geometry": "global_linear",
            "clean_region": "point_only",
            "reliability_resolution": "x_resolvable",
            "arm_structure": "asymmetric_signed",
            "cate_geometry": "quadratic",
            "bias_cate_alignment": "partly_aligned",
            "rct_support": "limited_interval",
            "transportability": "holds",
            "hidden_distribution": "gaussian",
        },
        "integrative_r_learner_kps_code": {
            "discrepancy_origin": "treatment_conditioned_hidden_association",
            "confounding_geometry": "global_linear",
            "clean_region": "point_only",
            "reliability_resolution": "x_resolvable",
            "arm_structure": "asymmetric_signed",
            "cate_geometry": "quadratic",
            "bias_cate_alignment": "partly_aligned",
            "rct_support": "common_on_target_support",
            "transportability": "holds",
            "hidden_distribution": "gaussian",
        },
        "integrative_r_learner_multivariate_paper": {
            "discrepancy_origin": "hidden_treatment_selection",
            "confounding_geometry": "global_smooth",
            "clean_region": "none",
            "reliability_resolution": "conditional_x_only",
            "arm_structure": "selection_induced_asymmetric",
            "cate_geometry": "sparse_linear",
            "bias_cate_alignment": "partly_aligned",
            "rct_support": "study_selected",
            "transportability": "holds",
            "hidden_distribution": "gaussian",
        },
        "elastic_integrative_gen_data_overlap": {
            "discrepancy_origin": "hidden_treatment_selection",
            "confounding_geometry": "global_smooth",
            "clean_region": "none",
            "reliability_resolution": "conditional_x_only",
            "arm_structure": "selection_induced_asymmetric",
            "cate_geometry": "linear_effect_modifiers",
            "bias_cate_alignment": "not_controlled",
            "rct_support": "study_selected",
            "transportability": "holds",
            "hidden_distribution": "gaussian",
        },
        "power_likelihood_paper_text": {
            "discrepancy_origin": "hidden_selection_and_source_baseline",
            "confounding_geometry": "global_smooth",
            "clean_region": "none",
            "reliability_resolution": "conditional_x_only",
            "arm_structure": "selection_induced_asymmetric",
            "cate_geometry": "binary_subgroup_plus_full_w_copula",
            "bias_cate_alignment": "not_controlled",
            "rct_support": "common",
            "transportability": "holds",
            "hidden_distribution": "binary",
        },
        "power_likelihood_supplement_code": {
            "discrepancy_origin": "hidden_treatment_selection",
            "confounding_geometry": "global_smooth",
            "clean_region": "none",
            "reliability_resolution": "conditional_x_only",
            "arm_structure": "selection_induced_asymmetric",
            "cate_geometry": "binary_subgroup_plus_full_w_copula",
            "bias_cate_alignment": "not_controlled",
            "rct_support": "common",
            "transportability": "holds",
            "hidden_distribution": "normal",
        },
        "cvci_official_linear_bias": {
            "discrepancy_origin": "source_outcome_shift",
            "confounding_geometry": "none",
            "clean_region": "none",
            "reliability_resolution": "x_resolvable",
            "arm_structure": "constant_source_cate_shift",
            "cate_geometry": "constant",
            "bias_cate_alignment": "not_applicable",
            "rct_support": "common",
            "transportability": "source_cate_differs",
            "hidden_distribution": "not_applicable",
        },
    }
)

PAPER_DGP_CARDS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "cornet_official_n_conf": {
            "title": "CorNet official n-conf simulation translation",
            "source": (
                "tobhatt/CorNet simulation_studies/sim_n_conf.py and simdata.py"
            ),
            "role": "method-favorable implementation reproduction",
            "fidelity": "numpy-architecture-translation-changed-rng",
            "preset": {
                "d": 25,
                "noise_std": 0.5,
                "cornet_outcome_norm": 10.0,
                "cornet_delta_rms": 2.0,
                "cornet_sparse_ratio": 0.666,
            },
            "specified": [
                "three-layer CorNet representation and linear outcome heads",
                "Gaussian covariates and randomized treatment in both sources",
                "normalized base-head weights and sparse correction weights",
                "OBS outcomes omit the correction used by RCT/target outcomes",
            ],
            "under_specified": [
                "the public scripts do not set NumPy or PyTorch seeds",
                "the parameter called confounded data uses Bernoulli(0.5) treatment",
                "the generator argument noise_scale is ignored in favor of 0.5 noise",
            ],
            "benchmark_conventions": [
                "mechanism_seed initializes a self-contained NumPy architecture translation",
                "NumPy PCG64 initialization and an independent 10,000-row RMS reference replace upstream RNG draws",
                "independent RCT train, tune, test, and evaluation draws replace one test draw",
            ],
            "scientific_warning": (
                "This construction contains no hidden treatment-selection confounding: "
                "both source assignments are Bernoulli(0.5). It is a source-specific "
                "outcome/CATE shift generated in CorNet's own representation and is "
                "therefore favorable to CorNet by construction."
            ),
        },
        "experimental_grounding_kps": {
            "title": "Experimental Grounding official simulation translation",
            "source": "CausalML/RemovingHiddenConfounding simulation.ipynb",
            "role": "method-favorable implementation reproduction",
            "fidelity": "official-equation-fixed-size-common-split-translation",
            "preset": {
                "d": 1,
                "noise_std": 0.5,
                "kps_kappa": 0.5,
                "kps_nu0": -0.25,
                "kps_nu1": 0.25,
            },
            "specified": [
                "Gaussian observational X,U conditional on treatment",
                "RCT X support restricted to [-1,1]",
                "quadratic arm-specific outcome surfaces",
                "linear CATE discrepancy and randomized treatment",
            ],
            "under_specified": [
                "the notebook performs stochastic learner search without seeds"
            ],
            "benchmark_conventions": [
                "mechanism coefficients are frozen across sample seeds",
                "independent fixed-size RCT train/tune/test draws",
            ],
        },
        "integrative_r_learner_kps_normal_target_extension": {
            "title": "Integrative R-learner normal-target benchmark extension",
            "source": "Wu--Yang (2022), Section 6.1.1",
            "role": "benchmark target-population extension",
            "fidelity": "paper-outcome-equations-normal-target-extension",
            "preset": {
                "d": 1,
                "noise_std": 0.5,
                "kps_kappa": 0.5,
                "kps_nu0": 0.0,
                "kps_nu1": 0.0,
                "int_r_cf": 0.75,
            },
            "specified": [
                "uniform RCT and Gaussian observational covariates",
                "treatment-dependent X,U correlation",
                "quadratic target CATE and randomized treatment",
            ],
            "under_specified": [],
            "scientific_warning": (
                "Normal evaluation covariates are a benchmark extension. "
                "The paper and released code both use Uniform[-1,1] test covariates."
            ),
            "benchmark_conventions": [
                "independent fixed-size RCT train/tune/test draws",
                "normal evaluation covariates are a benchmark extension of the uniform paper/code target",
            ],
        },
        "integrative_r_learner_kps_code": {
            "title": "Integrative R-learner paper and released-code KPS translation",
            "source": "lwu9/intRlearner R/intRlearner.R gen_data(KPS_data=TRUE)",
            "role": "method-favorable implementation reproduction",
            "fidelity": "official-code-equation-fixed-size-common-split-translation",
            "preset": {
                "d": 1,
                "noise_std": 0.5,
                "kps_kappa": 0.5,
                "kps_nu0": 0.0,
                "kps_nu1": 0.0,
                "int_r_cf": 0.75,
            },
            "specified": [
                "uniform RCT and Gaussian observational covariates",
                "treatment-dependent X,U correlation",
                "quadratic target CATE and randomized treatment",
            ],
            "under_specified": [],
            "benchmark_conventions": [
                "independent fixed-size RCT train/tune/test draws",
                "evaluation follows the paper and released gen_data Uniform[-1,1] target",
            ],
        },
        "integrative_r_learner_multivariate_paper": {
            "title": "Integrative R-learner paper Simulation II translation",
            "source": (
                "Wu--Yang (2022), Section 6.1.2, and lwu9/intRlearner "
                "R/intRlearner.R gen_data(KPS_data=FALSE)"
            ),
            "role": "method-favorable implementation reproduction",
            "fidelity": "official-equation-conditional-sampling-translation",
            "preset": {
                "d": 9,
                "noise_std": 1.0,
                "int_r_confounding_strength": 2.5,
                "int_r_rct_selection_intercept": 7.0,
                "int_r_zero_cate_coefficients": 0,
            },
            "specified": [
                "Gaussian observed and hidden covariates",
                "sparse linear CATE",
                "observational hidden-confounded treatment mechanism",
                "observed-X trial participation and treatment propensities",
            ],
            "under_specified": [
                "the official finite population produces a random RCT size"
            ],
            "benchmark_conventions": [
                "sample directly from the trial-participant conditional law",
                "retain known nonconstant trial propensity in the data contract",
                "use fixed requested sizes for train/tune/test comparability",
            ],
            "scientific_warning": (
                "The released README reverses the paper's Simulation I/II labels: "
                "the multivariate KPS_data=FALSE setting is Simulation II in the paper. "
                "Its example p0=1 also leaves only eight, not the paper's nine, "
                "nonzero covariate modifiers; the paper-target cell uses p0=0."
            ),
        },
        "elastic_integrative_gen_data_overlap": {
            "title": "Elastic Integrative official overlap DGP translation",
            "source": "Gaochenyin/ElasticIntegrative R/GenData.R",
            "role": "method-favorable implementation reproduction",
            "fidelity": "official-equation-conditional-sampling-translation",
            "preset": {
                "d": 2,
                "noise_std": 1.0,
                "elastic_mean_x": 1.0,
                "elastic_tlocalpar": 2.0,
                "elastic_rct_selection_intercept": -4.5,
            },
            "specified": [
                "linear baseline and HTE effect-modifier models",
                "Gaussian hidden baseline covariate",
                "hidden-confounded observational treatment assignment",
                "observed-X trial participation and randomized treatment",
            ],
            "under_specified": [
                "the official finite population produces a random RCT size"
            ],
            "benchmark_conventions": [
                "replace empirical propensity-centering root by its population value",
                "sample directly from the trial-participant conditional law",
                "use independent fixed-size role draws",
            ],
        },
        "power_likelihood_paper_text": {
            "title": "Power-likelihood Section 5 paper-equation translation",
            "source": "Lin--Tarp--Evans (Biometrics 2025), Section 5",
            "role": "method-favorable paper reproduction",
            "fidelity": "paper-equation-full-W-translation",
            "preset": {
                "d": 2,
                "noise_std": 1.0,
                "power_omega": 0.5,
                "power_u_distribution": "binary",
                "power_copula_treatment_coef": 2.5,
            },
            "specified": [
                "binary U and C distributions",
                "Z, treatment, and potential-outcome equations",
                "treatment-specific Gaussian-copula correlation",
                "RCT treatment probability and removal of U influence",
            ],
            "under_specified": [
                "paper conditions the target CATE on C after marginalizing Z, "
                "whereas the common benchmark predicts conditional on all W",
            ],
            "benchmark_conventions": [
                "report full-W CATE including the copula-induced Z term",
                "also store the paper's subgroup-marginal CATE tau(C) in truth",
                "independent RCT train, tune, and test samples",
            ],
            "scientific_warning": (
                "The published supplement does not implement these exact equations: "
                "it draws U as Gaussian and uses a different copula coefficient."
            ),
        },
        "power_likelihood_supplement_code": {
            "title": "Power-likelihood Section 5 supplement-code translation",
            "source": "ujaf008 supplemental sec_5_code.zip, run_simulations.R",
            "role": "method-favorable implementation reproduction",
            "fidelity": "supplement-equation-full-W-translation",
            "preset": {
                "d": 2,
                "noise_std": 1.0,
                "power_omega": 0.5,
                "power_u_distribution": "normal",
                "power_copula_treatment_coef": 1.0,
            },
            "specified": [
                "Gaussian U draw in sim_func",
                "causl formulas and parameter vectors",
                "n_obs=1000, n_rct=500, omega and eta grids",
            ],
            "under_specified": [
                "causl Git revision and complete R package versions",
                "parallel RNG stream; the supplement sets no seed",
            ],
            "benchmark_conventions": [
                "translate the Gaussian copula analytically into NumPy draws",
                "fix sample seeds and separate RCT train/tune/test roles",
                "evaluate the full-W CATE and retain tau(C) as auxiliary truth",
            ],
            "scientific_warning": (
                "This is intentionally separate from the paper-text variant because "
                "the hidden-variable distribution and copula coefficient differ."
            ),
        },
        "cvci_official_linear_bias": {
            "title": "CVCI official linear bias simulation translation",
            "source": "cross_validated_causal/linear_eps.py",
            "role": "method-favorable implementation reproduction",
            "fidelity": "official-equation-common-split-translation",
            "preset": {
                "d": 5,
                "noise_std": 1.0,
                "cvci_true_te": 0.5,
                "cvci_te_bias": 0.3,
                "cvci_obs_propensity": 0.2,
            },
            "specified": [
                "Gaussian covariates and outcome noise",
                "separate linear source outcome coefficients",
                "constant experimental and biased observational effects",
                "RCT and OBS treatment probabilities",
            ],
            "under_specified": [],
            "benchmark_conventions": [
                "mechanism coefficients are fixed across sample seeds",
                "independent RCT train/tune/test roles replace one RCT sample",
            ],
        },
    }
)


@dataclass(frozen=True)
class PaperDGPConfig:
    name: str
    paper_id: PaperDGP
    mechanism_seed: int = 2025
    n_obs: int = 10_000
    n_rct_train: int = 700
    n_rct_tune: int = 200
    n_rct_test: int = 500
    n_eval: int = 2_000
    d: int = 25
    noise_std: float = 1.0

    quadrature_order: int = 31

    # CorNet n-conf simulation translation.
    cornet_outcome_norm: float = 10.0
    cornet_delta_rms: float = 2.0
    cornet_sparse_ratio: float = 0.666

    # Kallus--Puli--Shalit and integrative R-learner translations.
    kps_kappa: float = 0.5
    kps_nu0: float = -0.25
    kps_nu1: float = 0.25
    int_r_cf: float = 0.75
    int_r_confounding_strength: float = 2.5
    int_r_rct_selection_intercept: float = 7.0
    int_r_zero_cate_coefficients: int = 1

    # Elastic Integrative official generator translation.
    elastic_mean_x: float = 1.0
    elastic_tlocalpar: float = 2.0
    elastic_rct_selection_intercept: float = -4.5

    # Power-likelihood Section 5 paper/supplement translations.
    power_omega: float = 0.5
    power_u_distribution: Literal["binary", "normal"] = "binary"
    power_copula_intercept: float = 1.0
    power_copula_treatment_coef: float = 2.5

    # Official CVCI linear simulation translation.
    cvci_true_te: float = 0.5
    cvci_te_bias: float = 0.3
    cvci_obs_propensity: float = 0.2

    def __post_init__(self) -> None:
        if self.paper_id not in PAPER_DGPS:
            raise ValueError(f"unknown paper DGP: {self.paper_id}")
        sample_fields = (
            "n_obs",
            "n_rct_train",
            "n_rct_tune",
            "n_rct_test",
            "n_eval",
        )
        integer_fields = sample_fields + (
            "mechanism_seed",
            "d",
            "quadrature_order",
            "int_r_zero_cate_coefficients",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field_name} must be an integer")
        if min(getattr(self, field_name) for field_name in sample_fields) < 4:
            raise ValueError("all sample roles must have at least four rows")
        if self.mechanism_seed < 0:
            raise ValueError("mechanism_seed must be nonnegative")
        if self.d < 1:
            raise ValueError("d must be at least 1")
        if self.quadrature_order < 9:
            raise ValueError("quadrature_order must be at least 9")
        numeric_fields = (
            "noise_std",
            "cornet_outcome_norm",
            "cornet_delta_rms",
            "cornet_sparse_ratio",
            "kps_kappa",
            "kps_nu0",
            "kps_nu1",
            "int_r_cf",
            "int_r_confounding_strength",
            "int_r_rct_selection_intercept",
            "elastic_mean_x",
            "elastic_tlocalpar",
            "elastic_rct_selection_intercept",
            "power_omega",
            "power_copula_intercept",
            "power_copula_treatment_coef",
            "cvci_true_te",
            "cvci_te_bias",
            "cvci_obs_propensity",
        )
        if any(
            not np.isfinite(float(getattr(self, field_name)))
            for field_name in numeric_fields
        ):
            raise ValueError("paper DGP numeric controls must be finite")
        if self.paper_id == "cornet_official_n_conf":
            if self.cornet_outcome_norm <= 0.0:
                raise ValueError("cornet_outcome_norm must be positive")
            if self.cornet_delta_rms < 0.0:
                raise ValueError("cornet_delta_rms must be nonnegative")
            if not 0.0 <= self.cornet_sparse_ratio < 1.0:
                raise ValueError("cornet_sparse_ratio must lie in [0,1)")
            zero_count = int(self.cornet_sparse_ratio * self.d)
            if zero_count > self.d - 1:
                raise ValueError(
                    "CorNet sparsification excludes coordinate zero and cannot "
                    "zero more than d-1 weights"
                )
        if self.paper_id in {
            "experimental_grounding_kps",
            "integrative_r_learner_kps_normal_target_extension",
            "integrative_r_learner_kps_code",
        }:
            if self.d != 1:
                raise ValueError("KPS paper translations require d=1")
            if not 0.0 <= abs(self.kps_kappa) < 1.0:
                raise ValueError("abs(kps_kappa) must be less than one")
        if self.paper_id == "integrative_r_learner_multivariate_paper":
            if self.d < 2:
                raise ValueError("integrative R-learner Simulation II requires d>=2")
            if self.int_r_confounding_strength < 0.0:
                raise ValueError("int_r_confounding_strength must be nonnegative")
            if not 0 <= self.int_r_zero_cate_coefficients <= self.d:
                raise ValueError("invalid number of zero CATE coefficients")
        if self.paper_id == "elastic_integrative_gen_data_overlap":
            if self.d != 2:
                raise ValueError("Elastic Integrative overlap translation requires d=2")
            if self.elastic_tlocalpar < 0.0:
                raise ValueError("elastic_tlocalpar must be nonnegative")
        if self.paper_id.startswith("power_likelihood"):
            if self.d < 2:
                raise ValueError("power-likelihood translations require d>=2")
            if self.power_u_distribution not in {"binary", "normal"}:
                raise ValueError("unknown power-likelihood hidden distribution")
            expected = (
                "binary"
                if self.paper_id.endswith("paper_text")
                else "normal"
            )
            if self.power_u_distribution != expected:
                raise ValueError(
                    "power-likelihood paper_id and hidden distribution disagree"
                )
            if self.power_omega < 0.0:
                raise ValueError("power_omega must be nonnegative")
        if self.paper_id == "cvci_official_linear_bias":
            if not 0.0 < self.cvci_obs_propensity < 1.0:
                raise ValueError("cvci_obs_propensity must lie in (0,1)")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative")


PAPER_SCIENTIFIC_DESIGN_FIELDS: tuple[str, ...] = (
    "noise_std",
    "cornet_outcome_norm",
    "cornet_delta_rms",
    "cornet_sparse_ratio",
    "kps_kappa",
    "kps_nu0",
    "kps_nu1",
    "int_r_cf",
    "int_r_confounding_strength",
    "int_r_rct_selection_intercept",
    "int_r_zero_cate_coefficients",
    "elastic_mean_x",
    "elastic_tlocalpar",
    "elastic_rct_selection_intercept",
    "power_omega",
    "power_u_distribution",
    "power_copula_intercept",
    "power_copula_treatment_coef",
    "cvci_true_te",
    "cvci_te_bias",
    "cvci_obs_propensity",
)


def _paper_reference_config(cfg: PaperDGPConfig) -> PaperDGPConfig:
    preset = dict(PAPER_DGP_CARDS[cfg.paper_id]["preset"])
    return PaperDGPConfig(name=cfg.name, paper_id=cfg.paper_id, **preset)


def paper_design_overrides(cfg: PaperDGPConfig) -> dict[str, Any]:
    reference = _paper_reference_config(cfg)
    return {
        field_name: getattr(cfg, field_name)
        for field_name in PAPER_SCIENTIFIC_DESIGN_FIELDS
        if getattr(cfg, field_name) != getattr(reference, field_name)
    }


def paper_axis_levels(cfg: PaperDGPConfig) -> dict[str, str]:
    levels = dict(PAPER_DGP_AXIS_LEVELS[cfg.paper_id])
    if not paper_design_overrides(cfg):
        return levels
    if cfg.paper_id.startswith("power_likelihood"):
        levels["hidden_distribution"] = cfg.power_u_distribution
        if cfg.power_omega == 0.0:
            levels.update(
                {
                    "discrepancy_origin": "none",
                    "confounding_geometry": "none",
                    "clean_region": "all",
                    "reliability_resolution": "not_applicable",
                    "arm_structure": "symmetric",
                    "bias_cate_alignment": "not_applicable",
                }
            )
        elif cfg.power_u_distribution == "binary":
            levels["discrepancy_origin"] = (
                "hidden_selection_and_source_baseline"
            )
        else:
            levels["discrepancy_origin"] = "hidden_treatment_selection"
        return levels
    if cfg.paper_id == "cvci_official_linear_bias":
        if cfg.cvci_te_bias == 0.0:
            levels["arm_structure"] = "baseline_source_shift_only"
            levels["transportability"] = "holds"
        else:
            levels["arm_structure"] = "constant_source_cate_shift"
            levels["transportability"] = "source_cate_differs"
        return levels
    if cfg.paper_id == "cornet_official_n_conf":
        if cfg.cornet_delta_rms == 0.0:
            levels.update(
                {
                    "discrepancy_origin": "none",
                    "clean_region": "all",
                    "reliability_resolution": "not_applicable",
                    "arm_structure": "symmetric",
                    "bias_cate_alignment": "not_applicable",
                    "transportability": "holds",
                }
            )
        return levels
    if cfg.paper_id in {
        "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code",
    }:
        hidden_association = (
            cfg.kps_kappa != 0.0 or cfg.kps_nu0 != cfg.kps_nu1
        )
        source_baseline = (cfg.kps_nu0 + cfg.kps_nu1) / 2.0 != 0.0
        if not hidden_association and not source_baseline:
            levels.update(
                {
                    "discrepancy_origin": "none",
                    "confounding_geometry": "none",
                    "clean_region": "all",
                    "reliability_resolution": "not_applicable",
                    "arm_structure": "symmetric",
                    "bias_cate_alignment": "not_applicable",
                }
            )
        elif hidden_association and source_baseline:
            levels["discrepancy_origin"] = (
                "treatment_conditioned_hidden_and_source_baseline"
            )
            levels["clean_region"] = "none"
        elif source_baseline:
            levels.update(
                {
                    "discrepancy_origin": "source_outcome_shift",
                    "confounding_geometry": "none",
                    "clean_region": "none",
                    "reliability_resolution": "global",
                    "arm_structure": "symmetric_source_shift",
                    "bias_cate_alignment": "not_applicable",
                }
            )
        else:
            levels["discrepancy_origin"] = (
                "treatment_conditioned_hidden_association"
            )
            levels["clean_region"] = (
                "point_only"
                if cfg.kps_kappa != 0.0
                and np.isclose(cfg.kps_nu0 + cfg.kps_nu1, 0.0)
                else "none"
            )
        return levels
    if cfg.paper_id == "integrative_r_learner_multivariate_paper":
        if cfg.int_r_confounding_strength == 0.0:
            levels.update(
                {
                    "discrepancy_origin": "none",
                    "confounding_geometry": "none",
                    "clean_region": "all",
                    "reliability_resolution": "not_applicable",
                    "arm_structure": "symmetric",
                    "bias_cate_alignment": "not_applicable",
                }
            )
        return levels
    if cfg.paper_id == "elastic_integrative_gen_data_overlap":
        if cfg.elastic_tlocalpar == 0.0:
            levels.update(
                {
                    "discrepancy_origin": "none",
                    "confounding_geometry": "none",
                    "clean_region": "all",
                    "reliability_resolution": "not_applicable",
                    "arm_structure": "symmetric",
                    "bias_cate_alignment": "not_applicable",
                }
            )
        return levels
    return levels


def make_paper_config(
    *, name: str, paper_id: str, **overrides: Any
) -> PaperDGPConfig:
    if paper_id not in PAPER_DGP_CARDS:
        raise ValueError(f"unknown paper DGP: {paper_id}")
    preset = dict(PAPER_DGP_CARDS[paper_id]["preset"])
    preset.update(overrides)
    return PaperDGPConfig(name=name, paper_id=paper_id, **preset)


def _sigmoid(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _fingerprint(arrays: Mapping[str, FloatArray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=np.float64)
        digest.update(key.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _cornet_official_mechanism(cfg: PaperDGPConfig) -> dict[str, FloatArray]:
    """Create the public CorNet DGP architecture with a local NumPy RNG.

    Three d-to-d representation layers use tanh after the first two layers;
    four linear heads supply base outcomes and source corrections. Linear
    parameters use the uniform bounds of torch.nn.Linear initialization.
    NumPy PCG64 replaces the source's unseeded PyTorch/NumPy RNGs, so seeded
    parameter values do not match the upstream implementation. No global RNG
    state, external code loader, or estimator implementation is involved.
    """

    rng = np.random.default_rng(cfg.mechanism_seed)
    bound = 1.0 / np.sqrt(cfg.d)
    mechanism: dict[str, FloatArray] = {}
    for index in range(3):
        mechanism[f"rep_weight_{index}"] = rng.uniform(
            -bound, bound, size=(cfg.d, cfg.d)
        )
        mechanism[f"rep_bias_{index}"] = rng.uniform(-bound, bound, size=cfg.d)
    for label in ("base", "correction"):
        for arm in (0, 1):
            weight = rng.uniform(-bound, bound, size=cfg.d)
            if label == "base":
                norm = float(np.linalg.norm(weight))
                if norm == 0.0:
                    raise RuntimeError("CorNet initialized a zero outcome head")
                weight *= cfg.cornet_outcome_norm / norm
            mechanism[f"{label}_weight_{arm}"] = weight
            mechanism[f"{label}_bias_{arm}"] = rng.uniform(-bound, bound, size=1)
    zero_count = int(cfg.cornet_sparse_ratio * cfg.d)
    for arm in (1, 0):
        indices = rng.choice(np.arange(1, cfg.d), size=zero_count, replace=False)
        mechanism[f"correction_weight_{arm}"][indices] = 0.0
    # An independent 10,000-row standard-normal reference fixes correction RMS.
    # Its separate seed domain makes the convention explicit and reproducible.
    reference_rng = np.random.default_rng(
        np.random.SeedSequence(cfg.mechanism_seed, spawn_key=(0x434F524E,))
    )
    reference = reference_rng.standard_normal((10_000, cfg.d))
    _, _, correction0, correction1, _, _ = _cornet_components(reference, mechanism)
    raw_rms = float(np.sqrt(np.mean((correction1 - correction0) ** 2)))
    if raw_rms == 0.0:
        raise RuntimeError("CorNet initialized a zero correction CATE")
    scale = cfg.cornet_delta_rms / raw_rms
    for arm in (0, 1):
        mechanism[f"correction_weight_{arm}"] *= scale
        mechanism[f"correction_bias_{arm}"] *= scale
    mechanism["numpy_setup"] = np.array(
        [3.0, cfg.d, cfg.cornet_outcome_norm, cfg.cornet_delta_rms,
         cfg.cornet_sparse_ratio, raw_rms], dtype=np.float64
    )
    return mechanism


def _cornet_components(
    X: FloatArray, mechanism: Mapping[str, FloatArray]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    representation = X
    activations: list[FloatArray] = []
    for index in range(3):
        representation = (
            representation @ mechanism[f"rep_weight_{index}"].T
            + mechanism[f"rep_bias_{index}"]
        )
        if index < 2:
            representation = np.tanh(representation)
            activations.append(representation)
    base0 = (
        representation @ mechanism["base_weight_0"]
        + mechanism["base_bias_0"][0]
    )
    base1 = (
        representation @ mechanism["base_weight_1"]
        + mechanism["base_bias_1"][0]
    )
    correction0 = (
        representation @ mechanism["correction_weight_0"]
        + mechanism["correction_bias_0"][0]
    )
    correction1 = (
        representation @ mechanism["correction_weight_1"]
        + mechanism["correction_bias_1"][0]
    )

    # Gradient norm of the target CATE through the two tanh layers.  This is a
    # diagnostic only and never enters training or method selection.
    target_head = (
        mechanism["base_weight_1"]
        + mechanism["correction_weight_1"]
        - mechanism["base_weight_0"]
        - mechanism["correction_weight_0"]
    )
    gradient = np.broadcast_to(target_head, representation.shape).copy()
    gradient = gradient @ mechanism["rep_weight_2"]
    gradient *= 1.0 - activations[1] ** 2
    gradient = gradient @ mechanism["rep_weight_1"]
    gradient *= 1.0 - activations[0] ** 2
    gradient = gradient @ mechanism["rep_weight_0"]
    complexity = np.linalg.norm(gradient, axis=1)
    return base0, base1, correction0, correction1, complexity, representation


def _normal_logistic_expectation(
    *, intercept: float, scale: float, order: int
) -> float:
    nodes, raw_weights = hermgauss(order)
    standard_normal = np.sqrt(2.0) * nodes
    weights = raw_weights / np.sqrt(np.pi)
    return float(np.sum(_sigmoid(intercept + scale * standard_normal) * weights))


def _mechanism(cfg: PaperDGPConfig) -> dict[str, FloatArray]:
    if cfg.paper_id == "cornet_official_n_conf":
        return _cornet_official_mechanism(cfg)
    rng = np.random.default_rng(cfg.mechanism_seed)
    if cfg.paper_id in {
        "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code",
    }:
        return {
            "fixed_parameters": np.array(
                [
                    cfg.kps_kappa,
                    cfg.kps_nu0,
                    cfg.kps_nu1,
                    cfg.int_r_cf,
                ],
                dtype=np.float64,
            )
        }
    if cfg.paper_id == "integrative_r_learner_multivariate_paper":
        coefficients = np.ones(cfg.d + 1, dtype=np.float64)
        if cfg.int_r_zero_cate_coefficients:
            coefficients[-cfg.int_r_zero_cate_coefficients :] = 0.0
        selection_probability = _normal_logistic_expectation(
            intercept=-cfg.int_r_rct_selection_intercept,
            scale=np.sqrt(2.0),
            order=cfg.quadrature_order,
        )
        return {
            "cate_coefficients": coefficients,
            "rct_selection_probability": np.array(
                [selection_probability], dtype=np.float64
            ),
        }
    if cfg.paper_id == "elastic_integrative_gen_data_overlap":
        selection_probability = _normal_logistic_expectation(
            intercept=(
                cfg.elastic_rct_selection_intercept
                - 4.0 * cfg.elastic_mean_x
            ),
            scale=np.sqrt(8.0),
            order=cfg.quadrature_order,
        )
        return {
            "baseline_coefficients": np.array(
                [0.0, 1.0, 1.0, 1.0], dtype=np.float64
            ),
            "cate_coefficients": np.array(
                [0.0, 1.0, 1.0], dtype=np.float64
            ),
            "rct_selection_probability": np.array(
                [selection_probability], dtype=np.float64
            ),
        }
    if cfg.paper_id.startswith("power_likelihood"):
        return {
            "fixed_parameters": np.array(
                [
                    cfg.power_omega,
                    cfg.power_copula_intercept,
                    cfg.power_copula_treatment_coef,
                ],
                dtype=np.float64,
            )
        }
    if cfg.paper_id == "cvci_official_linear_bias":
        coefficients = rng.standard_normal(2 * cfg.d)
        return {
            "target_baseline_coef": coefficients[: cfg.d],
            "obs_baseline_coef": coefficients[cfg.d :],
        }
    raise ValueError(f"unknown paper DGP: {cfg.paper_id}")


def _hidden_discrepancy_truth(
    *,
    mu0: FloatArray,
    tau: FloatArray,
    c0: FloatArray,
    c1: FloatArray,
    propensity: FloatArray,
    rct_density: FloatArray,
    cate_complexity: FloatArray,
    obs_density: FloatArray | None = None,
    source_shift_0: FloatArray | None = None,
    source_shift_1: FloatArray | None = None,
) -> dict[str, FloatArray]:
    rows = mu0.shape[0]
    zero = np.zeros(rows)
    one = np.ones(rows)
    d0 = zero if source_shift_0 is None else source_shift_0
    d1 = zero if source_shift_1 is None else source_shift_1
    r0 = c0 + d0
    r1 = c1 + d1
    confounding_magnitude = np.sqrt(0.5 * (c0**2 + c1**2))
    total_magnitude = np.sqrt(0.5 * (r0**2 + r1**2))
    return {
        "mu0_target": mu0,
        "mu1_target": mu0 + tau,
        "tau_target": tau,
        "confounding_bias_0": c0,
        "confounding_bias_1": c1,
        "source_shift_0": d0,
        "source_shift_1": d1,
        "regression_discrepancy_0": r0,
        "regression_discrepancy_1": r1,
        "confounding_cate_bias": c1 - c0,
        "source_cate_shift": d1 - d0,
        "total_cate_discrepancy": r1 - r0,
        "obs_propensity": propensity,
        "expected_severity": total_magnitude,
        "confounding_bias_magnitude": confounding_magnitude,
        "total_discrepancy_magnitude": total_magnitude,
        "bias_magnitude": total_magnitude,
        "region_score": total_magnitude,
        "cate_complexity_score": cate_complexity,
        "obs_density": one if obs_density is None else obs_density,
        "rct_density": rct_density,
    }


def _kps_truth(X: FloatArray, cfg: PaperDGPConfig) -> dict[str, FloatArray]:
    x = X[:, 0]
    if cfg.paper_id == "experimental_grounding_kps":
        mu0 = 1.0 + x + 0.5 * x**2
        tau = 1.0 + x + 0.25 * x**2
    else:
        mu0 = 1.0 + x + 0.5 * x**2
        tau = 1.0 + 2.0 * x + cfg.int_r_cf * x**2
    source_hidden_mean = np.full(
        X.shape[0], 0.5 * (cfg.kps_nu0 + cfg.kps_nu1)
    )
    c0 = cfg.kps_nu0 - cfg.kps_kappa * x - source_hidden_mean
    c1 = cfg.kps_nu1 + cfg.kps_kappa * x - source_hidden_mean
    if cfg.paper_id == "integrative_r_learner_kps_code":
        # The paper and released gen_data use Uniform[-1,1] for RCT/test X.
        rct_density = np.where(np.abs(x) <= 1.0, 1.0, 0.0)
        obs_density = (
            2.0 * np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
        )
    else:
        target_density = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
        rct_density = np.where(
            np.abs(x) <= 1.0,
            0.5 / np.maximum(target_density, 1e-300),
            0.0,
        )
        obs_density = np.ones(X.shape[0])
    derivative = (
        1.0 + 0.5 * x
        if cfg.paper_id == "experimental_grounding_kps"
        else 2.0 + 2.0 * cfg.int_r_cf * x
    )
    return _hidden_discrepancy_truth(
        mu0=mu0,
        tau=tau,
        c0=c0,
        c1=c1,
        propensity=np.full(X.shape[0], 0.5),
        rct_density=rct_density,
        cate_complexity=np.abs(derivative),
        obs_density=obs_density,
        source_shift_0=source_hidden_mean,
        source_shift_1=source_hidden_mean,
    )


def _normal_hidden_moments(
    base_logit: FloatArray,
    *,
    hidden_coefficient: float,
    hidden_mean: float,
    order: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    nodes, raw_weights = hermgauss(order)
    hidden = hidden_mean + np.sqrt(2.0) * nodes
    weights = raw_weights / np.sqrt(np.pi)
    individual = _sigmoid(
        base_logit[:, None] + hidden_coefficient * hidden[None, :]
    )
    propensity = np.sum(individual * weights[None, :], axis=1)
    propensity = np.clip(propensity, 1e-12, 1.0 - 1e-12)
    treated = np.sum(
        hidden[None, :] * individual * weights[None, :], axis=1
    ) / propensity
    control = np.sum(
        hidden[None, :]
        * (1.0 - individual)
        * weights[None, :],
        axis=1,
    ) / (1.0 - propensity)
    return propensity, control, treated


def _integrative_r_truth(
    X: FloatArray, cfg: PaperDGPConfig, mechanism: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    base_logit = -1.0 - X[:, 0] + X[:, 1]
    propensity, hidden0, hidden1 = _normal_hidden_moments(
        base_logit,
        hidden_coefficient=cfg.int_r_confounding_strength,
        hidden_mean=0.0,
        order=cfg.quadrature_order,
    )
    design = np.column_stack((np.ones(X.shape[0]), X))
    tau = design @ mechanism["cate_coefficients"]
    mu0 = X[:, 0]
    selection = _sigmoid(
        -cfg.int_r_rct_selection_intercept - X[:, 0] + X[:, 1]
    )
    rct_density = selection / mechanism["rct_selection_probability"][0]
    complexity = np.full(
        X.shape[0],
        np.linalg.norm(mechanism["cate_coefficients"][1:]),
    )
    return _hidden_discrepancy_truth(
        mu0=mu0,
        tau=tau,
        c0=hidden0,
        c1=hidden1,
        propensity=propensity,
        rct_density=rct_density,
        cate_complexity=complexity,
    )


def _elastic_truth(
    X: FloatArray, cfg: PaperDGPConfig, mechanism: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    base_logit = cfg.elastic_tlocalpar - X[:, 0] + X[:, 1]
    propensity, hidden0, hidden1 = _normal_hidden_moments(
        base_logit,
        hidden_coefficient=-cfg.elastic_tlocalpar,
        hidden_mean=cfg.elastic_mean_x,
        order=cfg.quadrature_order,
    )
    mu0 = X[:, 0] + X[:, 1] + cfg.elastic_mean_x
    tau = X[:, 0] + X[:, 1]
    c0 = hidden0 - cfg.elastic_mean_x
    c1 = hidden1 - cfg.elastic_mean_x
    selection = _sigmoid(
        cfg.elastic_rct_selection_intercept - 2.0 * X[:, 0] - 2.0 * X[:, 1]
    )
    rct_density = selection / mechanism["rct_selection_probability"][0]
    return _hidden_discrepancy_truth(
        mu0=mu0,
        tau=tau,
        c0=c0,
        c1=c1,
        propensity=propensity,
        rct_density=rct_density,
        cate_complexity=np.full(X.shape[0], np.sqrt(2.0)),
    )


def _cornet_truth(
    X: FloatArray, mechanism: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    base0, base1, correction0, correction1, complexity, _ = (
        _cornet_components(X, mechanism)
    )
    mu0 = base0 + correction0
    mu1 = base1 + correction1
    # In the official generator the so-called confounded/OBS outcome omits
    # delta, while the RCT and target outcome include it.  Hence this is a
    # source shift d_t=-delta_t, with c_t identically zero.
    d0 = -correction0
    d1 = -correction1
    zero = np.zeros(X.shape[0])
    one = np.ones(X.shape[0])
    magnitude = np.sqrt(0.5 * (d0**2 + d1**2))
    return {
        "mu0_target": mu0,
        "mu1_target": mu1,
        "tau_target": mu1 - mu0,
        "confounding_bias_0": zero,
        "confounding_bias_1": zero,
        "source_shift_0": d0,
        "source_shift_1": d1,
        "regression_discrepancy_0": d0,
        "regression_discrepancy_1": d1,
        "confounding_cate_bias": zero,
        "source_cate_shift": d1 - d0,
        "total_cate_discrepancy": d1 - d0,
        "obs_propensity": np.full(X.shape[0], 0.5),
        "expected_severity": magnitude,
        "confounding_bias_magnitude": zero,
        "total_discrepancy_magnitude": magnitude,
        "bias_magnitude": magnitude,
        "region_score": magnitude,
        "cate_complexity_score": complexity,
        "obs_density": one,
        "rct_density": one,
    }


def _power_hidden_moments(
    base_logit: FloatArray, cfg: PaperDGPConfig
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Return P(T=1|X), E[U|T=0,X], E[U|T=1,X], and E[U]."""

    if cfg.power_u_distribution == "binary":
        hidden_values = np.array([0.0, 1.0], dtype=np.float64)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        hidden_mean = 0.5
    else:
        nodes, raw_weights = hermgauss(cfg.quadrature_order)
        hidden_values = np.sqrt(2.0) * nodes
        weights = raw_weights / np.sqrt(np.pi)
        hidden_mean = 0.0
    individual = _sigmoid(
        base_logit[:, None]
        + cfg.power_omega * hidden_values[None, :]
    )
    propensity = np.sum(individual * weights[None, :], axis=1)
    propensity = np.clip(propensity, 1e-12, 1.0 - 1e-12)
    treated_mean = np.sum(
        hidden_values[None, :] * individual * weights[None, :], axis=1
    ) / propensity
    control_mean = np.sum(
        hidden_values[None, :]
        * (1.0 - individual)
        * weights[None, :],
        axis=1,
    ) / (1.0 - propensity)
    return propensity, control_mean, treated_mean, hidden_mean


def _power_truth(X: FloatArray, cfg: PaperDGPConfig) -> dict[str, FloatArray]:
    C = X[:, 0]
    Z = X[:, 1]
    z_residual = Z - (1.0 + C)
    base_logit = -3.0 + C + Z + C * Z
    propensity, hidden0, hidden1, hidden_mean = _power_hidden_moments(
        base_logit, cfg
    )
    rho0 = float(2.0 * _sigmoid(np.array([cfg.power_copula_intercept]))[0] - 1.0)
    rho1 = float(
        2.0
        * _sigmoid(
            np.array(
                [
                    cfg.power_copula_intercept
                    + cfg.power_copula_treatment_coef
                ]
            )
        )[0]
        - 1.0
    )
    mu0 = 1.0 + C + cfg.noise_std * rho0 * z_residual
    mu1 = (
        1.0
        + C
        + 0.1
        + 0.1 * C
        + cfg.noise_std * rho1 * z_residual
    )
    tau = mu1 - mu0
    c0 = cfg.power_omega * (hidden0 - hidden_mean)
    c1 = cfg.power_omega * (hidden1 - hidden_mean)
    d0 = np.full(X.shape[0], cfg.power_omega * hidden_mean)
    d1 = d0.copy()
    r0 = c0 + d0
    r1 = c1 + d1
    confounding_magnitude = np.sqrt(0.5 * (c0**2 + c1**2))
    total_magnitude = np.sqrt(0.5 * (r0**2 + r1**2))
    one = np.ones(X.shape[0])
    zero = np.zeros(X.shape[0])
    return {
        "mu0_target": mu0,
        "mu1_target": mu1,
        "tau_target": tau,
        "paper_subgroup_tau_c": 0.1 + 0.1 * C,
        "confounding_bias_0": c0,
        "confounding_bias_1": c1,
        "source_shift_0": d0,
        "source_shift_1": d1,
        "regression_discrepancy_0": r0,
        "regression_discrepancy_1": r1,
        "confounding_cate_bias": c1 - c0,
        "source_cate_shift": zero,
        "total_cate_discrepancy": c1 - c0,
        "obs_propensity": propensity,
        "expected_severity": total_magnitude,
        "confounding_bias_magnitude": confounding_magnitude,
        "total_discrepancy_magnitude": total_magnitude,
        "bias_magnitude": total_magnitude,
        "region_score": total_magnitude,
        "cate_complexity_score": np.full(X.shape[0], abs(rho1 - rho0)),
        "obs_density": one,
        "rct_density": one,
    }


def _cvci_truth(
    X: FloatArray, cfg: PaperDGPConfig, mechanism: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    mu0 = X @ mechanism["target_baseline_coef"]
    tau = np.full(X.shape[0], cfg.cvci_true_te)
    baseline_shift = X @ (
        mechanism["obs_baseline_coef"] - mechanism["target_baseline_coef"]
    )
    d0 = baseline_shift
    d1 = baseline_shift + cfg.cvci_te_bias
    zero = np.zeros(X.shape[0])
    one = np.ones(X.shape[0])
    magnitude = np.sqrt(0.5 * (d0**2 + d1**2))
    return {
        "mu0_target": mu0,
        "mu1_target": mu0 + tau,
        "tau_target": tau,
        "confounding_bias_0": zero,
        "confounding_bias_1": zero,
        "source_shift_0": d0,
        "source_shift_1": d1,
        "regression_discrepancy_0": d0,
        "regression_discrepancy_1": d1,
        "confounding_cate_bias": zero,
        "source_cate_shift": np.full(X.shape[0], cfg.cvci_te_bias),
        "total_cate_discrepancy": np.full(X.shape[0], cfg.cvci_te_bias),
        "obs_propensity": np.full(X.shape[0], cfg.cvci_obs_propensity),
        "expected_severity": magnitude,
        "confounding_bias_magnitude": zero,
        "total_discrepancy_magnitude": magnitude,
        "bias_magnitude": magnitude,
        "region_score": magnitude,
        "cate_complexity_score": zero,
        "obs_density": one,
        "rct_density": one,
    }


def paper_truth_at(
    X: FloatArray,
    cfg: PaperDGPConfig,
    mechanism: Mapping[str, FloatArray] | None = None,
) -> dict[str, FloatArray]:
    values = _mechanism(cfg) if mechanism is None else mechanism
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != cfg.d or not np.isfinite(X).all():
        raise ValueError("X must be a finite matrix with cfg.d columns")
    if cfg.paper_id == "cornet_official_n_conf":
        return _cornet_truth(X, values)
    if cfg.paper_id in {
        "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code",
    }:
        return _kps_truth(X, cfg)
    if cfg.paper_id == "integrative_r_learner_multivariate_paper":
        return _integrative_r_truth(X, cfg, values)
    if cfg.paper_id == "elastic_integrative_gen_data_overlap":
        return _elastic_truth(X, cfg, values)
    if cfg.paper_id.startswith("power_likelihood"):
        return _power_truth(X, cfg)
    if cfg.paper_id == "cvci_official_linear_bias":
        return _cvci_truth(X, cfg, values)
    raise ValueError(f"unknown paper DGP: {cfg.paper_id}")


def _draw_selected_normal(
    rng: np.random.Generator,
    n: int,
    *,
    d: int,
    mean: float,
    selection_probability: Any,
) -> FloatArray:
    accepted: list[FloatArray] = []
    accepted_rows = 0
    proposed_rows = 0
    maximum_proposals = max(1_000_000, 20_000 * n)
    while accepted_rows < n:
        batch_size = max(4_096, 8 * (n - accepted_rows))
        candidates = mean + rng.standard_normal((batch_size, d))
        probability = np.asarray(
            selection_probability(candidates), dtype=np.float64
        )
        if probability.shape != (batch_size,):
            raise RuntimeError("trial-selection probability has invalid shape")
        keep = rng.random(batch_size) < probability
        if keep.any():
            chosen = candidates[keep]
            accepted.append(chosen)
            accepted_rows += chosen.shape[0]
        proposed_rows += batch_size
        if proposed_rows > maximum_proposals and accepted_rows < n:
            raise RuntimeError("trial-participant rejection sampler stalled")
    return np.concatenate(accepted, axis=0)[:n]


def _draw_covariates(
    rng: np.random.Generator,
    n: int,
    cfg: PaperDGPConfig,
    *,
    source: Literal["obs", "rct", "eval"] = "eval",
) -> FloatArray:
    if cfg.paper_id in {
        "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code",
    }:
        if source == "rct" or (
            source == "eval"
            and cfg.paper_id == "integrative_r_learner_kps_code"
        ):
            return rng.uniform(-1.0, 1.0, size=(n, 1))
        return rng.standard_normal((n, 1))
    if cfg.paper_id == "integrative_r_learner_multivariate_paper":
        if source == "rct":
            return _draw_selected_normal(
                rng,
                n,
                d=cfg.d,
                mean=0.0,
                selection_probability=lambda values: _sigmoid(
                    -cfg.int_r_rct_selection_intercept
                    - values[:, 0]
                    + values[:, 1]
                ),
            )
        return rng.standard_normal((n, cfg.d))
    if cfg.paper_id == "elastic_integrative_gen_data_overlap":
        if source == "rct":
            return _draw_selected_normal(
                rng,
                n,
                d=2,
                mean=cfg.elastic_mean_x,
                selection_probability=lambda values: _sigmoid(
                    cfg.elastic_rct_selection_intercept
                    - 2.0 * values[:, 0]
                    - 2.0 * values[:, 1]
                ),
            )
        return cfg.elastic_mean_x + rng.standard_normal((n, 2))
    if not cfg.paper_id.startswith("power_likelihood"):
        return rng.standard_normal((n, cfg.d))
    C = rng.binomial(1, 0.5, size=n).astype(np.float64)
    Z = 1.0 + C + rng.standard_normal(n)
    columns = [C[:, None], Z[:, None]]
    if cfg.d > 2:
        columns.append(rng.standard_normal((n, cfg.d - 2)))
    return np.concatenate(columns, axis=1)


def _draw_source(
    rng: np.random.Generator,
    n: int,
    cfg: PaperDGPConfig,
    mechanism: Mapping[str, FloatArray],
    *,
    source: Literal["obs", "rct"],
    role: str,
) -> tuple[SourceSample, dict[str, FloatArray], dict[str, float]]:
    X = _draw_covariates(rng, n, cfg, source=source)
    truth = paper_truth_at(X, cfg, mechanism)
    realized: dict[str, FloatArray] = {}
    if cfg.paper_id == "cornet_official_n_conf":
        assignment = np.full(n, 0.5)
        T = rng.binomial(1, assignment).astype(np.float64)
        base0, base1, _, _, _, _ = _cornet_components(X, mechanism)
        if source == "obs":
            mean = np.where(T == 1.0, base1, base0)
        else:
            mean = np.where(
                T == 1.0, truth["mu1_target"], truth["mu0_target"]
            )
        Y = mean + cfg.noise_std * rng.standard_normal(n)
        realized = {"individual_treatment_probability": assignment}
    elif cfg.paper_id in {
        "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code",
    }:
        assignment = np.full(n, 0.5)
        T = rng.binomial(1, assignment).astype(np.float64)
        x = X[:, 0]
        if source == "obs":
            hidden_mean = np.where(T == 1.0, cfg.kps_nu1, cfg.kps_nu0)
            covariance = np.where(
                T == 1.0, cfg.kps_kappa, -cfg.kps_kappa
            )
            hidden = (
                hidden_mean
                + covariance * x
                + np.sqrt(1.0 - cfg.kps_kappa**2) * rng.standard_normal(n)
            )
        else:
            hidden = rng.standard_normal(n)
        if cfg.paper_id == "experimental_grounding_kps":
            base0 = 1.0 + x + 0.5 * x**2
            base1 = 2.0 + 2.0 * x + 0.75 * x**2
        else:
            base0 = 1.0 + x + 0.5 * x**2
            base1 = base0 + 1.0 + 2.0 * x + cfg.int_r_cf * x**2
        Y = (
            np.where(T == 1.0, base1, base0)
            + hidden
            + cfg.noise_std * rng.standard_normal(n)
        )
        realized = {
            "individual_treatment_probability": assignment,
            "hidden_confounder": hidden,
        }
    elif cfg.paper_id == "integrative_r_learner_multivariate_paper":
        hidden = rng.standard_normal(n)
        if source == "obs":
            assignment = _sigmoid(
                -1.0
                - X[:, 0]
                + X[:, 1]
                + cfg.int_r_confounding_strength * hidden
            )
        else:
            assignment = _sigmoid(-1.0 - X[:, 0] + X[:, 1])
        T = rng.binomial(1, assignment).astype(np.float64)
        design = np.column_stack((np.ones(n), X))
        tau = design @ mechanism["cate_coefficients"]
        Y = (
            X[:, 0]
            + hidden
            + T * tau
            + cfg.noise_std * rng.standard_normal(n)
        )
        realized = {
            "individual_treatment_probability": assignment,
            "hidden_confounder": hidden,
        }
    elif cfg.paper_id == "elastic_integrative_gen_data_overlap":
        hidden = cfg.elastic_mean_x + rng.standard_normal(n)
        if source == "obs":
            assignment = _sigmoid(
                cfg.elastic_tlocalpar
                - X[:, 0]
                + X[:, 1]
                - cfg.elastic_tlocalpar * hidden
            )
        else:
            assignment = np.full(n, 0.5)
        T = rng.binomial(1, assignment).astype(np.float64)
        tau = X[:, 0] + X[:, 1]
        Y = (
            X[:, 0]
            + X[:, 1]
            + hidden
            + T * tau
            + cfg.noise_std * rng.standard_normal(n)
        )
        realized = {
            "individual_treatment_probability": assignment,
            "hidden_confounder": hidden,
        }
    elif cfg.paper_id.startswith("power_likelihood"):
        C = X[:, 0]
        Z = X[:, 1]
        if cfg.power_u_distribution == "binary":
            hidden = rng.binomial(1, 0.5, size=n).astype(np.float64)
        else:
            hidden = rng.standard_normal(n)
        base_logit = -3.0 + C + Z + C * Z
        if source == "obs":
            assignment = _sigmoid(base_logit + cfg.power_omega * hidden)
            hidden_outcome = cfg.power_omega * hidden
        else:
            assignment = np.full(n, 0.5)
            hidden_outcome = np.zeros(n)
        T = rng.binomial(1, assignment).astype(np.float64)
        rho = 2.0 * _sigmoid(
            cfg.power_copula_intercept
            + cfg.power_copula_treatment_coef * T
        ) - 1.0
        z_residual = Z - (1.0 + C)
        Y = (
            1.0
            + C
            + 0.1 * T
            + 0.1 * C * T
            + hidden_outcome
            + cfg.noise_std
            * (
                rho * z_residual
                + np.sqrt(np.maximum(1.0 - rho**2, 0.0))
                * rng.standard_normal(n)
            )
        )
        realized = {
            "individual_treatment_probability": assignment,
            "hidden_confounder": hidden,
        }
    elif cfg.paper_id == "cvci_official_linear_bias":
        assignment = (
            np.full(n, cfg.cvci_obs_propensity)
            if source == "obs"
            else np.full(n, 0.5)
        )
        T = rng.binomial(1, assignment).astype(np.float64)
        baseline = X @ mechanism[
            "obs_baseline_coef" if source == "obs" else "target_baseline_coef"
        ]
        treatment_effect = (
            cfg.cvci_true_te + cfg.cvci_te_bias
            if source == "obs"
            else cfg.cvci_true_te
        )
        Y = (
            baseline
            + T * treatment_effect
            + cfg.noise_std * rng.standard_normal(n)
        )
        realized = {"individual_treatment_probability": assignment}
    else:
        raise ValueError(f"unknown paper DGP: {cfg.paper_id}")
    hidden = realized.get("hidden_confounder")
    correlation = 0.0
    if hidden is not None and float(hidden.std()) > 0.0 and float(T.std()) > 0.0:
        correlation = float(np.corrcoef(T, hidden)[0, 1])
    diagnostics = {
        "treated_fraction": float(T.mean()),
        "t_hidden_correlation": correlation,
    }
    return SourceSample(
        X,
        T,
        Y,
        source,
        role,
        known_propensity=assignment if source == "rct" else None,
    ), realized, diagnostics


def generate_paper_problem(cfg: PaperDGPConfig, seed: int) -> SimulationProblem:
    mechanism = _mechanism(cfg)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    obs, obs_realized, obs_diag = _draw_source(
        streams[0], cfg.n_obs, cfg, mechanism, source="obs", role="obs_train"
    )
    rct_train, _, train_diag = _draw_source(
        streams[1], cfg.n_rct_train, cfg, mechanism,
        source="rct", role="rct_train",
    )
    rct_tune, _, tune_diag = _draw_source(
        streams[2], cfg.n_rct_tune, cfg, mechanism,
        source="rct", role="rct_tune",
    )
    rct_test, _, test_diag = _draw_source(
        streams[3], cfg.n_rct_test, cfg, mechanism,
        source="rct", role="rct_test",
    )
    eval_X = _draw_covariates(streams[4], cfg.n_eval, cfg, source="eval")
    eval_truth = paper_truth_at(eval_X, cfg, mechanism)
    obs_truth = paper_truth_at(obs.X, cfg, mechanism)
    obs_truth.update(obs_realized)
    truth = SimulationTruth(
        eval=eval_truth,
        obs_train=obs_truth,
        n_eval=cfg.n_eval,
        n_obs=cfg.n_obs,
        definition=(
            "c_t=hidden-treatment-selection bias; d_t=OBS-source potential-outcome "
            "shift from target; r_t=c_t+d_t"
        ),
    )
    card = dict(PAPER_DGP_CARDS[cfg.paper_id])
    axis_levels = paper_axis_levels(cfg)
    design_overrides = paper_design_overrides(cfg)
    card["axis_levels"] = axis_levels
    card["target_population"] = PAPER_TARGET_POPULATIONS[cfg.paper_id]
    metadata = {
        "generator": "paper_translation_v1",
        "generator_version": PAPER_DGP_VERSION,
        "paper_id": cfg.paper_id,
        "paper_card": card,
        "target_population": PAPER_TARGET_POPULATIONS[cfg.paper_id],
        "axis_levels": axis_levels,
        "design_overrides": design_overrides,
        "sampling_axes": {
            "n_obs": cfg.n_obs,
            "n_rct_train": cfg.n_rct_train,
            "n_rct_tune": cfg.n_rct_tune,
            "n_rct_test": cfg.n_rct_test,
            "n_eval": cfg.n_eval,
            "dimension": cfg.d,
        },
        "config": asdict(cfg),
        "mechanism_sha256": _fingerprint(mechanism),
        "obs_diagnostics": obs_diag,
        "rct_train_diagnostics": train_diag,
        "rct_tune_diagnostics": tune_diag,
        "rct_test_diagnostics": test_diag,
    }
    return SimulationProblem(
        name=cfg.name,
        seed=seed,
        obs_train=obs,
        rct_train=rct_train,
        rct_tune=rct_tune,
        rct_test=rct_test,
        eval_X=eval_X,
        eval_tau=eval_truth["tau_target"],
        eval_region_score=eval_truth["total_discrepancy_magnitude"],
        metadata=metadata,
        truth=truth,
    )
