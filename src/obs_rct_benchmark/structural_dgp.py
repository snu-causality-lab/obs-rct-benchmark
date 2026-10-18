"""Versioned structural DGP assets for the OBS--RCT CATE benchmark.

The generator is deliberately mechanism-first: it exposes hidden-treatment
selection bias, source/target potential-outcome shift, and their total
regression discrepancy as separate evaluation-only quantities.  Normal
estimators see only the role-separated observed samples in ``TrainingView``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
from statistics import NormalDist
from typing import Any, Literal, Mapping

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray

from .data import (
    SimulationProblem,
    SimulationTruth,
    SourceSample,
    TargetCovariateSample,
)


FloatArray = NDArray[np.float64]
BiasGeometry = Literal[
    "constant",
    "smooth",
    "localized",
    "interaction",
    "multi_region",
    "sign_change",
    "fixed_sign_varying_magnitude",
]
OutcomeGeometry = Literal["linear", "smooth", "interaction"]
CateGeometry = Literal[
    "constant",
    "linear",
    "smooth",
    "rare",
    "discontinuous",
    "null",
    "centered_linear",
]
Alignment = Literal["aligned", "orthogonal", "opposed"]
HiddenDistribution = Literal["normal", "binary", "normal_mixture"]
NoiseDistribution = Literal["normal", "student_t"]
SignalBasis = Literal["axis_aligned", "dense_orthogonal"]
CovariateMarginals = Literal[
    "gaussian", "symmetric_heavy_tailed", "right_skewed"
]

STRUCTURAL_DGP_VERSION = "2.0.0"
STRUCTURAL_MECHANISMS = tuple(f"M{i}" for i in range(12))
STRUCTURAL_ADDENDUM_VERSION = "2.0.0"
STRUCTURAL_ADDENDUM_CELLS = tuple(f"C{i:02d}" for i in range(1, 11))
_MARGINAL_TRANSFORM_SCALE = 0.5


@dataclass(frozen=True)
class StructuralDGPConfig:
    name: str
    mechanism: str = "custom"
    n_obs: int = 2_000
    n_rct_train: int = 200
    n_rct_tune: int = 200
    n_rct_test: int = 500
    n_eval: int = 2_000
    n_target_reference: int = 0
    d: int = 5
    signal_basis: SignalBasis = "axis_aligned"
    signal_seed: int = 2025
    confounding_coordinate_offset: int = 0
    covariate_rho: float = 0.0
    covariate_marginals: CovariateMarginals = "gaussian"

    bias_geometry: BiasGeometry = "smooth"
    clean_fraction: float = 0.5
    bias_min: float = 0.0
    resolvability: float = 1.0
    u_predictability: float = 0.0
    confounding_treatment: float = 1.0
    confounding_outcome_0: float = 1.0
    confounding_outcome_1: float = 1.0
    hidden_outcome_correlation: float = 1.0
    hidden_distribution: HiddenDistribution = "normal"
    quadrature_order: int = 21

    treatment_linear_scale: float = 1.0
    treatment_nonlinearity: float = 0.0
    propensity_floor: float = 0.01
    rct_propensity: float = 0.5
    outcome_geometry: OutcomeGeometry = "smooth"
    cate_geometry: CateGeometry = "smooth"
    cate_alignment: Alignment = "orthogonal"
    cate_scale: float = 1.0

    obs_covariate_shift: float = 0.0
    rct_covariate_shift: float = 0.0
    obs_selection_strength: float = 0.0
    rct_selection_strength: float = 0.0
    rct_hidden_selection_strength: float = 0.0
    participation_outcome_loading_0: float = 0.0
    participation_outcome_loading_1: float = 0.0
    obs_baseline_shift: float = 0.0
    obs_cate_shift: float = 0.0
    source_shift_geometry: Literal[
        "constant",
        "smooth",
        "localized",
    ] = "smooth"


    noise_std: float = 1.0
    noise_distribution: NoiseDistribution = "normal"
    noise_heteroskedasticity: float = 0.0

    def __post_init__(self) -> None:
        sample_fields = (
            "n_obs",
            "n_rct_train",
            "n_rct_tune",
            "n_rct_test",
            "n_eval",
        )
        integer_fields = sample_fields + (
            "n_target_reference",
            "d",
            "signal_seed",
            "confounding_coordinate_offset",
            "quadrature_order",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field_name} must be an integer")
        for field_name in sample_fields:
            if getattr(self, field_name) < 4:
                raise ValueError(f"{field_name} must be at least 4")
        if self.n_target_reference != 0 and self.n_target_reference < 4:
            raise ValueError("n_target_reference must be zero or at least 4")
        if self.d < 3:
            raise ValueError("d must be at least 3")
        if not 0 <= self.confounding_coordinate_offset <= self.d - 3:
            raise ValueError(
                "confounding_coordinate_offset must select three coordinates "
                "within d"
            )
        if self.signal_basis not in {"axis_aligned", "dense_orthogonal"}:
            raise ValueError(f"unknown signal basis: {self.signal_basis}")
        if self.covariate_marginals not in {
            "gaussian",
            "symmetric_heavy_tailed",
            "right_skewed",
        }:
            raise ValueError(
                f"unknown covariate marginals: {self.covariate_marginals}"
            )
        if self.signal_seed < 0:
            raise ValueError("signal_seed must be nonnegative")
        if not -0.95 < self.covariate_rho < 0.95:
            raise ValueError("covariate_rho must lie strictly between -0.95 and 0.95")
        if self.mechanism != "custom" and self.mechanism not in (
            STRUCTURAL_MECHANISMS + STRUCTURAL_ADDENDUM_CELLS
        ):
            raise ValueError(f"unknown structural mechanism: {self.mechanism}")
        if self.bias_geometry not in {
            "constant",
            "smooth",
            "localized",
            "interaction",
            "multi_region",
            "sign_change",
            "fixed_sign_varying_magnitude",
        }:
            raise ValueError(f"unknown bias geometry: {self.bias_geometry}")
        if self.outcome_geometry not in {"linear", "smooth", "interaction"}:
            raise ValueError(f"unknown outcome geometry: {self.outcome_geometry}")
        if self.cate_geometry not in {
            "constant",
            "linear",
            "smooth",
            "rare",
            "discontinuous",
            "null",
            "centered_linear",
        }:
            raise ValueError(f"unknown CATE geometry: {self.cate_geometry}")
        if self.cate_alignment not in {"aligned", "orthogonal", "opposed"}:
            raise ValueError(f"unknown CATE alignment: {self.cate_alignment}")
        if self.hidden_distribution not in {"normal", "binary", "normal_mixture"}:
            raise ValueError(f"unknown hidden distribution: {self.hidden_distribution}")
        if self.noise_distribution not in {"normal", "student_t"}:
            raise ValueError(f"unknown noise distribution: {self.noise_distribution}")
        if self.source_shift_geometry not in {
            "constant",
            "smooth",
            "localized",
        }:
            raise ValueError(f"unknown source-shift geometry: {self.source_shift_geometry}")
        numeric_fields = (
            "covariate_rho",
            "clean_fraction",
            "bias_min",
            "resolvability",
            "u_predictability",
            "confounding_treatment",
            "confounding_outcome_0",
            "confounding_outcome_1",
            "hidden_outcome_correlation",
            "treatment_linear_scale",
            "treatment_nonlinearity",
            "propensity_floor",
            "rct_propensity",
            "cate_scale",
            "obs_covariate_shift",
            "rct_covariate_shift",
            "obs_selection_strength",
            "rct_selection_strength",
            "rct_hidden_selection_strength",
            "participation_outcome_loading_0",
            "participation_outcome_loading_1",
            "obs_baseline_shift",
            "obs_cate_shift",
            "noise_std",
            "noise_heteroskedasticity",
        )
        if any(
            not np.isfinite(float(getattr(self, field_name)))
            for field_name in numeric_fields
        ):
            raise ValueError("structural DGP numeric controls must be finite")
        for field_name in ("clean_fraction", "bias_min", "resolvability", "u_predictability"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if not -1.0 <= self.hidden_outcome_correlation <= 1.0:
            raise ValueError("hidden_outcome_correlation must lie in [-1, 1]")
        if not 0.0 <= self.propensity_floor < 0.5:
            raise ValueError("propensity_floor must lie in [0, 0.5)")
        if not 0.0 < self.rct_propensity < 1.0:
            raise ValueError("rct_propensity must lie strictly in (0, 1)")
        if self.noise_std < 0.0 or self.noise_heteroskedasticity < 0.0:
            raise ValueError("noise scales must be nonnegative")
        if self.quadrature_order < 5:
            raise ValueError("quadrature_order must be at least 5")
        has_hidden_participation_selection = (
            self.rct_hidden_selection_strength != 0.0
        )
        has_participation_outcome_loading = (
            self.participation_outcome_loading_0 != 0.0
            or self.participation_outcome_loading_1 != 0.0
        )
        if (
            has_hidden_participation_selection
            != has_participation_outcome_loading
        ):
            raise ValueError(
                "hidden RCT participation selection and participation outcome "
                "loading must be jointly active"
            )


MECHANISM_CARDS: Mapping[str, Mapping[str, Any]] = {
    "M0": {
        "title": "No hidden confounding",
        "purpose": "Negative control for full borrowing",
        "assumptions_varied": ["hidden treatment selection"],
        "preset": {
            "bias_geometry": "constant",
            "bias_min": 1.0,
            "confounding_treatment": 0.0,
            "confounding_outcome_0": 1.0,
            "confounding_outcome_1": 1.0,
            "cate_geometry": "smooth",
        },
    },
    "M1": {
        "title": "Global simple arm-specific bias",
        "purpose": "Correction-favorable reference",
        "assumptions_varied": ["arm-specific conditional bias"],
        "preset": {
            "bias_geometry": "constant",
            "bias_min": 1.0,
            "confounding_treatment": 1.0,
            "confounding_outcome_0": 0.5,
            "confounding_outcome_1": 1.0,
            "cate_geometry": "linear",
        },
    },
    "M2": {
        "title": "Observable smooth severity",
        "purpose": "Observable conditional discrepancy",
        "assumptions_varied": ["X-resolvable severity"],
        "preset": {
            "bias_geometry": "smooth",
            "bias_min": 0.15,
            "resolvability": 1.0,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": 1.5,
            "confounding_outcome_1": 1.5,
            "cate_geometry": "smooth",
            "cate_alignment": "orthogonal",
        },
    },
    "M3": {
        "title": "Localized bias with a clean complement",
        "purpose": "Localized conditional discrepancy",
        "assumptions_varied": ["clean subset", "sharp boundary"],
        "preset": {
            "bias_geometry": "localized",
            "clean_fraction": 0.75,
            "bias_min": 0.0,
            "resolvability": 1.0,
            "confounding_treatment": 2.0,
            "confounding_outcome_0": 2.0,
            "confounding_outcome_1": 2.0,
            "cate_geometry": "smooth",
            "cate_alignment": "orthogonal",
        },
    },
    "M4": {
        "title": "Localized severity without a clean subset",
        "purpose": "Test information loss from discounting",
        "assumptions_varied": ["minimum confounding"],
        "preset": {
            "bias_geometry": "localized",
            "clean_fraction": 0.75,
            "bias_min": 0.25,
            "resolvability": 1.0,
            "confounding_treatment": 2.0,
            "confounding_outcome_0": 2.0,
            "confounding_outcome_1": 2.0,
            "cate_geometry": "smooth",
        },
    },
    "M5": {
        "title": "Within-X unresolvable severity",
        "purpose": "Local weights should shrink to a global rule",
        "assumptions_varied": ["reliability observability"],
        "preset": {
            "bias_geometry": "localized",
            "clean_fraction": 0.5,
            "bias_min": 0.0,
            "resolvability": 0.0,
            "treatment_linear_scale": 0.0,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": 1.5,
            "confounding_outcome_1": 1.5,
            "cate_geometry": "smooth",
        },
    },
    "M6": {
        "title": "Sign-changing conditional bias",
        "purpose": "Expose cancellation hidden by global averages",
        "assumptions_varied": ["bias sign"],
        "preset": {
            "bias_geometry": "sign_change",
            "bias_min": 0.25,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": 1.5,
            "confounding_outcome_1": 1.5,
            "cate_geometry": "smooth",
        },
    },
    "M7": {
        "title": "Arm-asymmetric hidden outcome loading",
        "purpose": "Test symmetric reliability and hidden effect modification",
        "assumptions_varied": ["arm symmetry", "T-by-U effect"],
        "preset": {
            "bias_geometry": "interaction",
            "bias_min": 0.1,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": -0.5,
            "confounding_outcome_1": 2.0,
            "cate_geometry": "smooth",
        },
    },
    "M8": {
        "title": "Bias--CATE complexity alignment",
        "purpose": "Test local information collapse in the effect-critical region",
        "assumptions_varied": ["bias-CATE alignment"],
        "preset": {
            "bias_geometry": "localized",
            "clean_fraction": 0.8,
            "bias_min": 0.1,
            "confounding_treatment": 2.0,
            "confounding_outcome_0": 2.0,
            "confounding_outcome_1": 2.0,
            "cate_geometry": "rare",
            "cate_alignment": "aligned",
            "cate_scale": 1.5,
        },
    },
    "M9": {
        "title": "Poor local randomized support",
        "purpose": "Separate trial support from hidden confounding",
        "assumptions_varied": ["RCT study selection", "local support"],
        "preset": {
            "bias_geometry": "localized",
            "clean_fraction": 0.7,
            "bias_min": 0.1,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": 1.5,
            "confounding_outcome_1": 1.5,
            "rct_selection_strength": -2.0,
            "cate_geometry": "rare",
            "cate_alignment": "aligned",
        },
    },
    "M10": {
        "title": "Source-specific CATE",
        "purpose": "Deliberate transportability violation",
        "assumptions_varied": ["CATE transportability"],
        "preset": {
            "bias_geometry": "smooth",
            "bias_min": 0.1,
            "confounding_treatment": 1.0,
            "confounding_outcome_0": 1.0,
            "confounding_outcome_1": 1.0,
            "obs_cate_shift": 1.0,
            "source_shift_geometry": "smooth",
            "cate_geometry": "smooth",
        },
    },
    "M11": {
        "title": "Non-Gaussian multi-signal hidden structure",
        "purpose": "Robustness beyond one Gaussian severity variable",
        "assumptions_varied": ["hidden distribution", "heavy-tailed noise"],
        "preset": {
            "bias_geometry": "multi_region",
            "bias_min": 0.1,
            "confounding_treatment": 1.5,
            "confounding_outcome_0": 1.0,
            "confounding_outcome_1": 1.8,
            "hidden_outcome_correlation": 0.6,
            "hidden_distribution": "normal_mixture",
            "noise_distribution": "student_t",
            "noise_heteroskedasticity": 0.5,
            "cate_geometry": "discontinuous",
        },
    },
}


_M7_RMS_HIDDEN_LOADING = (17.0 / 8.0) ** 0.5

STRUCTURAL_ADDENDUM_CARDS: Mapping[str, Mapping[str, Any]] = {
    "C01": {
        "title": "Fixed-sign twin of M6",
        "purpose": "Isolate sign cancellation from discrepancy magnitude",
        "paired_reference": "M6",
        "isolated_contrast": "bias_sign",
        "matched_quantities": [
            "severity magnitude",
            "treatment propensity",
            "target surfaces",
            "pointwise absolute arm discrepancy",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M6"]["preset"]),
            "bias_geometry": "fixed_sign_varying_magnitude",
        },
    },
    "C02": {
        "title": "Arm-symmetric twin of M7",
        "purpose": "Isolate signed arm asymmetry at matched average loading variance",
        "paired_reference": "M7",
        "isolated_contrast": "hidden_outcome_arm_loading",
        "matched_quantities": [
            "confounding geometry",
            "treatment propensity",
            "target surfaces",
            "arm-averaged squared hidden loading",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M7"]["preset"]),
            "confounding_outcome_0": _M7_RMS_HIDDEN_LOADING,
            "confounding_outcome_1": _M7_RMS_HIDDEN_LOADING,
        },
    },
    "C03": {
        "title": "Gaussian-noise twin of M11",
        "purpose": "Isolate heavy tails at matched conditional noise variance",
        "paired_reference": "M11",
        "isolated_contrast": "outcome_noise_shape",
        "matched_quantities": [
            "all analytic truth",
            "noise conditional variance",
            "noise heteroskedasticity",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M11"]["preset"]),
            "noise_distribution": "normal",
        },
    },
    "C04": {
        "title": "CATE-neutral observational outcome shift",
        "purpose": "Negative control for equal arm-specific source shifts",
        "paired_reference": "M0",
        "isolated_contrast": "observational_baseline_shift",
        "matched_quantities": [
            "target surfaces",
            "RCT law",
            "zero hidden confounding",
            "zero source CATE shift",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M0"]["preset"]),
            "obs_baseline_shift": 1.0,
            "obs_cate_shift": 0.0,
            "source_shift_geometry": "smooth",
        },
    },
    "C05": {
        "title": "Null CATE",
        "purpose": "Measure false treatment-effect signal under an exact null",
        "paired_reference": "M0",
        "isolated_contrast": "cate_nullity",
        "matched_quantities": [
            "baseline outcome surface",
            "treatment and source-selection mechanisms",
            "source support and transport mechanisms",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M0"]["preset"]),
            "cate_geometry": "null",
        },
    },
    "C06": {
        "title": "Centered sign-changing CATE",
        "purpose": "Separate heterogeneous-effect learning from nonzero ATE",
        "paired_reference": "M0",
        "isolated_contrast": "zero_ate_heterogeneity",
        "matched_quantities": [
            "baseline outcome surface",
            "treatment and source-selection mechanisms",
            "source support and transport mechanisms",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M0"]["preset"]),
            "cate_geometry": "centered_linear",
        },
    },
    "C07": {
        "title": "Hidden RCT participation with a prognostic variable",
        "purpose": "Separate hidden study selection from effect transport",
        "paired_reference": "C08",
        "isolated_contrast": "participation_variable_role",
        "matched_quantities": [
            "hidden participation law",
            "observed covariate support",
            "arm-specific latent variance",
            "target surfaces",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M0"]["preset"]),
            "confounding_outcome_0": 0.0,
            "confounding_outcome_1": 0.0,
            "rct_hidden_selection_strength": 2.0,
            "participation_outcome_loading_0": 1.0,
            "participation_outcome_loading_1": 1.0,
        },
    },
    "C08": {
        "title": "Hidden RCT participation with an effect modifier",
        "purpose": "Expose an unobserved conditional-effect transport violation",
        "paired_reference": "C07",
        "isolated_contrast": "participation_variable_role",
        "matched_quantities": [
            "hidden participation law",
            "observed covariate support",
            "arm-specific latent variance",
            "target surfaces",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M0"]["preset"]),
            "confounding_outcome_0": 0.0,
            "confounding_outcome_1": 0.0,
            "rct_hidden_selection_strength": 2.0,
            "participation_outcome_loading_0": -1.0,
            "participation_outcome_loading_1": 1.0,
        },
    },
    "C09": {
        "title": "Shared-coordinate confounding representation",
        "purpose": "Anchor a discrepancy law that shares target signal coordinates",
        "paired_reference": "C10",
        "isolated_contrast": "representation_overlap",
        "matched_quantities": [
            "target surfaces",
            "confounding equations",
            "marginal discrepancy law",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M2"]["preset"]),
            "d": 6,
            "confounding_coordinate_offset": 0,
        },
    },
    "C10": {
        "title": "Disjoint-coordinate confounding representation",
        "purpose": "Test reliance on shared target and discrepancy features",
        "paired_reference": "C09",
        "isolated_contrast": "representation_overlap",
        "matched_quantities": [
            "target surfaces",
            "confounding equations",
            "marginal discrepancy law",
        ],
        "preset": {
            **dict(MECHANISM_CARDS["M2"]["preset"]),
            "d": 6,
            "confounding_coordinate_offset": 3,
        },
    },
}

STRUCTURAL_ADDENDUM_CONTRASTS: tuple[Mapping[str, str], ...] = (
    {
        "id": "M6_to_C01_bias_sign",
        "reference": "M6",
        "variant": "C01",
        "isolated_contrast": "bias_sign",
    },
    {
        "id": "M7_to_C02_arm_loading",
        "reference": "M7",
        "variant": "C02",
        "isolated_contrast": "hidden_outcome_arm_loading",
    },
    {
        "id": "M11_to_C03_noise_shape",
        "reference": "M11",
        "variant": "C03",
        "isolated_contrast": "outcome_noise_shape",
    },
    {
        "id": "M0_to_C04_source_shift",
        "reference": "M0",
        "variant": "C04",
        "isolated_contrast": "observational_baseline_shift",
    },
    {
        "id": "M0_to_C05_null_cate",
        "reference": "M0",
        "variant": "C05",
        "isolated_contrast": "cate_nullity",
    },
    {
        "id": "M0_to_C06_zero_ate_heterogeneity",
        "reference": "M0",
        "variant": "C06",
        "isolated_contrast": "zero_ate_heterogeneity",
    },
    {
        "id": "C07_to_C08_hidden_effect_modification",
        "reference": "C07",
        "variant": "C08",
        "isolated_contrast": "participation_variable_role",
    },
    {
        "id": "C09_to_C10_representation_overlap",
        "reference": "C09",
        "variant": "C10",
        "isolated_contrast": "representation_overlap",
    },
)

STRUCTURAL_ADDENDUM_CONFIG_DIFFERENCES: Mapping[str, tuple[str, ...]] = {
    "M6_to_C01_bias_sign": ("bias_geometry",),
    "M7_to_C02_arm_loading": (
        "confounding_outcome_0",
        "confounding_outcome_1",
    ),
    "M11_to_C03_noise_shape": ("noise_distribution",),
    "M0_to_C04_source_shift": ("obs_baseline_shift",),
    "M0_to_C05_null_cate": ("cate_geometry",),
    "M0_to_C06_zero_ate_heterogeneity": ("cate_geometry",),
    "C07_to_C08_hidden_effect_modification": (
        "participation_outcome_loading_0",
    ),
    "C09_to_C10_representation_overlap": (
        "confounding_coordinate_offset",
    ),
}


MECHANISM_AXIS_LEVELS: Mapping[str, Mapping[str, str]] = {
    "M0": {
        "discrepancy_origin": "none",
        "confounding_geometry": "none",
        "clean_region": "all",
        "reliability_resolution": "not_applicable",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "not_applicable",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M1": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "global_constant",
        "clean_region": "none",
        "reliability_resolution": "global",
        "arm_structure": "asymmetric_magnitude",
        "cate_geometry": "linear",
        "bias_cate_alignment": "not_applicable",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M2": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "smooth",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M3": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "localized",
        "clean_region": "exact_complement",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M4": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "localized_magnitude",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M5": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "within_x",
        "clean_region": "latent_only",
        "reliability_resolution": "unresolvable_from_x",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "unobservable",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M6": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "sign_changing",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M7": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "interaction",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "asymmetric_signed",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M8": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "localized",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "rare_subgroup",
        "bias_cate_alignment": "aligned",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M9": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "localized",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric",
        "cate_geometry": "rare_subgroup",
        "bias_cate_alignment": "aligned",
        "rct_support": "poor_in_effect_region_moderate",
        "transportability": "holds",
        "hidden_distribution": "gaussian",
    },
    "M10": {
        "discrepancy_origin": "hidden_selection_and_source_shift",
        "confounding_geometry": "smooth",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "hidden_symmetric_plus_source_cate_shift",
        "cate_geometry": "smooth",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "source_cate_differs",
        "hidden_distribution": "gaussian",
    },
    "M11": {
        "discrepancy_origin": "hidden_treatment_selection",
        "confounding_geometry": "multi_region",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "asymmetric_magnitude",
        "cate_geometry": "discontinuous",
        "bias_cate_alignment": "orthogonal",
        "rct_support": "common",
        "transportability": "holds",
        "hidden_distribution": "gaussian_mixture",
    },
}


STRUCTURAL_ADDENDUM_AXIS_LEVELS: Mapping[str, Mapping[str, str]] = {
    "C01": {
        **dict(MECHANISM_AXIS_LEVELS["M6"]),
        "confounding_geometry": "fixed_sign_varying_magnitude",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M6",
        "isolated_contrast": "bias_sign",
        "reporting_layer": "matched_structural_addendum",
    },
    "C02": {
        **dict(MECHANISM_AXIS_LEVELS["M7"]),
        "arm_structure": "symmetric_rms_matched",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M7",
        "isolated_contrast": "hidden_outcome_arm_loading",
        "reporting_layer": "matched_structural_addendum",
    },
    "C03": {
        **dict(MECHANISM_AXIS_LEVELS["M11"]),
        "noise_shape": "gaussian_variance_matched",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M11",
        "isolated_contrast": "outcome_noise_shape",
        "reporting_layer": "matched_structural_addendum",
    },
    "C04": {
        **dict(MECHANISM_AXIS_LEVELS["M0"]),
        "discrepancy_origin": "source_outcome_shift",
        "clean_region": "none",
        "reliability_resolution": "x_resolvable",
        "arm_structure": "symmetric_source_shift",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M0",
        "isolated_contrast": "observational_baseline_shift",
        "reporting_layer": "matched_structural_addendum",
    },
    "C05": {
        **dict(MECHANISM_AXIS_LEVELS["M0"]),
        "cate_geometry": "null",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M0",
        "isolated_contrast": "cate_nullity",
        "reporting_layer": "matched_structural_addendum",
    },
    "C06": {
        **dict(MECHANISM_AXIS_LEVELS["M0"]),
        "cate_geometry": "centered_linear_zero_ate",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "not_applicable",
        "paired_reference": "M0",
        "isolated_contrast": "zero_ate_heterogeneity",
        "reporting_layer": "matched_structural_addendum",
    },
    "C07": {
        **dict(MECHANISM_AXIS_LEVELS["M0"]),
        "discrepancy_origin": "rct_hidden_participation",
        "arm_structure": "symmetric_participation",
        "rct_support": "observed_x_common_hidden_selected",
        "transportability": "rct_baseline_differs_cate_holds",
        "noise_shape": "gaussian",
        "rct_participation": "hidden_prognostic_only",
        "representation_relation": "not_applicable",
        "paired_reference": "C08",
        "isolated_contrast": "participation_variable_role",
        "reporting_layer": "matched_structural_addendum",
    },
    "C08": {
        **dict(MECHANISM_AXIS_LEVELS["M0"]),
        "discrepancy_origin": "rct_hidden_participation",
        "arm_structure": "asymmetric_signed_participation",
        "rct_support": "observed_x_common_hidden_selected",
        "transportability": "rct_hidden_cate_differs",
        "noise_shape": "gaussian",
        "rct_participation": "hidden_effect_modifier",
        "representation_relation": "not_applicable",
        "paired_reference": "C07",
        "isolated_contrast": "participation_variable_role",
        "reporting_layer": "matched_structural_addendum",
    },
    "C09": {
        **dict(MECHANISM_AXIS_LEVELS["M2"]),
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "shared",
        "paired_reference": "C10",
        "isolated_contrast": "representation_overlap",
        "reporting_layer": "matched_structural_addendum",
    },
    "C10": {
        **dict(MECHANISM_AXIS_LEVELS["M2"]),
        "bias_cate_alignment": "disjoint_coordinate_blocks",
        "noise_shape": "gaussian",
        "rct_participation": "none",
        "representation_relation": "disjoint",
        "paired_reference": "C09",
        "isolated_contrast": "representation_overlap",
        "reporting_layer": "matched_structural_addendum",
    },
}


SCIENTIFIC_DESIGN_FIELDS: tuple[str, ...] = (
    "signal_basis",
    "signal_seed",
    "confounding_coordinate_offset",
    "covariate_rho",
    "covariate_marginals",
    "bias_geometry",
    "clean_fraction",
    "bias_min",
    "resolvability",
    "u_predictability",
    "confounding_treatment",
    "confounding_outcome_0",
    "confounding_outcome_1",
    "hidden_outcome_correlation",
    "hidden_distribution",
    "treatment_linear_scale",
    "treatment_nonlinearity",
    "propensity_floor",
    "rct_propensity",
    "outcome_geometry",
    "cate_geometry",
    "cate_alignment",
    "cate_scale",
    "obs_covariate_shift",
    "rct_covariate_shift",
    "obs_selection_strength",
    "rct_selection_strength",
    "rct_hidden_selection_strength",
    "participation_outcome_loading_0",
    "participation_outcome_loading_1",
    "obs_baseline_shift",
    "obs_cate_shift",
    "source_shift_geometry",
    "noise_std",
    "noise_distribution",
    "noise_heteroskedasticity",
)


def _reference_config(cfg: StructuralDGPConfig) -> StructuralDGPConfig:
    base = StructuralDGPConfig(name=cfg.name, mechanism=cfg.mechanism)
    if cfg.mechanism == "custom":
        return base
    cards = (
        STRUCTURAL_ADDENDUM_CARDS
        if cfg.mechanism in STRUCTURAL_ADDENDUM_CARDS
        else MECHANISM_CARDS
    )
    return replace(base, **dict(cards[cfg.mechanism]["preset"]))


def structural_design_overrides(cfg: StructuralDGPConfig) -> dict[str, Any]:
    reference = _reference_config(cfg)
    return {
        field_name: getattr(cfg, field_name)
        for field_name in SCIENTIFIC_DESIGN_FIELDS
        if getattr(cfg, field_name) != getattr(reference, field_name)
    }


def _m9_rct_support_level(selection_strength: float) -> str:
    if selection_strength > 0.0:
        return "effect_region_enriched"
    magnitude = abs(selection_strength)
    if magnitude <= 2.0:
        return "poor_in_effect_region_moderate"
    if magnitude <= 5.0:
        return "poor_in_effect_region_strong"
    return "poor_in_effect_region_severe"


def structural_axis_levels(cfg: StructuralDGPConfig) -> dict[str, str]:
    """Return regime labels implied by actual parameters, not only a card name."""

    levels = dict(STRUCTURAL_ADDENDUM_AXIS_LEVELS.get(
        cfg.mechanism, MECHANISM_AXIS_LEVELS.get(cfg.mechanism, {})
    ))
    if levels and not structural_design_overrides(cfg):
        return levels
    if not levels:
        levels = {
            "discrepancy_origin": "custom",
            "confounding_geometry": cfg.bias_geometry,
            "clean_region": "custom",
            "reliability_resolution": "custom",
            "arm_structure": "custom",
            "cate_geometry": cfg.cate_geometry,
            "bias_cate_alignment": cfg.cate_alignment,
            "rct_support": "common",
            "transportability": "holds",
            "hidden_distribution": cfg.hidden_distribution,
        }
    residual_hidden = np.sqrt(max(0.0, 1.0 - cfg.u_predictability**2))
    shared_residual_signal = (
        residual_hidden > 1e-12
        and abs(cfg.hidden_outcome_correlation) > 0.0
    )
    shared_latent_severity = (
        cfg.resolvability < 1.0 and abs(cfg.u_predictability) > 0.0
    )
    hidden_selection = (
        abs(cfg.confounding_treatment) > 0.0
        and (shared_residual_signal or shared_latent_severity)
        and (
            abs(cfg.confounding_outcome_0) > 0.0
            or abs(cfg.confounding_outcome_1) > 0.0
        )
    )
    source_shift = any(
        abs(value) > 0.0
        for value in (
            cfg.obs_baseline_shift,
            cfg.obs_cate_shift,
        )
    )
    if hidden_selection and source_shift:
        levels["discrepancy_origin"] = "hidden_selection_and_source_shift"
    elif hidden_selection:
        levels["discrepancy_origin"] = "hidden_treatment_selection"
    elif source_shift:
        levels["discrepancy_origin"] = "source_outcome_shift"
    else:
        levels["discrepancy_origin"] = "none"
    levels["confounding_geometry"] = (
        cfg.bias_geometry if hidden_selection else "none"
    )
    if not hidden_selection and not source_shift:
        levels["clean_region"] = "all"
        levels["reliability_resolution"] = "not_applicable"
        levels["bias_cate_alignment"] = "not_applicable"
    else:
        hidden_local = cfg.bias_geometry in {"localized", "multi_region"}
        hidden_exact_complement = (
            hidden_selection
            and hidden_local
            and cfg.bias_min == 0.0
            and cfg.resolvability == 1.0
        )
        source_exact_complement = (
            source_shift and cfg.source_shift_geometry == "localized"
        )
        if hidden_selection and source_shift:
            levels["clean_region"] = (
                "exact_complement"
                if hidden_exact_complement and source_exact_complement
                else "none"
            )
        elif hidden_selection:
            if hidden_local and cfg.bias_min > 0.0:
                levels["clean_region"] = "none"
            elif hidden_exact_complement:
                levels["clean_region"] = "exact_complement"
            elif hidden_local:
                levels["clean_region"] = "latent_or_partial"
            else:
                levels["clean_region"] = "none"
        else:
            levels["clean_region"] = (
                "source_shift_complement" if source_exact_complement else "none"
            )
        if not hidden_selection:
            levels["reliability_resolution"] = "x_resolvable"
        elif cfg.bias_geometry == "constant" and not (
            source_shift and cfg.source_shift_geometry != "constant"
        ):
            levels["reliability_resolution"] = "global"
        elif cfg.resolvability == 1.0:
            levels["reliability_resolution"] = "x_resolvable"
        elif cfg.resolvability == 0.0:
            levels["reliability_resolution"] = "unresolvable_from_x"
        else:
            levels["reliability_resolution"] = "partly_x_resolvable"
        levels["bias_cate_alignment"] = cfg.cate_alignment
    if np.isclose(cfg.confounding_outcome_0, cfg.confounding_outcome_1):
        hidden_arm_structure = "symmetric"
    elif cfg.confounding_outcome_0 * cfg.confounding_outcome_1 < 0.0:
        hidden_arm_structure = "asymmetric_signed"
    else:
        hidden_arm_structure = "asymmetric_magnitude"
    if source_shift and (
        cfg.obs_cate_shift != 0.0
    ):
        if hidden_selection and hidden_arm_structure == "symmetric":
            levels["arm_structure"] = (
                "hidden_symmetric_plus_source_cate_shift"
            )
        elif hidden_selection:
            levels["arm_structure"] = "combined_asymmetric"
        else:
            levels["arm_structure"] = "asymmetric_source_shift"
    elif hidden_selection:
        levels["arm_structure"] = hidden_arm_structure
    elif source_shift:
        levels["arm_structure"] = "symmetric_source_shift"
    else:
        levels["arm_structure"] = "symmetric"
    levels["cate_geometry"] = {
        "rare": "rare_subgroup",
        "discontinuous": "discontinuous",
        "centered_linear": "centered_linear_zero_ate",
    }.get(cfg.cate_geometry, cfg.cate_geometry)
    if cfg.rct_selection_strength != 0.0 and cfg.rct_covariate_shift != 0.0:
        levels["rct_support"] = "selected_and_shifted"
    elif cfg.rct_selection_strength != 0.0:
        levels["rct_support"] = (
            _m9_rct_support_level(cfg.rct_selection_strength)
            if cfg.mechanism == "M9"
            else "study_selected"
        )
    elif cfg.rct_covariate_shift != 0.0:
        levels["rct_support"] = "covariate_shifted"
    else:
        levels["rct_support"] = "common"
    hidden_participation = cfg.rct_hidden_selection_strength != 0.0
    participation_effect_modification = not np.isclose(
        cfg.participation_outcome_loading_0,
        cfg.participation_outcome_loading_1,
    )
    if hidden_participation and participation_effect_modification:
        levels["transportability"] = "rct_hidden_cate_differs"
    elif hidden_participation:
        levels["transportability"] = "rct_baseline_differs_cate_holds"
    else:
        levels["transportability"] = (
            "source_cate_differs"
            if cfg.obs_cate_shift != 0.0
            else "holds"
        )
    levels["hidden_distribution"] = {
        "normal": "gaussian",
        "binary": "binary",
        "normal_mixture": "gaussian_mixture",
    }[cfg.hidden_distribution]
    if cfg.mechanism in STRUCTURAL_ADDENDUM_CELLS:
        levels["noise_shape"] = (
            "student_t_variance_standardized"
            if cfg.noise_distribution == "student_t"
            else "gaussian"
        )
        if hidden_participation:
            levels["discrepancy_origin"] = "rct_hidden_participation"
            levels["rct_support"] = "observed_x_common_hidden_selected"
            levels["rct_participation"] = (
                "hidden_effect_modifier"
                if participation_effect_modification
                else "hidden_prognostic_only"
            )
            if np.isclose(
                cfg.participation_outcome_loading_0,
                cfg.participation_outcome_loading_1,
            ):
                levels["arm_structure"] = "symmetric_participation"
            elif (
                cfg.participation_outcome_loading_0
                * cfg.participation_outcome_loading_1
                < 0.0
            ):
                levels["arm_structure"] = "asymmetric_signed_participation"
            else:
                levels["arm_structure"] = "asymmetric_magnitude_participation"
        else:
            levels["rct_participation"] = "none"
        if cfg.mechanism in {"C09", "C10"}:
            if cfg.confounding_coordinate_offset == 0:
                levels["representation_relation"] = "shared"
            elif cfg.confounding_coordinate_offset >= 3:
                levels["representation_relation"] = "disjoint"
            else:
                levels["representation_relation"] = "partial_overlap"
        if structural_design_overrides(cfg):
            levels["isolated_contrast"] = "modified_from_canonical"
    return levels


def make_structural_config(
    *, name: str, mechanism: str, **overrides: Any
) -> StructuralDGPConfig:
    """Construct a named, versioned mechanism with explicit overrides."""

    if mechanism not in MECHANISM_CARDS:
        raise ValueError(f"unknown structural mechanism: {mechanism}")
    base = StructuralDGPConfig(name=name, mechanism=mechanism)
    preset = dict(MECHANISM_CARDS[mechanism]["preset"])
    preset.update(overrides)
    return replace(base, **preset)


def make_structural_addendum_config(
    *, name: str, cell: str, **overrides: Any
) -> StructuralDGPConfig:
    """Construct one canonical matched-contrast addendum cell."""

    if cell not in STRUCTURAL_ADDENDUM_CARDS:
        raise ValueError(f"unknown structural addendum cell: {cell}")
    base = StructuralDGPConfig(name=name, mechanism=cell)
    preset = dict(STRUCTURAL_ADDENDUM_CARDS[cell]["preset"])
    preset.update(overrides)
    return replace(base, **preset)


def _sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@lru_cache(maxsize=None)
def _target_covariance(d: int, rho: float) -> FloatArray:
    indices = np.arange(d)
    covariance = np.power(rho, np.abs(indices[:, None] - indices[None, :]))
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance.setflags(write=False)
    return covariance


@lru_cache(maxsize=None)
def _target_cholesky(d: int, rho: float) -> FloatArray:
    factor = np.linalg.cholesky(_target_covariance(d, rho))
    factor.setflags(write=False)
    return factor


@lru_cache(maxsize=None)
def _signal_transform(
    d: int, rho: float, basis: SignalBasis, seed: int
) -> FloatArray:
    if basis == "axis_aligned":
        rotation = np.identity(d)
    else:
        rng = np.random.default_rng(seed)
        rotation, triangular = np.linalg.qr(rng.standard_normal((d, d)))
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        rotation = rotation * signs[None, :]
    covariance = _target_covariance(d, rho)
    variances = np.diag(rotation.T @ covariance @ rotation)
    transform = rotation / np.sqrt(variances)[None, :]
    transform = np.asarray(transform, dtype=np.float64)
    transform.setflags(write=False)
    return transform


def _signal_coordinates(X: FloatArray, cfg: StructuralDGPConfig) -> FloatArray:
    latent = _inverse_covariate_marginals(X, cfg)
    transform = _signal_transform(
        cfg.d, cfg.covariate_rho, cfg.signal_basis, cfg.signal_seed
    )
    return latent @ transform


def _confounding_coordinates(
    X: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    coordinates = _signal_coordinates(X, cfg)
    start = cfg.confounding_coordinate_offset
    return coordinates[:, start : start + 3]


def _forward_covariate_marginals(
    latent: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    """Map latent Gaussian coordinates to standardized observed marginals."""

    values = np.asarray(latent, dtype=np.float64)
    if cfg.covariate_marginals == "gaussian":
        return values.copy()
    scale = _MARGINAL_TRANSFORM_SCALE
    if cfg.covariate_marginals == "symmetric_heavy_tailed":
        standard_deviation = np.sqrt((np.exp(2.0 * scale**2) - 1.0) / 2.0)
        return np.sinh(scale * values) / standard_deviation
    mean = np.exp(0.5 * scale**2)
    standard_deviation = np.sqrt(
        np.exp(scale**2) * (np.exp(scale**2) - 1.0)
    )
    return (np.exp(scale * values) - mean) / standard_deviation


def _inverse_covariate_marginals(
    X: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    """Recover latent Gaussian coordinates from the observed covariates."""

    values = np.asarray(X, dtype=np.float64)
    if cfg.covariate_marginals == "gaussian":
        return values
    scale = _MARGINAL_TRANSFORM_SCALE
    if cfg.covariate_marginals == "symmetric_heavy_tailed":
        standard_deviation = np.sqrt((np.exp(2.0 * scale**2) - 1.0) / 2.0)
        return np.arcsinh(values * standard_deviation) / scale
    mean = np.exp(0.5 * scale**2)
    standard_deviation = np.sqrt(
        np.exp(scale**2) * (np.exp(scale**2) - 1.0)
    )
    shifted = values * standard_deviation + mean
    if (shifted <= 0.0).any():
        raise ValueError("right-skewed covariates lie outside their support")
    return np.log(shifted) / scale


def _draw_target_covariates(
    rng: np.random.Generator, n: int, cfg: StructuralDGPConfig
) -> FloatArray:
    latent = rng.standard_normal((n, cfg.d)) @ _target_cholesky(
        cfg.d, cfg.covariate_rho
    ).T
    return _forward_covariate_marginals(latent, cfg)


def _clean_threshold(probability: float) -> float:
    if probability == 0.0:
        return -np.inf
    if probability == 1.0:
        return np.inf
    return NormalDist().inv_cdf(probability)


def _geometry(X: FloatArray, cfg: StructuralDGPConfig) -> tuple[FloatArray, FloatArray]:
    X = _confounding_coordinates(X, cfg)
    n = X.shape[0]
    if cfg.bias_geometry == "constant":
        return np.ones(n), np.ones(n)
    if cfg.bias_geometry == "smooth":
        return _sigmoid(2.0 * X[:, 0]), np.ones(n)
    if cfg.bias_geometry == "interaction":
        return _sigmoid(2.0 * X[:, 0] * X[:, 1]), np.ones(n)
    if cfg.bias_geometry == "sign_change":
        magnitude = 0.25 + 0.75 * np.tanh(np.abs(X[:, 0]))
        sign = np.where(X[:, 0] >= 0.0, 1.0, -1.0)
        return magnitude, sign
    if cfg.bias_geometry == "fixed_sign_varying_magnitude":
        magnitude = 0.25 + 0.75 * np.tanh(np.abs(X[:, 0]))
        return magnitude, np.ones(n)
    threshold = _clean_threshold(cfg.clean_fraction)
    if cfg.bias_geometry == "localized":
        return (X[:, 0] > threshold).astype(np.float64), np.ones(n)
    first = X[:, 0] > threshold
    second = X[:, 1] < -threshold
    return np.logical_or(first, second).astype(np.float64), np.ones(n)


def _severity(
    q_x: FloatArray, latent_severity: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    score = (
        cfg.resolvability * q_x
        + (1.0 - cfg.resolvability) * latent_severity
    )
    return cfg.bias_min + (1.0 - cfg.bias_min) * score


def _aligned_score(
    X: FloatArray, region: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    target_X = _signal_coordinates(X, cfg)
    confounding_X = _confounding_coordinates(X, cfg)
    if cfg.cate_alignment == "aligned":
        return region
    if cfg.cate_alignment == "opposed":
        if cfg.bias_geometry == "constant":
            return region
        if cfg.bias_geometry == "smooth":
            return _sigmoid(-2.0 * confounding_X[:, 0])
        if cfg.bias_geometry == "interaction":
            return _sigmoid(
                -2.0 * confounding_X[:, 0] * confounding_X[:, 1]
            )
        if cfg.bias_geometry == "localized":
            threshold = _clean_threshold(cfg.clean_fraction)
            return (confounding_X[:, 0] < -threshold).astype(np.float64)
        if cfg.bias_geometry == "multi_region":
            threshold = _clean_threshold(cfg.clean_fraction)
            return np.logical_or(
                confounding_X[:, 0] < -threshold,
                confounding_X[:, 1] > threshold,
            ).astype(np.float64)
        return 1.0 - region
    if cfg.cate_geometry == "rare" and cfg.bias_geometry in {
        "localized",
        "multi_region",
    }:
        threshold = _clean_threshold(cfg.clean_fraction)
        return (target_X[:, 1] > threshold).astype(np.float64)
    return _sigmoid(2.0 * target_X[:, 1])


def _target_surfaces(
    X: FloatArray, cfg: StructuralDGPConfig, region: FloatArray | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    raw_X = X
    if region is None:
        region, _ = _geometry(raw_X, cfg)
    X = _signal_coordinates(raw_X, cfg)
    mu0 = 0.5 * X[:, 0] + 0.25 * X[:, 1]
    if cfg.outcome_geometry == "smooth":
        mu0 = mu0 + 0.4 * np.sin(X[:, 0]) + 0.25 * X[:, 1] ** 2
    elif cfg.outcome_geometry == "interaction":
        mu0 = mu0 + 0.5 * X[:, 0] * X[:, 1] + 0.2 * np.cos(X[:, 2])

    score = _aligned_score(raw_X, region, cfg)
    if cfg.cate_geometry == "null":
        tau = np.zeros(X.shape[0])
        complexity = np.zeros(X.shape[0])
    elif cfg.cate_geometry == "constant":
        tau = np.ones(X.shape[0])
        complexity = np.zeros(X.shape[0])
    elif cfg.cate_geometry == "centered_linear":
        coordinate = (
            X[:, 0]
            if cfg.cate_alignment == "aligned"
            else -X[:, 0]
            if cfg.cate_alignment == "opposed"
            else X[:, 1]
        )
        tau = cfg.cate_scale * coordinate
        complexity = np.abs(coordinate)
    elif cfg.cate_geometry == "linear":
        coordinate = (
            X[:, 0]
            if cfg.cate_alignment == "aligned"
            else -X[:, 0]
            if cfg.cate_alignment == "opposed"
            else X[:, 1]
        )
        tau = 1.0 + 0.5 * cfg.cate_scale * coordinate
        complexity = np.abs(coordinate)
    elif cfg.cate_geometry == "smooth":
        coordinate = (
            X[:, 0]
            if cfg.cate_alignment == "aligned"
            else -X[:, 0]
            if cfg.cate_alignment == "opposed"
            else X[:, 1]
        )
        tau = 1.0 + cfg.cate_scale * (
            0.5 * np.tanh(coordinate) + 0.25 * np.sin(X[:, 2])
        )
        complexity = np.abs(np.tanh(coordinate))
    elif cfg.cate_geometry == "rare":
        tau = 0.5 + 1.5 * cfg.cate_scale * score
        complexity = score
    else:
        tau = 1.0 + cfg.cate_scale * np.where(score > 0.5, 1.0, -0.5)
        complexity = np.abs(score - 0.5) * 2.0
    return mu0, tau, complexity


def _source_shift(
    X: FloatArray, cfg: StructuralDGPConfig, region: FloatArray
) -> tuple[FloatArray, FloatArray]:
    X = _signal_coordinates(X, cfg)
    if cfg.source_shift_geometry == "constant":
        shape = np.ones(X.shape[0])
    elif cfg.source_shift_geometry == "localized":
        shape = region
    else:
        shape = np.tanh(X[:, 0])
    d0 = cfg.obs_baseline_shift * shape
    d1 = d0 + cfg.obs_cate_shift * shape
    return d0, d1


def _treatment_base(X: FloatArray, cfg: StructuralDGPConfig) -> FloatArray:
    X = _confounding_coordinates(X, cfg)
    base = cfg.treatment_linear_scale * (0.4 * X[:, 0] - 0.2 * X[:, 1])
    return base + cfg.treatment_nonlinearity * np.sin(X[:, 0] * X[:, 1])


def _hidden_proxy(X: FloatArray, cfg: StructuralDGPConfig) -> FloatArray:
    """Observed component of the latent treatment/outcome signal.

    Under the declared target Gaussian and standardized signal transform, this
    has unit marginal variance. Conditional on X it is not unmeasured
    confounding; it must therefore enter target potential-outcome means and the
    observed part of the treatment mechanism explicitly.
    """

    return _confounding_coordinates(X, cfg)[:, 2]


def _hidden_quadrature(cfg: StructuralDGPConfig) -> tuple[FloatArray, FloatArray]:
    if cfg.hidden_distribution == "binary":
        return np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    nodes, weights = hermgauss(cfg.quadrature_order)
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    if cfg.hidden_distribution == "normal":
        return nodes, weights
    low_scale = 0.5
    high_scale = np.sqrt(1.75)
    mixture_nodes = np.concatenate((low_scale * nodes, high_scale * nodes))
    mixture_weights = np.concatenate((0.5 * weights, 0.5 * weights))
    return mixture_nodes, mixture_weights


def _draw_hidden(rng: np.random.Generator, n: int, kind: HiddenDistribution) -> FloatArray:
    if kind == "normal":
        return rng.normal(size=n)
    if kind == "binary":
        return rng.choice(np.array([-1.0, 1.0]), size=n)
    high = rng.binomial(1, 0.5, size=n).astype(bool)
    scale = np.where(high, np.sqrt(1.75), 0.5)
    return rng.normal(size=n) * scale


def _participation_is_active(cfg: StructuralDGPConfig) -> bool:
    return cfg.rct_hidden_selection_strength != 0.0


@lru_cache(maxsize=256)
def _rct_participation_moments(
    strength: float, quadrature_order: int
) -> tuple[float, float]:
    if strength == 0.0:
        return 0.5, 0.0
    # The logistic-normal selected mean needs more nodes than the existing
    # hidden-confounding integral to reach stable truth-level precision.
    nodes, weights = hermgauss(max(64, quadrature_order))
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    acceptance = _sigmoid(strength * nodes)
    rate = float(np.sum(weights * acceptance))
    selected_mean = float(np.sum(weights * acceptance * nodes) / rate)
    return rate, selected_mean


def _draw_participation_hidden(
    rng: np.random.Generator,
    n: int,
    cfg: StructuralDGPConfig,
    *,
    source: Literal["obs", "rct"],
) -> FloatArray | None:
    if not _participation_is_active(cfg):
        return None
    if source == "obs" or cfg.rct_hidden_selection_strength == 0.0:
        return rng.normal(size=n)
    accepted: list[FloatArray] = []
    remaining = n
    proposed = 0
    maximum_proposals = max(1_000_000, 20_000 * n)
    while remaining > 0:
        if proposed >= maximum_proposals:
            raise RuntimeError("hidden participation sampling exceeded its proposal budget")
        batch = min(max(256, 3 * remaining), maximum_proposals - proposed)
        proposed += batch
        proposals = rng.normal(size=batch)
        probability = _sigmoid(
            cfg.rct_hidden_selection_strength * proposals
        )
        keep = rng.random(batch) < probability
        if keep.any():
            values = proposals[keep][:remaining]
            accepted.append(values)
            remaining -= values.shape[0]
    return np.concatenate(accepted)


def _selection_acceptance(
    X: FloatArray, cfg: StructuralDGPConfig, source: Literal["obs", "rct"]
) -> FloatArray:
    strength = (
        cfg.obs_selection_strength if source == "obs" else cfg.rct_selection_strength
    )
    if strength == 0.0:
        return np.ones(X.shape[0])
    region, _ = _geometry(X, cfg)
    centered = 2.0 * region - 1.0
    return _sigmoid(strength * centered)


def _draw_covariates(
    rng: np.random.Generator,
    n: int,
    cfg: StructuralDGPConfig,
    source: Literal["obs", "rct"],
) -> FloatArray:
    shift = cfg.obs_covariate_shift if source == "obs" else cfg.rct_covariate_shift
    strength = (
        cfg.obs_selection_strength if source == "obs" else cfg.rct_selection_strength
    )
    if strength == 0.0:
        latent = rng.standard_normal((n, cfg.d)) @ _target_cholesky(
            cfg.d, cfg.covariate_rho
        ).T
        latent[:, 0] += shift
        return _forward_covariate_marginals(latent, cfg)
    accepted: list[FloatArray] = []
    remaining = n
    proposed = 0
    maximum_proposals = max(1_000_000, 20_000 * n)
    while remaining > 0:
        if proposed >= maximum_proposals:
            raise RuntimeError("covariate sampling exceeded its proposal budget")
        batch = min(max(256, 3 * remaining), maximum_proposals - proposed)
        proposed += batch
        latent = rng.standard_normal((batch, cfg.d)) @ _target_cholesky(
            cfg.d, cfg.covariate_rho
        ).T
        latent[:, 0] += shift
        proposal = _forward_covariate_marginals(latent, cfg)
        keep = rng.random(batch) < _selection_acceptance(proposal, cfg, source)
        if keep.any():
            values = proposal[keep][:remaining]
            accepted.append(values)
            remaining -= values.shape[0]
    return np.concatenate(accepted, axis=0)


def _density_unnormalized(
    X: FloatArray, cfg: StructuralDGPConfig, source: Literal["obs", "rct"]
) -> FloatArray:
    shift = cfg.obs_covariate_shift if source == "obs" else cfg.rct_covariate_shift
    mean = np.zeros(cfg.d)
    mean[0] = shift
    precision_mean = np.linalg.solve(
        _target_covariance(cfg.d, cfg.covariate_rho), mean
    )
    latent = _inverse_covariate_marginals(X, cfg)
    log_ratio = latent @ precision_mean - 0.5 * float(mean @ precision_mean)
    ratio = np.exp(np.clip(log_ratio, -30.0, 30.0))
    return ratio * _selection_acceptance(X, cfg, source)


def structural_truth_at(
    X: FloatArray, cfg: StructuralDGPConfig
) -> dict[str, FloatArray]:
    """Return target surfaces and the ``c_t + d_t = r_t`` decomposition."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != cfg.d or not np.isfinite(X).all():
        raise ValueError("X must be a finite matrix with cfg.d columns")
    region, sign = _geometry(X, cfg)
    base_mu0, base_tau, cate_complexity = _target_surfaces(X, cfg, region)
    d0, d1 = _source_shift(X, cfg, region)

    z_nodes, z_weights = _hidden_quadrature(cfg)
    latent_nodes = np.array([0.0, 1.0])
    latent_weights = np.array([0.5, 0.5])
    severity = _severity(
        region[:, None], latent_nodes[None, :], cfg
    )
    proxy = _hidden_proxy(X, cfg)
    predictable_hidden = cfg.u_predictability * proxy
    hidden_scale = np.sqrt(max(0.0, 1.0 - cfg.u_predictability**2))
    base = _treatment_base(X, cfg)
    treatment_geometry = severity
    logits = (
        base[:, None, None]
        + cfg.confounding_treatment
        * treatment_geometry[:, :, None]
        * (
            predictable_hidden[:, None, None]
            + hidden_scale * z_nodes[None, None, :]
        )
    )
    individual_propensity = np.clip(
        _sigmoid(logits), cfg.propensity_floor, 1.0 - cfg.propensity_floor
    )
    joint_weights = latent_weights[None, :, None] * z_weights[None, None, :]
    propensity = np.sum(individual_propensity * joint_weights, axis=(1, 2))
    propensity = np.clip(propensity, 1e-10, 1.0 - 1e-10)

    hidden_outcome = (
        predictable_hidden[:, None, None]
        + hidden_scale
        * cfg.hidden_outcome_correlation
        * z_nodes[None, None, :]
    )
    beta0 = cfg.confounding_outcome_0 * severity * sign[:, None]
    beta1 = cfg.confounding_outcome_1 * severity * sign[:, None]
    signal0 = beta0[:, :, None] * hidden_outcome
    signal1 = beta1[:, :, None] * hidden_outcome
    selected1 = np.sum(
        signal1 * individual_propensity * joint_weights, axis=(1, 2)
    ) / propensity
    selected0 = np.sum(
        signal0 * (1.0 - individual_propensity) * joint_weights, axis=(1, 2)
    ) / (1.0 - propensity)
    expected_severity = np.sum(severity * latent_weights[None, :], axis=1)
    target_hidden0 = (
        cfg.confounding_outcome_0
        * expected_severity
        * sign
        * predictable_hidden
    )
    target_hidden1 = (
        cfg.confounding_outcome_1
        * expected_severity
        * sign
        * predictable_hidden
    )
    mu0 = base_mu0 + target_hidden0
    mu1 = base_mu0 + base_tau + target_hidden1
    tau = mu1 - mu0
    c1 = selected1 - target_hidden1
    c0 = selected0 - target_hidden0
    r0 = c0 + d0
    r1 = c1 + d1
    confounding_bias_magnitude = np.sqrt(0.5 * (c0**2 + c1**2))
    total_discrepancy_magnitude = np.sqrt(0.5 * (r0**2 + r1**2))
    truth = {
        "mu0_target": mu0,
        "mu1_target": mu1,
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
        "expected_severity": expected_severity,
        "hidden_proxy": proxy,
        "predictable_hidden_component": predictable_hidden,
        "confounding_bias_magnitude": confounding_bias_magnitude,
        "total_discrepancy_magnitude": total_discrepancy_magnitude,
        # Backward-compatible shorthand. Reliability of an OBS regression is
        # governed by total discrepancy, not by confounding alone when source
        # potential outcomes differ from the target.
        "bias_magnitude": total_discrepancy_magnitude,
        "region_score": region,
        "cate_complexity_score": cate_complexity,
        "obs_density_unnormalized": _density_unnormalized(X, cfg, "obs"),
        "rct_density_unnormalized": _density_unnormalized(X, cfg, "rct"),
    }
    if _participation_is_active(cfg):
        participation_rate, participant_mean = _rct_participation_moments(
            cfg.rct_hidden_selection_strength, cfg.quadrature_order
        )
        rct_shift_0 = np.full(
            X.shape[0], cfg.participation_outcome_loading_0 * participant_mean
        )
        rct_shift_1 = np.full(
            X.shape[0], cfg.participation_outcome_loading_1 * participant_mean
        )
        rct_cate_shift = rct_shift_1 - rct_shift_0
        truth.update({
            "rct_participation_rate": np.full(X.shape[0], participation_rate),
            "rct_participant_hidden_mean": np.full(X.shape[0], participant_mean),
            "rct_participation_common_offset": np.zeros(X.shape[0]),
            "rct_source_shift_0": rct_shift_0,
            "rct_source_shift_1": rct_shift_1,
            "rct_source_cate_shift": rct_cate_shift,
            "rct_participation_cate_shift": rct_cate_shift,
            "mu0_rct": mu0 + rct_shift_0,
            "mu1_rct": mu1 + rct_shift_1,
            "tau_rct": tau + rct_cate_shift,
        })
    return truth


def _draw_noise(
    rng: np.random.Generator, X: FloatArray, cfg: StructuralDGPConfig
) -> FloatArray:
    if cfg.noise_distribution == "normal":
        base = rng.normal(size=X.shape[0])
    else:
        base = rng.standard_t(df=3.0, size=X.shape[0]) / np.sqrt(3.0)
    coordinate = _signal_coordinates(X, cfg)[:, 0]
    scale = cfg.noise_std * (
        1.0 + cfg.noise_heteroskedasticity * np.abs(coordinate)
    )
    return base * scale


def _draw_source(
    rng: np.random.Generator,
    n: int,
    cfg: StructuralDGPConfig,
    *,
    source: Literal["obs", "rct"],
    role: str,
) -> tuple[SourceSample, dict[str, FloatArray], dict[str, float]]:
    X = _draw_covariates(rng, n, cfg, source)
    region, sign = _geometry(X, cfg)
    latent_severity = rng.binomial(1, 0.5, size=n).astype(np.float64)
    severity = _severity(region, latent_severity, cfg)
    z1 = _draw_hidden(rng, n, cfg.hidden_distribution)
    z2 = _draw_hidden(rng, n, cfg.hidden_distribution)
    hidden_scale = np.sqrt(max(0.0, 1.0 - cfg.u_predictability**2))
    rho = cfg.hidden_outcome_correlation
    predictable_hidden = cfg.u_predictability * _hidden_proxy(X, cfg)
    hidden_treatment = predictable_hidden + hidden_scale * z1
    hidden_outcome = predictable_hidden + hidden_scale * (
        rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2
    )
    participation_hidden = _draw_participation_hidden(
        rng, n, cfg, source=source
    )

    if source == "obs":
        treatment_geometry = severity
        logits = _treatment_base(X, cfg) + (
            cfg.confounding_treatment * treatment_geometry * hidden_treatment
        )
        individual_propensity = np.clip(
            _sigmoid(logits), cfg.propensity_floor, 1.0 - cfg.propensity_floor
        )
    else:
        individual_propensity = np.full(n, cfg.rct_propensity)
    T = rng.binomial(1, individual_propensity, size=n).astype(np.float64)

    mu0, tau, _ = _target_surfaces(X, cfg, region)
    if source == "obs":
        d0, d1 = _source_shift(X, cfg, region)
    else:
        d0 = np.zeros(n)
        d1 = np.zeros(n)
    mu0_source = mu0 + d0
    tau_source = tau + d1 - d0
    beta0 = cfg.confounding_outcome_0 * severity * sign
    beta1 = cfg.confounding_outcome_1 * severity * sign
    beta = np.where(T == 1.0, beta1, beta0)
    Y = (
        mu0_source
        + T * tau_source
        + beta * hidden_outcome
        + _draw_noise(rng, X, cfg)
    )
    if participation_hidden is not None:
        participation_loading = np.where(
            T == 1.0,
            cfg.participation_outcome_loading_1,
            cfg.participation_outcome_loading_0,
        )
        Y = Y + participation_loading * participation_hidden
    sample = SourceSample(
        X=X,
        T=T,
        Y=Y,
        source=source,
        role=role,
        known_propensity=(individual_propensity if source == "rct" else None),
    )

    if float(T.std()) == 0.0 or float(z1.std()) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(T, z1)[0, 1])
    diagnostics = {
        "treated_fraction": float(T.mean()),
        "t_hidden_correlation": correlation,
        "mean_realized_severity": float(severity.mean()),
    }
    realized = {
        "realized_severity": severity,
        "latent_severity": latent_severity,
        "hidden_treatment_signal": z1,
        "latent_treatment_input": hidden_treatment,
        "hidden_outcome_signal": hidden_outcome,
        "individual_treatment_probability": individual_propensity,
    }
    if participation_hidden is not None:
        diagnostics["participation_hidden_mean"] = float(
            participation_hidden.mean()
        )
        diagnostics["participation_hidden_std"] = float(
            participation_hidden.std()
        )
        if float(T.std()) == 0.0 or float(participation_hidden.std()) == 0.0:
            participation_t_correlation = 0.0
        else:
            participation_t_correlation = float(
                np.corrcoef(T, participation_hidden)[0, 1]
            )
        diagnostics["t_participation_hidden_correlation"] = (
            participation_t_correlation
        )
        realized["participation_hidden_signal"] = participation_hidden
    return sample, realized, diagnostics


def _normalize_density(
    truth: dict[str, FloatArray], normalizers: Mapping[str, float]
) -> None:
    for source in ("obs", "rct"):
        raw_key = f"{source}_density_unnormalized"
        truth[f"{source}_density"] = truth[raw_key] / normalizers[source]


def generate_structural_problem(
    cfg: StructuralDGPConfig, seed: int
) -> SimulationProblem:
    """Generate independent observed roles and separate evaluation-only truth."""

    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    obs, obs_realized, obs_diag = _draw_source(
        streams[0], cfg.n_obs, cfg, source="obs", role="obs_train"
    )
    rct_train, _, train_diag = _draw_source(
        streams[1], cfg.n_rct_train, cfg, source="rct", role="rct_train"
    )
    rct_tune, _, tune_diag = _draw_source(
        streams[2], cfg.n_rct_tune, cfg, source="rct", role="rct_tune"
    )
    rct_test, _, test_diag = _draw_source(
        streams[3], cfg.n_rct_test, cfg, source="rct", role="rct_test"
    )
    eval_X = _draw_target_covariates(streams[4], cfg.n_eval, cfg)
    target_reference = None
    if cfg.n_target_reference:
        target_rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(0x54415247,))
        )
        target_reference = TargetCovariateSample(
            _draw_target_covariates(target_rng, cfg.n_target_reference, cfg)
        )
    eval_truth = structural_truth_at(eval_X, cfg)
    normalizers = {
        source: max(float(eval_truth[f"{source}_density_unnormalized"].mean()), 1e-12)
        for source in ("obs", "rct")
    }
    _normalize_density(eval_truth, normalizers)
    obs_truth = structural_truth_at(obs.X, cfg)
    _normalize_density(obs_truth, normalizers)
    obs_truth.update(obs_realized)
    truth_definition = (
        "c_t=E[Y(t)|X,T=t,S=OBS]-E[Y(t)|X,S=OBS]; "
        "d_t=E[Y(t)|X,S=OBS]-E[Y(t)|X,target]; r_t=c_t+d_t"
    )
    if _participation_is_active(cfg):
        truth_definition += (
            "; RCT shifts integrate hidden participation under its selected law"
        )
    truth = SimulationTruth(
        eval=eval_truth, obs_train=obs_truth, n_eval=cfg.n_eval,
        n_obs=cfg.n_obs, definition=truth_definition,
    )
    is_addendum = cfg.mechanism in STRUCTURAL_ADDENDUM_CARDS
    cards = STRUCTURAL_ADDENDUM_CARDS if is_addendum else MECHANISM_CARDS
    card = dict(cards.get(cfg.mechanism, {}))
    axis_levels = structural_axis_levels(cfg)
    card["axis_levels"] = axis_levels
    signal_transform = np.ascontiguousarray(_signal_transform(
        cfg.d, cfg.covariate_rho, cfg.signal_basis, cfg.signal_seed
    ))
    covariance = np.ascontiguousarray(_target_covariance(cfg.d, cfg.covariate_rho))
    sampling_axes = {
        "n_obs": cfg.n_obs, "n_rct_train": cfg.n_rct_train,
        "n_rct_tune": cfg.n_rct_tune, "n_rct_test": cfg.n_rct_test,
        "n_eval": cfg.n_eval, "dimension": cfg.d,
    }
    if cfg.n_target_reference:
        sampling_axes["n_target_reference"] = cfg.n_target_reference
    metadata = {
        "generator": "structural_addendum_v1" if is_addendum else "structural_v1",
        "generator_version": (
            STRUCTURAL_ADDENDUM_VERSION if is_addendum else STRUCTURAL_DGP_VERSION
        ),
        "mechanism": cfg.mechanism,
        "target_population": (
            "standardized observed marginals declared by covariate_marginals "
            "with an AR(1) Gaussian copula declared by covariate_rho"
        ),
        "mechanism_card": card,
        "axis_levels": axis_levels,
        "design_overrides": structural_design_overrides(cfg),
        "signal_transform_sha256": hashlib.sha256(signal_transform.tobytes()).hexdigest(),
        "latent_gaussian_covariance_sha256": hashlib.sha256(covariance.tobytes()).hexdigest(),
        "sampling_axes": sampling_axes,
        "config": asdict(cfg),
        "obs_diagnostics": obs_diag,
        "rct_train_diagnostics": train_diag,
        "rct_tune_diagnostics": tune_diag,
        "rct_test_diagnostics": test_diag,
        "density_normalizers": normalizers,
    }
    return SimulationProblem(
        name=cfg.name, seed=seed, obs_train=obs, rct_train=rct_train,
        rct_tune=rct_tune, rct_test=rct_test, eval_X=eval_X,
        eval_tau=eval_truth["tau_target"],
        eval_region_score=eval_truth["bias_magnitude"],
        metadata=metadata, truth=truth, target_reference=target_reference,
    )
