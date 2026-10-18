"""Marginal-sensitivity-model data generation with exact binary-hidden truth.

This is a DGP factory, not a sensitivity estimator. The odds ratio compares
P(T=1 | X,U) with P(T=1 | X), as in Zhao, Small and Bhattacharya (2019).
All oracle quantities stay outside the estimator's TrainingView.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import SimulationProblem, SimulationTruth, SourceSample, TargetCovariateSample


FloatArray = NDArray[np.float64]
ScalarOrFunction = float | Callable[[FloatArray], ArrayLike]
GAMMA_DGP_VERSION = "1.0.0"


def _vector(values: ArrayLike, *, label: str, rows: int | None = None) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 and rows is not None:
        result = np.full(rows, float(result), dtype=np.float64)
    if result.ndim != 1 or (rows is not None and len(result) != rows):
        raise ValueError(f"{label} must be a scalar or one value per row")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _X(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError("X must be a nonempty finite two-dimensional array")
    return result


def _at(specification: ScalarOrFunction, X: FloatArray, label: str) -> FloatArray:
    return _vector(
        specification(X) if callable(specification) else specification,
        label=label,
        rows=len(X),
    )


def _expit(values: FloatArray) -> FloatArray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(probability: FloatArray) -> FloatArray:
    return np.log(probability) - np.log1p(-probability)


def msm_propensity_bounds(
    observed_propensity: ArrayLike, gamma: ArrayLike | float
) -> tuple[FloatArray, FloatArray]:
    """Return finite interior MSM endpoints for each marginal propensity.

    Scalars for Gamma broadcast over a one-dimensional propensity array.
    Numerically unrepresentable endpoints raise instead of being clipped.
    The caller remains responsible for marginal compatibility for a custom
    hidden-variable distribution.
    """
    e = _vector(observed_propensity, label="observed propensity")
    g = _vector(gamma, label="Gamma", rows=len(e))
    if len(e) == 0 or np.any((e <= 0.0) | (e >= 1.0)):
        raise ValueError("observed propensities must lie strictly in (0,1)")
    if np.any(g < 1.0):
        raise ValueError("Gamma must be finite and at least 1")
    log_e = _logit(e)
    lower = _expit(log_e - np.log(g))
    upper = _expit(log_e + np.log(g))
    # Gamma=1 is exact even for subnormal or near-one probabilities.
    lower = np.where(g == 1.0, e, lower)
    upper = np.where(g == 1.0, e, upper)
    if np.any((lower <= 0.0) | (upper >= 1.0)):
        raise ValueError("MSM endpoints are not representable in float64")
    if np.any((g > 1.0) & ((lower >= e) | (upper <= e))):
        raise ValueError("requested Gamma contrast is not representable in float64")
    return lower, upper


def validate_msm_propensities(
    observed_propensity: ArrayLike,
    full_propensity: ArrayLike,
    gamma: ArrayLike | float,
) -> FloatArray:
    """Validate pointwise MSM odds bounds and return represented local Gamma.

    This checks the odds bound only; it does not prove marginal compatibility.
    A tolerance of 64 machine epsilons on the log-odds scale covers arithmetic
    roundoff. Larger violations fail closed.
    """
    e = _vector(observed_propensity, label="observed propensity")
    full = _vector(full_propensity, label="full propensity", rows=len(e))
    g = _vector(gamma, label="Gamma", rows=len(e))
    if np.any((e <= 0.0) | (e >= 1.0) | (full <= 0.0) | (full >= 1.0)):
        raise ValueError("MSM propensities must lie strictly in (0,1)")
    if np.any(g < 1.0):
        raise ValueError("Gamma must be finite and at least 1")
    log_ratio = np.abs(_logit(full) - _logit(e))
    log_gamma = np.log(g)
    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, log_gamma)
    if np.any(log_ratio > log_gamma + tolerance):
        raise ValueError("represented propensity violates the requested MSM bound")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return np.exp(log_ratio)
        except FloatingPointError as error:
            raise ValueError("local Gamma is not representable in float64") from error


@dataclass(frozen=True)
class MSMPropensities:
    """Oracle assignment arrays for fair U in {-1,+1}; all are read-only."""

    observed: FloatArray
    gamma: FloatArray
    delta: FloatArray
    p_minus: FloatArray
    p_plus: FloatArray
    local_gamma: FloatArray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def construct_msm_propensities(
    observed_propensity: ArrayLike,
    gamma: ArrayLike | float,
    *,
    bias_direction: int = 1,
) -> MSMPropensities:
    """Preserve e(X) exactly in law while attaining each pointwise Gamma.

    For fair binary U, use e(X,U)=e(X)+direction*U*delta(X), where
    delta=min(e-expit(logit(e)-log(Gamma)),
              expit(logit(e)+log(Gamma))-e).
    At least one hidden branch attains the prescribed bound in real arithmetic.
    """
    if isinstance(bias_direction, bool) or bias_direction not in (-1, 1):
        raise ValueError("bias_direction must be -1 or +1")
    e = _vector(observed_propensity, label="observed propensity")
    g = _vector(gamma, label="Gamma", rows=len(e))
    lower, upper = msm_propensity_bounds(e, g)
    delta = np.minimum(e - lower, upper - e)
    minus = e - bias_direction * delta
    plus = e + bias_direction * delta
    local_minus = validate_msm_propensities(e, minus, g)
    local_plus = validate_msm_propensities(e, plus, g)
    local = np.maximum(local_minus, local_plus)
    # Losing sharpness is a numerical failure, rather than a weaker DGP.
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.log(g))
    if np.any(np.abs(np.log(local) - np.log(g)) > tolerance):
        raise ValueError("sharp MSM contrast is not representable in float64")
    return MSMPropensities(e, g, delta, minus, plus, local)


def msm_truth_at(
    X: ArrayLike,
    *,
    observed_propensity: ScalarOrFunction,
    gamma: ScalarOrFunction,
    mu0: ScalarOrFunction = 0.0,
    tau: ScalarOrFunction = 1.0,
    hidden_loading0: ScalarOrFunction = 1.0,
    hidden_loading1: ScalarOrFunction = 1.0,
    obs_shift0: ScalarOrFunction = 0.0,
    obs_shift1: ScalarOrFunction = 0.0,
    bias_direction: int = 1,
) -> dict[str, FloatArray]:
    """Exact evaluator truth from the two-point hidden distribution.

    Callback outputs must be finite scalars or vectors with one value per X row.
    Gamma callbacks are checked at the supplied rows. They do not establish a
    global bound outside those rows. Hidden outcome loadings and source shifts
    are independent controls, not determined by Gamma.
    """
    x = _X(X)
    e = _at(observed_propensity, x, "observed propensity")
    g = _at(gamma, x, "Gamma")
    assignments = construct_msm_propensities(e, g, bias_direction=bias_direction)
    baseline = _at(mu0, x, "mu0")
    effect = _at(tau, x, "tau")
    loading0 = _at(hidden_loading0, x, "hidden_loading0")
    loading1 = _at(hidden_loading1, x, "hidden_loading1")
    shift0 = _at(obs_shift0, x, "obs_shift0")
    shift1 = _at(obs_shift1, x, "obs_shift1")
    # Exact sums over U=-1,+1: P(U=u)=1/2 in every source and target.
    hidden_treated = 0.5 * (assignments.p_plus - assignments.p_minus) / e
    hidden_control = 0.5 * (assignments.p_minus - assignments.p_plus) / (1.0 - e)
    bias0 = loading0 * hidden_control
    bias1 = loading1 * hidden_treated
    mu1 = baseline + effect
    result = {
        "mu0_target": baseline,
        "mu1_target": mu1,
        "tau_target": effect,
        "obs_propensity": e,
        "gamma_requested": g,
        "gamma_local": assignments.local_gamma,
        "hidden_propensity_minus": assignments.p_minus,
        "hidden_propensity_plus": assignments.p_plus,
        "hidden_outcome_loading_0": loading0,
        "hidden_outcome_loading_1": loading1,
        "hidden_mean_treated": hidden_treated,
        "hidden_mean_control": hidden_control,
        "confounding_bias_0": bias0,
        "confounding_bias_1": bias1,
        "source_shift_0": shift0,
        "source_shift_1": shift1,
        "regression_discrepancy_0": bias0 + shift0,
        "regression_discrepancy_1": bias1 + shift1,
        "obs_conditional_mean_0": baseline + shift0 + bias0,
        "obs_conditional_mean_1": mu1 + shift1 + bias1,
        "source_cate_shift": shift1 - shift0,
        "confounding_cate_bias": bias1 - bias0,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("outcome/truth arithmetic overflowed")
    return result


def sample_msm_source(
    X: ArrayLike,
    *,
    source: Literal["obs", "rct"],
    role: str,
    rng: np.random.Generator,
    observed_propensity: ScalarOrFunction,
    gamma: ScalarOrFunction,
    mu0: ScalarOrFunction = 0.0,
    tau: ScalarOrFunction = 1.0,
    hidden_loading0: ScalarOrFunction = 1.0,
    hidden_loading1: ScalarOrFunction = 1.0,
    obs_shift0: ScalarOrFunction = 0.0,
    obs_shift1: ScalarOrFunction = 0.0,
    bias_direction: int = 1,
    rct_propensity: ScalarOrFunction = 0.5,
    noise_std: float = 1.0,
) -> tuple[SourceSample, dict[str, FloatArray]]:
    """Generate observed rows and separate oracle truth for an arbitrary X law.

    No sampled U or full OBS propensity is included in SourceSample. RCT
    treatment is randomized independently of U conditional on X, with its
    probability exposed through the normal known_propensity contract.
    """
    x = _X(X)
    if source not in {"obs", "rct"}:
        raise ValueError("source must be 'obs' or 'rct'")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and nonnegative")
    truth = msm_truth_at(
        x, observed_propensity=observed_propensity, gamma=gamma, mu0=mu0,
        tau=tau, hidden_loading0=hidden_loading0, hidden_loading1=hidden_loading1,
        obs_shift0=obs_shift0, obs_shift1=obs_shift1, bias_direction=bias_direction,
    )
    hidden = 2.0 * rng.integers(0, 2, size=len(x)) - 1.0
    if source == "obs":
        propensity = np.where(
            hidden > 0.0, truth["hidden_propensity_plus"], truth["hidden_propensity_minus"]
        )
        known = None
    else:
        propensity = _at(rct_propensity, x, "RCT propensity")
        if np.any((propensity <= 0.0) | (propensity >= 1.0)):
            raise ValueError("RCT propensities must lie strictly in (0,1)")
        known = propensity
    treatment = rng.binomial(1, propensity).astype(np.float64)
    loading = np.where(treatment == 1.0, truth["hidden_outcome_loading_1"], truth["hidden_outcome_loading_0"])
    outcome = truth["mu0_target"] + treatment * truth["tau_target"] + loading * hidden
    if source == "obs":
        outcome = outcome + np.where(treatment == 1.0, truth["source_shift_1"], truth["source_shift_0"])
    outcome = outcome + noise_std * rng.normal(size=len(x))
    if not np.isfinite(outcome).all():
        raise ValueError("sampled outcome arithmetic overflowed")
    return SourceSample(x, treatment, outcome, source, role, known), truth


@dataclass(frozen=True)
class GammaDGPConfig:
    """Serializable original Gaussian DGP with an explicit global Gamma bound."""

    name: str
    n_obs: int = 2_000
    n_rct_train: int = 200
    n_rct_tune: int = 200
    n_rct_test: int = 500
    n_eval: int = 2_000
    n_target_reference: int = 0
    d: int = 5
    gamma_profile: Literal["constant", "logistic"] = "constant"
    gamma: float = 2.0
    gamma_min: float = 1.0
    gamma_slope: float = 1.0
    gamma_intercept: float = 0.0
    observed_overlap: float = 0.5
    propensity_floor: float = 0.02
    hidden_loading0: float = 1.0
    hidden_loading1: float = 1.0
    bias_direction: int = 1
    obs_baseline_shift: float = 0.0
    obs_cate_shift: float = 0.0
    obs_covariate_shift: float = 0.0
    rct_covariate_shift: float = 0.0
    cate_complexity: Literal["constant", "linear", "smooth", "interaction"] = "smooth"
    cate_scale: float = 1.0
    noise_std: float = 1.0
    rct_propensity: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        for key in ("n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval", "n_target_reference", "d"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer")
            if value < (0 if key == "n_target_reference" else 1):
                raise ValueError(f"{key} must be positive, or zero for n_target_reference")
        if self.gamma_profile not in {"constant", "logistic"}:
            raise ValueError("gamma_profile must be 'constant' or 'logistic'")
        if self.cate_complexity not in {"constant", "linear", "smooth", "interaction"}:
            raise ValueError("unknown cate_complexity")
        for key in (
            "gamma", "gamma_min", "gamma_slope", "gamma_intercept", "observed_overlap",
            "propensity_floor", "hidden_loading0", "hidden_loading1", "obs_baseline_shift",
            "obs_cate_shift", "obs_covariate_shift", "rct_covariate_shift", "cate_scale",
            "noise_std", "rct_propensity",
        ):
            value = getattr(self, key)
            if isinstance(value, (bool, str, bytes)) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if not 1.0 <= self.gamma_min <= self.gamma:
            raise ValueError("Gamma must satisfy 1 <= gamma_min <= gamma")
        if self.observed_overlap < 0.0:
            raise ValueError("observed_overlap must be nonnegative")
        if not 0.0 < self.propensity_floor < 0.5:
            raise ValueError("propensity_floor must lie strictly in (0,0.5)")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative")
        if not 0.0 < self.rct_propensity < 1.0:
            raise ValueError("rct_propensity must lie strictly in (0,1)")
        if isinstance(self.bias_direction, bool) or self.bias_direction not in (-1, 1):
            raise ValueError("bias_direction must be -1 or +1")

    def resolve_gamma(self, X: ArrayLike) -> FloatArray:
        x = _X(X)
        if self.gamma_profile == "constant":
            return np.full(len(x), self.gamma)
        with np.errstate(over="raise", invalid="raise"):
            try:
                score = self.gamma_intercept + self.gamma_slope * x[:, 0]
            except FloatingPointError as error:
                raise ValueError("Gamma profile arithmetic overflowed") from error
        return self.gamma_min + (self.gamma - self.gamma_min) * _expit(score)


def _config_components(config: GammaDGPConfig) -> dict[str, ScalarOrFunction | int]:
    def propensity(x: FloatArray) -> FloatArray:
        with np.errstate(over="raise", invalid="raise"):
            try:
                score = config.observed_overlap * x[:, 0]
            except FloatingPointError as error:
                raise ValueError("observed propensity arithmetic overflowed") from error
        return config.propensity_floor + (1.0 - 2.0 * config.propensity_floor) * _expit(score)

    def baseline(x: FloatArray) -> FloatArray:
        return 0.5 * x[:, 0] + 0.25 * np.sin(x[:, min(1, x.shape[1] - 1)])

    def effect(x: FloatArray) -> FloatArray:
        first, second = x[:, 0], x[:, min(1, x.shape[1] - 1)]
        if config.cate_complexity == "constant":
            raw = np.ones(len(x))
        elif config.cate_complexity == "linear":
            raw = 1.0 + first
        elif config.cate_complexity == "smooth":
            raw = 1.0 + np.sin(first) + 0.5 * np.tanh(second)
        else:
            raw = 1.0 + np.tanh(first) * np.tanh(second)
        return config.cate_scale * raw

    return dict(
        observed_propensity=propensity, gamma=config.resolve_gamma, mu0=baseline,
        tau=effect, hidden_loading0=config.hidden_loading0,
        hidden_loading1=config.hidden_loading1, bias_direction=config.bias_direction,
        obs_shift0=config.obs_baseline_shift,
        obs_shift1=config.obs_baseline_shift + config.obs_cate_shift,
    )


def gamma_truth_at(
    X: ArrayLike,
    config: GammaDGPConfig,
    *,
    gamma: ScalarOrFunction | None = None,
) -> dict[str, FloatArray]:
    """Evaluate the serializable factory's exact two-point structural truth."""
    x = _X(X)
    if x.shape[1] != config.d:
        raise ValueError("X dimension does not match GammaDGPConfig.d")
    components = _config_components(config)
    if gamma is not None:
        components["gamma"] = gamma
    return msm_truth_at(x, **components)


def generate_gamma_problem(
    config: GammaDGPConfig,
    seed: int,
    *,
    gamma: ScalarOrFunction | None = None,
) -> SimulationProblem:
    """Generate independent OBS/train/tune/test/evaluation/reference streams.

    The optional Python-only Gamma override accepts an arbitrary scalar or
    deterministic function of X. Callback metadata records pointwise checks,
    without inferring a global bound or serializing executable code.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)]
    components = _config_components(config)
    if gamma is not None:
        components["gamma"] = gamma

    def draw(rows: int, shift: float, rng: np.random.Generator) -> FloatArray:
        x = rng.normal(size=(rows, config.d))
        x[:, 0] += shift
        return x

    samples = {}
    obs_truth = None
    for index, (role, rows, shift) in enumerate((
        ("obs_train", config.n_obs, config.obs_covariate_shift),
        ("rct_train", config.n_rct_train, config.rct_covariate_shift),
        ("rct_tune", config.n_rct_tune, config.rct_covariate_shift),
        ("rct_test", config.n_rct_test, 0.0),
    )):
        rng = streams[index]
        sample, truth = sample_msm_source(
            draw(rows, shift, rng), source="obs" if index == 0 else "rct",
            role=role, rng=rng, rct_propensity=config.rct_propensity,
            noise_std=config.noise_std, **components,
        )
        samples[role] = sample
        if index == 0:
            obs_truth = truth
    eval_x = draw(config.n_eval, 0.0, streams[4])
    eval_truth = msm_truth_at(eval_x, **components)
    target = None
    if config.n_target_reference:
        target = TargetCovariateSample(draw(config.n_target_reference, 0.0, streams[5]))
    global_upper = config.gamma if gamma is None else (None if callable(gamma) else float(gamma))
    gamma_specification = (
        {"kind": "config_profile", "profile": config.gamma_profile}
        if gamma is None else
        {"kind": "callback", "guarantee": "pointwise_at_evaluated_rows"}
        if callable(gamma) else
        {"kind": "constant_override", "value": float(gamma)}
    )
    return SimulationProblem(
        name=config.name, seed=seed, **samples, eval_X=eval_x,
        eval_tau=eval_truth["tau_target"],
        eval_region_score=np.maximum(np.abs(eval_truth["confounding_bias_0"]), np.abs(eval_truth["confounding_bias_1"])),
        target_reference=target,
        truth=SimulationTruth(
            eval=eval_truth, obs_train=obs_truth, n_eval=config.n_eval,
            n_obs=config.n_obs,
            definition="Fair binary U; exact two-point conditional sums; MSM odds ratio relative to marginal e(X).",
        ),
        metadata={
            "dgp_kind": "gamma", "dgp_version": GAMMA_DGP_VERSION,
            "config": asdict(config), "gamma_specification": gamma_specification,
            "gamma_global_upper": global_upper,
            "gamma_global_upper_is_bound": global_upper is not None,
            "truth_method": "exact_binary_sum",
            "source": "https://doi.org/10.1111/rssb.12327",
            "implementation": "original DGP construction; not a published DGP reproduction",
        },
    )
