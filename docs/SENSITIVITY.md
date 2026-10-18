# General marginal-sensitivity-model DGP factory

The factory generates observational and randomized samples with a configurable scalar `Gamma >= 1` or pointwise `Gamma(X) >= 1`. It is an original data-generation construction with known CATE, not a sensitivity estimator or a reproduction of a published simulation. The general Python API accepts functions of visible covariates. The serializable `GammaDGPConfig` supplies constant and logistic profiles for configuration files.

The model bounds the treatment odds conditional on hidden information relative to the marginal treatment odds given visible covariates. This is the marginal sensitivity model in [Zhao, Small and Bhattacharya (2019), Definition 1](https://doi.org/10.1111/rssb.12327), building on Tan (2006). It differs from a bound comparing two hidden-variable treatment odds at the same visible covariates. The construction below is original; the citation supports the sensitivity-model definition.

## Assignment construction and guarantee

For each visible `x`, choose any finite `Gamma(x) >= 1` and any observed propensity `0 < e(x) < 1`. Let `U` be independent of `X` with equal probability on `-1` and `+1` in the target, OBS, and RCT populations. Define

```text
p_lo(x) = expit(logit(e(x)) - log(Gamma(x)))
p_hi(x) = expit(logit(e(x)) + log(Gamma(x)))
delta(x) = min(e(x) - p_lo(x), p_hi(x) - e(x))
p(x,u) = e(x) + direction * u * delta(x),  direction in {-1,+1}.
```

Because `E[U | X] = 0`, the true OBS marginal propensity is exactly `e(x)` in the data-generating law. Both hidden branches lie in `[p_lo(x), p_hi(x)]`, so

```text
1 / Gamma(x) <= odds(p(x,u)) / odds(e(x)) <= Gamma(x).
```

At least one branch attains a boundary in real arithmetic. Thus the two-point construction has pointwise local maximum odds deviation `Gamma(x)`, even when `e(x) != 0.5`; simply shifting a logistic intercept by `U*log(Gamma)` would generally change the prescribed marginal `e(x)`. Reversing `direction` swaps hidden branches and reverses the hidden-selection biases while preserving `e(x)` and `Gamma(x)`.

`msm_propensity_bounds(e, gamma)` exposes the interval for custom finite-support constructions. The interval alone does not ensure that a custom hidden distribution marginalizes to `e(x)`. `validate_msm_propensities(e, full, gamma)` checks only a supplied pointwise odds bound. `construct_msm_propensities(e, gamma)` adds the fair-binary construction and returns read-only assignment arrays and its represented local maximum.

All parameters must be finite. The implementation rejects probabilities at zero or one and rejects requested endpoints or contrasts that cannot be represented in float64. It never clips an unrepresentable endpoint to make a request appear valid. The represented odds bound is checked with a log-scale roundoff tolerance of `64*eps*max(1,log(Gamma))`; a separate `128*eps` check rejects material loss of pointwise sharpness. Algebraic equalities refer to the generating law; ordinary floating-point rounding still applies to returned arrays.

## Outcomes, target, and exact truth

Users independently specify visible baseline `mu0(x)`, visible CATE `tau(x)`, hidden outcome loadings `a0(x), a1(x)`, and OBS outcome shifts `s0(x), s1(x)`:

```text
Y_target(t) = mu0(X) + t*tau(X) + a_t(X)*U + epsilon_t
Y_RCT(t)    = Y_target(t)
Y_OBS(t)    = Y_target(t) + s_t(X)
```

The outcome noise is independent, mean-zero Gaussian with configurable standard deviation. Sampling generates the realized noise only; these equations do not impose an unobserved cross-arm noise correlation. Hidden loading can modify individual effects when `a0 != a1`. The target CATE remains `tau(X)` because target `U` has mean zero. Gamma controls hidden treatment selection and does not determine either outcome loading, either source shift, or CATE complexity. At `Gamma=1`, hidden-selection bias is zero even with nonzero outcome loadings; source shifts can remain. Zero hidden outcome loadings remove hidden-selection bias at any Gamma.

For `p_minus=p(x,-1)` and `p_plus=p(x,+1)`, exact two-point conditioning gives

```text
E[U | X=x,T=1,OBS] = (p_plus - p_minus) / (2*e(x))
E[U | X=x,T=0,OBS] = (p_minus - p_plus) / (2*(1-e(x)))
b_t(x) = a_t(x) * E[U | X=x,T=t,OBS]
E[Y | X=x,T=t,OBS] = mu0(x) + t*tau(x) + s_t(x) + b_t(x).
```

`msm_truth_at` returns the marginal propensity, both hidden propensities, requested and represented local Gamma, exact arm means, arm confounding biases, source shifts, and their total regression discrepancies. `gamma_truth_at(X, config)` evaluates the serializable configuration. `SimulationTruth` stores these quantities for evaluators. `TrainingView` contains observed source samples and optional outcome-free target covariates only: no `U`, OBS oracle propensity, Gamma array, truth, held-out test outcomes, or evaluation CATE is supplied to methods.

RCT treatment is independent of `U` given `X`. The callback API accepts any finite `e_r(X)` strictly between zero and one and supplies it as `known_propensity`. The serializable factory uses a constant `rct_propensity`. RCT test rows and evaluation covariates follow the target law. OBS and training/tuning RCT covariate shifts affect their covariate distributions, and do not change the specified target.

## Python use

```python
import numpy as np
from obs_rct_benchmark.sensitivity import (
    GammaDGPConfig, generate_gamma_problem, sample_msm_source, msm_truth_at,
)

config = GammaDGPConfig(
    name="gamma_example", gamma_profile="logistic", gamma=5.0,
    gamma_min=1.0, gamma_slope=1.5, hidden_loading0=0.5,
    hidden_loading1=1.2, bias_direction=-1, obs_cate_shift=0.3,
    cate_complexity="interaction",
)
problem = generate_gamma_problem(config, seed=17)
training = problem.training_view()
# The complete role-separated factory also accepts a Python-only callback.
custom = generate_gamma_problem(
    config, seed=17, gamma=lambda x: 1.0 + 2.0 * np.exp(-x[:, 0] ** 2),
)

# Any finite X matrix can be supplied by a user-defined covariate sampler.
# Use separate RNG streams and independent rows for each benchmark role.
X = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
specification = dict(
    observed_propensity=lambda x: 0.2 + 0.6 / (1.0 + np.exp(-x[:, 0])),
    gamma=lambda x: 1.0 + 2.0 * np.exp(-x[:, 0] ** 2),
    mu0=lambda x: x[:, 0], tau=lambda x: 1.0 + x[:, 1],
    hidden_loading0=0.4, hidden_loading1=-0.7,
)
observed, evaluator_truth = sample_msm_source(
    X, source="obs", role="obs_train", rng=np.random.default_rng(31),
    **specification,
)
truth_at_rows = msm_truth_at(X, **specification)
```

Callbacks must return a finite scalar or a vector with one value per supplied row. They should be deterministic functions of `X`; the generator does not serialize, inspect, or globally optimize user callbacks. The guarantee is checked pointwise wherever a callback is evaluated. A callback must supply its own mathematical global bound if one is claimed, and generated sample maxima do not prove it. `generate_gamma_problem(config, seed, gamma=...)` overrides the configuration's Gamma profile only in the Python API; a callable override records `gamma_global_upper=None` and a pointwise-check label in metadata. A scalar override records its constant global bound. Preserve callback source and parameters externally when reproducing a custom Python run; the serializable configuration alone cannot reproduce executable callback code.

## Serializable configuration

`GammaDGPConfig` has role budgets `n_obs`, `n_rct_train`, `n_rct_tune`, `n_rct_test`, `n_eval`, and optional `n_target_reference`. Each role receives its own `SeedSequence` child stream. Changing a training-role budget does not change test/evaluation rows, and changing test/evaluation budgets does not change training samples. Methods must tune with observed training/tuning roles, never these evaluator truths.

The visible covariate law is independent standard Gaussian coordinates; `obs_covariate_shift` and `rct_covariate_shift` shift the first coordinate in OBS and training/tuning RCT respectively. Target, RCT test, and reference coordinates are standard Gaussian. The observed propensity is

```text
e(X) = floor + (1-2*floor)*expit(observed_overlap*X[0]).
```

`propensity_floor` lies strictly between zero and one half. Larger nonnegative `observed_overlap` makes marginal treatment propensities more separated. Baseline is `0.5*X[0]+0.25*sin(X[1])`. In dimension one the second coordinate is replaced by the first. CATE is `cate_scale` times one of `1`, `1+X[0]`, `1+sin(X[0])+0.5*tanh(X[1])`, or `1+tanh(X[0])*tanh(X[1])`, selected by `cate_complexity = "constant"`, `"linear"`, `"smooth"`, or `"interaction"`. OBS shifts are constant: `s0=obs_baseline_shift`, `s1=obs_baseline_shift+obs_cate_shift`.

The constant Gamma profile uses `Gamma(x)=gamma`. The logistic profile uses

```text
Gamma(x) = gamma_min + (gamma-gamma_min)
           * expit(gamma_intercept + gamma_slope*x[0]),
1 <= gamma_min <= gamma < infinity.
```

`gamma` is an explicit global upper bound for both profiles. For a nonconstant logistic profile with full Gaussian support it is the supremum, which need not be reached at any finite `x` or by a finite sample. If `gamma_slope=0`, the profile is constant at its intermediate value, so `gamma` is still a valid upper bound but need not be the smallest one. Metadata calls it an upper bound; it does not claim an observed maximum equals the bound. The complete dataclass configuration is serializable with `dataclasses.asdict`.

For CLI integration use scenario `kind="gamma"`, `id="constant"` or `id="logistic"`, and place the remaining dataclass controls in the scenario's `parameters` table; the runner supplies the scenario name. See [gamma_smoke.toml](../configs/gamma_smoke.toml) for the run schema.

## Verification scope

`tests/test_sensitivity.py` independently calculates rational propensity endpoints, enumerates the two hidden states with Bayes' rule, checks actual odds ratios on a bounded grid, and verifies the zero-Gamma-contrast and zero-outcome-loading limits. It also checks independent controls, role-stream separation, known nonconstant RCT probabilities, malformed callbacks, boundary probabilities, and numerically unrepresentable requests. These establish the implementation identities; they do not assert that every member of the marginal sensitivity model can be generated by this two-point construction or that any estimator achieves a sensitivity bound.
