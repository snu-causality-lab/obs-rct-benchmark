"""Independent finite-support checks for the general MSM DGP construction."""

from dataclasses import asdict, fields, replace
import math

import numpy as np
import pytest

from obs_rct_benchmark.sensitivity import (
    GammaDGPConfig,
    construct_msm_propensities,
    gamma_truth_at,
    generate_gamma_problem,
    msm_propensity_bounds,
    msm_truth_at,
    sample_msm_source,
    validate_msm_propensities,
)


SMALL = dict(n_obs=19, n_rct_train=13, n_rct_tune=11, n_rct_test=17, n_eval=23)


def _odds(probability):
    return probability / (1.0 - probability)


@pytest.mark.parametrize("direction", [-1, 1])
def test_binary_propensities_preserve_marginal_and_attain_pointwise_bound(direction):
    # Independent rational endpoints, without logit/expit or truth helpers.
    e = np.array([0.02, 0.15, 0.5, 0.71, 0.98])
    gamma = np.array([1.0, 1.2, 3.0, 11.0, 2.0])
    lower = e / (e + gamma * (1.0 - e))
    upper = gamma * e / (1.0 - e + gamma * e)
    delta = np.minimum(e - lower, upper - e)
    assignment = construct_msm_propensities(e, gamma, bias_direction=direction)
    np.testing.assert_allclose(assignment.p_minus, e - direction * delta, rtol=1e-14)
    np.testing.assert_allclose(assignment.p_plus, e + direction * delta, rtol=1e-14)
    np.testing.assert_allclose((assignment.p_minus + assignment.p_plus) / 2.0, e, atol=1e-16)
    odds_ratios = np.stack([
        _odds(assignment.p_minus) / _odds(e),
        _odds(assignment.p_plus) / _odds(e),
    ])
    assert np.all(odds_ratios < gamma[None, :] + 1e-12)
    assert np.all(odds_ratios > 1.0 / gamma[None, :] - 1e-12)
    attained = np.maximum(odds_ratios.max(axis=0), (1.0 / odds_ratios).max(axis=0))
    np.testing.assert_allclose(attained, gamma, rtol=1e-13)
    for item in fields(assignment):
        assert not getattr(assignment, item.name).flags.writeable


def test_property_grid_checks_actual_odds_and_fair_hidden_marginal():
    e = np.linspace(0.03, 0.97, 31)
    gamma = np.exp(np.linspace(0.0, math.log(19.0), 31))
    result = construct_msm_propensities(e, gamma)
    for index in range(len(e)):
        probabilities = [float(result.p_minus[index]), float(result.p_plus[index])]
        assert math.isclose(math.fsum(probabilities) / 2.0, float(e[index]), abs_tol=1e-16)
        ratios = [_odds(p) / _odds(float(e[index])) for p in probabilities]
        assert min(ratios) >= 1.0 / gamma[index] - 1e-13
        assert max(ratios) <= gamma[index] + 1e-12
        assert math.isclose(max(*ratios, *(1.0 / r for r in ratios)), gamma[index], rel_tol=1e-12)


def test_gamma_one_is_exact_at_representable_probability_extremes():
    e = np.array([np.nextafter(0.0, 1.0), 1e-250, 0.5, np.nextafter(1.0, 0.0)])
    assignment = construct_msm_propensities(e, 1.0)
    np.testing.assert_array_equal(assignment.p_minus, e)
    np.testing.assert_array_equal(assignment.p_plus, e)
    np.testing.assert_array_equal(assignment.local_gamma, 1.0)
    np.testing.assert_array_equal(assignment.delta, 0.0)


@pytest.mark.parametrize("e,gamma", [
    ([0.0], 2.0), ([1.0], 2.0), ([0.5], 0.9), ([0.5], math.inf),
    ([math.nan], 2.0), ([1e-300], 1e100), ([np.nextafter(1.0, 0.0)], 2.0),
])
def test_invalid_or_unrepresentable_msm_requests_fail_closed(e, gamma):
    with pytest.raises(ValueError):
        construct_msm_propensities(e, gamma)


def test_validation_rejects_custom_propensity_violation():
    with pytest.raises(ValueError, match="violates"):
        validate_msm_propensities([0.5], [0.9], 2.0)
    with pytest.raises(ValueError, match="strictly"):
        validate_msm_propensities([0.5], [1.0], 2.0)


def test_exact_truth_matches_independent_bayes_enumeration_with_hidden_effect_modifier():
    X = np.array([[-2.0], [0.5], [1.0]])
    e = np.array([0.2, 0.5, 0.8])
    g = np.array([1.0, 2.0, 4.0])
    truth = msm_truth_at(
        X, observed_propensity=lambda x: e, gamma=lambda x: g,
        mu0=lambda x: x[:, 0], tau=lambda x: 2.0 + x[:, 0],
        hidden_loading0=0.7, hidden_loading1=-1.3,
        obs_shift0=0.4, obs_shift1=-0.2, bias_direction=-1,
    )
    for row in range(len(X)):
        lower = e[row] / (e[row] + g[row] * (1.0 - e[row]))
        upper = g[row] * e[row] / (1.0 - e[row] + g[row] * e[row])
        delta = min(e[row] - lower, upper - e[row])
        arm_means = []
        for treatment in (0, 1):
            numerator = denominator = 0.0
            for hidden in (-1.0, 1.0):
                p_treated = e[row] - hidden * delta
                p_arm = p_treated if treatment else 1.0 - p_treated
                potential_mean = (
                    X[row, 0] + treatment * (2.0 + X[row, 0])
                    + (0.7 if treatment == 0 else -1.3) * hidden
                    + (0.4 if treatment == 0 else -0.2)
                )
                numerator += 0.5 * p_arm * potential_mean
                denominator += 0.5 * p_arm
            arm_means.append(numerator / denominator)
            assert truth[f"obs_conditional_mean_{treatment}"][row] == pytest.approx(arm_means[-1], abs=1e-13)
        # Marginal target U has zero mean despite arm-specific hidden loadings.
        target_potential = [
            math.fsum(X[row, 0] + t * (2.0 + X[row, 0]) + (0.7 if t == 0 else -1.3) * u for u in (-1.0, 1.0)) / 2.0
            for t in (0, 1)
        ]
        assert truth["tau_target"][row] == pytest.approx(target_potential[1] - target_potential[0])


def test_assignment_outcome_shift_and_cate_controls_are_independent():
    cfg = GammaDGPConfig(name="independent", **SMALL)
    X = np.array([[-1.0] * cfg.d, [0.0] * cfg.d, [1.0] * cfg.d])
    original = gamma_truth_at(X, cfg)
    no_loading = gamma_truth_at(X, replace(cfg, hidden_loading0=0.0, hidden_loading1=0.0))
    opposite = gamma_truth_at(X, replace(cfg, bias_direction=-1))
    shifted = gamma_truth_at(X, replace(cfg, obs_baseline_shift=2.0, obs_cate_shift=0.7))
    different_effect = gamma_truth_at(X, replace(cfg, cate_complexity="constant", cate_scale=3.0))
    for altered in (no_loading, opposite, shifted, different_effect):
        np.testing.assert_array_equal(altered["gamma_requested"], original["gamma_requested"])
        np.testing.assert_array_equal(altered["obs_propensity"], original["obs_propensity"])
    for arm in (0, 1):
        np.testing.assert_array_equal(no_loading[f"confounding_bias_{arm}"], 0.0)
        np.testing.assert_allclose(opposite[f"confounding_bias_{arm}"], -original[f"confounding_bias_{arm}"])
        np.testing.assert_array_equal(shifted[f"confounding_bias_{arm}"], original[f"confounding_bias_{arm}"])
    np.testing.assert_array_equal(shifted["tau_target"], original["tau_target"])
    np.testing.assert_array_equal(different_effect["tau_target"], 3.0)
    np.testing.assert_array_equal(different_effect["confounding_bias_1"], original["confounding_bias_1"])


@pytest.mark.parametrize("profile", ["constant", "logistic"])
def test_serializable_factory_reproducibility_and_training_boundary(profile):
    cfg = GammaDGPConfig(name="roles", gamma_profile=profile, gamma=4.0, gamma_min=1.2, n_target_reference=7, **SMALL)
    assert GammaDGPConfig(**asdict(cfg)) == cfg
    problem = generate_gamma_problem(cfg, 21)
    again = generate_gamma_problem(cfg, 21)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        first, second = getattr(problem, role), getattr(again, role)
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.Y, second.Y)
    assert problem.obs_train.known_propensity is None
    for role in ("rct_train", "rct_tune", "rct_test"):
        np.testing.assert_array_equal(getattr(problem, role).known_propensity, cfg.rct_propensity)
    view = problem.training_view()
    assert {field.name for field in fields(view)} == {"obs_train", "rct_train", "rct_tune", "target_reference"}
    assert not hasattr(view, "truth") and not hasattr(view, "rct_test")
    assert problem.metadata["gamma_global_upper"] == 4.0
    assert np.all(problem.truth.eval["gamma_requested"] <= 4.0)
    np.testing.assert_array_equal(problem.eval_tau, problem.truth.eval["tau_target"])
    bigger_train = generate_gamma_problem(replace(cfg, n_obs=29, n_rct_train=31), 21)
    bigger_test = generate_gamma_problem(replace(cfg, n_rct_test=29, n_eval=31, n_target_reference=13), 21)
    np.testing.assert_array_equal(problem.eval_X, bigger_train.eval_X)
    np.testing.assert_array_equal(problem.rct_test.Y, bigger_train.rct_test.Y)
    np.testing.assert_array_equal(problem.obs_train.Y, bigger_test.obs_train.Y)
    np.testing.assert_array_equal(problem.rct_tune.Y, bigger_test.rct_tune.Y)
    np.testing.assert_array_equal(problem.target_reference.X, bigger_test.target_reference.X[:7])


def test_gamma_one_zero_bias_and_nonconstant_known_trial_randomization():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    sample, truth = sample_msm_source(
        X, source="rct", role="rct_train", rng=np.random.default_rng(9),
        observed_propensity=0.4, gamma=lambda x: np.ones(len(x)),
        hidden_loading0=2.0, hidden_loading1=3.0,
        rct_propensity=lambda x: 0.2 + 0.1 * (x[:, 0] + 1.0), noise_std=0.0,
    )
    np.testing.assert_allclose(sample.known_propensity, [0.2, 0.3, 0.4, 0.5])
    for arm in (0, 1):
        np.testing.assert_array_equal(truth[f"confounding_bias_{arm}"], 0.0)
    assert not hasattr(sample, "U")


def test_logistic_profile_has_declared_global_bound_without_claiming_sample_maximum():
    cfg = GammaDGPConfig(name="profile", gamma_profile="logistic", gamma=7.0, gamma_min=1.5, gamma_slope=-2.0, **SMALL)
    X = np.zeros((3, cfg.d))
    X[:, 0] = [-1.0, 0.0, 1.0]
    expected = np.array([1.5 + 5.5 / (1.0 + math.exp(2.0 * x)) for x in X[:, 0]])
    np.testing.assert_allclose(cfg.resolve_gamma(X), expected)
    assert max(cfg.resolve_gamma(X)) < cfg.gamma
    np.testing.assert_allclose(gamma_truth_at(X, cfg)["gamma_local"], expected, rtol=1e-13)


def test_complete_factory_accepts_arbitrary_gamma_callback_without_global_bound_claim():
    cfg = GammaDGPConfig(name="callback", **SMALL)
    callback = lambda x: 1.0 + 3.0 * np.exp(-x[:, 0] ** 2)
    problem = generate_gamma_problem(cfg, 81, gamma=callback)
    np.testing.assert_allclose(problem.truth.eval["gamma_requested"], callback(problem.eval_X))
    assert problem.metadata["gamma_global_upper"] is None
    assert not problem.metadata["gamma_global_upper_is_bound"]
    assert problem.metadata["gamma_specification"]["kind"] == "callback"
    np.testing.assert_array_equal(problem.eval_tau, generate_gamma_problem(cfg, 81).eval_tau)
    np.testing.assert_array_equal(problem.rct_test.Y, generate_gamma_problem(cfg, 81).rct_test.Y)
    overridden = generate_gamma_problem(cfg, 81, gamma=3.0)
    np.testing.assert_array_equal(overridden.truth.eval["gamma_requested"], 3.0)
    assert overridden.metadata["gamma_global_upper"] == 3.0


@pytest.mark.parametrize("overrides", [
    {"gamma": 0.9}, {"gamma": math.inf}, {"gamma_min": 3.0},
    {"gamma_profile": "other"}, {"bias_direction": 0}, {"n_obs": True},
    {"d": 0}, {"propensity_floor": 0.0}, {"rct_propensity": 1.0},
    {"noise_std": -1.0}, {"cate_complexity": "other"},
])
def test_config_rejects_undefined_controls(overrides):
    with pytest.raises(ValueError):
        GammaDGPConfig(name="bad", **(SMALL | overrides))


def test_callback_api_rejects_bad_shapes_nonfinite_truth_and_gamma():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="one value per row"):
        msm_truth_at(X, observed_propensity=0.5, gamma=lambda x: np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        msm_truth_at(X, observed_propensity=0.5, gamma=2.0, tau=lambda x: np.full(len(x), math.inf))
    with pytest.raises(ValueError, match="at least 1"):
        msm_truth_at(X, observed_propensity=0.5, gamma=lambda x: np.full(len(x), 0.5))
    lower, upper = msm_propensity_bounds([0.25, 0.75], 2.0)
    np.testing.assert_allclose(lower, [1.0 / 7.0, 0.6])
    np.testing.assert_allclose(upper, [0.4, 6.0 / 7.0])
