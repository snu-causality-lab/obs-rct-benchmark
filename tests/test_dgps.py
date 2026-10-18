"""Public preset coverage and independent small analytic truth checks."""

from dataclasses import fields, replace
import math

import numpy as np
import pytest

from obs_rct_benchmark.data import SourceSample, TrainingView
from obs_rct_benchmark.paper_dgp import (
    PAPER_DGPS,
    PAPER_DGP_CARDS,
    _cornet_components,
    _mechanism,
    generate_paper_problem,
    make_paper_config,
    paper_truth_at,
)
from obs_rct_benchmark.structural_dgp import (
    MECHANISM_CARDS,
    STRUCTURAL_ADDENDUM_CARDS,
    STRUCTURAL_ADDENDUM_CELLS,
    STRUCTURAL_MECHANISMS,
    StructuralDGPConfig,
    generate_structural_problem,
    make_structural_addendum_config,
    make_structural_config,
    structural_axis_levels,
    structural_truth_at,
)


SMALL = dict(n_obs=16, n_rct_train=12, n_rct_tune=8, n_rct_test=8, n_eval=12)


def _structural_config(cell, **overrides):
    make = (
        make_structural_addendum_config
        if cell in STRUCTURAL_ADDENDUM_CELLS
        else make_structural_config
    )
    selector = {"cell": cell} if cell in STRUCTURAL_ADDENDUM_CELLS else {"mechanism": cell}
    return make(name=cell, **selector, **(SMALL | overrides))


def _assert_problem(problem, cfg):
    assert problem.truth is not None
    assert problem.eval_X.shape == (cfg.n_eval, cfg.d)
    assert problem.eval_tau.shape == (cfg.n_eval,)
    np.testing.assert_array_equal(problem.eval_tau, problem.truth.eval["tau_target"])
    view = problem.training_view()
    assert isinstance(view, TrainingView)
    assert {field.name for field in fields(view)} == {
        "obs_train", "rct_train", "rct_tune", "target_reference"
    }
    for forbidden in ("truth", "eval_tau", "rct_test", "metadata", "seed"):
        assert not hasattr(view, forbidden)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        sample = getattr(problem, role)
        assert sample.role == role
        expected_n = cfg.n_obs if role == "obs_train" else getattr(cfg, f"n_{role}")
        assert sample.X.shape == (expected_n, cfg.d)
        assert not sample.X.flags.writeable
        assert not sample.T.flags.writeable
        assert not sample.Y.flags.writeable
        assert np.isfinite(sample.Y).all()
    assert problem.obs_train.known_propensity is None
    for sample in (problem.rct_train, problem.rct_tune, problem.rct_test):
        assert sample.known_propensity.shape == (sample.n,)
    for truth in (problem.truth.eval, problem.truth.obs_train):
        for arm in (0, 1):
            np.testing.assert_allclose(
                truth[f"regression_discrepancy_{arm}"],
                truth[f"confounding_bias_{arm}"] + truth[f"source_shift_{arm}"],
                atol=1e-14,
            )
        np.testing.assert_allclose(truth["mu1_target"] - truth["mu0_target"], truth["tau_target"])
        assert all(not value.flags.writeable for value in truth.values())
    with pytest.raises(TypeError):
        problem.truth.eval["invented"] = np.zeros(cfg.n_eval)


@pytest.mark.parametrize("cell", STRUCTURAL_MECHANISMS + STRUCTURAL_ADDENDUM_CELLS)
def test_all_structural_presets_generate_reproducibly(cell):
    cfg = _structural_config(cell)
    first = generate_structural_problem(cfg, 17)
    second = generate_structural_problem(cfg, 17)
    _assert_problem(first, cfg)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        np.testing.assert_array_equal(getattr(first, role).Y, getattr(second, role).Y)
    np.testing.assert_array_equal(first.eval_X, second.eval_X)


@pytest.mark.parametrize("paper_id", PAPER_DGPS)
def test_all_paper_presets_generate_reproducibly(paper_id):
    cfg = make_paper_config(name=paper_id, paper_id=paper_id, **SMALL)
    first = generate_paper_problem(cfg, 23)
    second = generate_paper_problem(cfg, 23)
    _assert_problem(first, cfg)
    np.testing.assert_array_equal(first.obs_train.Y, second.obs_train.Y)
    np.testing.assert_array_equal(first.eval_X, second.eval_X)
    assert first.metadata["mechanism_sha256"] == second.metadata["mechanism_sha256"]


def test_public_catalogs_have_only_supported_presets():
    assert tuple(MECHANISM_CARDS) == tuple(f"M{i}" for i in range(12))
    assert tuple(STRUCTURAL_ADDENDUM_CARDS) == tuple(f"C{i:02d}" for i in range(1, 11))
    assert set(PAPER_DGPS) == set(PAPER_DGP_CARDS) == {
        "cornet_official_n_conf", "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension", "integrative_r_learner_kps_code",
        "integrative_r_learner_multivariate_paper",
        "elastic_integrative_gen_data_overlap", "power_likelihood_paper_text",
        "power_likelihood_supplement_code", "cvci_official_linear_bias",
    }
    with pytest.raises(ValueError, match="unknown structural mechanism"):
        StructuralDGPConfig(name="invalid", mechanism="unregistered")


@pytest.mark.parametrize("geometry", ["localized", "multi_region"])
@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_clean_fraction_endpoints_have_finite_outcomes(geometry, fraction):
    cfg = StructuralDGPConfig(
        name="endpoint", bias_geometry=geometry, clean_fraction=fraction,
        cate_alignment="opposed", **SMALL,
    )
    problem = generate_structural_problem(cfg, 71)
    assert np.isfinite(problem.eval_tau).all()
    assert np.isfinite(problem.obs_train.Y).all()
    if fraction == 1.0:
        np.testing.assert_array_equal(problem.eval_region_score, np.zeros(cfg.n_eval))
    else:
        assert np.all(problem.eval_region_score > 0.0)


def test_impossible_covariate_selection_fails_with_a_bounded_budget():
    cfg = make_structural_config(name="rare-selection", mechanism="M0",
                                 obs_selection_strength=-100.0, **SMALL)
    with pytest.raises(RuntimeError, match="proposal budget"):
        generate_structural_problem(cfg, 72)


def test_noise_override_does_not_change_m11_reliability_label():
    original = _structural_config("M11")
    changed = replace(original, noise_std=original.noise_std + 0.25)
    assert structural_axis_levels(original)["reliability_resolution"] == "x_resolvable"
    assert structural_axis_levels(changed)["reliability_resolution"] == "x_resolvable"


def test_binary_hidden_truth_matches_explicit_two_point_enumeration():
    cfg = StructuralDGPConfig(
        name="binary", d=3, bias_geometry="constant", bias_min=1.0,
        treatment_linear_scale=0.0, confounding_treatment=0.8,
        confounding_outcome_0=0.7, confounding_outcome_1=1.3,
        hidden_distribution="binary", **SMALL,
    )
    X = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 0.25]])
    truth = structural_truth_at(X, cfg)
    # Enumerate independent equally likely U=-1,+1 with Bayes' rule.
    p_minus, p_plus = 1.0 / (1.0 + math.exp(0.8)), 1.0 / (1.0 + math.exp(-0.8))
    marginal = 0.5 * (p_minus + p_plus)
    u_treated = (p_plus - p_minus) / (p_plus + p_minus)
    u_control = ((1.0 - p_plus) - (1.0 - p_minus)) / (2.0 - p_plus - p_minus)
    np.testing.assert_allclose(truth["obs_propensity"], marginal, atol=1e-14)
    np.testing.assert_allclose(truth["confounding_bias_0"], 0.7 * u_control, atol=1e-14)
    np.testing.assert_allclose(truth["confounding_bias_1"], 1.3 * u_treated, atol=1e-14)


def test_no_confounding_and_equal_source_shifts_leave_cate_unchanged():
    X = np.array([[-1.0, 0.0, 0.3, 0.0, 0.0], [1.0, 0.2, -0.4, 0.0, 0.0]])
    original = structural_truth_at(X, _structural_config("M0"))
    shifted = structural_truth_at(X, _structural_config("C04"))
    for arm in (0, 1):
        np.testing.assert_allclose(original[f"confounding_bias_{arm}"], 0.0, atol=1e-14)
    np.testing.assert_array_equal(shifted["source_shift_0"], shifted["source_shift_1"])
    np.testing.assert_array_equal(original["tau_target"], shifted["tau_target"])
    np.testing.assert_array_equal(shifted["source_cate_shift"], 0.0)


def test_matched_sign_and_noise_controls_preserve_declared_truth():
    X = np.array([[-1.0, 0.2, 0.3, 0.0, 0.0], [1.0, -0.2, 0.1, 0.0, 0.0]])
    signed = structural_truth_at(X, _structural_config("M6"))
    fixed = structural_truth_at(X, _structural_config("C01"))
    for arm in (0, 1):
        np.testing.assert_allclose(np.abs(signed[f"confounding_bias_{arm}"]), np.abs(fixed[f"confounding_bias_{arm}"]))
    heavy = structural_truth_at(X, _structural_config("M11"))
    normal = structural_truth_at(X, _structural_config("C03"))
    for key in heavy:
        np.testing.assert_array_equal(heavy[key], normal[key])


def test_participation_effect_modifier_changes_rct_cate_only():
    X = np.zeros((2, 5))
    prognostic = structural_truth_at(X, _structural_config("C07"))
    modifier = structural_truth_at(X, _structural_config("C08"))
    np.testing.assert_array_equal(prognostic["tau_target"], modifier["tau_target"])
    np.testing.assert_allclose(prognostic["rct_source_cate_shift"], 0.0)
    assert np.all(modifier["rct_source_cate_shift"] > 0.0)


def test_target_reference_is_independent_and_outcome_free():
    cfg = _structural_config("M0")
    plain = generate_structural_problem(cfg, 31)
    with_reference = generate_structural_problem(replace(cfg, n_target_reference=9), 31)
    longer_reference = generate_structural_problem(replace(cfg, n_target_reference=11), 31)
    np.testing.assert_array_equal(plain.obs_train.Y, with_reference.obs_train.Y)
    np.testing.assert_array_equal(plain.eval_X, with_reference.eval_X)
    np.testing.assert_array_equal(with_reference.target_reference.X, longer_reference.target_reference.X[:9])
    assert {field.name for field in fields(with_reference.target_reference)} == {"X", "role"}
    assert not with_reference.target_reference.X.flags.writeable


def test_kps_truth_and_normal_target_extension():
    X = np.array([[-2.0], [0.0], [2.0]])
    cfg = make_paper_config(name="kps", paper_id="experimental_grounding_kps", **SMALL)
    truth = paper_truth_at(X, cfg)
    np.testing.assert_allclose(truth["tau_target"], [0.0, 1.0, 4.0])
    np.testing.assert_allclose(truth["regression_discrepancy_0"], -0.25 - 0.5 * X[:, 0])
    np.testing.assert_allclose(truth["regression_discrepancy_1"], 0.25 + 0.5 * X[:, 0])
    extension = make_paper_config(name="extension", paper_id="integrative_r_learner_kps_normal_target_extension", **SMALL)
    code = make_paper_config(name="code", paper_id="integrative_r_learner_kps_code", **SMALL)
    pp = generate_paper_problem(extension, 53)
    cp = generate_paper_problem(code, 53)
    np.testing.assert_array_equal(pp.obs_train.X, cp.obs_train.X)
    assert np.all(np.abs(cp.eval_X) <= 1.0)
    assert not np.array_equal(pp.eval_X, cp.eval_X)


def test_power_likelihood_full_covariate_cate_has_copula_term():
    cfg = make_paper_config(name="power", paper_id="power_likelihood_paper_text", **SMALL)
    X = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0]])
    truth = paper_truth_at(X, cfg)
    # Corr(Z, Y(t)) = tanh((1 + 2.5*t)/2) in the paper-text variant.
    slope = math.tanh(3.5 / 2.0) - math.tanh(1.0 / 2.0)
    np.testing.assert_allclose(truth["tau_target"], [0.1, 0.1 + slope, 0.2])
    np.testing.assert_allclose(truth["paper_subgroup_tau_c"], [0.1, 0.1, 0.2])


def test_cornet_mechanism_has_declared_norm_sparsity_and_reference_rms():
    cfg = make_paper_config(name="cornet", paper_id="cornet_official_n_conf", d=4, **SMALL)
    mechanism = _mechanism(cfg)
    for arm in (0, 1):
        np.testing.assert_allclose(np.linalg.norm(mechanism[f"base_weight_{arm}"]), 10.0)
        assert np.count_nonzero(mechanism[f"correction_weight_{arm}"][1:] == 0.0) == 2
    ref_rng = np.random.default_rng(np.random.SeedSequence(cfg.mechanism_seed, spawn_key=(0x434F524E,)))
    X = ref_rng.standard_normal((10_000, cfg.d))
    _, _, c0, c1, _, _ = _cornet_components(X, mechanism)
    np.testing.assert_allclose(np.sqrt(np.mean((c1 - c0) ** 2)), 2.0, rtol=1e-14)
    truth = paper_truth_at(X[:3], cfg, mechanism)
    np.testing.assert_array_equal(truth["confounding_cate_bias"], 0.0)
    assert np.any(np.abs(truth["source_cate_shift"]) > 0.0)


def test_data_rejects_observational_propensity_and_copies_inputs():
    X = np.ones((4, 2))
    T = np.array([0, 1, 0, 1])
    Y = np.arange(4.0)
    with pytest.raises(ValueError, match="not exposed"):
        SourceSample(X, T, Y, "obs", "obs_train", known_propensity=np.full(4, 0.5))
    sample = SourceSample(X, T, Y, "obs", "obs_train")
    X[:] = -5.0
    np.testing.assert_array_equal(sample.X, 1.0)
    with pytest.raises(ValueError):
        sample.X[0, 0] = 7.0
