"""Independent target/loss checks and deliberately falsified retained evidence.

No estimator is fitted. Native CorNet setup and all numerical tests run only
inside the allocated-server correctness preflight.
"""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return load_script("verify_native_panel")


@pytest.fixture(scope="module")
def panel():
    return load_script("native_panel_data")


@pytest.fixture(scope="module")
def config(panel):
    return panel.load_config(ROOT / "configs/native_panel.toml")


@pytest.mark.parametrize("scenario_key", (
    "structural:M0", "structural:M2", "structural:M10", "addendum:C08@trial_target",
    "paper:integrative_r_learner_kps_code", "paper:cornet_official_n_conf",
))
def test_all_native_panel_producer_targets_have_independent_equations(checker, panel, config, scenario_key):
    case = next(case for case in panel.profile_cases(config)
                if case["scenario_key"] == scenario_key and case["n_rct"] == 128)
    arrays = panel.generate_case(config, case, profile=True)
    result = checker.verify_case(arrays, scenario_key)
    assert result["passed"], result
    assert result["checks"]["target"]["max_absolute_error"] <= 1e-12
    assert result["checks"]["query"]["moment_check_performed"]


def test_smooth_structural_effect_at_hand_computed_points(checker):
    X = np.array([[0., 0., 0., 0., 0.], [0., math.atanh(.5), math.pi / 2, 0., 0.]])
    for key in ("structural:M0", "structural:M2", "structural:M10"):
        np.testing.assert_allclose(checker.independent_truth(X, key), [1., 1.5], rtol=0, atol=1e-15)


def test_c08_integral_agrees_with_independent_stein_identity(checker):
    from scipy.integrate import quad

    # Stein's identity: E[Z sigmoid(2Z)] / .5 = 4 E[sigmoid(2Z)(1-sigmoid(2Z))].
    # Symmetry reduces its derivative integral to the nonnegative half-line.
    def derivative_integrand(z):
        decay = math.exp(-2 * z)
        return decay / (1 + decay) ** 2 * math.exp(-z*z/2) / math.sqrt(2*math.pi)

    stein_mean = 8 * quad(derivative_integrand, 0., np.inf, epsabs=2e-14, epsrel=2e-14)[0]
    assert checker.c08_selected_mean() == pytest.approx(stein_mean, rel=0, abs=1e-13)
    assert .60 < stein_mean < .61


def test_c08_canonical_eval_tau_is_rejected(checker):
    key = "addendum:C08@trial_target"
    X = np.zeros((4, 5))
    canonical = np.ones(4)
    rejected = checker.verify_case({"eval_X": X, "eval_tau": canonical}, key)
    assert not rejected["passed"]
    assert "trial target differs" in rejected["errors"][0]
    correct = canonical + 2 * checker.c08_selected_mean()
    assert checker.verify_case({"eval_X": X, "eval_tau": correct}, key)["passed"]


def test_native_kps_values_and_exact_query_support(checker):
    key = "paper:integrative_r_learner_kps_code"
    X = np.array([[-1.], [0.], [1.]])
    tau = np.array([-.25, 1., 3.75])
    np.testing.assert_array_equal(checker.independent_truth(X, key), tau)
    assert checker.verify_case({"eval_X": X, "eval_tau": tau}, key)["passed"]
    outside = np.array([[-1.0001], [0.], [1.]])
    result = checker.verify_case({"eval_X": outside,
                                  "eval_tau": checker.independent_truth(outside, key)}, key)
    assert not result["passed"]
    assert "support" in result["errors"][0]


def test_uniform_law_substituted_for_native_normal_fails_empirical_sanity(checker):
    X = np.tile(np.linspace(-1., 1., 2048)[:, None], (1, 5))
    result = checker.verify_case({"eval_X": X,
                                  "eval_tau": checker.independent_truth(X, "structural:M0")}, "structural:M0")
    assert not result["passed"]
    assert "empirical moments" in result["errors"][0]


def test_wrong_seed_and_permuted_cornet_network_are_rejected(checker):
    from obs_rct_benchmark.paper_dgp import _mechanism, make_paper_config, paper_truth_at

    key = "paper:cornet_official_n_conf"
    X = np.random.default_rng(15).standard_normal((8, 25))
    cfg = make_paper_config(name="native verifier counterexample", paper_id="cornet_official_n_conf")
    wrong_seed_tau = paper_truth_at(X, replace(cfg, mechanism_seed=2026))["tau_target"]
    parameters = _mechanism(cfg)
    parameters["rep_weight_0"] = parameters["rep_weight_0"][:, ::-1].copy()
    wrong_network_tau = paper_truth_at(X, cfg, parameters)["tau_target"]
    for tau in (wrong_seed_tau, wrong_network_tau):
        result = checker.verify_case({"eval_X": X, "eval_tau": tau}, key)
        assert not result["passed"]
        assert "trial target differs" in result["errors"][0]


def test_independent_cornet_forward_works_when_producer_helpers_are_disabled(checker, monkeypatch):
    import obs_rct_benchmark.paper_dgp as producer

    def forbidden(*args, **kwargs):
        raise AssertionError("independent checker called a producer helper")

    monkeypatch.setattr(producer, "_mechanism", forbidden)
    monkeypatch.setattr(producer, "_cornet_components", forbidden)
    monkeypatch.setattr(producer, "paper_truth_at", forbidden)
    checker._cornet_parameters.cache_clear()
    truth = checker.independent_truth(np.zeros((2, 25)), "paper:cornet_official_n_conf")
    assert truth.shape == (2,)
    assert np.isfinite(truth).all()


def test_score_has_hand_calculated_mse_and_root(checker):
    result = checker.score(np.array([2., 3., 4.]), np.ones(3))
    assert result["cate_mse"] == 14 / 3
    assert result["root_pehe"] == math.sqrt(14 / 3)
    # fsum preserves the many small terms next to one large squared error.
    prediction = np.array([1e8] + [1.] * 8)
    assert checker.score(prediction, np.zeros(9))["cate_mse"] == (10**16 + 8) / 9


def test_retained_prediction_perturbation_invalidates_claimed_score(checker):
    X = np.zeros((4, 5))
    arrays = {"eval_X": X, "eval_tau": np.ones(4), "prediction": np.ones(4),
              "cate_mse": 0., "root_pehe": 0.}
    assert checker.verify_case(arrays, "structural:M0")["passed"]
    arrays["prediction"] = np.array([1., 1., 1., 1.1])
    result = checker.verify_case(arrays, "structural:M0")
    assert not result["passed"]
    assert "cate_mse differs" in result["errors"][0]
    assert result["checks"]["score"]["cate_mse"] == pytest.approx(.0025)


def test_large_finite_loss_accepts_dot_reduction_and_rejects_wrong_relative_score(checker):
    # This synthetic vector exercises arithmetic tolerance without a model fit.
    X = np.zeros((8, 5))
    truth = np.ones(8)
    prediction = np.array([1e12, -1e12, 3e12, -3e12, 7e11, -7e11, 2e11, -2e11])
    error = prediction - truth
    dot_mse = float(np.dot(error, error) / len(error))
    arrays = {"eval_X": X, "eval_tau": truth, "prediction": prediction,
              "cate_mse": dot_mse, "root_pehe": math.sqrt(dot_mse)}
    assert checker.verify_case(arrays, "structural:M0")["passed"]

    independent = checker.score(prediction, truth)
    # A fixed sub-tolerance perturbation makes the absolute-only regression
    # deterministic even on BLAS builds whose dot reduction happens to agree.
    arrays["cate_mse"] = independent["cate_mse"] * (1 + .5 * checker.SCORE_RELATIVE_TOLERANCE)
    assert abs(arrays["cate_mse"] - independent["cate_mse"]) > checker.SCORE_ABSOLUTE_TOLERANCE
    assert checker.verify_case(arrays, "structural:M0")["passed"]

    arrays["cate_mse"] = independent["cate_mse"] * (1 + 4 * checker.SCORE_RELATIVE_TOLERANCE)
    result = checker.verify_case(arrays, "structural:M0")
    assert not result["passed"]
    assert "cate_mse differs" in result["errors"][0]


def test_large_finite_loss_tolerance_does_not_relax_analytic_target(checker):
    X = np.zeros((4, 5))
    arrays = {"eval_X": X, "eval_tau": np.ones(4) + 1e-10,
              "prediction": np.full(4, 1e12)}
    result = checker.verify_case(arrays, "structural:M0")
    assert not result["passed"]
    assert "trial target differs" in result["errors"][0]
    assert result["checks"]["target"]["absolute_tolerance"] == 1e-12


@pytest.mark.parametrize("prediction,truth", [([], []), ([[1.]], [1.]), ([1.], [1., 2.]),
                                               ([np.nan], [1.]), ([1.], [np.inf])])
def test_score_rejects_invalid_vectors(checker, prediction, truth):
    with pytest.raises(ValueError, match="matching nonempty finite vectors"):
        checker.score(prediction, truth)


def test_retained_manifest_combines_separate_evaluator_archive(checker, tmp_path, monkeypatch):
    np.savez(tmp_path / "observed.npz", eval_X=np.zeros((4, 5)))
    np.savez(tmp_path / "evaluation.npz", eval_tau=np.ones(4))
    np.save(tmp_path / "prediction.npy", np.full(4, 2.))
    (tmp_path / "record.json").write_text(json.dumps({"status": "success",
        "metrics": {"cate_mse": 1., "root_pehe": 1.}}))
    manifest = {"cases": [{"scenario_key": "structural:M0", "arrays": "observed.npz",
                            "evaluation": "evaluation.npz", "prediction": "prediction.npy",
                            "record": "record.json"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["verify_native_panel.py", "--manifest", str(tmp_path / "manifest.json"),
                                      "--output", str(tmp_path / "verification.json")])
    checker.main()
    result = json.loads((tmp_path / "verification.json").read_text())
    assert result["passed"]
    assert result["cases"][0]["checks"]["score"] == {"cate_mse": 1., "root_pehe": 1.}
    assert set(result["cases"][0]["input_sha256"]) == {"arrays", "evaluation", "prediction", "record"}
