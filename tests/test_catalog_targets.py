"""Independent fixed-point examples and a falsified-target detection test."""

import importlib.util
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark.paper_dgp import make_paper_config
from obs_rct_benchmark.runner import load_config
from obs_rct_benchmark.structural_dgp import (
    make_structural_addendum_config, make_structural_config,
)


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "catalog_target_check", ROOT / "scripts/verify_catalog_targets.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def structural(name):
    if name.startswith("C"):
        return make_structural_addendum_config(name=name, cell=name)
    return make_structural_config(name=name, mechanism=name)


def test_structural_targets_match_fixed_value_examples():
    X = np.array([[0., -1., 0., 0., 0.], [2., 1., 0., 0., 0.]])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("M1")), [.5, 1.5])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("M8")), [.5, 2.75])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("M9")), [.5, 2.])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("M11")), [.5, 2.])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("C03")), [.5, 2.])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("C05")), [0., 0.])
    np.testing.assert_array_equal(checker.independent_structural_target(X, structural("C06")), [-1., 1.])
    # Every other canonical target is one at the origin, including both
    # hidden-participation cases whose trial-population effects differ.
    for name in ("M0", "M2", "M3", "M4", "M5", "M6", "M7", "M10",
                 "C01", "C02", "C04", "C07", "C08", "C09", "C10"):
        config = structural(name)
        assert checker.independent_structural_target(np.zeros((1, config.d)), config)[0] == 1.
    point = np.array([[0., math.atanh(.5), math.pi / 2, 0., 0.]])
    assert checker.independent_structural_target(point, structural("M2"))[0] == pytest.approx(1.5)


def test_paper_targets_match_independent_fixed_points():
    X = np.array([[-2.], [0.], [2.]])
    cfg = make_paper_config(name="kps", paper_id="experimental_grounding_kps")
    np.testing.assert_array_equal(checker.independent_paper_target(X, cfg), [0., 1., 4.])
    for name in ("integrative_r_learner_kps_code", "integrative_r_learner_kps_normal_target_extension"):
        cfg = make_paper_config(name=name, paper_id=name)
        np.testing.assert_array_equal(checker.independent_paper_target(X, cfg), [0., 1., 8.])
    cfg = make_paper_config(name="intR", paper_id="integrative_r_learner_multivariate_paper")
    np.testing.assert_array_equal(checker.independent_paper_target(np.ones((1, 9)), cfg), [10.])
    cfg = make_paper_config(name="elastic", paper_id="elastic_integrative_gen_data_overlap")
    np.testing.assert_array_equal(checker.independent_paper_target(np.array([[1., 2.], [0., -1.]]), cfg), [3., -1.])
    cfg = make_paper_config(name="cvci", paper_id="cvci_official_linear_bias")
    np.testing.assert_array_equal(checker.independent_paper_target(np.ones((2, 5)), cfg), [.5, .5])


def test_power_target_uses_full_displayed_covariates():
    cfg = make_paper_config(name="power", paper_id="power_likelihood_paper_text")
    X = np.array([[0., 1.], [1., 2.], [0., 2.]])
    slope = math.tanh(1.75) - math.tanh(.5)
    np.testing.assert_allclose(checker.independent_paper_target(X, cfg), [.1, .2, .1 + slope], atol=1e-14)


def test_independent_neural_forward_matches_scalar_hand_calculation():
    parameters = {f"rep_weight_{i}": np.ones((1, 1)) for i in range(3)}
    parameters.update({f"rep_bias_{i}": np.zeros(1) for i in range(3)})
    for kind in ("base", "correction"):
        for arm in (0, 1):
            parameters[f"{kind}_weight_{arm}"] = np.array([float(arm)])
            parameters[f"{kind}_bias_{arm}"] = np.zeros(1)
    X = np.array([[0.], [1.]])
    np.testing.assert_allclose(checker.independent_cornet_target(X, parameters),
                               [0., 2 * math.tanh(math.tanh(1.))], atol=1e-14)


def test_catalog_selection_keeps_native_paper_reference_setting():
    config = load_config(ROOT / "configs/catalog_study.toml")
    selected = checker._select_cases(config)
    assert len(selected) == 31
    assert sum(s.kind == "paper" for s in selected) == 9
    paper = next(s for s in config.scenarios if s.kind == "paper")
    assert "n_target_reference" not in paper.parameters
    changed = replace(paper, parameters={**paper.parameters, "noise_std": 1.0})
    invalid = replace(config, scenarios=tuple(changed if s is paper else s
                                               for s in config.scenarios))
    with pytest.raises(ValueError, match="declared role budgets"):
        checker._select_cases(invalid)


def test_modified_evaluator_target_is_rejected_before_other_cases(monkeypatch):
    config = load_config(ROOT / "configs/catalog_study.toml")
    calls = []
    def changed_target(scenario, seed):
        calls.append(scenario.key)
        return SimpleNamespace(eval_X=np.zeros((2, 5)), eval_tau=np.full(2, 1.0001))
    monkeypatch.setattr(checker, "_generate", changed_target)
    with pytest.raises(ValueError, match="independent target differs.*M0"):
        checker.verify_targets(config)
    assert calls == ["structural:M0@o512_r128"]
