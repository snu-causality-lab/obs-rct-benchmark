"""Scientific target, information and frozen-grid checks; no estimator fits.

Run this cohort on the server with the panel's numerical environment. Generator
calls retain native sizes and mechanisms; these are not miniature performance
experiments or source-method admission tests.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panel():
    spec = importlib.util.spec_from_file_location("native_panel_data_test", ROOT / "scripts/native_panel_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(panel):
    return panel.load_config(ROOT / "configs/native_panel.toml")


def _case(panel, config, scenario_key, *, profile=False, budget=128):
    collection = panel.profile_cases(config) if profile else panel.cases(config)
    return next(case for case in collection if case["scenario_key"] == scenario_key and case["n_rct"] == budget)


def test_campaign_and_preflight_have_disjoint_complete_stable_ids(panel, config):
    planned = panel.cases(config)
    profiles = panel.profile_cases(config)
    assert len(planned) == 6 * 2 * 64
    assert len({row["id"] for row in planned}) == len(planned)
    assert len(planned) * len(config["methods"]) == 8448
    assert len(profiles) == 12
    assert {row["seed"] for row in profiles} == {49999}
    assert {row["id"] for row in planned}.isdisjoint({row["id"] for row in profiles})
    assert panel.digest(config) == panel.digest(deepcopy(config))


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("n_obs", 2048), ("n_eval", 1024),
    ("rct_budgets", [512, 128]), ("seeds", [50000]),
    ("preflight_seed", 50000), ("factory_rct_tune", 2),
    ("factory_rct_test", 0), ("tune_role", "independent_validation"),
    ("analysis", "mean_risk_t_test"),
])
def test_scientific_grid_cannot_change_under_same_contract(panel, config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        panel.validate_config(config)


def test_target_metadata_and_recipe_options_are_frozen(panel, config):
    changed = deepcopy(config)
    changed["conditions"][3]["scenario_key"] = "addendum:C08"
    with pytest.raises(ValueError, match="conditions"):
        panel.validate_config(changed)
    changed = deepcopy(config)
    changed["method_options"]["cornet_official"]["source_epochs"] = 10
    with pytest.raises(ValueError, match="method_options"):
        panel.validate_config(changed)
    changed = deepcopy(config)
    changed["method_options"]["datafusionppi"]["learner"] = "R"
    with pytest.raises(ValueError, match="method_options"):
        panel.validate_config(changed)


def test_reviewed_resources_require_evidence_without_claiming_success(panel, config):
    changed = deepcopy(config)
    changed["resources"]["status"] = "provisional_pending_preflight"
    changed["resources"]["preflight_manifest_sha256"] = ""
    assert panel.validate_config(changed) == changed
    changed["resources"]["status"] = "preflight_reviewed"
    with pytest.raises(ValueError, match="requires a preflight"):
        panel.validate_config(changed)
    changed["resources"]["preflight_manifest_sha256"] = "a" * 64
    assert panel.validate_config(changed) == changed
    assert changed["method_options"]["integrative_r_quadratic"]["timeout_seconds"] == 1200
    obsolete = deepcopy(changed)
    obsolete["resources"]["status"] = "preflight_verified"
    with pytest.raises(ValueError, match="readiness"):
        panel.validate_config(obsolete)
    changed["resources"]["method_fit_timeout_seconds"].pop("cornet_official")
    with pytest.raises(ValueError, match="every recipe"):
        panel.validate_config(changed)


def test_resource_validation_rejects_boolean_nan_and_unknown_fields(panel, config):
    for key, value in [("memory_gib", float("nan")), ("num_cpus", True),
                       ("preflight_manifest_sha256", "not-a-hash"), ("posthoc_retry", True)]:
        changed = deepcopy(config)
        changed["resources"][key] = value
        with pytest.raises(ValueError):
            panel.validate_config(changed)


def test_preflight_seed_and_case_identity_cannot_enter_campaign(panel, config):
    case = panel.profile_cases(config)[0]
    with pytest.raises(ValueError, match="not in"):
        panel.factory_config(config, case)
    bad = {**panel.cases(config)[0], "id": "changed"}
    with pytest.raises(ValueError, match="not in"):
        panel.factory_config(config, bad)
    assert panel.factory_config(config, case, profile=True).d == 5


def test_native_dimensions_and_mechanism_seeds_are_preserved(panel, config):
    for condition in config["conditions"]:
        case = _case(panel, config, condition["scenario_key"])
        cfg = panel.factory_config(config, case)
        assert cfg.d == condition["native_dimension"]
        assert getattr(cfg, "mechanism_seed", getattr(cfg, "signal_seed", None)) == 2025
        assert cfg.n_rct_tune == cfg.n_rct_test == 4
        assert getattr(cfg, "n_target_reference", 0) == 0


def test_hidden_participation_scores_trial_effect_instead_of_canonical_effect(panel, config):
    case = _case(panel, config, "addendum:C08@trial_target")
    arrays = panel.generate_case(config, case)
    x = arrays["eval_X"]
    canonical = 1 + .5 * np.tanh(x[:, 1]) + .25 * np.sin(x[:, 2])
    shift = arrays["eval_tau"] - canonical
    assert np.min(shift) > 0
    np.testing.assert_allclose(shift, np.full(len(x), shift[0]), atol=1e-12, rtol=0)
    metadata = panel.case_metadata(config, case)
    assert metadata["target_mapping"] == "tau_rct_new_trial_target"
    assert set(arrays) == set(panel.OBSERVED_ARRAYS) | {"eval_tau"}


def test_c08_panel_quadrature_meets_fixed_independent_accuracy_and_preserves_inputs(panel, config):
    # The independent verifier derives its moment with adaptive integration and
    # does not import producer truth helpers or this Hermite calculation.
    spec = importlib.util.spec_from_file_location("native_panel_c08_accuracy_check", ROOT / "scripts/verify_native_panel.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert panel.c08_selected_mean() == pytest.approx(checker.c08_selected_mean(), rel=0, abs=1e-13)
    case = _case(panel, config, "addendum:C08@trial_target")
    cfg = panel.factory_config(config, case)
    from obs_rct_benchmark.structural_dgp import generate_structural_problem
    native = generate_structural_problem(cfg, case["seed"])
    produced = panel.generate_case(config, case)
    for prefix, sample in (("obs", native.obs_train), ("rct", native.rct_train)):
        for field in ("X", "T", "Y"):
            np.testing.assert_array_equal(produced[prefix + "_" + field], getattr(sample, field))
    np.testing.assert_array_equal(produced["eval_X"], native.eval_X)
    assert cfg.quadrature_order == 21  # native sampling/truth configuration retained
    result = checker.verify_case(produced, case["scenario_key"])
    assert result["passed"], result
    assert result["checks"]["target"]["max_absolute_error"] <= 1e-12
    precision = panel.case_metadata(config, case)["truth_precision"]
    assert precision == {"target_atol": 1e-12, "target_rtol": 0.0,
                         "selected_mean_quadrature_order": 256,
                         "native_observed_factory_unchanged": True}
    changed = deepcopy(config)
    changed["truth_precision"]["target_atol"] = 1e-9
    with pytest.raises(ValueError, match="truth_precision"):
        panel.validate_config(changed)


def test_native_kps_uses_uniform_trial_queries_and_quadratic_effect(panel, config):
    case = _case(panel, config, "paper:integrative_r_learner_kps_code")
    arrays = panel.generate_case(config, case)
    x = arrays["eval_X"][:, 0]
    assert np.max(np.abs(x)) <= 1
    assert arrays["eval_X"].shape == (2048, 1)
    np.testing.assert_allclose(arrays["eval_tau"], 1 + 2*x + .75*x*x, atol=1e-12, rtol=0)
    assert np.max(np.abs(arrays["obs_X"])) > 1


def test_method_views_exclude_truth_test_and_discarded_labels(panel, config):
    case = _case(panel, config, "structural:M0")
    arrays = panel.generate_case(config, case)
    observed = {key: arrays[key] for key in panel.OBSERVED_ARRAYS}
    view = panel.make_training_view(observed)
    np.testing.assert_array_equal(view.rct_tune.Y, view.rct_train.Y[:2])
    np.testing.assert_array_equal(view.rct_tune.X, view.rct_train.X[:2])
    assert not hasattr(view, "eval_tau") and not hasattr(view, "rct_test")
    assert view.target_reference is None
    assert not view.rct_tune.Y.flags.writeable
    sample = panel.make_source_sample(observed, "rct")
    assert sample.n == 128 and sample.source == "rct"
    metadata = panel.case_metadata(config, case)
    assert metadata["generated_unique_outcomes"] == {"obs": 512, "rct": 136}
    assert metadata["method_view_unique_outcomes"] == {"obs": 512, "rct": 128}
    assert metadata["discarded_factory_outcomes"] == {"rct_tune": 4, "rct_test": 4}
    assert metadata["compatibility_tuning_unique_outcomes"] == 0
    assert metadata["query_outcomes_generated"] == 0


def test_gp_control_uses_matched_native_random_stream(panel, config):
    case = panel.cases(config)[0]
    assert panel.case_method_seed(case, "causal_icm_fixed") == panel.case_method_seed(case, "causal_icm_rct_only")
    changed_budget = {**case, "n_rct": 512}
    assert panel.case_method_seed(case, "cornet_official") == panel.case_method_seed(changed_budget, "cornet_official")
    with pytest.raises(ValueError, match="method"):
        panel.case_method_seed(case, "not_a_recipe")


def test_hash_records_shapes_and_bytes_including_array_changes(panel):
    arrays = {"x": np.arange(6., dtype=np.float64).reshape(2, 3)}
    identity = panel.array_identity(arrays)
    assert identity["x"]["shape"] == [2, 3]
    assert identity["x"]["dtype"] == "float64"
    changed = {"x": arrays["x"].copy()}
    changed["x"][0, 0] = 1
    assert identity != panel.array_identity(changed)
