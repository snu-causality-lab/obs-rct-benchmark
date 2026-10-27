"""Hand-calculated paired means and independent inference declarations.

Finite recorded values establish neither metric moments nor interval eligibility.
These fixtures separate coverage, descriptive arithmetic, and declared assumptions.
No generator or estimator fitting is needed for the runner integration check.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark import runner
from obs_rct_benchmark.evaluation import summarize_records
from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, load_config


METHODS = ("rct_t_ridge", "obs_t_ridge")
TINY_BUDGET = {
    "n_obs": 24, "n_rct_train": 20, "n_rct_tune": 10,
    "n_rct_test": 10, "n_eval": 32,
}


def declaration(mse=True, root=True):
    return {
        "cate_mse": {"eligible": mse, "reason": "Declared MSE moment assumption"},
        "root_pehe": {"eligible": root, "reason": "Declared root-PEHE moment assumption"},
    }


def record(scenario, seed, method, mse):
    return {
        "scenario": scenario, "seed": seed, "method": method,
        "status": "ok", "metrics": {"cate_mse": mse, "root_pehe": math.sqrt(mse)},
    }


def paired_records(scenario, differences=(2, 4)):
    rows = []
    for seed, difference in zip((3, 5), differences):
        rows.extend((record(scenario, seed, METHODS[0], 1),
                     record(scenario, seed, METHODS[1], 1 + difference)))
    return rows


def summarize(rows, scenarios, policy=None):
    return summarize_records(
        rows, scenarios=scenarios, seeds=(3, 5), methods=METHODS,
        reference_method=METHODS[0], scenario_inference=policy,
    )


def comparison(summary, scenario=None):
    result = summary["paired_comparisons"][METHODS[1]]
    return result["overall"] if scenario is None else result["by_scenario"][scenario]


def assert_withheld(metric, reason):
    assert metric["eligible_for_inference"] is False
    assert metric["standard_error"] is None
    assert metric["ci95"] is None
    assert metric["ci_method"] is None
    assert reason in metric["withheld_reason"]


def test_declared_gaussian_pair_has_hand_calculated_seed_interval():
    result = summarize(paired_records("gaussian"), ["gaussian"],
                       {"gaussian": declaration()})
    pair = comparison(result)
    metric = pair["metrics"]["cate_mse"]
    # Differences 2 and 4: mean 3, sample variance 2, SE sqrt(2/2)=1.
    assert pair["complete_coverage"] is True
    assert pair["eligible_for_comparison"] is True
    assert pair["withheld_reason"] is None
    assert metric["n_seed_blocks"] == 2
    assert metric["mean"] == 3
    assert metric["standard_error"] == 1
    assert metric["ci95"] == pytest.approx([3 - 12.7062047364, 3 + 12.7062047364])
    assert metric["ci_method"] == "student_t_conservative_table"
    assert metric["eligible_for_inference"] is True
    assert metric["withheld_reason"] is None


def test_complete_m11_keeps_mse_mean_without_uncertainty():
    policy = declaration(mse=False, root=False)
    policy["cate_mse"]["reason"] = "Student t3 noise: MSE moment condition not established"
    result = summarize(paired_records("structural:M11"), ["structural:M11"],
                       {"structural:M11": policy})
    pair = comparison(result)
    assert result["status"] == "complete"
    assert result["coverage"]["successful"] == 4
    assert pair["complete_coverage"] is True
    assert pair["eligible_for_comparison"] is False
    assert pair["metrics"]["cate_mse"]["mean"] == 3
    assert pair["metrics"]["cate_mse"]["n_seed_blocks"] == 2
    assert_withheld(pair["metrics"]["cate_mse"], "Student t3")
    assert "Student t3" in pair["withheld_reason"]
    assert result["scenarios"]["structural:M11"][METHODS[1]]["mean_cate_mse"] == 4


def test_mixed_heavy_tail_blocks_overall_mse_but_preserves_gaussian_interval():
    rows = paired_records("gaussian") + paired_records("structural:M11", (6, 8))
    heavy = declaration(mse=False, root=False)
    heavy["cate_mse"]["reason"] = "Student t3 noise"
    result = summarize(rows, ["gaussian", "structural:M11"],
                       {"gaussian": declaration(), "structural:M11": heavy})
    overall = comparison(result)
    assert overall["complete_coverage"] is True
    # Within-seed averages are (2+6)/2=4 and (4+8)/2=6.
    assert overall["metrics"]["cate_mse"]["mean"] == 5
    assert overall["metrics"]["cate_mse"]["n_seed_blocks"] == 2
    assert_withheld(overall["metrics"]["cate_mse"], "Student t3")
    gaussian = comparison(result, "gaussian")["metrics"]["cate_mse"]
    assert gaussian["mean"] == 3
    assert gaussian["standard_error"] == 1
    assert gaussian["ci95"] == pytest.approx([3 - 12.7062047364, 3 + 12.7062047364])


def test_root_permission_is_independent_of_mse_permission_and_aliases():
    result = summarize(paired_records("s"), ["s"], {"s": declaration(mse=False)})
    pair = comparison(result)
    assert pair["eligible_for_comparison"] is False
    assert_withheld(pair["metrics"]["cate_mse"], "Declared MSE")
    root = pair["metrics"]["root_pehe"]
    # Root differences are sqrt(3)-1 and sqrt(5)-1, not sqrt(2), sqrt(4).
    assert root["mean"] == pytest.approx((math.sqrt(3) + math.sqrt(5)) / 2 - 1)
    assert root["standard_error"] == pytest.approx((math.sqrt(5) - math.sqrt(3)) / 2)
    assert root["eligible_for_inference"] is True
    assert root["ci95"] is not None
    assert root["withheld_reason"] is None


def test_root_withheld_or_undeclared_does_not_withhold_declared_mse():
    for policy in (declaration(root=False), {"cate_mse": declaration()["cate_mse"]}):
        pair = comparison(summarize(paired_records("s"), ["s"], {"s": policy}))
        assert pair["complete_coverage"] is True
        assert pair["eligible_for_comparison"] is True
        assert pair["withheld_reason"] is None
        mse = pair["metrics"]["cate_mse"]
        assert mse["eligible_for_inference"] is True
        assert mse["mean"] == 3
        assert mse["ci95"] == pytest.approx([3 - 12.7062047364, 3 + 12.7062047364])
        root = pair["metrics"]["root_pehe"]
        assert root["mean"] == pytest.approx((math.sqrt(3) + math.sqrt(5)) / 2 - 1)
        reason = "Declared root-PEHE" if "root_pehe" in policy else "not declared"
        assert_withheld(root, reason)


@pytest.mark.parametrize("policy", [None, {}, {"other": declaration()},
                                    {"s": {"root_pehe": declaration()["root_pehe"]}}])
def test_absent_or_unknown_mse_permission_keeps_mean_but_fails_closed(policy):
    pair = comparison(summarize(paired_records("s"), ["s"], policy))
    assert pair["complete_coverage"] is True
    assert pair["eligible_for_comparison"] is False
    assert pair["metrics"]["cate_mse"]["mean"] == 3
    assert_withheld(pair["metrics"]["cate_mse"], "not declared")


@pytest.mark.parametrize("failed", [False, True], ids=["missing", "failed"])
def test_missing_or_failed_pair_cannot_restore_permission(failed):
    rows = paired_records("s")
    last = rows.pop()
    if failed:
        rows.append(dict(last, status="failed", metrics=None))
    for declared_mse in (True, False):
        pair = comparison(summarize(rows, ["s"], {"s": declaration(mse=declared_mse)}))
        assert pair["complete_coverage"] is False
        assert pair["complete_seed_blocks"] == 1
        assert pair["eligible_for_comparison"] is False
        assert pair["withheld_reason"] == "failed or missing requested cells"
        for metric in pair["metrics"].values():
            assert metric["mean"] is None
            assert_withheld(metric, "failed or missing requested cells")


@pytest.mark.parametrize("kind,identifier", [
    ("structural", "M0"), ("addendum", "C03"),
    ("paper", "experimental_grounding_kps"),
])
def test_canonical_presets_with_only_budget_and_label_changes_use_declared_default(kind, identifier):
    scenario = ScenarioConfig(kind, identifier, TINY_BUDGET, label="budget20")
    policy = runner.resolve_inference_policy(scenario, METHODS)
    for metric in ("cate_mse", "root_pehe"):
        assert policy[metric]["eligible"] is True
        assert policy[metric]["source"] == "builtin_declared_policy"
        assert policy[metric]["reason"]


@pytest.mark.parametrize("method", [
    "rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
    "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
    "rct_dr_ridge", "rct_dr_extra_trees",
])
def test_each_generic_default_method_is_inside_builtin_policy(method):
    policy = runner.resolve_inference_policy(ScenarioConfig("structural", "M0"), (method,))
    assert policy["cate_mse"]["eligible"] is True
    assert policy["root_pehe"]["eligible"] is True


@pytest.mark.parametrize("kind,identifier,parameters", [
    ("structural", "M0", {"noise_distribution": "student_t"}),
    ("structural", "M11", {}),
    ("addendum", "C03", {"noise_distribution": "student_t"}),
])
def test_resolved_student_t3_blocks_mse_even_when_user_declares_it_eligible(kind, identifier, parameters):
    scenario = ScenarioConfig(kind, identifier, parameters, inference_policy=declaration())
    assert runner.build_dgp_config(scenario).noise_distribution == "student_t"
    policy = runner.resolve_inference_policy(scenario, METHODS)
    assert policy["cate_mse"]["eligible"] is False
    assert policy["cate_mse"]["source"] == "resolved_noise"
    assert "t3" in policy["cate_mse"]["reason"].lower() or "t(3)" in policy["cate_mse"]["reason"].lower()
    # Root-PEHE has its own explicit assumption; the MSE restriction is not copied.
    assert policy["root_pehe"]["eligible"] is True
    assert policy["root_pehe"]["source"] == "scenario_declaration"


def test_undeclared_student_t3_root_permission_is_not_inferred_from_squaring():
    policy = runner.resolve_inference_policy(ScenarioConfig("structural", "M11"), METHODS)
    assert policy["cate_mse"]["eligible"] is False
    assert policy["cate_mse"]["source"] == "resolved_noise"
    assert policy["root_pehe"]["eligible"] is False
    assert "not established" in policy["root_pehe"]["reason"]


def test_m11_overridden_to_normal_follows_resolved_noise_and_explicit_assumptions():
    scenario = ScenarioConfig("structural", "M11", {"noise_distribution": "normal"},
                              inference_policy=declaration())
    assert runner.build_dgp_config(scenario).noise_distribution == "normal"
    policy = runner.resolve_inference_policy(scenario, METHODS)
    for metric in ("cate_mse", "root_pehe"):
        assert policy[metric]["eligible"] is True
        assert policy[metric]["source"] == "scenario_declaration"


@pytest.mark.parametrize("parameters", [
    {"noise_std": 2.0}, {"noise_distribution": "normal"},
    {"cate_scale": 2.0}, {"signal_seed": 7},
])
def test_scientific_overrides_do_not_inherit_canonical_permission(parameters):
    scenario = ScenarioConfig("structural", "M0", parameters)
    policy = runner.resolve_inference_policy(scenario, METHODS)
    for metric in ("cate_mse", "root_pehe"):
        assert policy[metric]["eligible"] is False
        assert policy[metric]["source"] == "unknown_default"


def test_future_n_prefixed_generator_control_does_not_inherit_permission(monkeypatch):
    # A future generator may accept a scientific control beginning with n_.
    # Its name alone must not make it an approved sample-budget variation.
    monkeypatch.setattr(runner, "build_dgp_config", lambda scenario: SimpleNamespace(
        noise_distribution="normal"))
    scenario = ScenarioConfig("structural", "M0", {"n_unreviewed_control": 1})
    policy = runner.resolve_inference_policy(scenario, METHODS)
    assert policy["cate_mse"]["eligible"] is False
    assert policy["root_pehe"]["eligible"] is False


@pytest.mark.parametrize("methods,options", [
    (("rct_t_ridge", "author_defined_method"), {}),
    (("rct_t_ridge", "datafusionppi"), {}),
    (METHODS, {"obs_t_ridge": {"alpha": 2.0}}),
])
def test_custom_methods_and_changed_options_need_explicit_assumptions(methods, options):
    scenario = ScenarioConfig("structural", "M0")
    policy = runner.resolve_inference_policy(scenario, methods, options)
    for metric in ("cate_mse", "root_pehe"):
        assert policy[metric]["eligible"] is False
        assert policy[metric]["source"] == "unknown_default"
    declared = ScenarioConfig("structural", "M0", inference_policy=declaration())
    permitted = runner.resolve_inference_policy(declared, methods, options)
    assert permitted["cate_mse"]["eligible"] is True
    assert permitted["cate_mse"]["source"] == "scenario_declaration"


@pytest.mark.parametrize("kind,identifier,parameters", [
    ("gamma", "constant", {}),
    ("empirical", "ihdp", {"hidden_columns": ("hidden",)}),
])
def test_unestablished_scenario_kinds_require_explicit_assumptions(kind, identifier, parameters):
    scenario = ScenarioConfig(kind, identifier, parameters)
    policy = runner.resolve_inference_policy(scenario, METHODS)
    for metric in ("cate_mse", "root_pehe"):
        assert policy[metric]["eligible"] is False
        assert policy[metric]["source"] == "unknown_default"
    declared = ScenarioConfig(kind, identifier, parameters, inference_policy=declaration())
    permitted = runner.resolve_inference_policy(declared, METHODS)
    assert permitted["cate_mse"]["eligible"] is True
    assert permitted["root_pehe"]["eligible"] is True


@pytest.mark.parametrize("invalid", [
    [], {"cate_mse": True}, {"unknown_metric": {"eligible": True, "reason": "Assumed"}},
    {"cate_mse": {"eligible": 1, "reason": "Assumed"}},
    {"cate_mse": {"eligible": True}},
    {"cate_mse": {"eligible": True, "reason": ""}},
    {"cate_mse": {"eligible": True, "reason": "   "}},
    {"cate_mse": {"eligible": True, "reason": "first\nsecond"}},
    {"cate_mse": {"eligible": True, "reason": 3}},
    {"cate_mse": {"eligible": True, "reason": "Assumed", "source": "builtin_declared_policy"}},
])
def test_malformed_user_policy_and_forged_provenance_are_rejected(invalid):
    with pytest.raises(ValueError):
        ScenarioConfig("structural", "M0", inference_policy=invalid)


def test_toml_policy_is_preserved_in_payload_and_config_identity(tmp_path):
    path = tmp_path / "policy.toml"
    path.write_text('''schema_version = 1
[run]
seeds = [3, 5]
methods = ["rct_t_ridge", "obs_t_ridge"]
reference_method = "rct_t_ridge"
[[scenarios]]
kind = "structural"
id = "M0"
[scenarios.inference_policy.cate_mse]
eligible = false
reason = "MSE inference withheld by design"
[scenarios.inference_policy.root_pehe]
eligible = true
reason = "Root-PEHE moment assumption declared"
''')
    loaded = load_config(path)
    expected = {
        "cate_mse": {"eligible": False, "reason": "MSE inference withheld by design"},
        "root_pehe": {"eligible": True, "reason": "Root-PEHE moment assumption declared"},
    }
    assert loaded.scenarios[0].inference_policy == expected
    assert loaded.payload()["scenarios"][0]["inference_policy"] == expected
    undeclared = BenchmarkConfig((ScenarioConfig("structural", "M0"),),
                                 seeds=(3, 5), methods=METHODS)
    assert loaded.config_hash != undeclared.config_hash
    assert undeclared.scenarios[0].inference_policy == {}
    assert "inference_policy" not in undeclared.payload()["scenarios"][0]
    resolved = runner.resolve_inference_policy(loaded.scenarios[0], METHODS)
    assert resolved["cate_mse"]["eligible"] is False
    assert resolved["root_pehe"]["eligible"] is True
    assert resolved["cate_mse"]["source"] == "scenario_declaration"
    assert resolved["cate_mse"]["reason"] == expected["cate_mse"]["reason"]
    assert "source" not in loaded.payload()["scenarios"][0]["inference_policy"]["cate_mse"]


def test_mocked_runner_exports_resolved_policy_and_report_keeps_m11_mean(tmp_path, monkeypatch):
    scenario = ScenarioConfig("structural", "M11", TINY_BUDGET,
                              inference_policy=declaration())

    def fake_problem(requested, seed):
        assert requested.parameters == TINY_BUDGET
        return SimpleNamespace(
            metadata={}, eval_X=np.full((32, 3), seed), eval_tau=np.zeros(32),
            training_view=lambda: object(),
        )

    def fake_method(name, **options):
        def predict(training, eval_X, *, seed):
            # No fitting: reference MSE 1, contender MSE 3 then 5.
            mse = 1 if name == METHODS[0] else (3 if eval_X[0, 0] == 3 else 5)
            return np.full(len(eval_X), math.sqrt(mse))
        return SimpleNamespace(fit_predict=predict)

    monkeypatch.setattr(runner, "_generate", fake_problem)
    monkeypatch.setattr(runner, "make_method", fake_method)
    config = BenchmarkConfig((scenario,), seeds=(3, 5), methods=METHODS)
    output = tmp_path / "mocked"
    result = runner.run_benchmark(config, output)
    assert result["status"] == "complete"
    policy = result["scenario_inference"]["structural:M11"]
    assert policy["cate_mse"]["eligible"] is False
    assert policy["cate_mse"]["source"] == "resolved_noise"
    assert policy["root_pehe"]["eligible"] is True
    assert policy["root_pehe"]["source"] == "scenario_declaration"
    metric = comparison(result, "structural:M11")["metrics"]["cate_mse"]
    assert metric["mean"] == pytest.approx(3)
    assert metric["ci95"] is None
    report = (output / "README.md").read_text()
    assert "| `structural:M11` | `obs_t_ridge` | 3 | unavailable | 2 / 2 | withheld:" in report
    assert metric["withheld_reason"] in report
    saved = json.loads((output / "summary.json").read_text())
    assert saved["scenario_inference"] == result["scenario_inference"]
    previous_records = (output / "records.jsonl").read_bytes()
    runner.run_benchmark(config, output, resume=True)
    assert (output / "records.jsonl").read_bytes() == previous_records
