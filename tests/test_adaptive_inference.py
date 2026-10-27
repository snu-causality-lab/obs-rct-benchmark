"""Hand-record checks for recruitment inference and descriptive coverage."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path

import pytest

from obs_rct_benchmark import adaptive_runner
from obs_rct_benchmark.adaptive_runner import AdaptiveConfig
from obs_rct_benchmark.runner import ScenarioConfig, resolved_scenarios


spec = importlib.util.spec_from_file_location(
    "adaptive_inference_analysis", Path(__file__).parents[1]/"scripts/analyze_adaptive_study.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


ROOT_POLICY = {"root_pehe": {
    "eligible": True,
    "reason": "Prospectively assume finite variance for the root-PEHE seed contrasts.",
}}
GAUSSIAN_POLICY = {
    "cate_mse": {"eligible": True, "reason": "Declared Gaussian-control score-moment assumption."},
    **ROOT_POLICY,
}


def config(*scenarios, budgets=(16,)):
    if not scenarios:
        scenarios = (ScenarioConfig("structural", "M0", {"n_obs": 8, "n_eval": 8}),)
    return AdaptiveConfig(tuple(scenarios), seeds=(1, 2, 3),
                          policies=("uniform", "geometry"), estimators=("rct_t_ridge",),
                          budgets=budgets, pilot_size=8, tune_size=4, pool_size=32,
                          purpose="reference")


def records(config, *, failed=None):
    """Use square scores so MSE and root-PEHE have separate simple oracles.

    Uniform MSE is (1, 4, 9); geometry MSE is (0, 1, 4). Therefore the
    MSE differences are (-1, -3, -5), and root-PEHE differences are all -1.
    """
    rows = []
    for scenario in config.scenarios:
        for budget in config.budgets:
            for index, seed in enumerate(config.seeds):
                for policy in config.policies:
                    row = adaptive_runner._identity(
                        config, scenario.key, seed, policy, budget, config.estimators[0])
                    mse = (index + (policy == "uniform"))**2
                    row.update(status="ok", stage="complete", query_indices=list(range(budget)),
                               query_cost=budget, pilot_cost=config.pilot_size,
                               tune_cost=config.tune_size, train_cost=budget-config.tune_size,
                               metrics={"cate_mse": mse, "root_pehe": math.sqrt(mse)})
                    if (scenario.key, seed, policy) == failed:
                        row.update(status="failed", stage="method_or_metric", metrics=None,
                                   error_type="ValueError")
                    rows.append(row)
    return rows


def test_adaptive_policy_uses_original_scenario_and_estimator_names(monkeypatch):
    scenario = ScenarioConfig("structural", "M11", {"n_obs": 8, "n_eval": 8}, label="named")
    c = config(scenario, budgets=(16, 24))
    resolved = {"cate_mse": {"eligible": False, "reason": "Heavy-tailed scores.",
                             "source": "fixture"},
                "root_pehe": {"eligible": False, "reason": "Undeclared root score policy.",
                              "source": "fixture"}}
    calls = []

    def resolve(original, methods, method_options=None):
        calls.append((original, methods, method_options))
        return resolved

    monkeypatch.setattr(adaptive_runner, "resolve_inference_policy", resolve)
    policies = adaptive_runner.adaptive_inference_policy(c)
    assert calls == [(scenario, c.estimators, c.method_options)]
    assert policies == {"structural:M11@named/budget_16": resolved,
                        "structural:M11@named/budget_24": resolved}


def test_adaptive_config_roundtrip_preserves_declarations_and_generator_controls(tmp_path):
    plain = ScenarioConfig("structural", "M11", {"n_obs": 8, "n_eval": 8})
    declared = replace(plain, inference_policy=ROOT_POLICY)
    old, new = config(plain), config(declared)
    assert "inference_policy" not in old.payload()["scenarios"][0]
    assert new.payload()["scenarios"][0]["inference_policy"] == ROOT_POLICY
    assert old.config_hash != new.config_hash
    assert new.pool_scenario(declared).inference_policy == ROOT_POLICY
    assert resolved_scenarios(old) == resolved_scenarios(new)
    path = tmp_path/"adaptive.toml"
    path.write_text('''schema_version = 1
[run]
seeds = [1, 2, 3]
policies = ["uniform", "geometry"]
estimators = ["rct_t_ridge"]
budgets = [16]
pilot_size = 8
tune_size = 4
pool_size = 32
purpose = "reference"
[[scenarios]]
kind = "structural"
id = "M11"
[scenarios.parameters]
n_obs = 8
n_eval = 8
[scenarios.inference_policy.root_pehe]
eligible = true
reason = "Prospectively assume finite variance for the root-PEHE seed contrasts."
''')
    assert adaptive_runner.load_adaptive_config(path).payload() == new.payload()


def test_adaptive_analyzer_withholds_m11_and_mixed_mse_but_keeps_means():
    heavy = ScenarioConfig("structural", "M11", {"n_obs": 8, "n_eval": 8},
                           inference_policy=ROOT_POLICY)
    normal = ScenarioConfig("structural", "M0", {"n_obs": 8, "n_eval": 8},
                            inference_policy=GAUSSIAN_POLICY)
    c = config(heavy, normal)
    summary, scores, contrasts = analysis.analyze_records(c, records(c))
    assert summary["coverage"] == {"expected": 12, "recorded": 12,
                                   "successful": 12, "failed": 0, "missing": 0}
    for score in (item for item in scores if item["scenario"] == heavy.key):
        assert score["mean_mse"] == pytest.approx(14/3 if score["policy"] == "uniform" else 5/3)
        assert score["mcse_mse"] is None
        assert score["ci95_lower"] is score["ci95_upper"] is None
        assert not score["eligible_for_inference"] and score["withheld_reason"]
    heavy_contrast = next(item for item in contrasts if item["scenario"] == heavy.key)
    assert heavy_contrast["mean_difference"] == -3
    assert heavy_contrast["successful_pairs"] == 3
    assert heavy_contrast["complete_coverage"]
    assert not heavy_contrast["eligible_for_inference"]
    assert heavy_contrast["mcse_difference"] is None
    assert heavy_contrast["ci95_lower"] is heavy_contrast["ci95_upper"] is None
    assert heavy_contrast["withheld_reason"]
    method = "geometry__rct_t_ridge"
    comparisons = summary["within_estimator_policy_comparisons"]["rct_t_ridge"][method]
    overall = comparisons["overall"]
    assert overall["complete_coverage"] and overall["complete_seed_blocks"] == 3
    assert overall["metrics"]["cate_mse"]["mean"] == -3
    assert overall["metrics"]["cate_mse"]["ci95"] is None
    assert overall["metrics"]["cate_mse"]["withheld_reason"]
    # A separate declaration governs the transformed metric, not the MSE flag.
    root = comparisons["by_scenario"][heavy.key+"/budget_16"]["metrics"]["root_pehe"]
    assert root["eligible_for_inference"]
    assert root["mean"] == -1 and root["ci95"] == [-1, -1]
    gaussian = next(item for item in contrasts if item["scenario"] == normal.key)
    assert gaussian["eligible_for_inference"]
    assert gaussian["mean_difference"] == -3
    assert gaussian["mcse_difference"] == pytest.approx(2/math.sqrt(3))
    assert gaussian["ci95_lower"] == pytest.approx(-3-4.3026527297*2/math.sqrt(3))
    assert gaussian["ci95_upper"] == pytest.approx(-3+4.3026527297*2/math.sqrt(3))


def test_adaptive_analyzer_failed_scores_keep_successful_mean_without_interval():
    scenario = ScenarioConfig("structural", "M0", {"n_obs": 8, "n_eval": 8},
                              inference_policy=GAUSSIAN_POLICY)
    c = config(scenario)
    rows = records(c, failed=(scenario.key, 3, "geometry"))
    summary, scores, contrasts = analysis.analyze_records(c, rows)
    geometry = next(item for item in scores if item["policy"] == "geometry")
    assert geometry["successful"] == 2 and geometry["failed"] == 1
    assert geometry["mean_mse"] == .5
    assert geometry["mcse_mse"] is None
    assert geometry["ci95_lower"] is geometry["ci95_upper"] is None
    assert geometry["withheld_reason"] == "failed or missing requested cells"
    assert summary["coverage"]["failed"] == 1
    assert contrasts[0]["successful_pairs"] == 2
    assert not contrasts[0]["complete_coverage"]
    assert contrasts[0]["mean_difference"] is None


def test_adaptive_policy_report_discloses_metric_reason_for_complete_records():
    comparison = {"complete_seed_blocks": 3, "expected_seed_blocks": 3,
                  "complete_coverage": True, "eligible_for_comparison": False,
                  "withheld_reason": "MSE uncertainty is unsupported.",
                  "metrics": {"cate_mse": {"mean": -3, "ci95": None,
                              "eligible_for_inference": False,
                              "withheld_reason": "MSE uncertainty is unsupported."}}}
    summary = {"within_estimator_policy_comparisons": {
        "rct_t_ridge": {"geometry__rct_t_ridge": {"by_scenario": {
            "structural:M11/budget_16": comparison}}}}}
    report = adaptive_runner._policy_report(summary)
    assert "| -3 | unavailable | 3 / 3 |" in report
    assert "withheld: MSE uncertainty is unsupported." in report


def test_adaptive_runner_uses_policy_in_both_comparison_views(tmp_path, monkeypatch):
    scenario = ScenarioConfig("structural", "M11", {"n_obs": 8, "n_eval": 8},
                              inference_policy=ROOT_POLICY)
    c = config(scenario)
    hand_rows = records(c)
    monkeypatch.setattr(adaptive_runner, "_generate", lambda *args: object())
    monkeypatch.setattr(adaptive_runner, "run_adaptive_problem", lambda problem, **kwargs: [
        row for row in hand_rows if row["seed"] == kwargs["seed"]])
    output = tmp_path/"run"
    summary = adaptive_runner.run_adaptive_benchmark(c, output)
    assert summary["coverage"]["successful"] == 6
    for comparisons in (summary["paired_comparisons"],
                        summary["within_estimator_policy_comparisons"]["rct_t_ridge"]):
        estimate = comparisons["geometry__rct_t_ridge"]["overall"]["metrics"]["cate_mse"]
        assert estimate["mean"] == -3 and estimate["ci95"] is None
        assert estimate["withheld_reason"]
    manifest = json.loads((output/"run_manifest.json").read_text())
    assert manifest["config"]["scenarios"][0]["inference_policy"] == ROOT_POLICY
    assert manifest["scenario_inference"] == adaptive_runner.adaptive_inference_policy(c)
    assert "inference_policy" not in manifest["resolved_pools"][scenario.key]["config"]
    before = (output/"records.jsonl").read_bytes()
    monkeypatch.setattr(adaptive_runner, "_generate", lambda *args: pytest.fail("resume generated a new pool"))
    adaptive_runner.run_adaptive_benchmark(c, output, resume=True)
    assert (output/"records.jsonl").read_bytes() == before
    assert "withheld:" in (output/"README.md").read_text()
    manifest["scenario_inference"][scenario.key+"/budget_16"]["cate_mse"]["reason"] = "Altered declaration."
    (output/"run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        adaptive_runner.run_adaptive_benchmark(c, output, resume=True)


def test_adaptive_analyzer_report_includes_complete_heavy_tail_withholding(tmp_path, monkeypatch):
    heavy = ScenarioConfig("structural", "M11", {"n_obs": 8, "n_eval": 8})
    c = config(heavy)
    monkeypatch.setattr(analysis, "collect_records", lambda *args: (records(c), []))
    monkeypatch.setattr(analysis, "plot_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis.importlib.metadata, "version", lambda name: "fixture")
    output = tmp_path/"analysis"
    summary = analysis.write_analysis(c, [], output)
    contrast = summary["policy_contrasts"][0]
    assert contrast["complete_coverage"] and contrast["mean_difference"] == -3
    assert contrast["withheld_reason"] in (output/"README.md").read_text()
    assert "| -3 | unavailable | 3 / 3 | withheld:" in (output/"README.md").read_text()
