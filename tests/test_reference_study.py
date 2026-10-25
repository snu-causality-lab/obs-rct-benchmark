"""Independent small calculations for the reference study's aggregation."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from obs_rct_benchmark import runner


spec = importlib.util.spec_from_file_location(
    "reference_analysis", Path(__file__).parents[1] / "scripts/analyze_reference_study.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_seed_summary_matches_three_value_hand_calculation():
    result = analysis.summarize_values([1., 2., 3.])
    assert result["n"] == 3
    assert result["mean"] == 2.
    assert result["mcse"] == pytest.approx(1 / math.sqrt(3))
    # t_(2,.975) is 4.3026527297; this is not the benchmark's table lookup.
    half = 4.3026527297 / math.sqrt(3)
    assert result["ci95"] == pytest.approx([2 - half, 2 + half])


def test_holm_matches_independent_three_test_example():
    assert analysis.holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert analysis.holm_adjust([0., 1., .02]) == pytest.approx([0., 1., .04])


def test_incomplete_primary_family_withholds_all_decisions():
    rows = [{"complete": True, "p_two_sided_zero": .001},
            {"complete": False, "p_two_sided_zero": None}]
    assert not analysis.apply_primary_family(rows)
    assert all(row["holm_p"] is None and row["holm_reject_05"] is None for row in rows)


def test_complete_family_uses_every_declared_contrast():
    rows = [{"complete": True, "p_two_sided_zero": p} for p in (.01, .04, .03)]
    assert analysis.apply_primary_family(rows)
    assert [row["holm_p"] for row in rows] == pytest.approx([.03, .06, .06])
    assert [row["holm_reject_05"] for row in rows] == [True, False, False]


def test_shipped_study_declares_full_fixed_recipes_and_budget_grid():
    config = analysis.load_config(Path(__file__).parents[1] / "configs/reference_study.toml")
    assert len(config.scenarios) == 16 and len(config.seeds) == 32
    assert len(config.scenarios) * len(config.seeds) * len(config.methods) == 3072
    assert config.method_options["cornet_official"]["source_epochs"] == 500
    assert config.method_options["cornet_official"]["correction_epochs"] == 200
    assert {s.parameters["n_rct_train"] + s.parameters["n_rct_tune"]
            for s in config.scenarios} == {160, 544}


def _labelled_config():
    return runner.BenchmarkConfig(
        scenarios=(runner.ScenarioConfig("structural", "M2", {
            "n_obs": 8, "n_rct_train": 8, "n_rct_tune": 4,
            "n_rct_test": 4, "n_eval": 8,
        }, label="small_budget"),),
        seeds=(7,), methods=("rct_t_ridge", "obs_t_ridge"), purpose="reference")


def _write_synthetic_shard(path, config, methods):
    """Create runner-contract artifacts without generating data or fitting."""
    path.mkdir()
    source = {"package_file_sha256": {"fixture.py": "0" * 64}}
    source["aggregate_sha256"] = analysis.digest(source["package_file_sha256"])
    bound = {"source": source, "custom_methods": {}, "environment": {},
             "external_methods": {}, "resolved_scenarios": runner.resolved_scenarios(config),
             "empirical_inputs": {}}
    manifest = {**bound, "source_fingerprint": analysis.digest(bound),
                "config": config.payload(), "config_hash": config.config_hash}
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    records = []
    for method in methods:
        row = runner._base_record(config, config.scenarios[0].key, 7, method)
        row.update(status="ok", metrics={"cate_mse": 4., "root_pehe": 2.})
        records.append(row)
    (path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return records


def test_collector_accepts_labelled_runner_record_with_preset_method_seed(tmp_path):
    config = _labelled_config()
    run = tmp_path / "shard"
    expected = _write_synthetic_shard(run, config, config.methods)
    records, provenance = analysis.collect_records(config, [run])
    assert len(records) == 2 and len(provenance) == 1
    row = records[(config.scenarios[0].key, 7, "rct_t_ridge")]
    assert row == expected[0]
    assert row["method_seed"] == runner.method_seed("structural:M2", 7, "rct_t_ridge")


def test_collector_rejects_overlapping_declared_seeds_with_complementary_records(tmp_path):
    config = _labelled_config()
    first, second = tmp_path / "first", tmp_path / "second"
    _write_synthetic_shard(first, config, config.methods[:1])
    _write_synthetic_shard(second, config, config.methods[1:])
    with pytest.raises(ValueError, match="overlapping seed"):
        analysis.collect_records(config, [first, second])
