"""Small external calculations and adversarial catalog-artifact fixtures."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark import runner
from obs_rct_benchmark.data import SourceSample


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("analyze_catalog_study")
repro = load_script("verify_catalog_reproducibility")


def config(kind="structural", identity="M0"):
    scenario = runner.ScenarioConfig(kind, identity, {
        "n_obs": 512, "n_rct_train": 128, "n_rct_tune": 32,
        "n_rct_test": 64, "n_eval": 2048}, label="o512_r128")
    return runner.BenchmarkConfig((scenario,), seeds=(3, 5),
        methods=("rct_t_extra_trees", "rct_dr_extra_trees", "cornet_official", "datafusionppi"),
        reference_method="rct_dr_extra_trees", purpose="reference")


def write_shard(path, canonical, *, seed=None, missing=None, failed=None):
    selected = canonical.scenarios
    methods = analysis.applicable_methods(canonical, selected[0])
    shard = replace(canonical, scenarios=selected,
                    seeds=canonical.seeds if seed is None else (seed,), methods=methods,
                    method_options={k: v for k, v in canonical.method_options.items() if k in methods})
    bound = {"source": runner.source_provenance(), "custom_methods": {},
             "environment": runner.environment_provenance(),
             "external_methods": runner.external_method_provenance(shard.methods),
             "resolved_scenarios": runner.resolved_scenarios(shard), "empirical_inputs": {}}
    manifest = {**bound, "source_fingerprint": runner._hash(bound),
                "config": shard.payload(), "config_hash": shard.config_hash}
    path.mkdir()
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for scenario in shard.scenarios:
        for sample_seed in shard.seeds:
            for method in shard.methods:
                if method == missing:
                    continue
                row = runner._base_record(shard, scenario.key, sample_seed, method)
                # Each paired difference is exactly -1: a separate arithmetic oracle.
                mse = float(sample_seed + (0 if method in analysis.FUSION else 1))
                row.update(status="ok", metrics={"cate_mse": mse, "root_pehe": math.sqrt(mse)})
                if method == failed:
                    row.update(status="failed", metrics=None, stage="method_or_metric", error_type="RuntimeError")
                rows.append(row)
    (path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows, manifest


def test_scalar_summary_and_holm_match_independent_small_calculations():
    result = analysis.summarize_values([1., 2., 3.])
    assert result["mean"] == 2.
    assert result["mcse"] == pytest.approx(1 / math.sqrt(3))
    half = 4.3026527297 / math.sqrt(3)
    assert result["ci95"] == pytest.approx([2 - half, 2 + half])
    assert analysis.holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_complete_collection_handles_distinct_seed_shards_and_labelled_rng(tmp_path):
    c = config()
    paths = [tmp_path / "one", tmp_path / "two"]
    for path, seed in zip(paths, c.seeds):
        write_shard(path, c, seed=seed)
    records, inputs = analysis.collect_records(c, paths)
    assert len(records) == 8 and len(inputs) == 2
    scores, contrasts, coverage = analysis.analyze_records(c, records)
    assert coverage["expected"] == coverage["successful"] == 8
    assert coverage["mean_t_family_size"] == 4 and coverage["mean_t_family_complete"]
    assert all(row["mean"] == -1. and row["mcse"] == 0. for row in contrasts)
    assert all(row["decision"] == "favorable" for row in contrasts)
    assert next(row for row in scores if row["method"] == "cornet_official")["mean"] == 4.


def test_declared_rectangles_cannot_be_spliced_even_when_records_do_not_overlap(tmp_path):
    c = config()
    one, two = tmp_path / "one", tmp_path / "two"
    write_shard(one, c, seed=3, missing="cornet_official")
    write_shard(two, c, seed=3, missing="datafusionppi")
    with pytest.raises(ValueError, match="declarations overlap"):
        analysis.collect_records(c, [one, two], require_full=False)


def test_missing_cells_fail_closed_and_terminal_failures_withhold_entire_family(tmp_path):
    c = config()
    missing, failed = tmp_path / "missing", tmp_path / "failed"
    write_shard(missing, c, missing="cornet_official")
    with pytest.raises(ValueError, match="missing requested"):
        analysis.collect_records(c, [missing])
    write_shard(failed, c, failed="cornet_official")
    records, _ = analysis.collect_records(c, [failed])
    _, contrasts, coverage = analysis.analyze_records(c, records)
    assert coverage["failed"] == 2 and not coverage["mean_t_family_complete"]
    assert all(row["holm_p"] is None for row in contrasts)
    assert all(row["decision"] == "withheld_incomplete_family" for row in contrasts)


def test_source_environment_and_record_identity_tampering_are_rejected(tmp_path):
    c = config()
    path = tmp_path / "run"
    rows, manifest = write_shard(path, c)
    wrong = dict(manifest)
    wrong["environment"] = {**manifest["environment"], "python": "0.0.0"}
    (path / "run_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="numerical environment"):
        analysis.collect_records(c, [path])
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    rows[0]["method_seed"] = int(runner._hash([
        "method_seed_v1", rows[0]["scenario"], rows[0]["seed"], rows[0]["method"]])[:8], 16)
    (path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="record identity"):
        analysis.collect_records(c, [path])


def test_unsupported_ppi_is_explicit_na_instead_of_failure(tmp_path):
    c = config("paper", "integrative_r_learner_multivariate_paper")
    path = tmp_path / "variable"
    write_shard(path, c)
    records, _ = analysis.collect_records(c, [path])
    scores, contrasts, coverage = analysis.analyze_records(c, records)
    assert coverage["nominal_cells"] == 8
    assert coverage["not_applicable_cells"] == 2
    assert coverage["expected"] == coverage["successful"] == 6
    assert coverage["failed"] == 0
    ppi = next(row for row in scores if row["method"] == "datafusionppi")
    assert not ppi["applicable"] and ppi["expected"] == 0 and ppi["mean"] is None
    assert len(contrasts) == 2


def test_m11_means_and_quartiles_are_descriptive_without_mean_inference(tmp_path):
    c = config("structural", "M11")
    path = tmp_path / "heavy"
    write_shard(path, c)
    records, _ = analysis.collect_records(c, [path])
    scores, contrasts, coverage = analysis.analyze_records(c, records)
    assert coverage["successful"] == 8 and coverage["mean_t_family_size"] == 0
    assert coverage["descriptive_heavy_tail_contrasts"] == 4
    for row in scores:
        assert row["mean"] is not None and row["sample_median"] is not None
        assert row["sample_q25"] <= row["sample_median"] <= row["sample_q75"]
        assert row["mcse"] is row["ci95"] is row["p_two_sided_zero"] is None
    assert all(row["decision"] == "not_assessed_heavy_tail" and row["holm_p"] is None for row in contrasts)


def test_canonical_plan_and_independent_analyzer_counts_agree():
    c = runner.load_config(Path(__file__).parents[1] / "configs/catalog_study.toml")
    plan = load_script("make_catalog_configs")
    assert analysis.expected_cells(c) == plan.expected_cells(c)
    assert len(analysis.expected_cells(c)) == 79104
    assert sum(row["seed_count"] for row in analysis.not_applicable_rows(c)) == 256
    expected_primary = {(scenario.key, fusion, control) for scenario in c.scenarios
                        if analysis.inference_reason(scenario) is None
                        for fusion in analysis.FUSION if fusion in analysis.applicable_methods(c, scenario)
                        for control in analysis.CONTROLS}
    assert expected_primary == plan.primary_contrasts(c)
    assert len(expected_primary) == 472


def tiny_problem():
    X = np.arange(4., dtype=float).reshape(-1, 1)
    def sample(source, role):
        return SourceSample(X, [0, 1, 0, 1], [0, 1, 0, 1], source, role)
    return SimpleNamespace(obs_train=sample("obs", "obs_train"),
        rct_train=sample("rct", "rct_train"), rct_tune=sample("rct", "rct_tune"),
        rct_test=sample("rct", "rct_test"), eval_X=np.array([[0.], [1.]]),
        eval_tau=np.array([1., 1.]), target_reference=None,
        training_view=lambda: None)


def repro_fixture(monkeypatch, *, failed=False, simulation_failure=False):
    c = replace(config(), seeds=(3,), methods=("rct_t_extra_trees",), reference_method="rct_t_extra_trees")
    scenario = c.scenarios[0]
    row = {"record_id": "fixture", "status": "ok", "metrics": {"cate_mse": 1., "root_pehe": 1.}}
    if failed:
        row.update(status="failed", metrics=None, stage="simulation" if simulation_failure else "method_or_metric",
                   error_type="ValueError" if simulation_failure else "RuntimeError")
    monkeypatch.setattr(repro.analysis, "collect_records", lambda *a, **k: ({(scenario.key, 3, c.methods[0]): row}, []))
    if simulation_failure:
        def broken(*args):
            raise ValueError("generation failed")
        monkeypatch.setattr(repro, "_generate", broken)
    else:
        monkeypatch.setattr(repro, "_generate", lambda *a: tiny_problem())
    return c


def test_repro_checker_hashes_two_generated_roles_and_refits_twice_against_hand_mse(tmp_path, monkeypatch):
    c = repro_fixture(monkeypatch)
    class Constant:
        def fit_predict(self, *a, **k):
            return np.array([2., 2.])
    monkeypatch.setattr(repro, "make_method", lambda *a, **k: Constant())
    result = repro.verify(c, [tmp_path / "ignored"], tmp_path / "result.json")
    assert result["selected_cells"] == result["matched_cells"] == 1
    assert result["planned_refit_attempts"] == result["actual_refit_attempts"] == 2
    assert result["checks"][0]["predictions_byte_identical"]
    assert result["max_absolute_mse_difference"] == 0.
    assert result["data_checks"][0]["repeat_data_sha256"][0] == result["data_checks"][0]["repeat_data_sha256"][1]
    assert repro.array_hash(np.array([1., 2.])) != repro.array_hash(np.array([[1., 2.]]))


def test_repro_checker_preserves_and_repeats_terminal_method_failures(tmp_path, monkeypatch):
    c = repro_fixture(monkeypatch, failed=True)
    class Broken:
        def fit_predict(self, *a, **k):
            raise RuntimeError("failure remains visible")
    monkeypatch.setattr(repro, "make_method", lambda *a, **k: Broken())
    result = repro.verify(c, [], tmp_path / "result.json")
    assert result["matched_cells"] == result["consistent_terminal_failures"] == 1
    assert result["actual_refit_attempts"] == 2 and result["successful_refit_attempts"] == 0


def test_repro_checker_distinguishes_generation_failure_from_refit_attempt(tmp_path, monkeypatch):
    c = repro_fixture(monkeypatch, failed=True, simulation_failure=True)
    result = repro.verify(c, [], tmp_path / "result.json")
    assert result["matched_cells"] == result["consistent_terminal_failures"] == 1
    assert result["actual_refit_attempts"] == result["successful_refit_attempts"] == 0


def test_repro_checker_rejects_changed_saved_score_without_substituting_another_seed(tmp_path, monkeypatch):
    c = repro_fixture(monkeypatch)
    class Different:
        def fit_predict(self, *a, **k):
            return np.array([3., 3.])
    monkeypatch.setattr(repro, "make_method", lambda *a, **k: Different())
    result = repro.verify(c, [], tmp_path / "result.json")
    assert result["mismatched_cells"] == 1 and result["seed"] == 3
