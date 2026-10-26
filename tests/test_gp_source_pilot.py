"""Pilot data accounting, pairing, provenance and descriptive-analysis contracts.

All estimator calls in this file are fake. Real native controls and the
independent mathematical design checks run in the server preflight driver.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def pilot():
    path = Path(__file__).resolve().parents[1] / "scripts/run_gp_source_pilot.py"
    spec = importlib.util.spec_from_file_location("gp_source_pilot_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return {
        "schema_version": 1, "scope": "development diagnostic", "generator": "paired_gp_source_v1",
        "d": 2, "n_obs": 12, "rct_budgets": [8, 16], "n_eval": 8,
        "deltas": [0, 1, 2], "seeds": [11, 19, 37], "rho": .5,
        "fit_timeout_seconds": 10, "tune_role": "first_two_training_rows_alias",
        "inference": "pointwise_paired_t_descriptive",
        "methods": {"full": "causal_icm_fixed", "rct_only": "causal_icm_rct_only",
                    "obs_start": "causal_icm_obs_start"},
    }


def test_fixed_function_values_have_independent_hand_values(pilot):
    points = np.array([[0., 0.], [.5, 0.], [0., .5]])
    np.testing.assert_allclose(pilot.true_effect(points), [1.25, 1.75, 1.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pilot.baseline_mean(points), [0., .25, .125], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pilot.bias_function(points), [0., 0., 1.], atol=1e-12, rtol=0)


def test_generator_pairs_bias_and_budget_without_changing_trial_or_target(pilot, config):
    first = pilot.generate_case(config, seed=11, delta=0, n_rct=8)
    changed = pilot.generate_case(config, seed=11, delta=2, n_rct=16)
    for name in ("obs_X", "obs_T", "eval_X", "eval_tau"):
        np.testing.assert_array_equal(first[name], changed[name])
    for name in ("rct_X", "rct_T", "rct_Y", "rct_known_propensity"):
        np.testing.assert_array_equal(first[name], changed[name][:8])
    x, t = first["obs_X"], first["obs_T"]
    expected_change = 2 * (t - .5) * np.cos(np.pi*x[:, 0]) * np.sin(np.pi*x[:, 1])
    np.testing.assert_allclose(changed["obs_Y"]-first["obs_Y"], expected_change, atol=1e-12, rtol=0)
    again = pilot.generate_case(config, seed=11, delta=0, n_rct=8)
    assert pilot.array_identity(again) == pilot.array_identity(first)


def test_tuning_alias_reuses_charged_rows_and_does_not_expose_truth(pilot, config):
    arrays = pilot.generate_case(config, seed=11, delta=1, n_rct=8)
    view = pilot.make_training_view(arrays)
    np.testing.assert_array_equal(view.rct_tune.X, view.rct_train.X[:2])
    np.testing.assert_array_equal(view.rct_tune.Y, view.rct_train.Y[:2])
    assert not view.rct_tune.X.flags.writeable
    assert not hasattr(view, "eval_tau") and not hasattr(view, "rct_test")
    assert view.target_reference is None
    assert set(pilot.OBSERVED_ARRAYS).isdisjoint({"eval_tau", "rct_test"})


@pytest.mark.parametrize("key,value", [
    ("generator", "changed"), ("schema_version", True), ("d", 3), ("n_obs", 1),
    ("n_eval", 0), ("rct_budgets", [8, 8]), ("seeds", [11, 11]),
    ("seeds", [True]), ("rho", float("nan")), ("rho", 1.1),
    ("tune_role", "new_unaccounted_labels"), ("inference", "posthoc_holm"),
    ("deltas", [0, float("inf")]), ("methods", {"full": "one"}),
])
def test_configuration_rejects_ambiguous_or_changed_contract(pilot, config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        pilot.validate_config(config)


def test_configuration_rejects_missing_unknown_fields_and_unregistered_case(pilot, config):
    for changed in ({**config, "oracle_features": [0]}, {k:v for k,v in config.items() if k != "rho"}):
        with pytest.raises(ValueError):
            pilot.validate_config(changed)
    with pytest.raises(ValueError, match="not in"):
        pilot.generate_case(config, seed=42, delta=0, n_rct=8)
    identifiers = [case["id"] for case in pilot.cases(config)]
    assert len(identifiers) == len(set(identifiers)) == 18


def test_source_recipe_ids_and_factories_cannot_be_swapped(pilot, config):
    from obs_rct_benchmark.causal_icm import CausalICMSourceControl, OfficialCausalICM
    config["methods"]["rct_only"], config["methods"]["obs_start"] = (
        config["methods"]["obs_start"], config["methods"]["rct_only"])
    with pytest.raises(ValueError, match="exact source-access"):
        pilot.validate_config(config)
    pilot._check_estimator(OfficialCausalICM(), "full", .5)
    pilot._check_estimator(CausalICMSourceControl(source="rct"), "rct_only", .5)
    pilot._check_estimator(CausalICMSourceControl(source="obs"), "obs_start", .5)
    for estimator, slot in ((CausalICMSourceControl(source="rct"), "obs_start"),
                            (CausalICMSourceControl(source="obs"), "rct_only"),
                            (OfficialCausalICM(effect_modifier_columns=(0,)), "full")):
        with pytest.raises(ValueError, match="factory"):
            pilot._check_estimator(estimator, slot, .5)


def test_descriptive_pair_summary_uses_seed_units_and_withholds_incomplete_intervals(pilot):
    result = pilot.paired_summary([1., 2., 3.], 3)
    assert result["mean"] == 2.
    assert result["mcse"] == pytest.approx(1/np.sqrt(3))
    # Student-t(2) has CDF 1/2+t/(2*sqrt(2+t^2)); this is an independent quantile.
    quantile = np.sqrt(1.805/.0975)
    assert result["ci95"] == pytest.approx([2-quantile/np.sqrt(3), 2+quantile/np.sqrt(3)], abs=1e-9)
    incomplete = pilot.paired_summary([1., 2.], 3)
    assert incomplete["mean"] == 1.5 and incomplete["mcse"] is None and incomplete["ci95"] is None
    empty = pilot.paired_summary([], 3)
    assert empty["mean"] is None and not empty["complete"]
    assert not any("p_value" in key or "holm" in key for key in result)


def _prepared_run(pilot, config, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(pilot, "guard_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(pilot, "source_identity", lambda: {"fixture_source": "fixed"})
    output = tmp_path / "run"
    args = SimpleNamespace(config=config_path, output=output, shard_count=2)
    assert pilot.init_run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    return args, manifest


def test_manifest_exclusive_and_resume_bound_to_config_and_source(pilot, config, tmp_path, monkeypatch):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    before = (args.output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        pilot.init_run(args)
    assert pilot.read_manifest(args.config, args.output)[1] == manifest
    monkeypatch.setattr(pilot, "source_identity", lambda: {"fixture_source": "changed"})
    with pytest.raises(ValueError, match="changed"):
        pilot.read_manifest(args.config, args.output)
    assert (args.output / "manifest.json").read_bytes() == before
    monkeypatch.setattr(pilot, "source_identity", lambda: {"fixture_source": "fixed"})
    args.config.write_text(json.dumps({**config, "rho": .4}))
    with pytest.raises(ValueError, match="changed"):
        pilot.read_manifest(args.config, args.output)


@pytest.mark.parametrize("key,value", [("schema_version", True), ("shard_count", 0),
                                       ("shard_count", 999),
                                       ("expected_records", 999)])
def test_manifest_execution_header_cannot_be_changed(pilot, config, tmp_path, monkeypatch, key, value):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    manifest[key] = value
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="execution header"):
        pilot.read_manifest(args.config, args.output)


def test_manifest_digest_and_record_request_bind_operational_payload(pilot, config, tmp_path, monkeypatch):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    request = pilot.expected_request(manifest["cases"][0], "full", config, manifest)
    assert request["manifest_sha256"] == manifest["manifest_sha256"]
    manifest["created_utc"] = "changed"
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical payload digest"):
        pilot.read_manifest(args.config, args.output)


def _fake_terminal(pilot, config, args, manifest, case, slot, monkeypatch):
    terminal = pilot.record_path(args.output, case, slot)
    attempt = terminal.parent / "attempts" / "fixture"
    attempt.mkdir(parents=True)
    request = pilot.expected_request(case, slot, config, manifest)
    pilot.write_json(attempt / "request.json", request)
    class Fake:
        provenance_ = {"fake_test_only": True}
        def fit_predict(self, training, query, *, seed):
            assert slot == "full"
            assert not hasattr(training, "eval_tau")
            assert seed == case["seed"]
            return np.ones(len(query), dtype=np.float64)
        def fit_predict_sample(self, sample, query, *, seed):
            assert slot != "full" and not hasattr(sample, "rct_train") and not hasattr(sample, "obs_train")
            assert sample.source == ("rct" if slot == "rct_only" else "obs")
            assert sample.n == (case["n_rct"] if slot == "rct_only" else config["n_obs"])
            assert seed == case["seed"]
            return np.ones(len(query), dtype=np.float64)
    monkeypatch.setattr(pilot, "_make_estimator", lambda method, rho: Fake())
    monkeypatch.setattr(pilot, "_check_estimator", lambda estimator, slot, rho: None)
    assert pilot.fit_worker(SimpleNamespace(config=args.config, output=args.output, attempt=attempt)) == 0
    record = json.loads((attempt / "worker.json").read_text())
    record.update(attempt=str(attempt.relative_to(args.output)),
                  artifact_sha256={p.name: pilot.file_hash(p) for p in attempt.iterdir() if p.is_file()})
    pilot.write_json(terminal, record)
    return terminal, attempt, record


@pytest.mark.parametrize("slot,trial_labels,obs_labels", [("full", 8, 12), ("rct_only", 8, 0), ("obs_start", 0, 12)])
def test_worker_boundary_label_budget_saved_predictions_and_metric(pilot, config, tmp_path, monkeypatch,
                                                                  slot, trial_labels, obs_labels):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    case = manifest["cases"][0]
    terminal, attempt, record = _fake_terminal(pilot, config, args, manifest, case, slot, monkeypatch)
    assert record["label_budget"]["consumed_rct_labels"] == trial_labels
    assert record["label_budget"]["consumed_obs_labels"] == obs_labels
    assert record["label_budget"]["supplied_unique_rct_labels"] == trial_labels
    assert record["label_budget"]["supplied_unique_obs_labels"] == obs_labels
    assert record["label_budget"]["additional_tuning_labels"] == 0
    if slot != "full":
        excluded_prefix = "obs_" if slot == "rct_only" else "rct_"
        assert not any(key.startswith(excluded_prefix) for key in record["method_argument_array_identity"])
        assert record["label_budget"]["tune_alias_train_indices"] == []
    observed = np.load(attempt / "observed_inputs.npz", allow_pickle=False)
    assert "eval_tau" not in observed.files
    truth = np.load(attempt / "evaluation.npz", allow_pickle=False)["eval_tau"]
    assert record["mse"] == pytest.approx(float(np.mean((1-truth)**2)))
    assert pilot.verify_record(terminal, args.output, case, slot, config, manifest) == record
    args.resume = True
    before = terminal.read_bytes()
    assert pilot.execute_fit(args, args.output, case, slot, config, manifest) == record
    assert terminal.read_bytes() == before
    data = bytearray((attempt / "prediction.npy").read_bytes())
    data[-1] ^= 1
    (attempt / "prediction.npy").write_bytes(data)
    with pytest.raises(ValueError, match="artifacts changed"):
        pilot.verify_record(terminal, args.output, case, slot, config, manifest)


def test_analysis_requires_exact_terminal_union_and_has_no_favorable_imputation(pilot, config, tmp_path, monkeypatch):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    args.analysis_output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="coverage differs"):
        pilot.analyze(args)
    assert not args.analysis_output.exists()


def test_timeout_terminates_entire_group_and_retains_terminal_failure(pilot, config, tmp_path, monkeypatch):
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    args.resume = False
    signals = []
    class Process:
        pid, returncode, calls = 12345, None, 0
        def wait(self, *, timeout):
            self.calls += 1
            if self.calls == 1:
                raise pilot.subprocess.TimeoutExpired("fixture", timeout)
            self.returncode = -9
            return self.returncode
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(pilot.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(pilot.time, "sleep", lambda seconds: None)
    case = manifest["cases"][0]
    record = pilot.execute_fit(args, args.output, case, "full", config, manifest)
    assert record["status"] == "timeout" and record["error_type"] == "FitTimeLimitExceeded"
    assert record["process_returncode"] == -9 and record["process_group_reaped"]
    assert [sig for _, sig in signals] == [pilot.signal.SIGTERM, pilot.signal.SIGKILL]
    assert "mse" not in record
    attempt = args.output / record["attempt"]
    assert (attempt / "stdout.log").exists() and (attempt / "request.json").exists()


def test_complete_analysis_pairs_all_seeds_and_checks_access_invariances(pilot, config, tmp_path, monkeypatch):
    config.update(n_obs=4, rct_budgets=[2, 4], n_eval=3, deltas=[0, 1], seeds=[11, 19])
    args, manifest = _prepared_run(pilot, config, tmp_path, monkeypatch)
    for case in manifest["cases"]:
        for slot in pilot.SLOTS:
            _fake_terminal(pilot, config, args, manifest, case, slot, monkeypatch)
    args.analysis_output = tmp_path / "analysis"
    assert pilot.analyze(args) == 0
    summary = json.loads((args.analysis_output / "summary.json").read_text())
    assert summary["expected_records"] == summary["success"] == 24
    assert summary["missing"] == summary["failed"] == 0
    for row in summary["conditions"]:
        for contrast in row["paired"].values():
            assert contrast["mean"] == 0 and contrast["successful_pairs"] == 2
            assert contrast["ci95"] == [0., 0.]
    for invariant in summary["invariances"].values():
        assert invariant["all_groups_checked"]
    assert "no p-values" in summary["inference"]
