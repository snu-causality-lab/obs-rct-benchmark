"""Observed projections, immutable attempts, and bounded worker failures.

All estimators here are mocks. No native adapter or scientific fit is run.
The process-group regression launches only sleeping Python processes.
"""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_native_panel.py"
spec = importlib.util.spec_from_file_location("native_panel_execution_test_subject", SCRIPT)
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def test_resolved_condition_contract_survives_json_manifest_roundtrip(monkeypatch):
    data_spec = importlib.util.spec_from_file_location("panel_resolved_test_data", SCRIPT.with_name("native_panel_data.py"))
    data = importlib.util.module_from_spec(data_spec)
    data_spec.loader.exec_module(data)
    config = data.load_config(SCRIPT.parents[1] / "configs/native_panel.toml")
    monkeypatch.setattr(panel, "_data", lambda: data)
    resolved = panel.resolved_conditions(config)
    assert len(resolved) == 12
    assert json.loads(json.dumps(resolved)) == resolved
    assert {row["native_factory_config"]["d"] for row in resolved} == {1, 5, 25}
    assert all(row["target_and_access"]["compatibility_tuning_unique_outcomes"] == 0 for row in resolved)


def test_preflight_identity_and_approved_caps_cannot_be_fabricated(tmp_path, monkeypatch):
    resources = {"num_cpus": 1, "num_threads": 1, "memory_gib": 8,
                 "default_fit_timeout_seconds": 1800,
                 "method_fit_timeout_seconds": {"method": 1800}}
    proof = {"execution_contract_passed": True, "comparative_fits": 0, "approved_resources": resources,
             "profiles": {"method": {"status": "success", "metrics": None, "process_group_reaped": True}}}
    location = tmp_path / "results/native_panel_preflight/summary.json"
    location.parent.mkdir(parents=True)
    location.write_text(json.dumps(proof))
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    config = {"methods": ["method"], "resources": dict(resources,
        status="preflight_reviewed", preflight_manifest_sha256=panel.file_hash(location))}
    panel.verify_preflight_contract(config)
    changed = deepcopy(config)
    changed["resources"]["memory_gib"] = 4
    with pytest.raises(ValueError, match="authorize"):
        panel.verify_preflight_contract(changed)
    proof["profiles"]["method"]["status"] = "failure"
    location.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="identity"):
        panel.verify_preflight_contract(config)
    config["resources"]["preflight_manifest_sha256"] = panel.file_hash(location)
    with pytest.raises(ValueError, match="authorize"):
        panel.verify_preflight_contract(config)
    # Only the explicitly reviewed native timeout can be authorized as a
    # budget limitation. An arbitrary failed adapter cannot use that label.
    proof["profiles"] = {"integrative_r_quadratic": dict(
        method="integrative_r_quadratic", status="failure", metrics=None,
        native_status="timeout", process_group_reaped=True)}
    config["methods"] = ["integrative_r_quadratic"]
    location.write_text(json.dumps(proof))
    config["resources"]["preflight_manifest_sha256"] = panel.file_hash(location)
    panel.verify_preflight_contract(config)
    proof["profiles"]["integrative_r_quadratic"]["native_status"] = "native_error"
    location.write_text(json.dumps(proof))
    config["resources"]["preflight_manifest_sha256"] = panel.file_hash(location)
    with pytest.raises(ValueError, match="authorize"):
        panel.verify_preflight_contract(config)


def arrays():
    x = np.arange(16., dtype=np.float64).reshape(8, 2)
    return {"obs_X": x + 100., "obs_T": np.tile([0., 1.], 4), "obs_Y": np.arange(8.) + 1000.,
            "rct_X": x, "rct_T": np.tile([0., 1.], 4), "rct_Y": np.arange(8.),
            "rct_known_propensity": np.full(8, .5), "eval_X": x[:3] - 20.,
            "eval_tau": np.full(3, 2.)}


def make_manifest(mode="campaign", methods=("rct_t_extra_trees",)):
    config = {"methods": list(methods), "method_options": {}, "resources": {
        "default_fit_timeout_seconds": 1, "method_fit_timeout_seconds": {},
        "memory_gib": 1, "num_cpus": 1, "num_threads": 1,
        "status": "preflight_reviewed", "preflight_manifest_sha256": "a" * 64}}
    case = {"id": "case1", "scenario_key": "structural:M0", "n_rct": 8, "seed": 3}
    return {"config": config, "mode": mode, "cases": [case], "methods": list(methods),
            "manifest_sha256": "frozen", "bindings": {
                "source_files_sha256": {"driver": "bound"},
                "runtime_inventory": {method: {"bound": True} for method in methods},
                "runtime_map": {"methods": {method: {"python": sys.executable, "env": {}}
                                             for method in methods}}}}


@pytest.fixture
def fake_data(monkeypatch):
    result = SimpleNamespace(
        generate_case=lambda config, case, profile=False: arrays(),
        case_method_seed=lambda case, method: 17,
        case_metadata=lambda config, case, profile=False: {"generated_rct": 16, "profile": profile},
    )
    monkeypatch.setattr(panel, "_data", lambda: result)
    monkeypatch.setattr(panel, "_checker", lambda: SimpleNamespace(
        SCORE_RELATIVE_TOLERANCE=1e-12, SCORE_ABSOLUTE_TOLERANCE=1e-12,
        verify_case=lambda inputs, scenario: {"passed": True, "errors": []},
        independent_truth=lambda inputs, scenario: np.full(len(inputs), 2.),
        score=panel.metrics,
    ))
    return result


@pytest.mark.parametrize("method,expected", [
    ("rct_causal_forest", panel.RCT_KEYS + ("eval_X",)),
    ("rct_t_extra_trees", panel.RCT_KEYS + ("eval_X",)),
    ("rct_dr_extra_trees", panel.RCT_KEYS + ("eval_X",)),
    ("obs_t_extra_trees", panel.OBS_KEYS + ("eval_X",)),
    ("naive_pooled_t_extra_trees", panel.OBSERVED_KEYS),
    ("cornet_official", panel.OBSERVED_KEYS),
    ("datafusionppi", panel.OBSERVED_KEYS),
    ("integrative_r_linear", panel.OBSERVED_KEYS),
    ("integrative_r_quadratic", panel.OBSERVED_KEYS),
    ("causal_icm_fixed", panel.OBSERVED_KEYS),
    ("causal_icm_rct_only", panel.RCT_KEYS + ("eval_X",)),
])
def test_every_panel_method_has_exact_observed_projection(method, expected):
    assert panel.projected_keys(method) == expected
    assert "eval_tau" not in panel.projected_keys(method)


@pytest.mark.parametrize("method,included", [
    ("rct_t_extra_trees", "rct"), ("rct_dr_extra_trees", "rct"),
    ("obs_t_extra_trees", "obs"),
])
def test_compatibility_placeholders_only_alias_included_observations(method, included):
    original = arrays()
    supplied = {key: original[key] for key in panel.projected_keys(method)}
    training, entry = panel.make_method_arguments(supplied, method)
    assert entry == "training_view"
    assert np.array_equal(training.obs_train.Y, original[included + "_Y"])
    assert np.array_equal(training.rct_train.Y, original[included + "_Y"])
    assert np.array_equal(training.rct_tune.Y, original[included + "_Y"][:2])
    assert not hasattr(training, "eval_tau")
    assert panel.label_budget(supplied, method)["additional_tuning_outcomes"] == 0


@pytest.mark.parametrize("method", ["rct_causal_forest", "causal_icm_rct_only"])
def test_source_restricted_entries_receive_one_sample_and_no_training_view(method):
    original = arrays()
    supplied = {key: original[key] for key in panel.projected_keys(method)}
    sample, entry = panel.make_method_arguments(supplied, method)
    assert entry == "sample" and sample.source == "rct"
    assert not hasattr(sample, "obs_train") and not hasattr(sample, "rct_tune")
    assert panel.label_budget(supplied, method)["supplied_unique_obs_outcomes"] == 0


def test_worker_rejects_even_an_unused_extra_truth_array():
    supplied = {key: arrays()[key] for key in panel.projected_keys("rct_t_extra_trees")}
    supplied["eval_tau"] = np.zeros(3)
    with pytest.raises(ValueError, match="projection"):
        panel.make_method_arguments(supplied, "rct_t_extra_trees")


def test_profile_never_persists_truth_and_campaign_keeps_it_separate(tmp_path, fake_data):
    for mode in ("profile", "campaign"):
        output = tmp_path / mode
        (output / "cases").mkdir(parents=True)
        manifest = make_manifest(mode)
        case = manifest["cases"][0]
        path = panel.prepare_case(output, case, manifest)
        observed = panel.load_arrays(path / "observed.npz")
        assert set(observed) == set(panel.OBSERVED_KEYS)
        assert "eval_tau" not in observed
        assert (path / "evaluation.npz").exists() == (mode == "campaign")
        if mode == "campaign":
            assert set(panel.load_arrays(path / "evaluation.npz")) == {"eval_tau"}
        else:
            assert panel.read_json(path / "identity.json")["evaluation_arrays"] == {}


def test_case_resume_rejects_incomplete_extra_and_changed_artifacts(tmp_path, fake_data):
    (tmp_path / "cases").mkdir()
    manifest = make_manifest()
    case = manifest["cases"][0]
    path = panel.prepare_case(tmp_path, case, manifest)
    previous = (path / "identity.json").read_bytes()
    panel.prepare_case(tmp_path, case, manifest)
    assert previous == (path / "identity.json").read_bytes()
    (path / "unbound").write_text("unbound")
    with pytest.raises(ValueError, match="union"):
        panel.prepare_case(tmp_path, case, manifest)
    (path / "unbound").unlink()
    (path / "observed.npz").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="bytes"):
        panel.prepare_case(tmp_path, case, manifest)


def test_json_publication_never_overwrites_and_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "record.json"
    panel.write_json(path, {"complete": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        panel.write_json(path, {"complete": False})
    assert path.read_bytes() == original
    path.write_text('{"status":"success","status":"failure"}')
    with pytest.raises(ValueError, match="duplicate"):
        panel.read_json(path)
    path.write_text('{"incomplete":')
    with pytest.raises(json.JSONDecodeError):
        panel.read_json(path)


def test_runtime_map_rejects_unbound_methods_reserved_overrides_and_relative_python(tmp_path):
    path = tmp_path / "runtimes.json"
    runtime = {"schema_version": 1, "methods": {"m": {"python": sys.executable, "env": {}}}}
    path.write_text(json.dumps(runtime))
    assert panel.load_runtimes(path, ["m"]) == runtime
    with pytest.raises(ValueError, match="exactly"):
        panel.load_runtimes(path, ["m", "missing"])
    runtime["methods"]["m"]["env"] = {"HOME": "/replacement"}
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="reserved"):
        panel.load_runtimes(path, ["m"])
    runtime["methods"]["m"] = {"python": "python", "env": {}}
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="absolute"):
        panel.load_runtimes(path, ["m"])


def test_provisional_caps_cannot_initialize_production_and_allocation_is_checked(monkeypatch):
    config = make_manifest()["config"]
    config["resources"]["status"] = "provisional_pending_preflight"
    with pytest.raises(ValueError, match="final caps"):
        panel.validate_resources(config, production=True)
    panel.validate_resources(config, production=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1023")
    with pytest.raises(ValueError, match="RAM"):
        panel.guard_allocation(config)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    panel.guard_allocation(config)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="one CPU"):
        panel.guard_allocation(config)


def mock_worker_process(monkeypatch, *, fail=False, timeout=False):
    class Process:
        pid = 123456789
        returncode = None
        def __init__(self, command, **kwargs):
            request_path = Path(command[-1])
            request = panel.read_json(request_path)
            assert "eval_tau" not in panel.load_arrays(request_path.parent / "observed_inputs.npz")
            if timeout:
                return
            worker = {"status": "failure" if fail else "success",
                      "request_sha256": panel.digest(request), "fit_predict_seconds": .01,
                      "method_entry": "training_view", "label_budget": panel.label_budget(
                          panel.load_arrays(request_path.parent / "observed_inputs.npz"), request["method"])}
            if not fail:
                prediction = np.array([1., 2., 3.])
                np.save(request_path.parent / "prediction.npy", prediction, allow_pickle=False)
                worker["prediction_array_identity"] = panel.array_identity({"prediction": prediction})
            else:
                worker["error_type"] = "MockFailure"
            panel.write_json(request_path.parent / "worker.json", worker)
        def wait(self, timeout=None):
            if timeout == 1 and self.returncode is None and timeout_mode:
                raise subprocess.TimeoutExpired("mock", timeout)
            self.returncode = 1 if fail else 0
            return self.returncode
    timeout_mode = timeout
    monkeypatch.setattr(panel.subprocess, "Popen", Process)
    monkeypatch.setattr(panel, "stop_process_group", lambda process: True)
    monkeypatch.setattr(panel, "enable_subreaper", lambda: None)
    return Process


@pytest.mark.parametrize("mode", ["profile", "campaign"])
def test_fresh_mock_fit_retains_projection_predictions_and_separate_scores(tmp_path, monkeypatch, fake_data, mode):
    mock_worker_process(monkeypatch)
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest(mode)
    case = manifest["cases"][0]
    row = panel.execute_fit(tmp_path, case, manifest["methods"][0], manifest, resume=False)
    assert row["status"] == "success"
    assert row["label_budget"]["supplied_unique_obs_outcomes"] == 0
    assert row["metrics"] == ({"cate_mse": 2 / 3, "root_pehe": np.sqrt(2 / 3)} if mode == "campaign" else None)
    assert panel.verify_record(tmp_path, case, manifest["methods"][0], manifest) == row


@pytest.mark.parametrize("failure_kind", ["failure", "timeout"])
def test_failure_and_timeout_are_terminal_resume_never_refits(tmp_path, monkeypatch, fake_data, failure_kind):
    mock_worker_process(monkeypatch, fail=failure_kind == "failure", timeout=failure_kind == "timeout")
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest()
    case, method = manifest["cases"][0], manifest["methods"][0]
    row = panel.execute_fit(tmp_path, case, method, manifest, resume=False)
    assert row["status"] == failure_kind and row["metrics"] is None
    path = panel.record_path(tmp_path, case, method)
    original = path.read_bytes()
    monkeypatch.setattr(panel.subprocess, "Popen", lambda *a, **k: pytest.fail("terminal fit retried"))
    assert panel.execute_fit(tmp_path, case, method, manifest, resume=True) == row
    assert original == path.read_bytes()


@pytest.mark.parametrize("mode", ["profile", "campaign"])
@pytest.mark.parametrize("late_status", ["failure", "success"])
def test_timeout_retains_late_worker_metadata_without_rescuing_prediction(
        tmp_path, monkeypatch, fake_data, mode, late_status):
    """SIGTERM cleanup can publish worker.json after wait() has timed out."""
    mock_worker_process(monkeypatch, timeout=True)
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest(mode)
    case, method = manifest["cases"][0], manifest["methods"][0]
    late_worker = {}

    def publish_during_cleanup(process):
        request_path = next((tmp_path / "records").rglob("request.json"))
        request = panel.read_json(request_path)
        supplied = panel.load_arrays(request_path.parent / "observed_inputs.npz")
        prediction = np.array([1., 2., 3.])
        np.save(request_path.parent / "prediction.npy", prediction, allow_pickle=False)
        late_worker.update(
            status=late_status, request_sha256=panel.digest(request),
            method_entry="training_view", label_budget=panel.label_budget(supplied, method),
            fit_predict_seconds=1.01, worker_seconds=1.02,
            self_usage={"maxrss_kib": 1024, "user_cpu_seconds": .8},
            children_usage={"maxrss_kib": 2048, "user_cpu_seconds": .1},
            cgroup_memory={"available": True, "counters": {"memory.peak": "4096"}},
            prediction_array_identity=panel.array_identity({"prediction": prediction}),
            provenance_={"status": "success" if late_status == "success" else "interrupted"},
            last_run_provenance_={"native_source": "mock", "status": "interrupted"},
            attempt_provenance_={"native_seed": 17},
        )
        if late_status == "failure":
            late_worker["error_type"] = "KeyboardInterrupt"
        panel.write_json(request_path.parent / "worker.json", late_worker)
        process.returncode = 0 if late_status == "success" else 1
        return True

    monkeypatch.setattr(panel, "stop_process_group", publish_during_cleanup)
    row = panel.execute_fit(tmp_path, case, method, manifest, resume=False)
    assert row["status"] == "timeout"
    assert row["error_type"] == "FitTimeLimitExceeded"
    assert row["metrics"] is None
    assert row["process_group_reaped"] is True
    assert row["late_worker_status"] == late_status
    assert row.get("late_worker_error_type") == late_worker.get("error_type")
    for field in ("method_entry", "label_budget", "fit_predict_seconds", "worker_seconds",
                  "self_usage", "children_usage", "cgroup_memory", "prediction_array_identity",
                  "provenance_", "last_run_provenance_", "attempt_provenance_"):
        assert row[field] == late_worker[field]
    assert panel.verify_record(tmp_path, case, method, manifest) == row
    terminal = panel.record_path(tmp_path, case, method)
    original = terminal.read_bytes()
    monkeypatch.setattr(panel.subprocess, "Popen", lambda *a, **k: pytest.fail("timed-out fit retried"))
    assert panel.execute_fit(tmp_path, case, method, manifest, resume=True) == row
    assert terminal.read_bytes() == original


def test_fit_replay_rejects_score_edits_extra_files_and_native_worker_drift(tmp_path, monkeypatch, fake_data):
    mock_worker_process(monkeypatch)
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest()
    case, method = manifest["cases"][0], manifest["methods"][0]
    row = panel.execute_fit(tmp_path, case, method, manifest, resume=False)
    path = panel.record_path(tmp_path, case, method)
    original = path.read_bytes()
    row["metrics"]["cate_mse"] = 99.
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="metrics disagree"):
        panel.verify_record(tmp_path, case, method, manifest)
    path.write_bytes(original)
    row = panel.read_json(path)
    extra = tmp_path / row["attempt"] / "unbound"
    extra.write_text("unbound")
    with pytest.raises(ValueError, match="unbound"):
        panel.verify_record(tmp_path, case, method, manifest)
    extra.unlink()
    row["fit_predict_seconds"] = 999.
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="worker fields"):
        panel.verify_record(tmp_path, case, method, manifest)


def test_collection_rejects_missing_extra_and_live_writer(tmp_path, monkeypatch, fake_data):
    mock_worker_process(monkeypatch)
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest()
    monkeypatch.setattr(panel, "read_manifest", lambda *a, **k: manifest)
    with pytest.raises(ValueError, match="missing"):
        panel.verified_records(tmp_path)
    case, method = manifest["cases"][0], manifest["methods"][0]
    panel.execute_fit(tmp_path, case, method, manifest, resume=False)
    config, records = panel.verified_records(tmp_path)
    assert config == manifest["config"] and len(records) == 1
    lock = panel.record_path(tmp_path, case, method).parent / "running.lock"
    lock.mkdir()
    with pytest.raises(ValueError, match="locks"):
        panel.verified_records(tmp_path)
    lock.rmdir()
    extra = tmp_path / "records" / "extra"
    extra.mkdir()
    (extra / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="extra"):
        panel.verified_records(tmp_path)


def test_collection_requires_the_independent_target_and_score_checks(tmp_path, monkeypatch, fake_data):
    mock_worker_process(monkeypatch)
    (tmp_path / "cases").mkdir()
    (tmp_path / "records").mkdir()
    manifest = make_manifest()
    monkeypatch.setattr(panel, "read_manifest", lambda *a, **k: manifest)
    panel.execute_fit(tmp_path, manifest["cases"][0], manifest["methods"][0], manifest, resume=False)
    monkeypatch.setattr(panel, "_checker", lambda: SimpleNamespace(
        SCORE_RELATIVE_TOLERANCE=1e-12, SCORE_ABSOLUTE_TOLERANCE=1e-12,
        verify_case=lambda inputs, scenario: {"passed": False, "errors": ["independent target differs"]}))
    with pytest.raises(ValueError, match="query/target"):
        panel.verified_records(tmp_path)
    monkeypatch.setattr(panel, "_checker", lambda: SimpleNamespace(
        SCORE_RELATIVE_TOLERANCE=1e-12, SCORE_ABSOLUTE_TOLERANCE=1e-12,
        verify_case=lambda inputs, scenario: {"passed": True, "errors": []},
        independent_truth=lambda inputs, scenario: np.full(len(inputs), 99.),
        score=panel.metrics,
    ))
    with pytest.raises(ValueError, match="prediction/score"):
        panel.verified_records(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and process groups")
def test_timeout_kills_and_reaps_descendants_after_worker_exit(tmp_path):
    panel.enable_subreaper()
    child_pid = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)")
    process = subprocess.Popen([sys.executable, "-c", code, str(child_pid)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid.exists()
        descendant = int(child_pid.read_text())
        assert panel.stop_process_group(process)
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
