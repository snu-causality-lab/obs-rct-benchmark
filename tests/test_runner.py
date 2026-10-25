"""Output coverage, interruption/resume invariants, and estimator isolation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark import runner
from obs_rct_benchmark.data import TrainingView
from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, load_config, run_benchmark


def config():
    return BenchmarkConfig(
        scenarios=(ScenarioConfig("structural", "M0", {
            "n_obs": 24, "n_rct_train": 20, "n_rct_tune": 10,
            "n_rct_test": 10, "n_eval": 16,
        }),), seeds=(3, 5), methods=("rct_t_ridge", "obs_t_ridge"),
    )


def records(path):
    return [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]


class SpyMethod:
    calls = []
    def fit_predict(self, training, eval_X, *, seed):
        assert isinstance(training, TrainingView)
        assert not hasattr(training, "truth")
        assert not hasattr(training, "eval_tau")
        assert not hasattr(training, "rct_test")
        assert eval_X.ndim == 2
        self.calls.append(seed)
        return np.zeros(len(eval_X))


def test_runner_excludes_truth_and_records_independent_method_seeds(tmp_path, monkeypatch):
    SpyMethod.calls = []
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    output = tmp_path / "run"
    summary = run_benchmark(config(), output)
    rows = records(output)
    assert summary["status"] == "complete"
    assert len(rows) == 4
    assert len(set(SpyMethod.calls)) == 4
    assert all(row["method_seed"] != row["seed"] for row in rows)
    assert all(row["metrics"]["root_pehe"] ** 2 == pytest.approx(row["metrics"]["cate_mse"])
               for row in rows)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["config_hash"] == config().config_hash
    assert manifest["source"]["package_file_sha256"]
    resolved = manifest["resolved_scenarios"]["structural:M0"]
    assert resolved["generator_version"]
    assert resolved["config"]["n_eval"] == 16
    assert "signal_seed" in resolved["config"]
    assert "| Mean CATE MSE | Mean root-PEHE |" in (output / "README.md").read_text()
    artifact_text = "".join(p.read_text() for p in output.iterdir())
    assert str(tmp_path) not in artifact_text
    assert "NaN" not in artifact_text
    assert set(p.name for p in output.iterdir()) == {
        "records.jsonl", "run_manifest.json", "summary.json", "README.md"}


def test_interrupt_then_resume_reproduces_uninterrupted_records(tmp_path, monkeypatch):
    class InterruptAfterOne(SpyMethod):
        count = 0
        def fit_predict(self, *args, **kwargs):
            self.__class__.count += 1
            if self.__class__.count == 2:
                raise KeyboardInterrupt
            return super().fit_predict(*args, **kwargs)
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: InterruptAfterOne())
    interrupted = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(config(), interrupted)
    assert len(records(interrupted)) == 1
    assert not (interrupted / ".run.lock").exists()
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    resumed = run_benchmark(config(), interrupted, resume=True)
    clean = tmp_path / "clean"
    run_benchmark(config(), clean)
    assert resumed["status"] == "complete"
    assert records(interrupted) == records(clean)
    previous = (interrupted / "records.jsonl").read_bytes()
    run_benchmark(config(), interrupted, resume=True)
    assert (interrupted / "records.jsonl").read_bytes() == previous


def test_failures_are_terminal_sanitized_and_not_scores(tmp_path, monkeypatch):
    class Failure:
        def fit_predict(self, *args, **kwargs):
            raise RuntimeError("secret /private/runtime/path must not be exported")
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: Failure())
    output = tmp_path / "failure"
    result = run_benchmark(config(), output)
    assert result["status"] == "incomplete"
    assert result["coverage"]["failed"] == 4
    assert all(row["metrics"] is None for row in records(output))
    assert "private/runtime" not in (output / "records.jsonl").read_text()
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    result = run_benchmark(config(), output, resume=True)
    assert result["coverage"]["failed"] == 4
    assert not result["paired_comparisons"]["obs_t_ridge"]["overall"]["eligible_for_comparison"]


def test_resume_rejects_duplicate_partial_records_changed_config_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    output = tmp_path / "run"
    run_benchmark(config(), output)
    path = output / "records.jsonl"
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        run_benchmark(config(), output, resume=True)
    path.write_text(original.rstrip("\n"))
    with pytest.raises(ValueError, match="incomplete"):
        run_benchmark(config(), output, resume=True)
    path.write_text(original)
    changed = BenchmarkConfig(scenarios=config().scenarios, seeds=(3,), methods=config().methods)
    with pytest.raises(ValueError, match="config or source"):
        run_benchmark(changed, output, resume=True)
    monkeypatch.setattr(runner, "source_provenance", lambda: {"aggregate_sha256": "modified"})
    with pytest.raises(ValueError, match="config or source"):
        run_benchmark(config(), output, resume=True)


def test_external_source_change_is_bound_to_resume_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    snapshot = {"datafusionppi": {"availability": "installed",
                                 "package_file_sha256": {"api.py": "original"}}}
    monkeypatch.setattr(runner, "external_method_provenance", lambda methods: snapshot)
    output = tmp_path / "external"
    run_benchmark(config(), output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["external_methods"] == snapshot
    snapshot["datafusionppi"]["package_file_sha256"]["api.py"] = "modified"
    with pytest.raises(ValueError, match="config or source"):
        run_benchmark(config(), output, resume=True)


def test_all_shipped_configs_validate_and_smoke_covers_all_31_assets():
    root = Path(__file__).parents[1]
    for name in ("quickstart.toml", "smoke.toml", "official_smoke.toml"):
        loaded = load_config(root / "configs" / name)
        assert loaded.seeds == (0, 1)
    smoke = load_config(root / "configs" / "smoke.toml")
    assert len(smoke.scenarios) == 31
    assert len(set(scenario.key for scenario in smoke.scenarios)) == 31


def test_default_payload_preserves_legacy_config_identity():
    original = config()
    legacy = {
        "schema_version": 1,
        "run": {"seeds": [3, 5], "methods": ["rct_t_ridge", "obs_t_ridge"],
                "reference_method": "rct_t_ridge", "scope": "smoke/development-inspected"},
        "scenarios": [{"kind": "structural", "id": "M0", "parameters": {
            "n_obs": 24, "n_rct_train": 20, "n_rct_tune": 10,
            "n_rct_test": 10, "n_eval": 16,
        }}],
        "method_options": {},
    }
    expected_hash = hashlib.sha256(json.dumps(
        legacy, ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    ).encode()).hexdigest()
    assert original.payload() == legacy
    assert original.config_hash == expected_hash
    assert original.scenarios[0].key == "structural:M0"
    explicit_smoke = BenchmarkConfig(
        scenarios=original.scenarios, seeds=original.seeds, methods=original.methods,
        purpose="smoke",
    )
    assert explicit_smoke.config_hash == original.config_hash


def test_labelled_budget_grid_loads_runs_and_resumes(tmp_path, monkeypatch):
    path = tmp_path / "grid.toml"
    path.write_text('''schema_version = 1
[run]
seeds = [3, 5]
methods = ["rct_t_ridge", "obs_t_ridge"]
reference_method = "rct_t_ridge"
purpose = "reference"
scope = "fixed budget grid for testing"
[[scenarios]]
kind = "structural"
id = "M0"
label = "rct20"
[scenarios.parameters]
n_obs = 24
n_rct_train = 20
n_rct_tune = 10
n_rct_test = 10
n_eval = 16
[[scenarios]]
kind = "structural"
id = "M0"
label = "rct40"
[scenarios.parameters]
n_obs = 24
n_rct_train = 40
n_rct_tune = 10
n_rct_test = 10
n_eval = 16
''')
    loaded = load_config(path)
    assert [s.key for s in loaded.scenarios] == [
        "structural:M0@rct20", "structural:M0@rct40"]
    assert loaded.payload()["run"]["purpose"] == "reference"
    monkeypatch.setattr(runner, "make_method", lambda *a, **kw: SpyMethod())
    output = tmp_path / "grid"
    result = run_benchmark(loaded, output)
    assert result["coverage"] == {
        "expected": 8, "recorded": 8, "successful": 8, "failed": 0, "missing": 0}
    assert result["scenario_definitions"]["structural:M0@rct40"]["sample_sizes"]["n_rct_train"] == 40
    assert len({row["record_id"] for row in records(output)}) == 8
    previous = (output / "records.jsonl").read_bytes()
    run_benchmark(loaded, output, resume=True)
    assert (output / "records.jsonl").read_bytes() == previous
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["purpose"] == "reference"
    assert "development smoke" not in manifest["interpretation"]
    report = (output / "README.md").read_text()
    assert "Purpose: reference" in report
    assert "development example" not in report
    assert "structural:M0@rct20" in report and "structural:M0@rct40" in report
    assert "RCT train + tune" in report
    assert "| `structural:M0@rct20` | 24 | 20 | 10 | 30 |" in report
    assert "| `structural:M0@rct40` | 24 | 40 | 10 | 50 |" in report


def test_labels_do_not_change_samples_or_method_randomization():
    original = config().scenarios[0]
    labelled = ScenarioConfig(original.kind, original.id, original.parameters,
                              label="same_setting")
    plain_problem = runner._generate(original, 3)
    labelled_problem = runner._generate(labelled, 3)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        plain, named = getattr(plain_problem, role), getattr(labelled_problem, role)
        for field in ("X", "T", "Y"):
            np.testing.assert_array_equal(getattr(plain, field), getattr(named, field))
    np.testing.assert_array_equal(plain_problem.eval_X, labelled_problem.eval_X)
    np.testing.assert_array_equal(plain_problem.eval_tau, labelled_problem.eval_tau)
    assert runner.method_seed(original.key, 3, "rct_t_ridge") == runner.method_seed(
        labelled.key, 3, "rct_t_ridge")


@pytest.mark.parametrize("label", ["", "has space", "has@sign", "a/b", "a|b", "한글", 7])
def test_invalid_scenario_labels_are_rejected(label):
    with pytest.raises(ValueError, match="scenario label"):
        ScenarioConfig("structural", "M0", label=label)


def test_duplicate_grid_keys_and_invalid_run_purpose_are_rejected():
    first = ScenarioConfig("structural", "M0", label="same")
    changed_budget = ScenarioConfig("structural", "M0", {"n_rct_train": 80}, label="same")
    with pytest.raises(ValueError, match="unique"):
        BenchmarkConfig(scenarios=(first, changed_budget))
    with pytest.raises(ValueError, match="purpose"):
        BenchmarkConfig(scenarios=(first,), purpose="confirmed_winner")
    original = config()
    reference = BenchmarkConfig(
        scenarios=original.scenarios, seeds=original.seeds, methods=original.methods,
        purpose="reference",
    )
    assert reference.config_hash != original.config_hash


def test_scenario_report_exposes_paired_effects_and_withholds_failed_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_generate", lambda *args: SimpleNamespace(
        training_view=lambda: None, eval_X=np.zeros((4, 3)),
        eval_tau=np.zeros(4), metadata={},
    ))

    class Constant:
        def __init__(self, value):
            self.value = value
        def fit_predict(self, training, eval_X, *, seed):
            return np.full(len(eval_X), self.value)

    monkeypatch.setattr(runner, "make_method", lambda name, **kw: Constant(
        1 if name == "rct_t_ridge" else 2))
    completed = tmp_path / "complete"
    run_benchmark(config(), completed)
    report = (completed / "README.md").read_text()
    # Independent hand calculation: 2^2 - 1^2 = 3 for each seed.
    assert "| `structural:M0` | `obs_t_ridge` | 3 | [3, 3] | 2 / 2 | complete |" in report
    assert "## Results by scenario" in report
    assert "## Aggregate across requested scenarios" in report
    assert "original outcome scale" in report
    assert "without multiplicity adjustment" in report

    class Failure:
        def fit_predict(self, *args, **kwargs):
            raise RuntimeError("test failure")

    monkeypatch.setattr(runner, "make_method", lambda name, **kw: (
        Constant(1) if name == "rct_t_ridge" else Failure()))
    failed = tmp_path / "failed"
    run_benchmark(config(), failed)
    report = (failed / "README.md").read_text()
    assert "| `structural:M0` | `obs_t_ridge` | unavailable | unavailable | 0 / 2 | 2 | 0 |" in report
    assert "unavailable | unavailable | 0 / 2 | withheld: failed or missing requested cells" in report
