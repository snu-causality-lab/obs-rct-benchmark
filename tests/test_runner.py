"""Output coverage, interruption/resume invariants, and estimator isolation."""

import json
from pathlib import Path

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
