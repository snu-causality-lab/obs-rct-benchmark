"""Factory integration with synthetic files, input identity, and safe resume."""

import hashlib
import json

import numpy as np
import pytest

from obs_rct_benchmark import runner
from obs_rct_benchmark.data import TrainingView
from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, load_config, run_benchmark


SMALL = dict(n_obs=12, n_rct_train=8, n_rct_tune=4, n_rct_test=4, n_eval=12)
HIDDEN = tuple(f"x{i}" for i in range(2, 26))


class SpyMethod:
    def __init__(self, calls):
        self.calls = calls

    def fit_predict(self, training, eval_X, *, seed):
        assert isinstance(training, TrainingView)
        assert not hasattr(training, "truth")
        assert not hasattr(training, "rct_test")
        assert not hasattr(training, "population")
        self.calls.append(seed)
        return np.zeros(len(eval_X))


@pytest.fixture
def spy(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "make_method", lambda *args, **kwargs: SpyMethod(calls))
    return calls


def _rows(output):
    return [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]


def _write_ihdp_fixture(path, *, offset=0.0):
    """Eight invented rows in the two supported IHDP file schemas."""
    X = np.zeros((8, 25))
    X[:, 0] = np.arange(8)
    X[:, 1] = np.arange(8) % 2
    mu0 = 0.5 * X[:, 0] + X[:, 1] + offset
    mu1 = mu0 + 2 + 0.25 * X[:, 0] + X[:, 1]
    treatment = np.arange(8) % 2
    factual = np.full(8, 12345.0)
    counterfactual = -factual
    if path.suffix == ".csv":
        np.savetxt(path, np.column_stack((treatment, factual, counterfactual, mu0, mu1, X)),
                   delimiter=",")
    else:
        # Two replications exercise selecting a nonzero replication index.
        np.savez(path, x=np.stack((X, X), axis=-1),
                 t=np.column_stack((treatment, treatment)),
                 yf=np.column_stack((factual, factual)),
                 ycf=np.column_stack((counterfactual, counterfactual)),
                 mu0=np.column_stack((mu0 + 100, mu0)),
                 mu1=np.column_stack((mu1 + 100, mu1)))


def _empirical_config(*, replication=0):
    return BenchmarkConfig(
        scenarios=(ScenarioConfig("empirical", "ihdp", {
            **SMALL, "hidden_columns": HIDDEN, "visible_binning": (("x1", 2),),
            "replication": replication, "gamma": 2.0, "noise_std": 0.0,
        }),), seeds=(3, 5), methods=("rct_t_ridge",),
    )


def _write_config(path, *, replication):
    hidden = json.dumps(list(HIDDEN))
    budgets = "\n".join(f"{key} = {value}" for key, value in SMALL.items())
    path.write_text(
        "schema_version = 1\n"
        "[run]\nseeds = [3, 5]\nmethods = [\"rct_t_ridge\"]\n"
        "[[scenarios]]\nkind = \"empirical\"\nid = \"ihdp\"\n"
        "[scenarios.parameters]\n"
        f"hidden_columns = {hidden}\nvisible_binning = [[\"x1\", 2]]\n"
        f"replication = {replication}\ngamma = 2.0\nnoise_std = 0.0\n{budgets}\n",
        encoding="utf-8",
    )


def test_gamma_profiles_run_and_resume_without_repeating_cells(tmp_path, spy):
    config = BenchmarkConfig(
        scenarios=tuple(ScenarioConfig("gamma", profile, {
            **SMALL, "gamma": 3.0, "gamma_min": 1.0,
        }) for profile in ("constant", "logistic")),
        seeds=(3, 5), methods=("rct_t_ridge",),
    )
    output = tmp_path / "gamma"
    summary = run_benchmark(config, output)
    assert summary["status"] == "complete"
    assert len(spy) == 4
    rows = _rows(output)
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["generator_diagnostics"] == {
        "gamma_global_upper": 3.0, "gamma_global_upper_is_bound": True,
    } for row in rows)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["empirical_inputs"] == {}
    for profile in ("constant", "logistic"):
        resolved = manifest["resolved_scenarios"][f"gamma:{profile}"]
        assert resolved["generator_version"]
        assert resolved["config"]["gamma_profile"] == profile
        assert resolved["config"]["n_eval"] == SMALL["n_eval"]
    previous = (output / "records.jsonl").read_bytes()
    assert run_benchmark(config, output, resume=True)["status"] == "complete"
    assert len(spy) == 4
    assert (output / "records.jsonl").read_bytes() == previous


@pytest.mark.parametrize("suffix,replication,source_role", [
    (".npz", 1, "ihdp_train"), (".csv", 0, "ihdp_csv"),
])
def test_empirical_file_runner_binds_hash_and_normalizes_tuple_config(
    tmp_path, monkeypatch, spy, suffix, replication, source_role,
):
    source = tmp_path / f"nonpublic_input_name_2718{suffix}"
    _write_ihdp_fixture(source)
    monkeypatch.setenv("OBSRCT_IHDP_TRAIN", str(source))
    monkeypatch.delenv("OBSRCT_IHDP_TEST", raising=False)
    config = _empirical_config(replication=replication)
    config_path = tmp_path / "run.toml"
    _write_config(config_path, replication=replication)
    loaded = load_config(config_path)
    # Programmatic tuples and TOML lists must represent the same run identity.
    assert loaded.payload() == config.payload()
    assert loaded.config_hash == config.config_hash
    output = tmp_path / "empirical"
    assert run_benchmark(config, output)["status"] == "complete"
    assert len(spy) == 2
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["empirical_inputs"]["ihdp"] == {
        "dataset_id": "ihdp", "availability": "available",
        "files": {"ihdp_train": {"sha256": digest}}, "missing_roles": [],
    }
    controls = manifest["config"]["scenarios"][0]["parameters"]
    assert controls["hidden_columns"] == list(HIDDEN)
    assert controls["visible_binning"] == [["x1", 2]]
    for row in _rows(output):
        assert row["status"] == "ok"
        assert row["dataset_provenance"]["files"] == [{"role": source_role, "sha256": digest}]
        assert row["generator_diagnostics"]["has_hidden_ambiguity"] is True
        assert row["generator_diagnostics"]["visible_groups"] == 2
    previous = (output / "records.jsonl").read_bytes()
    assert run_benchmark(loaded, output, resume=True)["status"] == "complete"
    assert len(spy) == 2
    assert (output / "records.jsonl").read_bytes() == previous
    artifacts = "".join(path.read_text() for path in output.iterdir())
    assert str(tmp_path) not in artifacts
    assert source.name not in artifacts


def test_empirical_resume_rejects_changed_input_bytes(tmp_path, monkeypatch, spy):
    source = tmp_path / "synthetic.npz"
    _write_ihdp_fixture(source)
    monkeypatch.setenv("OBSRCT_IHDP_TRAIN", str(source))
    monkeypatch.delenv("OBSRCT_IHDP_TEST", raising=False)
    output = tmp_path / "run"
    config = _empirical_config(replication=1)
    assert run_benchmark(config, output)["status"] == "complete"
    previous = (output / "records.jsonl").read_bytes()
    _write_ihdp_fixture(source, offset=1.0)
    with pytest.raises(ValueError, match="config or source provenance"):
        run_benchmark(config, output, resume=True)
    assert len(spy) == 2
    assert (output / "records.jsonl").read_bytes() == previous


def test_missing_empirical_inputs_are_explicit_failed_cells(tmp_path, monkeypatch, spy):
    monkeypatch.delenv("OBSRCT_IHDP_TRAIN", raising=False)
    monkeypatch.delenv("OBSRCT_IHDP_TEST", raising=False)
    output = tmp_path / "missing"
    summary = run_benchmark(_empirical_config(), output)
    assert summary["status"] == "incomplete"
    assert summary["coverage"]["failed"] == 2
    assert spy == []
    rows = _rows(output)
    assert len(rows) == 2
    assert all(row["stage"] == "simulation" and row["metrics"] is None for row in rows)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["empirical_inputs"]["ihdp"]["availability"] == "unavailable"
    assert manifest["empirical_inputs"]["ihdp"]["missing_roles"] == ["ihdp_train"]


def test_input_replaced_between_seeds_cannot_produce_mixed_scores(tmp_path, monkeypatch, spy):
    source = tmp_path / "synthetic.npz"
    _write_ihdp_fixture(source)
    monkeypatch.setenv("OBSRCT_IHDP_TRAIN", str(source))
    monkeypatch.delenv("OBSRCT_IHDP_TEST", raising=False)
    original_generate = runner._generate
    generated = []

    def replacing_generate(scenario, seed):
        problem = original_generate(scenario, seed)
        generated.append(seed)
        if len(generated) == 1:
            # First problem retains its original file fingerprint; the next
            # generation reads changed bytes and must fail before model fitting.
            _write_ihdp_fixture(source, offset=1.0)
        return problem

    monkeypatch.setattr(runner, "_generate", replacing_generate)
    output = tmp_path / "changed_midrun"
    summary = run_benchmark(_empirical_config(replication=1), output)
    assert summary["status"] == "incomplete"
    assert len(spy) == 1
    rows = _rows(output)
    assert [(row["seed"], row["status"]) for row in rows] == [(3, "ok"), (5, "failed")]
    assert rows[1]["stage"] == "simulation"
    assert rows[1]["error_type"] == "ValueError"
    assert rows[1]["metrics"] is None
    assert summary["coverage"]["failed"] == 1
