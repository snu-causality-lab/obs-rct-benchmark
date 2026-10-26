"""Optional R bridge contracts; native admission is a separate server check."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from obs_rct_benchmark import integrative_r as adapter
from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView


def _training():
    def sample(source, role, offset):
        x = np.column_stack((np.arange(12.0) + offset,
                             np.arange(12.0)**2 + offset))
        return SourceSample(X=x, T=np.arange(12) % 2, Y=x[:, 0] + 3,
                            source=source, role=role)
    return TrainingView(
        obs_train=sample("obs", "obs_train", 10),
        rct_train=sample("rct", "rct_train", 0),
        rct_tune=sample("rct", "rct_tune", 100),
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    root.mkdir()
    hashes = {}
    for name in adapter.INTEGRATIVE_R_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture: " + name, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("INTEGRATIVE_R_SOURCE", str(root))
    monkeypatch.setattr(adapter, "INTEGRATIVE_R_FILES", hashes)
    state = {"revision": adapter.INTEGRATIVE_R_REVISION,
             "tracked": "", "all": "", "toplevel": str(root)}

    def git(command, **kwargs):
        assert kwargs["timeout"] == 15
        if command[3:] == ["rev-parse", "--show-toplevel"]:
            return state["toplevel"]
        if command[3:] == ["rev-parse", "HEAD"]:
            return state["revision"]
        if command[-1] == "--untracked-files=no":
            return state["tracked"]
        if command[-1] == "--untracked-files=all":
            return state["all"]
        raise AssertionError(command)

    monkeypatch.setattr(adapter.subprocess, "check_output", git)
    return root, state


@pytest.fixture
def fake_r(tmp_path, monkeypatch, checkout):
    root, _ = checkout
    executable = tmp_path / "Rscript"
    executable.write_text("mock Rscript", encoding="utf-8")
    monkeypatch.setattr(adapter, "_rscript", lambda: executable)
    records = ["R\tR version test"]
    for package in ("base", *adapter.INTEGRATIVE_R_PACKAGES):
        where = tmp_path / "library" / package
        where.mkdir(parents=True)
        (where / "DESCRIPTION").write_text(f"Package: {package}\nVersion: 1.0\n")
        records.append(f"package\t{package}\t1.0\t{where}")
    runtime = "\n".join(records) + "\n"
    calls = []

    def run(command, **kwargs):
        if command[2] == "-e":
            return subprocess.CompletedProcess(command, 0, stdout=runtime, stderr="")
        assert command[1] == "--vanilla"
        work = Path(command[4])
        nr, no, nq, d, p = map(int, command[8:13])
        x = np.fromfile(work / "train_X.bin", dtype="<f8").reshape(nr + no, d)
        query = np.fromfile(work / "query_X.bin", dtype="<f8").reshape(nq, d)
        expanded = adapter._basis_matrix(query, command[6])
        # Synthetic process output, not a scientific fitting oracle.
        beta = np.r_[1.0, np.arange(2, p+2, dtype=float), np.full(p, 1e6)]
        prediction = 1 + expanded @ np.arange(2, p+2, dtype=float)
        for name, values in {
            "prediction.bin": prediction, "tau_beta.bin": beta,
            "center.bin": np.zeros(p), "scale.bin": np.ones(p),
        }.items():
            np.asarray(values, dtype="<f8").tofile(work / name)
        (work / "sources.txt").write_text(
            "\n".join(str(root / name) for name in ("R/utils.R", "R/intRlearner.R")) + "\n")
        (work / "runtime.tsv").write_text(runtime)
        kwargs["stdout"].write(b"Parameter searching ...\nnot predictions\n")
        calls.append({"x": x, "t": np.fromfile(work / "train_T.bin", dtype="<f8"),
                      "y": np.fromfile(work / "train_Y.bin", dtype="<f8"),
                      "foldid": np.fromfile(work / "foldid.bin", dtype="<f8"),
                      "query": query, "command": command, "kwargs": kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    return calls, run


def test_source_is_optional_and_missing_source_fails_closed(monkeypatch):
    monkeypatch.delenv("INTEGRATIVE_R_SOURCE", raising=False)
    assert not adapter._source_provenance()["verified"]
    with pytest.raises(RuntimeError, match="INTEGRATIVE_R_SOURCE"):
        adapter.OfficialIntegrativeR().fit_predict(_training(), [[0, 1]], seed=1)


def test_full_problem_is_rejected_before_optional_runtime(monkeypatch):
    def forbidden():
        raise AssertionError("optional runtime was inspected")
    monkeypatch.setattr(adapter, "_rscript", forbidden)
    with pytest.raises(TypeError, match="TrainingView"):
        adapter.OfficialIntegrativeR().fit_predict(object(), [[0, 1]], seed=1)


@pytest.mark.parametrize("options", [
    {"basis": "flexible"}, {"k_folds": True}, {"k_folds": 2},
    {"timeout_seconds": 0}, {"timeout_seconds": 1.5},
])
def test_options_are_checked_without_r(options):
    with pytest.raises(ValueError):
        adapter.OfficialIntegrativeR(**options)


def test_clean_pinned_checkout_is_verified(checkout):
    root, _ = checkout
    evidence = adapter._source_provenance()
    assert evidence["verified"]
    assert evidence["package_file_sha256"] == adapter.INTEGRATIVE_R_FILES
    assert str(root) not in json.dumps(evidence)


@pytest.mark.parametrize("field,value", [
    ("revision", "0" * 40), ("tracked", " M R/utils.R"),
    ("all", "?? untracked.R"), ("toplevel", "/unrelated/repo"),
])
def test_changed_git_identity_is_rejected(checkout, field, value):
    _, state = checkout
    state[field] = value
    assert not adapter._source_provenance()["verified"]


def test_hash_rejects_changes_hidden_from_git_status(checkout):
    root, _ = checkout
    (root / "R/utils.R").write_text("modified source")
    assert not adapter._source_provenance()["verified"]


def test_added_ignored_r_source_is_rejected(checkout):
    root, _ = checkout
    (root / "R/extra.R").write_text("untracked and ignored source")
    evidence = adapter._source_provenance()
    assert not evidence["verified"]
    assert evidence["extra_source_files"] == ["R/extra.R"]


def test_source_symlink_is_rejected(checkout, tmp_path):
    root, _ = checkout
    outside = tmp_path / "outside.R"
    original = root / "R/utils.R"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    assert not adapter._source_provenance()["verified"]


def test_quadratic_basis_matches_explicit_monomial_order():
    # An independent fixed answer catches interaction ordering for d >= 4.
    actual = adapter._basis_matrix([[2, 3, 5, 7]], "quadratic")
    expected = [[2, 3, 5, 7, 4, 9, 25, 49, 6, 10, 14, 15, 21, 35]]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_native_cate_map_omits_confounding_coefficients():
    actual = adapter._native_prediction_map(
        [[3, 8], [5, 4]], [7, 2, -3, 1000, -1000],
        [1, 2], [2, 3], basis="linear",
    )
    # 7 + 2*(x1-1)/2 - 3*(x2-2)/3, independently expanded.
    np.testing.assert_allclose(actual, [3, 9], atol=1e-12, rtol=0)


def test_folds_are_reproducible_and_stratified_by_source_and_arm():
    training = _training()
    first = adapter._stratified_foldid(training, 3, 17)
    np.testing.assert_array_equal(first, adapter._stratified_foldid(training, 3, 17))
    for sample, folds in zip((training.rct_train, training.obs_train),
                             np.split(first, [training.rct_train.n])):
        for arm in (0, 1):
            assert set(folds[sample.T == arm]) == {1, 2, 3}


def test_inadequate_arms_fail_before_r():
    with pytest.raises(ValueError, match="each source and treatment arm"):
        adapter.OfficialIntegrativeR(k_folds=7).fit_predict(_training(), [[0, 1]], seed=1)


@pytest.mark.parametrize("query", [[[1, 2, 3]], [[1, np.nan]], [], [1, 2]])
def test_invalid_query_fails_before_r(query):
    with pytest.raises(ValueError):
        adapter.OfficialIntegrativeR().fit_predict(_training(), query, seed=1)


def test_float64_bridge_and_prediction_only_roles(fake_r):
    calls, _ = fake_r
    training = _training()
    altered = replace(
        training,
        rct_tune=replace(training.rct_tune, Y=training.rct_tune.Y + 100000),
        rct_train=replace(training.rct_train,
                          known_propensity=np.linspace(0.1, 0.9, training.rct_train.n)),
        target_reference=TargetCovariateSample([[9000, -9000]]),
    )
    query = np.array([[1.0000000000000002, 2], [3, 4]], dtype=np.float64)
    first_method = adapter.OfficialIntegrativeR(k_folds=3)
    first = first_method.fit_predict(training, query, seed=2**32-1)
    second = adapter.OfficialIntegrativeR(k_folds=3).fit_predict(altered, query, seed=2**32-1)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, 1 + 2*query[:, 0] + 3*query[:, 1], atol=1e-12, rtol=0)
    assert first.shape == (2,) and first.dtype == np.float64
    assert calls[0]["command"][5] == "1"  # R's bounded set.seed input.
    np.testing.assert_array_equal(calls[0]["x"][:12], training.rct_train.X)
    np.testing.assert_array_equal(calls[0]["x"][12:], training.obs_train.X)
    np.testing.assert_array_equal(calls[0]["query"], query)
    for key in ("x", "t", "y", "foldid", "query"):
        np.testing.assert_array_equal(calls[0][key], calls[1][key])
    assert first_method.provenance_["known_rct_propensity"] == "not_used_native_estimation"
    assert first_method.last_run_provenance_["status"] == "success"
    assert "not predictions" in first_method.last_run_diagnostics_["stdout"]
    assert str(Path(calls[0]["command"][3]).parent) not in json.dumps(first_method.provenance_)


def test_explicit_quadratic_native_recipe(fake_r):
    calls, _ = fake_r
    result = adapter.OfficialIntegrativeR(basis="quadratic", k_folds=3).fit_predict(
        _training(), [[2, 3]], seed=5,
    )
    # B(x) = (2,3,4,9,6), slopes = (2,3,4,5,6).
    np.testing.assert_allclose(result, [111], atol=1e-12, rtol=0)
    assert calls[0]["command"][6] == "quadratic"
    assert calls[0]["command"][12] == "5"


def test_runtime_provenance_is_sanitized(fake_r):
    evidence = adapter.integrative_r_provenance()
    assert evidence["verified"]
    assert evidence["runtime"]["availability"] == "available"
    assert set(evidence["runtime"]["namespace_packages"]) == {"base", *adapter.INTEGRATIVE_R_PACKAGES}
    encoded = json.dumps(evidence)
    assert "/Users/" not in encoded and "/private/" not in encoded
    assert "namespace_path_sha256" in encoded


@pytest.mark.skipif(not os.environ.get("INTEGRATIVE_R_RSCRIPT"), reason="external R runtime not configured")
def test_real_r_namespace_probe_includes_builtin_base():
    # This reproduces R's special base namespace behavior rather than relying
    # on the fake process. It imports namespaces but performs no model fitting.
    executable = adapter._rscript()
    completed = subprocess.run(
        [str(executable), "--vanilla", "-e", adapter._R_PROBE],
        capture_output=True, text=True, timeout=60, check=True,
    )
    runtime = adapter._runtime_provenance(completed.stdout, executable)
    assert "base" in runtime["namespace_packages"]
    assert len(runtime["namespace_packages"]["base"]["description_sha256"]) == 64


def test_runtime_parser_rejects_unexpected_stdout(fake_r, monkeypatch):
    _, run = fake_r

    def malformed(command, **kwargs):
        result = run(command, **kwargs)
        return subprocess.CompletedProcess(command, 0,
                                           stdout=result.stdout + "unexpected output\n", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", malformed)
    assert adapter.integrative_r_provenance()["runtime"]["availability"] == "unavailable"


@pytest.mark.parametrize("failure", ["exit", "nonfinite", "truncated", "wrong_map", "sources"])
def test_native_process_and_output_fail_closed(fake_r, monkeypatch, failure):
    _, run = fake_r

    def fail(command, **kwargs):
        result = run(command, **kwargs)
        work = Path(command[4])
        if failure == "exit":
            return subprocess.CompletedProcess(command, 9)
        if failure == "sources":
            (work / "sources.txt").write_text("/unverified/source.R\n")
        else:
            values = {"nonfinite": [np.nan], "truncated": [], "wrong_map": [999.]}
            np.asarray(values[failure], dtype="<f8").tofile(work / "prediction.bin")
        return result

    monkeypatch.setattr(adapter.subprocess, "run", fail)
    method = adapter.OfficialIntegrativeR(k_folds=3)
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        method.fit_predict(_training(), [[2, 3]], seed=1)
    assert method.last_run_provenance_["status"] == (
        "native_error" if failure == "exit" else "validation_error")
    assert method.last_run_provenance_["source"]["verified"]
    assert method.last_run_provenance_["basis"] == "linear"
    assert method.last_run_provenance_["bridge_sha256"]


def test_source_change_during_r_call_is_rejected(fake_r, checkout, monkeypatch):
    _, state = checkout
    _, run = fake_r

    def change(command, **kwargs):
        result = run(command, **kwargs)
        state["tracked"] = " M R/intRlearner.R"
        return result

    monkeypatch.setattr(adapter.subprocess, "run", change)
    with pytest.raises(RuntimeError, match="changed during fitting"):
        adapter.OfficialIntegrativeR(k_folds=3).fit_predict(_training(), [[2, 3]], seed=1)


def test_timeout_is_terminal(fake_r, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    method = adapter.OfficialIntegrativeR(k_folds=3)
    with pytest.raises(RuntimeError, match="exceeded timeout_seconds"):
        method.fit_predict(_training(), [[2, 3]], seed=1)
    assert method.last_run_provenance_["status"] == "timeout"
    assert method.last_run_provenance_["returncode"] is None
    assert method.last_run_provenance_["source"]["verified"]
    assert method.last_run_diagnostics_ == {"stdout": "", "stderr": ""}
