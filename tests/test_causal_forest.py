"""Mocked process and information-boundary checks, not native fitting evidence."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from obs_rct_benchmark import causal_forest as adapter
from obs_rct_benchmark.data import SourceSample, TrainingView


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample():
    x = np.column_stack((np.arange(12.0), np.arange(12.0)**2))
    return SourceSample(x, np.arange(12) % 2, x[:, 0] + 3, "rct", "rct_train",
                        known_propensity=np.linspace(.25, .75, 12))


@pytest.fixture
def installation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    files = {"r-package/grf/DESCRIPTION": "Package: grf\nVersion: 2.6.1\n",
             "core/prediction.cpp": "independent fixture source\n",
             "NEWS.md": "noncritical source fixture\n"}
    for name, value in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    hashes = {name: _hash(source / name) for name in files}
    monkeypatch.setattr(adapter, "GRF_SOURCE_FILES_SHA256", {
        name: value for name, value in hashes.items() if name != "NEWS.md"})
    pinned_hashes = dict(hashes)
    monkeypatch.setattr(adapter, "_pinned_source_inventory", lambda root: (dict(pinned_hashes), {}))
    library = tmp_path / "library"
    package = library / "grf"
    for name, value in {"DESCRIPTION": "Package: grf\nVersion: 2.6.1\n",
                        "NAMESPACE": "useDynLib(grf)\n", "R/grf.rdb": "R bytecode",
                        "R/grf.rdx": "R index", "libs/grf.so": "native code"}.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    dependencies = {}
    for name in ("Rcpp", "RcppEigen"):
        where = library / name
        where.mkdir(parents=True)
        description = where / "DESCRIPTION"
        description.write_text(f"Package: {name}\nVersion: 1.0\n")
        dependencies[name] = {"version": "1.0", "description_sha256": _hash(description),
                              "library_root": str(library)}
    executable = tmp_path / "Rscript"
    executable.write_text("mock Rscript executable\n")
    manifest = {
        "schema_version": 1, "repo_revision": adapter.GRF_REVISION,
        "repo_tree": "a" * 40, "source_files_sha256": dict(hashes),
        "installed_package": {"name": "grf", "version": "2.6.1", "files_sha256": {
            path.relative_to(package).as_posix(): _hash(path)
            for path in package.rglob("*") if path.is_file()}},
        "rscript_sha256": _hash(executable), "r_version": "R version test",
        "dependencies": dependencies, "source_root": str(source),
        "library_root": str(library), "build_command": ["R", "CMD", "INSTALL", "grf"],
    }
    manifest_path = tmp_path / "build.json"
    manifest_path.write_text(json.dumps(manifest))
    for name, value in {"GRF_SOURCE": source, "GRF_LIBRARY": library,
                        "GRF_BUILD_MANIFEST": manifest_path}.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(adapter, "_rscript", lambda: executable)
    state = {"revision": adapter.GRF_REVISION, "tree": manifest["repo_tree"], "changes": ""}

    def git(root, *args):
        assert root == source
        if args == ("rev-parse", "--show-toplevel"):
            return str(source)
        if args == ("rev-parse", "HEAD"):
            return state["revision"]
        if args == ("rev-parse", "HEAD^{tree}"):
            return state["tree"]
        if args[0] == "status":
            return state["changes"]
        if args == ("ls-files", "-z"):
            return "\0".join(sorted(files)) + "\0"
        raise AssertionError(args)

    monkeypatch.setattr(adapter, "_git", git)
    runtime = ["R\tR version test", f"namespace\tgrf\t2.6.1\t{package}",
               f"dll\tgrf\t{package / 'libs/grf.so'}"]
    for name in ("grf", "Rcpp", "RcppEigen"):
        runtime.append(f"package\t{name}\t{'2.6.1' if name == 'grf' else '1.0'}\t{library/name}")
    return {"source": source, "package": package, "library": library,
            "executable": executable, "manifest_path": manifest_path,
            "manifest": manifest, "state": state, "runtime": "\n".join(runtime) + "\n"}


@pytest.fixture
def fake_r(installation, monkeypatch):
    calls = []
    real_run = subprocess.run

    def run(command, **kwargs):
        # NumPy may independently probe CPU capabilities while importing its
        # assertion helpers. Only intercept the selected native R executable.
        if (not isinstance(command, (tuple, list)) or not command
                or str(command[0]) != str(installation["executable"])):
            return real_run(command, **kwargs)
        assert command[1] == "--vanilla"
        if command[2] == "-e":
            return subprocess.CompletedProcess(command, 0, stdout=installation["runtime"], stderr="")
        work = Path(command[4])
        nr, nq, d = map(int, command[6:9])
        arrays = {
            "X": np.fromfile(work / "rct_X.bin", dtype="<f8").reshape(nr, d),
            "T": np.fromfile(work / "rct_T.bin", dtype="<f8"),
            "Y": np.fromfile(work / "rct_Y.bin", dtype="<f8"),
            "propensity": np.fromfile(work / "rct_propensity.bin", dtype="<f8"),
            "query": np.fromfile(work / "query_X.bin", dtype="<f8").reshape(nq, d),
        }
        # Deliberately simple synthetic output; native identity has a separate oracle.
        (arrays["query"][:, 0] + arrays["Y"].sum()).astype("<f8").tofile(work / "prediction.bin")
        (work / "runtime_before.tsv").write_text(installation["runtime"])
        (work / "runtime.tsv").write_text(installation["runtime"])
        kwargs["stdout"].write(b"native diagnostic only\n")
        calls.append({"command": command, "kwargs": kwargs, "arrays": arrays,
                      "input_files": sorted(p.name for p in work.glob("*.bin"))})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    return calls, run


def test_constructor_is_lightweight_and_validates_declared_counts(monkeypatch):
    monkeypatch.setattr(adapter, "_rscript", lambda: pytest.fail("constructor inspected R"))
    assert adapter.RCTCausalForest().num_trees == 2000
    for options in ({"num_trees": True}, {"num_trees": 0}, {"num_trees": 7},
                    {"num_trees": 6}, {"num_threads": 0}, {"num_threads": 1.5},
                    {"timeout_seconds": False}):
        with pytest.raises(ValueError):
            adapter.RCTCausalForest(**options)


def test_pinned_git_blobs_reject_hidden_file_and_link_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "source.R"
    path.write_bytes(b"pinned source\n")
    link = source / "alias.R"
    link.symlink_to("source.R")
    entries = []
    for name, mode, raw in (("source.R", "100644", path.read_bytes()),
                            ("alias.R", "120000", b"source.R")):
        object_id = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        entries.append(f"{mode} blob {object_id}\t{name}")

    def git(root, *args):
        assert root == source and args == ("ls-tree", "-rz", adapter.GRF_REVISION)
        return "\0".join(entries) + "\0"

    monkeypatch.setattr(adapter, "_git", git)
    files, links = adapter._pinned_source_inventory(source)
    assert files == {"source.R": _hash(path), "alias.R": _hash(path)}
    assert links == {"alias.R": "source.R"}
    path.write_bytes(b"modified despite a clean status\n")
    with pytest.raises(RuntimeError, match="pinned Git blob"):
        adapter._pinned_source_inventory(source)
    path.write_bytes(b"pinned source\n")
    (source / "copy.R").write_bytes(path.read_bytes())
    link.unlink()
    link.symlink_to("copy.R")
    with pytest.raises(RuntimeError, match="pinned Git blob"):
        adapter._pinned_source_inventory(source)


def test_optional_installation_fails_closed_without_starting_r(monkeypatch):
    monkeypatch.delenv("GRF_SOURCE", raising=False)
    monkeypatch.setattr(adapter, "_rscript", lambda: pytest.fail("unconfigured R invoked"))
    assert not adapter.causal_forest_provenance()["verified"]
    with pytest.raises(RuntimeError):
        adapter.RCTCausalForest().fit_predict_sample(_sample(), [[0, 1]], seed=2)


@pytest.mark.parametrize("failure", ["revision", "tree", "dirty", "source", "installed",
                                     "extra", "manifest", "executable", "symlink", "dependency",
                                     "rewritten_source_record", "hidden_noncritical_source"])
def test_source_install_and_build_record_are_all_bound(installation, fake_r, failure):
    data = installation
    if failure == "revision":
        data["state"]["revision"] = "b" * 40
    elif failure == "tree":
        data["state"]["tree"] = "b" * 40
    elif failure == "dirty":
        data["state"]["changes"] = " M core/prediction.cpp"
    elif failure == "source":
        (data["source"] / "core/prediction.cpp").write_text("changed")
    elif failure == "installed":
        (data["package"] / "libs/grf.so").write_text("changed DLL")
    elif failure == "extra":
        (data["package"] / "R/extra.R").write_text("injected")
    elif failure == "manifest":
        data["manifest"]["source_root"] = str(data["source"].parent)
        data["manifest_path"].write_text(json.dumps(data["manifest"]))
    elif failure == "executable":
        data["executable"].write_text("changed executable")
    elif failure == "dependency":
        (data["library"] / "Rcpp/DESCRIPTION").write_text("changed dependency")
    elif failure == "rewritten_source_record":
        source = data["source"] / "core/prediction.cpp"
        source.write_text("modified source and matching untrusted build metadata")
        data["manifest"]["source_files_sha256"]["core/prediction.cpp"] = _hash(source)
        data["manifest_path"].write_text(json.dumps(data["manifest"]))
    elif failure == "hidden_noncritical_source":
        source = data["source"] / "NEWS.md"
        source.write_text("modified with a clean status and updated build record")
        data["manifest"]["source_files_sha256"]["NEWS.md"] = _hash(source)
        data["manifest_path"].write_text(json.dumps(data["manifest"]))
    else:
        target = data["package"] / "libs/grf.so"
        target.unlink()
        target.symlink_to(data["executable"])
    assert not adapter.causal_forest_provenance()["verified"]


def test_runtime_checks_actual_grf_namespace_dll_and_dependencies(installation, fake_r, monkeypatch):
    assert adapter.causal_forest_provenance()["verified"]
    _, run = fake_r
    original = installation["runtime"]
    for wrong in (original.replace(str(installation["package"] / "libs/grf.so"), "/wrong/grf.so"),
                  original.replace("namespace\tgrf\t2.6.1", "namespace\tgrf\t0.0"),
                  original.replace("package\tRcpp\t1.0", "package\tRcpp\t9.0")):
        installation["runtime"] = wrong
        assert not adapter.causal_forest_provenance()["verified"]
    installation["runtime"] = original
    encoded = json.dumps(adapter.causal_forest_provenance())
    assert str(installation["source"].parent) not in encoded


@pytest.mark.parametrize("failure", ["schema", "package", "dependency_files", "dependency_library"])
def test_malformed_build_records_report_unavailable(installation, fake_r, failure):
    manifest = installation["manifest"]
    if failure == "schema":
        manifest["schema_version"] = 1.0
    elif failure == "package":
        manifest["installed_package"] = []
    elif failure == "dependency_files":
        manifest["dependencies"]["Rcpp"]["files_sha256"] = []
    else:
        del manifest["dependencies"]["Rcpp"]["library_root"]
    installation["manifest_path"].write_text(json.dumps(manifest))
    assert not adapter.causal_forest_provenance()["verified"]


def test_single_sample_and_wrapper_have_identical_inputs_without_other_roles(fake_r):
    calls, _ = fake_r
    sample, query = _sample(), np.array([[1., 2.], [3., 4.]])
    # Missing excluded attributes are hostile: any inspection raises AttributeError.
    training = object.__new__(TrainingView)
    object.__setattr__(training, "rct_train", sample)
    direct, wrapped = adapter.RCTCausalForest(), adapter.RCTCausalForest()
    first = direct.fit_predict_sample(sample, query, seed=2**32-1)
    second = wrapped.fit_predict(training, query, seed=2**32-1)
    np.testing.assert_array_equal(first, second)
    assert direct.provenance_ == wrapped.provenance_
    assert len(calls) == 2
    assert calls[0]["command"][5] == str((2**32-1) % (2**31-1))
    for name in calls[0]["arrays"]:
        np.testing.assert_array_equal(calls[0]["arrays"][name], calls[1]["arrays"][name])
    np.testing.assert_array_equal(calls[0]["arrays"]["propensity"], sample.known_propensity)
    assert set(calls[0]["input_files"]) == {"rct_X.bin", "rct_T.bin", "rct_Y.bin",
                                           "rct_propensity.bin", "query_X.bin", "prediction.bin"}
    assert direct.provenance_["known_propensity"] == "supplied_as_W.hat"
    assert direct.last_run_provenance_["status"] == "success"


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_bad_seeds_are_rejected_before_optional_runtime(monkeypatch, seed):
    monkeypatch.setattr(adapter, "_rscript", lambda: pytest.fail("R inspected"))
    with pytest.raises(ValueError):
        adapter.RCTCausalForest().fit_predict_sample(_sample(), [[0, 1]], seed=seed)


def test_wrong_type_source_role_and_query_are_rejected_before_runtime(monkeypatch):
    monkeypatch.setattr(adapter, "_rscript", lambda: pytest.fail("R inspected"))
    method, sample = adapter.RCTCausalForest(), _sample()
    with pytest.raises(TypeError):
        method.fit_predict(object(), [[0, 1]], seed=1)
    with pytest.raises(TypeError):
        method.fit_predict_sample(object(), [[0, 1]], seed=1)
    for invalid in (replace(sample, role="rct_tune"),
                    SourceSample(sample.X, sample.T, sample.Y, "obs", "obs_train")):
        with pytest.raises(ValueError):
            method.fit_predict_sample(invalid, [[0, 1]], seed=1)
    for query in ([], [1, 2], [[1]], [[np.nan, 1]], np.empty((0, 2))):
        with pytest.raises(ValueError):
            method.fit_predict_sample(sample, query, seed=1)
    with pytest.raises(ValueError):
        method.fit_predict_sample(replace(sample, T=np.ones(sample.n)), [[0, 1]], seed=1)


@pytest.mark.parametrize("failure", ["exit", "nonfinite", "truncated", "runtime", "tamper", "timeout"])
def test_failed_native_attempts_keep_terminal_status_and_clear_old_predictions(
        installation, fake_r, monkeypatch, failure):
    _, run = fake_r
    method = adapter.RCTCausalForest()
    method.fit_predict_sample(_sample(), [[0, 1]], seed=1)

    def fail(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        result = run(command, **kwargs)
        work = Path(command[4])
        if failure == "exit":
            return subprocess.CompletedProcess(command, 9)
        if failure in {"nonfinite", "truncated"}:
            np.asarray([np.nan] if failure == "nonfinite" else [], dtype="<f8").tofile(work / "prediction.bin")
        elif failure == "runtime":
            (work / "runtime.tsv").write_text(installation["runtime"].replace("2.6.1", "0.0"))
        elif failure == "tamper":
            (installation["package"] / "R/grf.rdb").write_text("changed during fit")
        return result

    monkeypatch.setattr(adapter.subprocess, "run", fail)
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        method.fit_predict_sample(_sample(), [[0, 1]], seed=1)
    assert method.provenance_ is None
    assert method.last_run_provenance_["status"] in {"native_error", "validation_error", "timeout"}
    assert method.last_run_diagnostics_ is not None


def test_query_rows_are_only_serialized_for_prediction(fake_r):
    calls, _ = fake_r
    method, sample = adapter.RCTCausalForest(), _sample()
    first = method.fit_predict_sample(sample, [[2, 4]], seed=0)
    second = method.fit_predict_sample(sample, [[-9, -8], [2, 4], [2, 4]], seed=0)
    np.testing.assert_array_equal(first, second[1:2])
    assert second[1] == second[2]
    for name in ("X", "T", "Y", "propensity"):
        np.testing.assert_array_equal(calls[0]["arrays"][name], calls[1]["arrays"][name])


def test_private_artifact_retention_keeps_complete_attempt_without_public_paths(
        tmp_path, fake_r, monkeypatch):
    root = tmp_path / "attempts"
    root.mkdir()
    monkeypatch.setenv("GRF_ARTIFACT_ROOT", str(root))
    method = adapter.RCTCausalForest()
    method.fit_predict_sample(_sample(), [[0, 1]], seed=1)
    work = Path(method.last_run_artifact_path_)
    assert work.parent == root
    assert {"bridge.R", "prediction.bin", "runtime.tsv", "runtime_before.tsv",
            "stdout.log", "stderr.log", "rct_X.bin", "rct_T.bin", "rct_Y.bin",
            "rct_propensity.bin", "query_X.bin"}.issubset(p.name for p in work.iterdir())
    assert str(root) not in json.dumps(method.provenance_)
    _, run = fake_r

    def failure(command, **kwargs):
        run(command, **kwargs)
        kwargs["stderr"].write(b"full failure diagnostics\n")
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(adapter.subprocess, "run", failure)
    with pytest.raises(RuntimeError):
        method.fit_predict_sample(_sample(), [[0, 1]], seed=1)
    failed = Path(method.last_run_artifact_path_)
    assert failed != work and failed.parent == root
    assert "full failure diagnostics" in (failed / "stderr.log").read_text()
    assert (failed / "rct_Y.bin").is_file()


def test_preparation_failures_get_terminal_status(installation, fake_r, monkeypatch):
    monkeypatch.setenv("GRF_ARTIFACT_ROOT", str(installation["source"] / "missing-directory"))
    method = adapter.RCTCausalForest()
    with pytest.raises(FileNotFoundError):
        method.fit_predict_sample(_sample(), [[0, 1]], seed=1)
    assert method.provenance_ is None
    assert method.last_run_provenance_["status"] == "preparation_error"
    assert method.last_run_provenance_["error_type"] == "FileNotFoundError"
