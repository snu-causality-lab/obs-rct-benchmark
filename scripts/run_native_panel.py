#!/usr/bin/env python3
"""Bound native-panel profiles and fits to immutable inputs and fresh workers.

Numerical commands require GSDS Slurm. Profiling records observed inputs,
predictions and costs only; evaluator truth is discarded before persistence.
Runtime paths and full diagnostics stay in the new server output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path("/shared/s2/lab05/sanghack/work")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "RCPP_PARALLEL_NUM_THREADS")
OBS_KEYS = ("obs_X", "obs_T", "obs_Y")
RCT_KEYS = ("rct_X", "rct_T", "rct_Y", "rct_known_propensity")
OBSERVED_KEYS = OBS_KEYS + RCT_KEYS + ("eval_X",)
TERMINAL = ("success", "failure", "timeout")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject,
                      object_pairs_hook=unique)


def write_json(path, value):
    """Publish a complete, fsynced file without replacing an existing record."""
    path = Path(path)
    contents = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise ValueError("unresolved lock; verify the previous writer before recovery") from exc
    try:
        yield
    finally:
        path.rmdir()


def _data():
    return importlib.import_module("native_panel_data")


def _checker():
    return importlib.import_module("verify_native_panel")


def guard_path(path):
    raw = os.environ.get("GSDS_WORK_ROOT")
    if (sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID")
            or raw is None or Path(raw).resolve() != WORK_ROOT):
        raise RuntimeError("execution requires GSDS Slurm and the declared shared work root")
    target = Path(path).expanduser().resolve()
    if target == WORK_ROOT or not target.is_relative_to(WORK_ROOT):
        raise ValueError("output must be below the declared shared work root")
    if any(os.environ.get(name) != "1" for name in THREADS[:4]):
        raise ValueError("all numerical thread limits must equal one")
    return target


def source_identity():
    paths = sorted((ROOT / "src/obs_rct_benchmark").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*native_panel*.py"))
    if not paths or not (ROOT / "scripts/native_panel_data.py").is_file():
        raise ValueError("native panel source checkout is incomplete")
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def load_runtimes(path, methods):
    raw = read_json(path)
    if (set(raw) != {"schema_version", "methods"} or raw["schema_version"] != 1
            or not isinstance(raw["methods"], dict)
            or set(raw["methods"]) != set(methods)):
        raise ValueError("runtime map must bind exactly the declared methods")
    for value in raw["methods"].values():
        if (not isinstance(value, dict) or set(value) != {"python", "env"}
                or not isinstance(value["python"], str)
                or not Path(value["python"]).is_absolute()
                or not Path(value["python"]).is_file()
                or not isinstance(value["env"], dict)):
            raise ValueError("runtime requires an absolute Python executable and env mapping")
        for key, item in value["env"].items():
            if (not isinstance(key, str) or not isinstance(item, str)
                    or key in {"HOME", "PYTHONPATH", "PYTHONHOME", "GSDS_WORK_ROOT"}
                    or key.startswith("SLURM_") or "\x00" in key + item
                    or "=" in key):
                raise ValueError("invalid or reserved runtime environment override")
    return raw


def worker_environment(runtime):
    environment = dict(os.environ)
    environment.update(runtime["env"])
    environment.update({name: "1" for name in THREADS})
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       PYTHONHASHSEED="0", PYTHONPATH=str(ROOT / "src"))
    return environment


def runtime_inventory(method):
    """Inspect installed identities without fitting a method."""
    from obs_rct_benchmark.runner import external_method_provenance, source_provenance
    actual = source_provenance()
    expected = {path.name: file_hash(path)
                for path in sorted((ROOT / "src/obs_rct_benchmark").glob("*.py"))}
    if actual["package_file_sha256"] != expected:
        raise ValueError("imported package differs from the driver checkout")
    distributions = []
    for distribution in importlib.metadata.distributions():
        distributions.append({"name": distribution.metadata.get("Name"),
                              "version": distribution.version,
                              "metadata_sha256": hashlib.sha256(
                                  (distribution.read_text("METADATA") or "").encode()).hexdigest()})
    distributions.sort(key=canonical)
    external = external_method_provenance((method,))
    for name, declaration in external.items():
        verified = declaration.get("verified", False)
        if name == "datafusionppi":
            verified = (declaration.get("matches_pinned_source") is True
                        and declaration.get("availability") == "installed"
                        and declaration.get("declared_git_revision") == declaration.get("expected_revision")
                        and declaration.get("import_resolution") == "verified_distribution")
        if not verified:
            raise ValueError("native/external method provenance is unavailable or unverified")
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": file_hash(Path(sys.executable).resolve()),
            "distributions": distributions, "package_source": actual,
            "external_methods": external}


def inspect_runtime(method, runtime):
    completed = subprocess.run(
        [runtime["python"], str(Path(__file__).resolve()), "_probe", "--method", method],
        env=worker_environment(runtime), capture_output=True, text=True,
        timeout=240, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("runtime inspection failed: " + completed.stderr[-1000:])
    result = json.loads(completed.stdout)
    if result["executable_sha256"] != file_hash(Path(runtime["python"]).resolve()):
        raise ValueError("runtime executable resolution changed")
    return result


def bindings(config_path, runtimes_path, protocol_path, config, *, methods=None):
    runtimes = load_runtimes(runtimes_path, config["methods"])
    selected = config["methods"] if methods is None else methods
    return {"source_files_sha256": source_identity(),
            "config_sha256": file_hash(config_path), "protocol_sha256": file_hash(protocol_path),
            "runtimes_sha256": file_hash(runtimes_path), "runtime_map": runtimes,
            "runtime_inventory": {method: inspect_runtime(method, runtimes["methods"][method])
                                  for method in selected}}


def timeout_seconds(config, method):
    resources = config["resources"]
    value = resources.get("method_fit_timeout_seconds", {}).get(
        method, resources["default_fit_timeout_seconds"])
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("fit timeout must be a positive integer")
    return value


def validate_resources(config, *, production):
    resources = config["resources"]
    if resources["num_cpus"] != 1 or resources["num_threads"] != 1:
        raise ValueError("native panel requires one CPU and one thread per task")
    if production and (resources["status"] != "preflight_reviewed"
                       or len(resources.get("preflight_manifest_sha256", "")) != 64):
        raise ValueError("production initialization requires final caps and preflight identity")
    for method in config["methods"]:
        timeout_seconds(config, method)


def guard_allocation(config):
    """Check requested allocation; kernel RAM enforcement is not established."""
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("request exactly one CPU per Slurm task")
    declared_mib = os.environ.get("SLURM_MEM_PER_NODE", "")
    if not declared_mib.isdigit() or int(declared_mib) != math.ceil(config["resources"]["memory_gib"] * 1024):
        raise ValueError("Slurm --mem allocation differs from the frozen RAM budget")


def _manifest_digest(manifest):
    return digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def resolved_conditions(config):
    resolved = [{"scenario_key": case["scenario_key"], "n_rct": case["n_rct"],
             "native_factory_config": asdict(_data().factory_config(config, case)),
             "target_and_access": _data().case_metadata(config, case)}
            for case in _data().cases(config) if case["seed"] == config["seeds"][0]]
    return json.loads(canonical(resolved))


def verify_preflight_contract(config):
    evidence = ROOT / "results/native_panel_preflight/summary.json"
    if file_hash(evidence) != config["resources"]["preflight_manifest_sha256"]:
        raise ValueError("published preflight evidence identity changed")
    proof = read_json(evidence)
    resources = {key: value for key, value in config["resources"].items()
                 if key not in {"status", "preflight_manifest_sha256"}}
    if (proof.get("execution_contract_passed") is not True or proof.get("comparative_fits") != 0
            or proof.get("approved_resources") != resources
            or set(proof.get("profiles", {})) != set(config["methods"])
            or any(row.get("method", method) != method
                   or row.get("metrics") is not None or not row.get("process_group_reaped")
                   or (row.get("status") != "success" and not (
                       row.get("status") == "failure" and row.get("native_status") == "timeout"
                       and method == "integrative_r_quadratic"))
                   for method, row in proof["profiles"].items())):
        raise ValueError("preflight does not authorize these resource caps and recipe rows")
    return proof


def init_run(args):
    config = _data().load_config(args.config)
    validate_resources(config, production=True)
    proof = verify_preflight_contract(config)
    output = guard_path(args.output)
    guard_allocation(config)
    cases = _data().cases(config)
    if (isinstance(args.shard_count, bool) or not 1 <= args.shard_count <= len(cases)):
        raise ValueError("invalid shard count")
    if args.shard_count != proof["execution_plan"]["shard_count"]:
        raise ValueError("shard count differs from the reviewed execution plan")
    identity = bindings(args.config, args.runtimes, args.protocol, config)
    from analyze_native_panel import analysis_contract
    resolved = resolved_conditions(config)
    manifest = {"schema_version": 1, "mode": "campaign", "config": config,
                "cases": cases, "methods": config["methods"], "shard_count": args.shard_count,
                "execution_plan": proof["execution_plan"],
                "resolved_conditions": resolved, "analysis_contract": analysis_contract(config),
                "expected_records": len(cases) * len(config["methods"]), "bindings": identity,
                "paths": {key: str(Path(getattr(args, key)).resolve())
                          for key in ("config", "runtimes", "protocol")},
                "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    (output / "records").mkdir()
    write_json(output / "manifest.json", manifest)
    print(canonical({"status": "initialized", "expected_records": manifest["expected_records"],
                     "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


def read_manifest(output, *, check_runtime=True):
    manifest = read_json(Path(output) / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("mode") not in {"campaign", "profile"}
            or manifest.get("manifest_sha256") != _manifest_digest(manifest)):
        raise ValueError("manifest identity changed")
    paths, config = manifest["paths"], manifest["config"]
    loaded = _data().load_config(paths["config"])
    if loaded != config or manifest["bindings"]["source_files_sha256"] != source_identity():
        raise ValueError("configuration or source changed")
    selected = manifest["methods"]
    if check_runtime:
        actual = bindings(paths["config"], paths["runtimes"], paths["protocol"], config,
                          methods=selected)
        if actual != manifest["bindings"]:
            raise ValueError("config, protocol, runtime or source identity changed")
    else:
        for field, name in (("config_sha256", "config"), ("protocol_sha256", "protocol"),
                            ("runtimes_sha256", "runtimes")):
            if file_hash(paths[name]) != manifest["bindings"][field]:
                raise ValueError("bound configuration, protocol or runtime map changed")
    if manifest["mode"] == "campaign":
        from analyze_native_panel import analysis_contract
        validate_resources(config, production=True)
        proof = verify_preflight_contract(config)
        if manifest["cases"] != _data().cases(config) or selected != config["methods"]:
            raise ValueError("campaign case or method union changed")
        if (manifest.get("resolved_conditions") != resolved_conditions(config)
                or manifest.get("analysis_contract") != analysis_contract(config)
                or manifest.get("execution_plan") != proof["execution_plan"]
                or manifest["shard_count"] != proof["execution_plan"]["shard_count"]):
            raise ValueError("resolved target/access or analysis contract changed")
    if manifest["expected_records"] != len(manifest["cases"]) * len(selected):
        raise ValueError("manifest expected record count changed")
    return manifest


def array_identity(arrays):
    import numpy as np
    return {name: {"shape": list(np.asarray(values).shape), "dtype": str(np.asarray(values).dtype),
                   "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
            for name, values in sorted(arrays.items())}


def load_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def prepare_case(output, case, manifest):
    import numpy as np
    path = Path(output) / "cases" / case["id"]
    if path.exists():
        verify_case_artifacts(path, case, manifest)
        return path
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        arrays = _data().generate_case(manifest["config"], case,
                                       profile=manifest["mode"] == "profile")
        observed = {key: arrays[key] for key in OBSERVED_KEYS}
        np.savez(temporary / "observed.npz", **observed)
        evaluation = {} if manifest["mode"] == "profile" else {"eval_tau": arrays["eval_tau"]}
        if evaluation:
            np.savez(temporary / "evaluation.npz", **evaluation)
        identity = {"case": case, "mode": manifest["mode"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "case_metadata": _data().case_metadata(
                        manifest["config"], case, profile=manifest["mode"] == "profile"),
                    "observed_arrays": array_identity(observed),
                    "evaluation_arrays": array_identity(evaluation),
                    "artifact_sha256": {p.name: file_hash(p) for p in temporary.iterdir()}}
        write_json(temporary / "identity.json", identity)
        os.rename(temporary, path)
    except BaseException:
        # Preserve every completed input artifact for inspection after interruption.
        raise
    return path


def verify_case_artifacts(path, case, manifest):
    path = Path(path)
    identity = read_json(path / "identity.json")
    if (identity.get("case") != case or identity.get("mode") != manifest["mode"]
            or identity.get("manifest_sha256") != manifest["manifest_sha256"]
            or identity.get("case_metadata") != _data().case_metadata(
                manifest["config"], case, profile=manifest["mode"] == "profile")):
        raise ValueError("case artifact identity changed")
    expected_names = {"observed.npz"} | ({"evaluation.npz"} if manifest["mode"] == "campaign" else set())
    if (set(identity["artifact_sha256"]) != expected_names
            or {p.name for p in path.iterdir()} != expected_names | {"identity.json"}):
        raise ValueError("case artifact union is incomplete or changed")
    if any(file_hash(path / name) != sha for name, sha in identity["artifact_sha256"].items()):
        raise ValueError("case artifact bytes changed")
    observed = load_arrays(path / "observed.npz")
    if set(observed) != set(OBSERVED_KEYS) or array_identity(observed) != identity["observed_arrays"]:
        raise ValueError("case observed arrays changed")
    if manifest["mode"] == "campaign":
        evaluation = load_arrays(path / "evaluation.npz")
        if set(evaluation) != {"eval_tau"} or array_identity(evaluation) != identity["evaluation_arrays"]:
            raise ValueError("case evaluation arrays changed")
    elif identity["evaluation_arrays"]:
        raise ValueError("profiles must not retain evaluator truth")
    return identity


def projected_keys(method):
    if method.startswith("obs_t_") or method == "causal_icm_obs_start":
        return OBS_KEYS + ("eval_X",)
    if method.startswith("rct_") or method == "causal_icm_rct_only":
        return RCT_KEYS + ("eval_X",)
    if method.startswith("naive_pooled_t_") or method in {"cornet_official", "datafusionppi", "integrative_r_linear",
                  "integrative_r_quadratic", "causal_icm_fixed"}:
        return OBSERVED_KEYS
    raise ValueError("method has no declared observed-only projection")


def make_method_arguments(arrays, method):
    """Build compatibility placeholders solely from the included observations."""
    from obs_rct_benchmark.data import SourceSample, TrainingView
    if set(arrays) != set(projected_keys(method)):
        raise ValueError("worker input projection differs from the method contract")
    included = "obs" if method.startswith("obs_t_") or method == "causal_icm_obs_start" else "rct"
    both = set(OBS_KEYS + RCT_KEYS) <= set(arrays)
    def sample(source, role, *, tune=False):
        origin = source if both else included
        indices = slice(0, 2) if tune else slice(None)
        propensity = (arrays["rct_known_propensity"][indices]
                      if source == "rct" and origin == "rct" else None)
        return SourceSample(arrays[origin + "_X"][indices], arrays[origin + "_T"][indices],
                            arrays[origin + "_Y"][indices], source, role, propensity)
    if method in {"rct_causal_forest", "causal_icm_rct_only", "causal_icm_obs_start"}:
        return sample(included, included + "_train"), "sample"
    return TrainingView(sample("obs", "obs_train"), sample("rct", "rct_train"),
                        sample("rct", "rct_tune", tune=True)), "training_view"


def label_budget(arrays, method):
    rct = len(arrays["rct_Y"]) if "rct_Y" in arrays else 0
    obs = len(arrays["obs_Y"]) if "obs_Y" in arrays else 0
    return {"supplied_unique_rct_outcomes": rct, "supplied_unique_obs_outcomes": obs,
            "consumed_rct_outcomes": rct, "consumed_obs_outcomes": obs,
            "additional_tuning_outcomes": 0,
            "compatibility_tune": ("absent_sample_entry" if method in {
                "rct_causal_forest", "causal_icm_rct_only", "causal_icm_obs_start"}
                else "first_two_included_rows_alias_unused"),
            "excluded_source_placeholders": "included_rows_relabelled_never_excluded_data"}


def process_usage(which):
    usage = resource.getrusage(which)
    return {"maxrss_kib": usage.ru_maxrss, "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "memory_scope": "separate process high-water mark; not simultaneous whole-job peak"}


def cgroup_memory():
    """Read current job/step cgroup counters; do not reset a shared peak."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split(":", 2)[2] for line in lines if line.startswith("0::"))
        base = Path("/sys/fs/cgroup") / relative.lstrip("/")
        values = {name: (base / name).read_text().strip()
                  for name in ("memory.current", "memory.peak", "memory.max") if (base / name).exists()}
        return {"available": bool(values), "cgroup_path_sha256": hashlib.sha256(str(base).encode()).hexdigest(),
                "counters": values, "scope": "enclosing job/step cgroup; peak may include earlier fits"}
    except (OSError, StopIteration):
        return {"available": False, "scope": "cgroup v2 counters unavailable"}


def fit_worker(request_path):
    import numpy as np
    request_path = Path(request_path).resolve()
    guard_path(request_path)
    request = read_json(request_path)
    if request["source_files_sha256"] != source_identity():
        raise ValueError("worker source differs from request")
    actual_runtime = runtime_inventory(request["method"])
    if actual_runtime != request["runtime_inventory"]:
        raise ValueError("worker runtime differs from frozen inventory")
    inputs = request_path.parent / "observed_inputs.npz"
    if file_hash(inputs) != request["observed_file_sha256"]:
        raise ValueError("worker observed input bytes changed")
    arrays = load_arrays(inputs)
    if array_identity(arrays) != request["observed_arrays"]:
        raise ValueError("worker observed input arrays changed")
    arguments, entry = make_method_arguments(arrays, request["method"])
    from obs_rct_benchmark.methods import make_method
    record = {"status": "failure", "request_sha256": digest(request),
              "method_entry": entry, "label_budget": label_budget(arrays, request["method"])}
    started = time.perf_counter()
    estimator = None
    try:
        estimator = make_method(request["method"], **request["method_options"])
        fit_started = time.perf_counter()
        try:
            entrypoint = estimator.fit_predict_sample if entry == "sample" else estimator.fit_predict
            prediction = np.asarray(entrypoint(arguments, arrays["eval_X"],
                                               seed=request["method_seed"]), dtype=np.float64)
        finally:
            record["fit_predict_seconds"] = time.perf_counter() - fit_started
        if prediction.shape != (len(arrays["eval_X"]),) or not np.isfinite(prediction).all():
            raise ValueError("invalid prediction vector")
        np.save(request_path.parent / "prediction.npy", prediction, allow_pickle=False)
        record.update(status="success", prediction_array_identity=array_identity({"prediction": prediction}))
    except BaseException as exc:
        record["error_type"] = type(exc).__name__
        (request_path.parent / "traceback.log").write_text(traceback.format_exc())
    finally:
        record.update(worker_seconds=time.perf_counter() - started,
                      self_usage=process_usage(resource.RUSAGE_SELF),
                      children_usage=process_usage(resource.RUSAGE_CHILDREN),
                      cgroup_memory=cgroup_memory())
        if estimator is not None:
            for attribute in ("provenance_", "last_run_provenance_", "attempt_provenance_"):
                value = getattr(estimator, attribute, None)
                if value is not None:
                    record[attribute] = value
            diagnostics = getattr(estimator, "last_run_diagnostics_", None)
            if diagnostics:
                write_json(request_path.parent / "native_diagnostics.json", diagnostics)
        write_json(request_path.parent / "worker.json", record)
    return 0 if record["status"] == "success" else 1


def enable_subreaper():
    if sys.platform == "linux":
        import ctypes
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError("could not establish worker-descendant reaping")


def _reap_group(group):
    while True:
        try:
            child, _ = os.waitpid(-group, os.WNOHANG)
            if child == 0:
                return
        except ChildProcessError:
            return


def stop_process_group(process):
    """Terminate descendants even after the worker leader exits."""
    group = process.pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        time.sleep(.1)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        _reap_group(group)
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return True
        time.sleep(.02)
    return False


def expected_request(manifest, case, method, observed_path, arrays):
    return {"schema_version": 1, "mode": manifest["mode"], "case": case,
            "method": method, "manifest_sha256": manifest["manifest_sha256"],
            "source_files_sha256": manifest["bindings"]["source_files_sha256"],
            "runtime_inventory": manifest["bindings"]["runtime_inventory"][method],
            "method_seed": _data().case_method_seed(case, method),
            "method_options": manifest["config"].get("method_options", {}).get(method, {}),
            "timeout_seconds": timeout_seconds(manifest["config"], method),
            "observed_file_sha256": file_hash(observed_path),
            "observed_arrays": array_identity(arrays)}


def record_path(output, case, method):
    return Path(output) / "records" / case["id"] / method / "record.json"


def metrics(prediction, truth):
    import numpy as np
    if (prediction.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError("invalid prediction or evaluation truth")
    with np.errstate(over="raise", invalid="raise"):
        error = prediction.astype(np.float64) - truth.astype(np.float64)
        mse = float(np.dot(error, error) / len(error))
    if not math.isfinite(mse):
        raise ValueError("nonfinite score")
    return {"cate_mse": mse, "root_pehe": math.sqrt(mse)}


def execute_fit(output, case, method, manifest, *, resume):
    import numpy as np
    terminal = record_path(output, case, method)
    if terminal.exists():
        if not resume:
            raise ValueError("terminal record exists; explicit resume required")
        return verify_record(output, case, method, manifest)
    parent = terminal.parent
    if parent.exists() and not resume:
        raise ValueError("unfinished attempt exists; explicit resume required")
    parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(parent / "running.lock"):
        case_path = prepare_case(output, case, manifest)
        all_observed = load_arrays(case_path / "observed.npz")
        observed = {key: all_observed[key] for key in projected_keys(method)}
        attempt = parent / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        np.savez(attempt / "observed_inputs.npz", **observed)
        request = expected_request(manifest, case, method, attempt / "observed_inputs.npz", observed)
        write_json(attempt / "request.json", request)
        runtime = manifest["bindings"]["runtime_map"]["methods"][method]
        command = [runtime["python"], str(Path(__file__).resolve()), "_worker",
                   "--request", str(attempt / "request.json")]
        record = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
                  "case_id": case["id"], "case": case, "method": method,
                  "mode": manifest["mode"], "status": "failure", "metrics": None,
                  "method_seed": request["method_seed"], "request_sha256": digest(request),
                  "attempt": str(attempt.relative_to(output)), "label_budget": label_budget(observed, method)}
        process, interrupted = None, None
        started = time.perf_counter()
        record["cgroup_before"] = cgroup_memory()
        enable_subreaper()
        try:
            with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
                process = subprocess.Popen(command, env=worker_environment(runtime), stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                try:
                    returncode = process.wait(timeout=request["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    returncode = None
                    record.update(status="timeout", error_type="FitTimeLimitExceeded")
                if returncode is not None:
                    if not (attempt / "worker.json").is_file():
                        record["error_type"] = "WorkerRecordMissing"
                    else:
                        worker = read_json(attempt / "worker.json")
                        if worker["request_sha256"] != digest(request):
                            raise ValueError("worker request identity differs")
                        if worker["status"] not in {"success", "failure"}:
                            raise ValueError("invalid worker status")
                        if returncode != (0 if worker["status"] == "success" else 1):
                            raise ValueError("worker return code differs from result")
                        record.update(worker)
                        if worker["status"] == "success" and manifest["mode"] == "campaign":
                            prediction = np.load(attempt / "prediction.npy", allow_pickle=False)
                            truth = load_arrays(case_path / "evaluation.npz")["eval_tau"]
                            record["metrics"] = metrics(prediction, truth)
        except BaseException as exc:
            record.update(status="failure", error_type=type(exc).__name__, metrics=None)
            (attempt / "launcher_traceback.log").write_text(traceback.format_exc())
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupted = exc
        finally:
            record["process_group_reaped"] = stop_process_group(process) if process is not None else True
            record["process_returncode"] = process.returncode if process is not None else None
            if not record["process_group_reaped"]:
                record.update(status="failure", error_type="UnreapedProcessGroup", metrics=None)
            record["launcher_seconds"] = time.perf_counter() - started
            record["cgroup_after"] = cgroup_memory()
            record["artifact_sha256"] = {p.name: file_hash(p) for p in sorted(attempt.iterdir()) if p.is_file()}
            write_json(attempt / "interrupted.json" if interrupted is not None else terminal, record)
        if interrupted is not None:
            raise interrupted
    return verify_record(output, case, method, manifest)


def verify_record(output, case, method, manifest):
    import numpy as np
    output = Path(output).resolve()
    record = read_json(record_path(output, case, method))
    expected = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
                "case_id": case["id"], "case": case, "method": method, "mode": manifest["mode"]}
    if any(record.get(key) != value for key, value in expected.items()) or record.get("status") not in TERMINAL:
        raise ValueError("terminal record identity or status changed")
    attempt = (output / record["attempt"]).resolve()
    required_parent = record_path(output, case, method).parent / "attempts"
    if attempt.parent != required_parent or (output / record["attempt"]).is_symlink():
        raise ValueError("attempt path escapes the declared fit")
    hashes = record["artifact_sha256"]
    if (any(Path(name).name != name for name in hashes)
            or {p.name for p in attempt.iterdir()} != set(hashes)
            or not {"request.json", "observed_inputs.npz", "stdout.log", "stderr.log"} <= set(hashes)):
        raise ValueError("fit artifacts are incomplete or contain unbound files")
    if any((attempt / name).is_symlink() or file_hash(attempt / name) != sha for name, sha in hashes.items()):
        raise ValueError("retained fit artifact bytes changed")
    observed = load_arrays(attempt / "observed_inputs.npz")
    case_path = output / "cases" / case["id"]
    verify_case_artifacts(case_path, case, manifest)
    all_observed = load_arrays(case_path / "observed.npz")
    projected = {key: all_observed[key] for key in projected_keys(method)}
    if set(observed) != set(projected) or array_identity(observed) != array_identity(projected):
        raise ValueError("worker projection differs from the paired observed case")
    request = expected_request(manifest, case, method, attempt / "observed_inputs.npz", projected)
    if (read_json(attempt / "request.json") != request or record["request_sha256"] != digest(request)
            or record["method_seed"] != request["method_seed"]
            or record["label_budget"] != label_budget(observed, method)):
        raise ValueError("fit request or label accounting changed")
    if "worker.json" in hashes:
        worker = read_json(attempt / "worker.json")
        if worker.get("request_sha256") != digest(request) or worker.get("status") not in {"success", "failure"}:
            raise ValueError("worker result identity changed")
        if record["status"] != "timeout" and "launcher_traceback.log" not in hashes:
            if record["status"] != worker["status"]:
                raise ValueError("terminal status differs from worker")
        for field in ("prediction_array_identity", "method_entry", "fit_predict_seconds",
                      "worker_seconds", "self_usage", "children_usage", "cgroup_memory",
                      "provenance_", "last_run_provenance_", "attempt_provenance_"):
            if field in worker and record.get(field) != worker[field]:
                raise ValueError("terminal worker fields changed")
    if record["status"] == "success":
        if not {"worker.json", "prediction.npy"} <= set(hashes) or not record["process_group_reaped"]:
            raise ValueError("successful fit has incomplete artifacts or live descendants")
        prediction = np.load(attempt / "prediction.npy", allow_pickle=False)
        if (prediction.shape != (len(observed["eval_X"]),) or not np.isfinite(prediction).all()
                or array_identity({"prediction": prediction}) != record["prediction_array_identity"]):
            raise ValueError("prediction shape, values or identity changed")
        if manifest["mode"] == "campaign":
            checker = _checker()
            truth = load_arrays(case_path / "evaluation.npz")["eval_tau"]
            expected_metrics = metrics(prediction, truth)
            if (set(record["metrics"]) != set(expected_metrics)
                    or any(not math.isclose(record["metrics"][key], value,
                                            rel_tol=checker.SCORE_RELATIVE_TOLERANCE,
                                            abs_tol=checker.SCORE_ABSOLUTE_TOLERANCE)
                           for key, value in expected_metrics.items())):
                raise ValueError("retained predictions and recorded metrics disagree")
        elif record["metrics"] is not None:
            raise ValueError("profiles cannot contain scores")
    elif record["metrics"] is not None:
        raise ValueError("failed or timed-out fits cannot contain scores")
    return record


def verified_records(output, *, require_complete=True, check_runtime=True):
    """Authenticate the exact union before a separate scientific analyzer runs."""
    output = Path(output).resolve()
    manifest = read_manifest(output, check_runtime=check_runtime)
    expected = {record_path(output, case, method): (case, method)
                for case in manifest["cases"] for method in manifest["methods"]}
    actual = set((output / "records").rglob("record.json"))
    if actual - expected.keys() or (require_complete and actual != expected.keys()):
        raise ValueError("terminal record union has missing, duplicate or extra fits")
    if any((output / "records").rglob("running.lock")):
        raise ValueError("unresolved running locks prevent collection acceptance")
    rows = [verify_record(output, *expected[path], manifest) for path in sorted(actual)]
    if manifest["mode"] == "campaign":
        checker = _checker()
        targets = {}
        for case in manifest["cases"]:
            if not any(row["case_id"] == case["id"] for row in rows):
                continue
            path = output / "cases" / case["id"]
            arrays = load_arrays(path / "observed.npz")
            arrays.update(load_arrays(path / "evaluation.npz"))
            checked = checker.verify_case(arrays, case["scenario_key"])
            if not checked["passed"]:
                raise ValueError("independent query/target verification failed: " + canonical(checked["errors"]))
            targets[case["id"]] = checker.independent_truth(arrays["eval_X"], case["scenario_key"])
        import numpy as np
        for row in rows:
            if row["status"] != "success":
                continue
            prediction = np.load(output / row["attempt"] / "prediction.npy", allow_pickle=False)
            independent = checker.score(prediction, targets[row["case_id"]])
            if any(not math.isclose(row["metrics"][name], value,
                                   rel_tol=checker.SCORE_RELATIVE_TOLERANCE,
                                   abs_tol=checker.SCORE_ABSOLUTE_TOLERANCE)
                   for name, value in independent.items()):
                raise ValueError("independent prediction/score verification failed")
    return manifest["config"], rows


def run_shard(args):
    output = guard_path(args.output)
    manifest = read_manifest(output)
    guard_allocation(manifest["config"])
    if manifest["mode"] != "campaign" or not 0 <= args.shard_index < manifest["shard_count"]:
        raise ValueError("invalid campaign shard")
    # Shards own disjoint case rectangles. Other active shards may have locks;
    # validate their completed records without treating their live work as drift.
    expected = {record_path(output, case, method): (case, method)
                for case in manifest["cases"] for method in manifest["methods"]}
    for path in (output / "records").rglob("record.json"):
        if path not in expected:
            raise ValueError("unrequested terminal fit")
        verify_record(output, *expected[path], manifest)
    states = []
    with exclusive_lock(output / f"shard-{args.shard_index}.lock"):
        for index, case in enumerate(manifest["cases"]):
            if index % manifest["shard_count"] != args.shard_index:
                continue
            for method in manifest["methods"]:
                row = execute_fit(output, case, method, manifest, resume=args.resume)
                states.append(row["status"])
                print(canonical({"case_id": case["id"], "method": method, "status": row["status"]}), flush=True)
    return 0 if all(state == "success" for state in states) else 1


def profile(args):
    config = _data().load_config(args.config)
    validate_resources(config, production=False)
    if args.method not in config["methods"]:
        raise ValueError("profile method is not in the declared panel")
    case = read_json(args.case_json)
    if case not in _data().profile_cases(config):
        raise ValueError("profile case is not in the reserved-seed preflight union")
    output = guard_path(args.output)
    guard_allocation(config)
    identity = bindings(args.config, args.runtimes, args.protocol, config, methods=[args.method])
    manifest = {"schema_version": 1, "mode": "profile", "config": config,
                "cases": [case], "methods": [args.method], "expected_records": 1,
                "bindings": identity,
                "paths": {key: str(Path(getattr(args, key)).resolve())
                          for key in ("config", "runtimes", "protocol")},
                "scope": "observed-only resource profile; producer truth computed then discarded; no scores",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    (output / "records").mkdir()
    write_json(output / "manifest.json", manifest)
    row = execute_fit(output, case, args.method, manifest, resume=False)
    print(canonical({"case_id": case["id"], "method": args.method, "status": row["status"],
                     "launcher_seconds": row["launcher_seconds"], "metrics": None}), flush=True)
    return 0 if row["status"] == "success" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "profile"):
        command = commands.add_parser(name)
        for field in ("config", "runtimes", "protocol", "output"):
            command.add_argument("--" + field, required=True, type=Path)
        if name == "init":
            command.add_argument("--shard-count", type=int, default=128)
        else:
            command.add_argument("--case-json", required=True, type=Path)
            command.add_argument("--method", required=True)
    command = commands.add_parser("run")
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--shard-index", required=True, type=int)
    command.add_argument("--resume", action="store_true")
    command = commands.add_parser("verify")
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("_worker")
    command.add_argument("--request", required=True, type=Path)
    command = commands.add_parser("_probe")
    command.add_argument("--method", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "_probe":
        guard_path(WORK_ROOT / "runtime-probe")
        print(canonical(runtime_inventory(args.method)))
        return 0
    def terminated(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminated)
    if args.command == "_worker":
        return fit_worker(args.request)
    if args.command == "verify":
        output = guard_path(args.output)
        _, rows = verified_records(output)
        print(canonical({"verified_records": len(rows)}))
        return 0
    return {"init": init_run, "profile": profile, "run": run_shard}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
