#!/usr/bin/env python3
"""Run and analyze the prospectively specified three-source GP pilot on GSDS.

``init`` creates a fresh immutable manifest; ``run`` streams separate bounded
fit attempts; ``analyze`` authenticates the complete terminal collection before
reporting condition-specific paired descriptive summaries. Raw arrays and logs
remain at the server output root. The script does not select recipes or seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import tomllib
import traceback
import uuid

import numpy as np


SLOTS = ("full", "rct_only", "obs_start")
METHOD_IDS = {"full": "causal_icm_fixed", "rct_only": "causal_icm_rct_only",
              "obs_start": "causal_icm_obs_start"}
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
REQUIRED_CONFIG = {"schema_version", "scope", "generator", "d", "n_obs", "rct_budgets",
                   "n_eval", "deltas", "seeds", "rho", "fit_timeout_seconds", "tune_role",
                   "inference", "methods"}
OBSERVED_ARRAYS = ("obs_X", "obs_T", "obs_Y", "rct_X", "rct_T", "rct_Y",
                   "rct_known_propensity", "eval_X")
METRIC_ATOL, METRIC_RTOL = 1e-12, 1e-10


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if replace:
        temporary.replace(path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_config(config):
    if not isinstance(config, dict) or set(config) - REQUIRED_CONFIG - {"description"}:
        raise ValueError("unknown configuration fields")
    if REQUIRED_CONFIG - set(config):
        raise ValueError("incomplete configuration")
    if (type(config["schema_version"]) is not int or config["schema_version"] != 1
            or config["generator"] != "paired_gp_source_v1"):
        raise ValueError("unsupported configuration or generator version")
    if not isinstance(config["scope"], str) or not config["scope"].strip():
        raise ValueError("scope must be explicitly declared")
    if config["d"] != 2:
        raise ValueError("paired_gp_source_v1 has exactly two covariates")
    for name in ("n_obs", "n_eval", "fit_timeout_seconds"):
        if not _positive_integer(config[name]):
            raise ValueError(f"{name} must be a positive integer")
    if config["n_obs"] < 2:
        raise ValueError("n_obs must be at least two")
    budgets, seeds, deltas = config["rct_budgets"], config["seeds"], config["deltas"]
    if (not isinstance(budgets, list) or not budgets or len(set(budgets)) != len(budgets)
            or any(not _positive_integer(n) or n < 2 for n in budgets)):
        raise ValueError("RCT budgets must be distinct integers at least two")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < 2**32 for n in seeds)):
        raise ValueError("seeds must be distinct uint32 integers")
    if (not isinstance(deltas, list) or not deltas or len(set(deltas)) != len(deltas)
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < 0 for x in deltas)):
        raise ValueError("deltas must be distinct finite nonnegative numbers")
    rho = config["rho"]
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 <= rho <= 1:
        raise ValueError("rho must be in [0,1]")
    if config["tune_role"] != "first_two_training_rows_alias":
        raise ValueError("this pilot aliases two charged fitting rows and never tunes")
    if config["inference"] != "pointwise_paired_t_descriptive":
        raise ValueError("only the fixed descriptive paired analysis is supported")
    methods = config["methods"]
    if methods != METHOD_IDS:
        raise ValueError("paired_gp_source_v1 requires its exact source-access method IDs")
    canonical(config)
    return config


def load_config(path):
    path = Path(path)
    config = (tomllib.loads(path.read_text()) if path.suffix == ".toml"
              else json.loads(path.read_text()))
    return validate_config(config)


def baseline_mean(X):
    X = np.asarray(X, dtype=np.float64)
    return .5 * X[:, 0] + .25 * X[:, 1]


def true_effect(X):
    X = np.asarray(X, dtype=np.float64)
    return 1. + .5 * np.sin(np.pi * X[:, 0]) + .25 * np.cos(np.pi * X[:, 1])


def bias_function(X):
    X = np.asarray(X, dtype=np.float64)
    return np.cos(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])


def _rng(seed, role):
    return np.random.default_rng(np.random.SeedSequence([seed, role]))


def generate_case(config, *, seed, delta, n_rct):
    """Generate shared observations and evaluator-only truth, without fitting.

    The latent SCM is U_OBS=T_OBS, U_RCT=1/2 and
    Y(t)=m(X)+(t-1/2)tau(X)+delta(U-1/2)b(X)+epsilon.
    Thus the causal effect is tau in both populations; the observed OBS arm
    contrast is tau+delta*b. OBS positivity conditional on latent U is absent.
    """
    if seed not in config["seeds"] or delta not in config["deltas"] or n_rct not in config["rct_budgets"]:
        raise ValueError("case is not in the frozen configuration")
    no, nr, nq = config["n_obs"], max(config["rct_budgets"]), config["n_eval"]
    ox = _rng(seed, 101).uniform(-1., 1., (no, 2))
    ot = _rng(seed, 102).binomial(1, .5, no).astype(np.float64)
    oe = _rng(seed, 103).standard_normal(no)
    rx = _rng(seed, 201).uniform(-1., 1., (nr, 2))[:n_rct]
    rt = _rng(seed, 202).binomial(1, .5, nr).astype(np.float64)[:n_rct]
    re = _rng(seed, 203).standard_normal(nr)[:n_rct]
    query = _rng(seed, 301).uniform(-1., 1., (nq, 2))
    return {
        "obs_X": ox, "obs_T": ot,
        "obs_Y": baseline_mean(ox) + (ot - .5) * (true_effect(ox) + delta * bias_function(ox)) + oe,
        "rct_X": rx, "rct_T": rt,
        "rct_Y": baseline_mean(rx) + (rt - .5) * true_effect(rx) + re,
        "rct_known_propensity": np.full(n_rct, .5),
        "eval_X": query, "eval_tau": true_effect(query),
    }


def make_training_view(arrays):
    from obs_rct_benchmark.data import SourceSample, TrainingView
    return TrainingView(
        SourceSample(arrays["obs_X"], arrays["obs_T"], arrays["obs_Y"], "obs", "obs_train"),
        SourceSample(arrays["rct_X"], arrays["rct_T"], arrays["rct_Y"], "rct", "rct_train",
                     known_propensity=arrays["rct_known_propensity"]),
        SourceSample(arrays["rct_X"][:2], arrays["rct_T"][:2], arrays["rct_Y"][:2], "rct", "rct_tune",
                     known_propensity=arrays["rct_known_propensity"][:2]),
    )


def make_source_sample(arrays, source):
    from obs_rct_benchmark.data import SourceSample
    if source not in {"obs", "rct"}:
        raise ValueError("source must be obs or rct")
    return SourceSample(arrays[source + "_X"], arrays[source + "_T"], arrays[source + "_Y"],
                        source, source + "_train",
                        known_propensity=arrays["rct_known_propensity"] if source == "rct" else None)


def array_identity(arrays):
    return {name: {"shape": list(x.shape), "dtype": str(x.dtype),
                   "sha256": hashlib.sha256(x.tobytes(order="C")).hexdigest()}
            for name, x in sorted(arrays.items())}


def cases(config):
    return [{"id": f"d{di:02d}_r{budget}_s{seed}", "delta_index": di,
             "delta": delta, "n_rct": budget, "seed": seed}
            for di, delta in enumerate(config["deltas"])
            for budget in config["rct_budgets"] for seed in config["seeds"]]


def source_identity():
    from obs_rct_benchmark import __version__
    from obs_rct_benchmark.causal_icm import causal_icm_provenance
    from obs_rct_benchmark.runner import environment_provenance, source_provenance
    repo = Path(__file__).resolve().parents[1]
    files = {p.name: file_hash(p) for p in sorted((repo / "src/obs_rct_benchmark").glob("*.py"))}
    actual = source_provenance()
    if actual["package_file_sha256"] != files:
        raise RuntimeError("imported producer differs from this driver checkout")
    external = causal_icm_provenance()
    if not external["verified"]:
        raise RuntimeError("native Causal-ICM source/numerical environment is unverified")
    return {"driver_sha256": file_hash(__file__), "package_source": actual,
            "module_version": __version__, "environment": environment_provenance(),
            "native_source": external}


def bindings(config_path):
    return {"config_sha256": file_hash(config_path), "source": source_identity()}


def guard_path(path):
    raw = os.environ.get("GSDS_WORK_ROOT")
    if (sys.platform != "linux" or not raw or not os.environ.get("SLURM_JOB_ID")
            or not Path(raw).is_absolute()):
        raise RuntimeError("pilot execution requires a GSDS Slurm job and shared work root")
    work = Path(raw).resolve(strict=True)
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to(work) or target == work:
        raise ValueError("pilot paths must be below GSDS_WORK_ROOT")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("all numerical thread limits must equal one")
    return target


def init_run(args):
    config = load_config(args.config)
    output = guard_path(args.output)
    if not _positive_integer(args.shard_count) or args.shard_count > len(cases(config)):
        raise ValueError("shard-count must be positive and no greater than the case count")
    identity = bindings(args.config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    (output / "shards").mkdir()
    manifest = {"schema_version": 1, "bindings": identity, "config": config,
                "cases": cases(config), "slots": list(SLOTS), "shard_count": args.shard_count,
                "expected_records": len(cases(config)) * 3,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest["manifest_sha256"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    print(canonical({"status": "initialized", "expected_records": manifest["expected_records"],
                     "manifest_sha256": file_hash(output / "manifest.json")}))
    return 0


def read_manifest(config_path, output):
    config = load_config(config_path)
    manifest = json.loads((output / "manifest.json").read_text())
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or not _positive_integer(manifest.get("shard_count"))
            or manifest["shard_count"] > len(cases(config))
            or type(manifest.get("expected_records")) is not int
            or manifest["expected_records"] != len(cases(config)) * len(SLOTS)):
        raise ValueError("manifest execution header changed")
    if manifest.get("manifest_sha256") != digest({k:v for k,v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("manifest canonical payload digest changed")
    if manifest["config"] != config or manifest["cases"] != cases(config) or manifest["slots"] != list(SLOTS):
        raise ValueError("manifest configuration/case contract changed")
    if manifest["bindings"] != bindings(config_path):
        raise ValueError("config, producer, numerical environment or author source changed")
    return config, manifest


def record_path(output, case, slot):
    return output / "records" / case["id"] / slot / "record.json"


def expected_request(case, slot, config, manifest):
    return {"case": case, "slot": slot, "method_id": config["methods"][slot],
            "bindings": manifest["bindings"], "manifest_sha256": manifest["manifest_sha256"],
            "rho": config["rho"]}


def _usage(who):
    usage = resource.getrusage(who)
    return {"maxrss_kib": usage.ru_maxrss, "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime}


def _make_estimator(method_id, rho):
    from obs_rct_benchmark.methods import make_method
    return make_method(method_id, rho=rho)


def _check_estimator(estimator, slot, rho):
    from obs_rct_benchmark.causal_icm import CausalICMSourceControl, OfficialCausalICM
    if slot == "full":
        valid = isinstance(estimator, OfficialCausalICM) and estimator.effect_modifier_columns is None
    else:
        valid = (isinstance(estimator, CausalICMSourceControl)
                 and estimator.source == {"rct_only": "rct", "obs_start": "obs"}[slot])
    if not valid or estimator.rho != rho:
        raise ValueError("factory does not implement the frozen source-access recipe")


def fit_worker(args):
    output = guard_path(args.output)
    attempt = guard_path(args.attempt)
    config, manifest = read_manifest(args.config, output)
    request = json.loads((attempt / "request.json").read_text())
    case, slot = request["case"], request["slot"]
    if case not in manifest["cases"] or slot not in SLOTS or request != expected_request(case, slot, config, manifest):
        raise ValueError("worker request does not match its frozen manifest")
    record = {**request, "status": "failed", "schema_version": 1}
    start = time.perf_counter()
    try:
        arrays = generate_case(config, seed=case["seed"], delta=case["delta"], n_rct=case["n_rct"])
        observed = {k: arrays[k] for k in OBSERVED_ARRAYS}
        np.savez(attempt / "observed_inputs.npz", **observed)
        np.savez(attempt / "evaluation.npz", eval_tau=arrays["eval_tau"])
        record["observed_array_identity"] = array_identity(observed)
        record["evaluation_array_identity"] = array_identity({"eval_tau": arrays["eval_tau"]})
        record["label_budget"] = {
            "supplied_unique_rct_labels": case["n_rct"] if slot != "obs_start" else 0,
            "supplied_unique_obs_labels": config["n_obs"] if slot != "rct_only" else 0,
            "consumed_rct_labels": case["n_rct"] if slot != "obs_start" else 0,
            "consumed_obs_labels": config["n_obs"] if slot != "rct_only" else 0,
            "additional_tuning_labels": 0, "tune_alias_train_indices": [0, 1] if slot == "full" else [],
        }
        supplied_keys = OBSERVED_ARRAYS if slot == "full" else tuple(
            key for key in OBSERVED_ARRAYS if key.startswith("rct_" if slot == "rct_only" else "obs_")
            or key == "eval_X")
        record["method_argument_array_identity"] = array_identity({k: arrays[k] for k in supplied_keys})
        record["method_argument_roles"] = (["obs_train", "rct_train", "rct_tune_alias", "eval_X"]
                                            if slot == "full" else
                                            ["rct_train" if slot == "rct_only" else "obs_train", "eval_X"])
        write_json(attempt / "inputs.json", record)
        estimator = _make_estimator(request["method_id"], config["rho"])
        _check_estimator(estimator, slot, config["rho"])
        fit_start = time.perf_counter()
        try:
            if slot == "full":
                raw_prediction = estimator.fit_predict(make_training_view(arrays), arrays["eval_X"], seed=case["seed"])
            else:
                sample = make_source_sample(arrays, "rct" if slot == "rct_only" else "obs")
                raw_prediction = estimator.fit_predict_sample(sample, arrays["eval_X"], seed=case["seed"])
            prediction = np.asarray(raw_prediction, dtype=np.float64)
        finally:
            record["fit_predict_seconds"] = time.perf_counter() - fit_start
            record["estimator_provenance"] = getattr(estimator, "provenance_", None)
        if prediction.shape != (config["n_eval"],) or not np.isfinite(prediction).all():
            raise ValueError("invalid pilot prediction vector")
        np.save(attempt / "prediction.npy", prediction, allow_pickle=False)
        error = prediction - arrays["eval_tau"]
        record.update(status="success", mse=float(np.dot(error, error) / error.size),
                      prediction_array_identity=array_identity({"prediction": prediction})["prediction"])
    except BaseException as exc:
        record["error_type"] = type(exc).__name__
        (attempt / "traceback.log").write_text(traceback.format_exc())
    record.update(worker_elapsed_seconds=time.perf_counter() - start,
                  worker_self_usage=_usage(resource.RUSAGE_SELF),
                  worker_children_usage=_usage(resource.RUSAGE_CHILDREN))
    write_json(attempt / "worker.json", record)
    return 0 if record["status"] == "success" else 1


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        time.sleep(.25)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
        return True
    except subprocess.TimeoutExpired:
        return False


def execute_fit(args, output, case, slot, config, manifest):
    terminal = record_path(output, case, slot)
    parent = terminal.parent
    if terminal.exists():
        if not args.resume:
            raise ValueError("terminal record already exists; use --resume")
        return verify_record(terminal, output, case, slot, config, manifest)
    if parent.exists() and not args.resume:
        raise ValueError("unfinished attempt exists; use --resume to retain and restart it")
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / "running.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("case is already running or has an unresolved previous lock") from exc
    attempt = parent / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    request = expected_request(case, slot, config, manifest)
    write_json(attempt / "request.json", request)
    environment = dict(os.environ)
    environment.update({name: "1" for name in THREADS + ("BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--config", str(Path(args.config).resolve()),
               "--output", str(output), "--attempt", str(attempt)]
    process = None
    record = {**request, "status": "failed", "schema_version": 1,
              "attempt": str(attempt.relative_to(output))}
    start = time.perf_counter()
    interruption = None
    try:
        with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                returncode = process.wait(timeout=config["fit_timeout_seconds"])
            except subprocess.TimeoutExpired:
                returncode = None
                record.update(status="timeout", error_type="FitTimeLimitExceeded")
            if returncode is not None and (attempt / "worker.json").exists():
                worker = json.loads((attempt / "worker.json").read_text())
                if any(worker.get(k) != v for k, v in request.items()):
                    raise ValueError("worker identity changed")
                record.update(worker)
                if returncode != (0 if worker["status"] == "success" else 1):
                    raise ValueError("worker return code disagrees with record")
            elif returncode is not None:
                record["error_type"] = "WorkerRecordMissing"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        (attempt / "launcher_traceback.log").write_text(traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            record["status"] = "interrupted"
            interruption = exc
    finally:
        record["process_group_reaped"] = _stop_group(process) if process is not None else True
        record["process_returncode"] = process.returncode if process is not None else None
        if not record["process_group_reaped"]:
            record.update(status="failed", error_type="UnreapedProcessGroup")
        record["launcher_seconds"] = time.perf_counter() - start
        record["attempt"] = str(attempt.relative_to(output))
        record["artifact_sha256"] = {p.name: file_hash(p) for p in sorted(attempt.iterdir()) if p.is_file()}
        write_json(attempt / "interrupted.json" if interruption is not None else terminal, record)
        lock.rmdir()
    if interruption is not None:
        raise interruption
    return verify_record(terminal, output, case, slot, config, manifest)


def verify_record(path, output, case, slot, config, manifest):
    record = json.loads(Path(path).read_text())
    if any(record.get(k) != v for k, v in expected_request(case, slot, config, manifest).items()):
        raise ValueError("record identity does not match frozen case")
    if record.get("status") not in {"success", "failed", "timeout"}:
        raise ValueError("record is not terminal")
    attempt = (output / record["attempt"]).resolve()
    if not attempt.is_relative_to(output):
        raise ValueError("record attempt escapes output root")
    if any(Path(name).name != name for name in record["artifact_sha256"]):
        raise ValueError("artifact names must be local basenames")
    if any(file_hash(attempt / name) != sha for name, sha in record["artifact_sha256"].items()):
        raise ValueError("retained fit artifacts changed")
    if record["status"] == "success":
        arrays = generate_case(config, seed=case["seed"], delta=case["delta"], n_rct=case["n_rct"])
        if record["observed_array_identity"] != array_identity({k: arrays[k] for k in OBSERVED_ARRAYS}):
            raise ValueError("record input arrays do not match its declared generator")
        prediction = np.load(attempt / "prediction.npy", allow_pickle=False)
        if array_identity({"prediction": prediction})["prediction"] != record["prediction_array_identity"]:
            raise ValueError("prediction identity changed")
        error = prediction - arrays["eval_tau"]
        mse = float(np.dot(error, error) / error.size)
        if not math.isclose(mse, record["mse"], abs_tol=METRIC_ATOL, rel_tol=METRIC_RTOL):
            raise ValueError("stored MSE disagrees with retained predictions")
    return record


def run_shard(args):
    output = guard_path(args.output)
    config, manifest = read_manifest(args.config, output)
    if not 0 <= args.shard_index < manifest["shard_count"]:
        raise ValueError("invalid shard index")
    records = []
    for index, case in enumerate(manifest["cases"]):
        if index % manifest["shard_count"] != args.shard_index:
            continue
        for slot in SLOTS:
            record = execute_fit(args, output, case, slot, config, manifest)
            records.append({"case": case["id"], "slot": slot, "status": record["status"]})
            print(canonical(records[-1]), flush=True)
    summary = {"shard_index": args.shard_index, "records": records,
               "status": "complete" if all(r["status"] == "success" for r in records) else "contains_failures"}
    write_json(output / "shards" / (f"shard-{args.shard_index}-" + uuid.uuid4().hex + ".json"), summary)
    return 0 if summary["status"] == "complete" else 1


def paired_summary(values, expected):
    n = len(values)
    result = {"expected_pairs": expected, "successful_pairs": n, "complete": n == expected,
              "mean": float(np.mean(values)) if n else None, "mcse": None, "ci95": None}
    if n == expected and n >= 2:
        from scipy.stats import t
        mean, se = result["mean"], float(np.std(values, ddof=1) / np.sqrt(n))
        width = float(t.ppf(.975, n - 1)) * se
        result.update(mcse=se, ci95=[mean - width, mean + width])
    return result


def analyze(args):
    output, analysis = guard_path(args.output), guard_path(args.analysis_output)
    config, manifest = read_manifest(args.config, output)
    expected_paths = {str(record_path(output, case, slot).relative_to(output))
                      for case in manifest["cases"] for slot in SLOTS}
    actual_paths = {str(p.relative_to(output)) for p in (output / "records").rglob("record.json")}
    if actual_paths != expected_paths:
        raise ValueError(f"terminal record coverage differs: missing={len(expected_paths-actual_paths)}, extra={len(actual_paths-expected_paths)}")
    records = {(case["id"], slot): verify_record(record_path(output, case, slot), output, case, slot, config, manifest)
               for case in manifest["cases"] for slot in SLOTS}
    groups = {}
    for case in manifest["cases"]:
        for slot, group in (("rct_only", ("rct", case["seed"], case["n_rct"])),
                            ("obs_start", ("obs", case["seed"], case["delta_index"]))):
            record = records[(case["id"], slot)]
            groups.setdefault(group, []).append(record)
    if any(len({r["prediction_array_identity"]["sha256"] for r in group if r["status"] == "success"}) > 1
           for group in groups.values()):
        raise ValueError("excluded-source condition invariance failed")
    invariants = {}
    for prefix, name in (("rct", "rct_only_delta_invariant"), ("obs", "obs_start_budget_invariant")):
        subset = [group for key, group in groups.items() if key[0] == prefix]
        complete = sum(all(r["status"] == "success" for r in group) for group in subset)
        invariants[name] = {"expected_groups": len(subset), "complete_groups": complete,
                            "all_groups_checked": complete == len(subset),
                            "discrepant_successful_groups": 0}
    conditions = []
    for di, delta in enumerate(config["deltas"]):
        for budget in config["rct_budgets"]:
            case_group = [case for case in manifest["cases"] if case["delta_index"] == di and case["n_rct"] == budget]
            row = {"delta": delta, "n_rct": budget, "recipes": {}, "paired": {}}
            for slot in SLOTS:
                subset = [records[(case["id"], slot)] for case in case_group]
                risks = [x["mse"] for x in subset if x["status"] == "success"]
                row["recipes"][slot] = {"successful": len(risks), "failed": len(subset)-len(risks),
                                        "mean_mse_success_only": float(np.mean(risks)) if risks else None}
            for control in ("rct_only", "obs_start"):
                differences = [records[(case["id"], "full")]["mse"] - records[(case["id"], control)]["mse"]
                               for case in case_group if records[(case["id"], "full")]["status"] == "success"
                               and records[(case["id"], control)]["status"] == "success"]
                row["paired"]["full_minus_" + control] = paired_summary(differences, len(case_group))
            conditions.append(row)
    summary = {"schema_version": 1, "scope": config["scope"], "bindings": manifest["bindings"],
               "expected_records": len(records), "success": sum(r["status"] == "success" for r in records.values()),
               "failed": sum(r["status"] != "success" for r in records.values()), "missing": 0,
               "conditions": conditions, "invariances": invariants,
               "inference": "Descriptive paired Monte Carlo means/approximate pointwise t intervals over configured sample seeds; no p-values, multiplicity decisions, or unseen-mechanism claim.",
               "record_sha256": {name: file_hash(output / name) for name in sorted(expected_paths)}}
    analysis.mkdir(parents=True, exist_ok=False)
    write_json(analysis / "summary.json", summary)
    lines = ["# GP source-access development pilot", "", summary["inference"], "",
             f"Records: {summary['success']} successful, {summary['failed']} failed, zero missing.", "",
             "| Delta | RCT rows | Contrast | Pairs | Mean difference | MCSE | Pointwise 95% interval |",
             "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in conditions:
        for name, item in row["paired"].items():
            fmt = lambda x: "withheld" if x is None else f"{x:.8g}"
            interval = "withheld" if item["ci95"] is None else "[" + ", ".join(map(fmt, item["ci95"])) + "]"
            lines.append(f"| {row['delta']} | {row['n_rct']} | {name} | {item['successful_pairs']}/{item['expected_pairs']} | {fmt(item['mean'])} | {fmt(item['mcse'])} | {interval} |")
    (analysis / "README.md").write_text("\n".join(lines) + "\n")
    print(canonical({"status": "analyzed", "success": summary["success"], "failed": summary["failed"], "missing": 0}))
    return 0 if not summary["failed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "analyze", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        if name == "init":
            p.add_argument("--shard-count", type=int, default=4)
        elif name == "run":
            p.add_argument("--shard-index", type=int, required=True)
            p.add_argument("--resume", action="store_true")
        elif name == "analyze":
            p.add_argument("--analysis-output", required=True, type=Path)
        else:
            p.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    return {"init": init_run, "run": run_shard, "analyze": analyze, "_worker": fit_worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
