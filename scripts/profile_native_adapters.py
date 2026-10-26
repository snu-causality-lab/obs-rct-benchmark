#!/usr/bin/env python3
"""Measure one native-adapter fit inside a bounded GSDS Slurm worker.

This is resource profiling only: it creates observed fitting arrays and query
covariates, with no evaluation truth, scores, admission decision or campaign.
Each invocation requires a fresh output directory. Inputs, predictions, raw
logs and tracebacks stay on the server; profile.json contains sanitized metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


METHODS = ("integrative_r_linear", "integrative_r_quadratic", "causal_icm_fixed")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")
EXTRA_THREAD_VARIABLES = ("VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                          "RCPP_PARALLEL_NUM_THREADS")
SHARED_WORK_PREFIX = Path("/shared/s2/lab05")
SCOPE = "resource_profile_only_not_native_admission_or_performance_evidence"


def _write_json(path: Path, value: dict) -> None:
    contents = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(contents)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict:
    driver = Path(__file__).resolve()
    package = driver.parents[1] / "src" / "obs_rct_benchmark"
    files = {path.name: _sha256(path) for path in sorted(package.glob("*.py"))}
    if not files:
        raise ValueError("driver must be run from its complete source checkout")
    return {"driver": "scripts/profile_native_adapters.py",
            "driver_sha256": _sha256(driver), "package_file_sha256": files}


def _parameters(args) -> dict:
    return {"method": args.method, "n_obs": args.n_obs, "n_rct_train": args.n_rct,
            "d": args.d, "n_eval": args.n_eval, "seed": args.seed,
            "timeout_seconds": args.timeout_seconds, "unused_rct_tune_rows": 2}


def _guard(args) -> Path:
    raw = os.environ.get("GSDS_WORK_ROOT")
    if not raw or not Path(raw).is_absolute() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("profiling requires a GSDS Slurm job and an absolute GSDS_WORK_ROOT")
    work = Path(raw).resolve(strict=True)
    if not work.is_relative_to(SHARED_WORK_PREFIX):
        raise ValueError("GSDS_WORK_ROOT must be below the GSDS lab shared directory")
    relative_work = work.relative_to(SHARED_WORK_PREFIX).parts
    if len(relative_work) < 2 or relative_work[1] != "work":
        raise ValueError("GSDS_WORK_ROOT must be under an account's shared work directory")
    output = Path(args.output_dir).expanduser().resolve()
    if not output.is_relative_to(work) or output == work:
        raise ValueError("output-dir must be below GSDS_WORK_ROOT")
    if not args.worker and output.exists():
        raise ValueError("output-dir must not exist; preserve the previous attempt")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError("OMP, MKL, OPENBLAS and NUMEXPR thread limits must all equal 1")
    if sys.platform != "linux":
        raise ValueError("this resource profile requires the GSDS Linux environment")
    for name in ("n_obs", "n_rct", "d", "n_eval", "timeout_seconds"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.n_obs < 2 or args.n_rct < 2:
        raise ValueError("both fitting samples require at least two rows")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    return output


def _usage(which: int) -> dict:
    usage = resource.getrusage(which)
    return {"maxrss_kib": int(usage.ru_maxrss), "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime}


def _make_observed_inputs(args):
    import numpy as np

    # Independent named streams preserve matching arrays across method IDs;
    # separate X/arm/noise streams also preserve prefixes as budgets grow.
    streams = np.random.SeedSequence(args.seed).spawn(13)

    def source(n, offset, randomized):
        x = np.random.default_rng(streams[offset]).normal(size=(n, args.d))
        latent = np.random.default_rng(streams[offset + 1]).normal(size=n)
        arm_rng = np.random.default_rng(streams[offset + 2])
        noise = np.random.default_rng(streams[offset + 3]).normal(size=n)
        second = x[:, 1] if args.d > 1 else np.zeros(n)
        probability = (np.full(n, .5) if randomized
                       else 1 / (1 + np.exp(-(.7 * x[:, 0] - .3 * second + .6 * latent))))
        arm = arm_rng.binomial(1, probability).astype(np.float64)
        outcome = (.5 * x[:, 0] - .25 * second + .4 * latent
                   + arm * (1 + .4 * x[:, 0] - .2 * second) + noise)
        return x, arm, outcome

    obs = source(args.n_obs, 0, False)
    trial = source(args.n_rct, 4, True)
    unused_tune = source(2, 8, True)
    query = np.random.default_rng(streams[12]).normal(size=(args.n_eval, args.d))
    arrays = {"query_X": query, "rct_known_propensity": np.full(args.n_rct, .5),
              "unused_tune_known_propensity": np.full(2, .5)}
    for prefix, values in (("obs", obs), ("rct", trial), ("unused_tune", unused_tune)):
        arrays.update({f"{prefix}_{field}": np.asarray(value, dtype=np.float64)
                       for field, value in zip(("X", "T", "Y"), values)})
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite constructed observed input")
    return arrays


def _array_identity(arrays) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
            for name, value in sorted(arrays.items())}


def _worker(args, output: Path) -> int:
    # The exclusive marker rejects direct replay into an existing attempt.
    with (output / "worker_started.json").open("x", encoding="utf-8") as handle:
        handle.write('{"started": true}\n')
    request = json.loads((output / "request.json").read_text())
    record = {"schema_version": 1, "scope": SCOPE, "parameters": _parameters(args),
              "status": "failed", "source_identity": _source_identity()}
    start = time.perf_counter()
    try:
        if (request["parameters"] != record["parameters"]
                or request["source_identity"] != record["source_identity"]):
            raise ValueError("worker request or source changed")
        import numpy as np
        from obs_rct_benchmark.data import SourceSample, TrainingView
        from obs_rct_benchmark.methods import make_method
        from obs_rct_benchmark.runner import (
            _validate_public_values, environment_provenance,
            external_method_provenance, source_provenance,
        )

        record["imported_package_source"] = source_provenance()
        if (record["imported_package_source"]["package_file_sha256"]
                != record["source_identity"]["package_file_sha256"]):
            raise ValueError("imported benchmark package differs from the profiled source")
        record["environment"] = environment_provenance()
        arrays = _make_observed_inputs(args)
        np.savez(output / "observed_inputs.npz", **arrays)
        record["observed_input_arrays"] = _array_identity(arrays)
        record["input_recipe"] = {
            "id": "fixed_observed_gaussian_covariates_binary_arms_v1",
            "rct_assignment_probability": .5,
            "obs_assignment": "logistic_covariate_and_unobserved_variable",
            "outcome": "fixed_covariate_mean_plus_arm_covariate_term_and_noise",
            "tuning_role": "two_independent_observed_rows_unused_by_all_requested_recipes",
            "rejection_or_resampling": False,
        }
        training = TrainingView(
            SourceSample(arrays["obs_X"], arrays["obs_T"], arrays["obs_Y"], "obs", "obs_train"),
            SourceSample(arrays["rct_X"], arrays["rct_T"], arrays["rct_Y"], "rct", "rct_train",
                         known_propensity=arrays["rct_known_propensity"]),
            SourceSample(arrays["unused_tune_X"], arrays["unused_tune_T"], arrays["unused_tune_Y"],
                         "rct", "rct_tune", known_propensity=arrays["unused_tune_known_propensity"]),
        )
        record["fitting_arm_counts"] = {
            source: {str(arm): int(np.count_nonzero(arrays[source + "_T"] == arm))
                     for arm in (0, 1)} for source in ("obs", "rct")}
        _validate_public_values(record)
        _write_json(output / "worker_inputs.json", {**record, "status": "preparing"})
        provenance_start = time.perf_counter()
        record["external_methods"] = external_method_provenance((args.method,))
        record["prefit_provenance_seconds"] = time.perf_counter() - provenance_start
        estimator = make_method(args.method)
        record["recipe_options"] = dict(vars(estimator))
        _validate_public_values(record)
        _write_json(output / "worker_preflight.json", {**record, "status": "fitting"})
        fit_start = time.perf_counter()
        try:
            prediction = np.asarray(estimator.fit_predict(
                training, arrays["query_X"], seed=args.seed), dtype=np.float64)
        finally:
            record["fit_predict_seconds"] = time.perf_counter() - fit_start
            for attribute in ("provenance_", "last_run_provenance_", "attempt_provenance_"):
                value = getattr(estimator, attribute, None)
                if value is not None:
                    record[attribute] = value
            diagnostics = getattr(estimator, "last_run_diagnostics_", None)
            if diagnostics:
                record["native_log_files"] = {}
                for name in ("stdout", "stderr"):
                    if name in diagnostics:
                        path = output / f"native_{name}.log"
                        path.write_text(diagnostics[name], encoding="utf-8")
                        record["native_log_files"][name] = {
                            "file": path.name, "sha256": _sha256(path)}
        if prediction.shape != (args.n_eval,) or not np.isfinite(prediction).all():
            raise ValueError("native adapter returned an invalid prediction vector")
        np.save(output / "predictions.npy", prediction, allow_pickle=False)
        record["prediction"] = {"shape": list(prediction.shape), "dtype": str(prediction.dtype),
                                "sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}
        native_parameters = getattr(estimator, "native_parameters_", None)
        if native_parameters:
            np.savez(output / "native_parameters.npz", **native_parameters)
            record["native_parameter_arrays"] = _array_identity(native_parameters)
        record["status"] = "completed"
        _validate_public_values(record)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        with (output / "worker_traceback.txt").open("x", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
    finally:
        record["worker_elapsed_seconds"] = time.perf_counter() - start
        record["worker_self_usage"] = _usage(resource.RUSAGE_SELF)
        record["worker_children_usage"] = _usage(resource.RUSAGE_CHILDREN)
        record["memory_interpretation"] = (
            "Linux ru_maxrss in KiB: self and largest reaped child are separate high-water marks; "
            "neither is the simultaneous process-group or whole-job peak")
        _write_json(output / "worker_result.json", record)
    return 0 if record["status"] == "completed" else 1


def _kill_group(process: subprocess.Popen) -> dict:
    # The R bridge inherits this group. Always send SIGKILL after the grace
    # period, even if the Python leader exits while an R child remains alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        signal_sent = "group_already_absent"
    else:
        time.sleep(.25)
        try:
            os.killpg(process.pid, signal.SIGKILL)
            signal_sent = "SIGTERM_then_SIGKILL"
        except ProcessLookupError:
            signal_sent = "SIGTERM_group_exited"
    try:
        process.wait(timeout=10)
        return {"signal": signal_sent, "leader_reaped": True}
    except subprocess.TimeoutExpired:
        # A pending SIGKILL cannot reap a process stuck in kernel I/O. Preserve
        # that unresolved cleanup condition instead of losing the time-limit record.
        return {"signal": signal_sent, "leader_reaped": False}


def _parent(args, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    request = {"schema_version": 1, "scope": SCOPE, "parameters": _parameters(args),
               "source_identity": _source_identity(),
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "thread_limits": {name: "1" for name in THREAD_VARIABLES + EXTRA_THREAD_VARIABLES}}
    _write_json(output / "request.json", request)
    environment = dict(os.environ)
    environment.update({name: "1" for name in THREAD_VARIABLES + EXTRA_THREAD_VARIABLES})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # A killed worker can leave an unfinished R bridge. Keep its temporary
    # inputs and raw logs inside this attempt instead of losing their location.
    temporary = output / "tmp"
    temporary.mkdir()
    environment["TMPDIR"] = str(temporary)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--method", args.method, "--n-obs", str(args.n_obs),
               "--n-rct", str(args.n_rct), "--d", str(args.d),
               "--n-eval", str(args.n_eval), "--seed", str(args.seed),
               "--timeout-seconds", str(args.timeout_seconds), "--output-dir", str(output)]
    process = None
    cleanup = {"signal": "not_started", "leader_reaped": False}
    status, returncode, error_type = "failed", None, None
    start = time.perf_counter()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       env=environment, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
                status = "completed" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status, error_type = "timeout", "WallTimeLimitExceeded"
            finally:
                cleanup = _kill_group(process)
                returncode = process.returncode
    except BaseException as exc:
        status, error_type = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", type(exc).__name__
        if process is not None:
            cleanup = _kill_group(process)
            returncode = process.returncode
        with (output / "parent_traceback.txt").open("x", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    profile = {**request, "status": status, "returncode": returncode,
               "launcher_elapsed_seconds": time.perf_counter() - start,
               "launcher_self_usage": _usage(resource.RUSAGE_SELF),
               "launcher_children_usage": _usage(resource.RUSAGE_CHILDREN),
               "termination": "worker_process_group_including_R_children",
               "process_group_cleanup": cleanup,
               "memory_interpretation": (
                   "Linux ru_maxrss is in KiB; separate self/child high-water marks "
                   "do not establish the simultaneous process-group or whole-job peak"),
               "server_only_artifacts": ["observed_inputs.npz", "predictions.npy", "stdout.log",
                                         "stderr.log", "native_stdout.log", "native_stderr.log",
                                         "native_parameters.npz", "worker_traceback.txt",
                                         "parent_traceback.txt", "tmp/"]}
    if error_type:
        profile["error_type"] = error_type
    for name in ("worker_result.json", "worker_preflight.json", "worker_inputs.json"):
        result_path = output / name
        if not result_path.exists():
            continue
        try:
            worker = json.loads(result_path.read_text())
            if not isinstance(worker, dict):
                raise ValueError("worker metadata must be an object")
            profile["worker"], profile["worker_record"] = worker, name
            if status == "completed" and (name != "worker_result.json" or worker.get("status") != "completed"):
                profile.update(status="failed", error_type="WorkerResultMismatch")
            break
        except (OSError, ValueError, AttributeError):
            profile.setdefault("unreadable_worker_records", []).append(name)
    if "worker" not in profile and status == "completed":
        profile.update(status="failed", error_type="WorkerResultMissing")
    _write_json(output / "profile.json", profile)
    print(json.dumps({"scope": SCOPE, "method": args.method, "status": profile["status"],
                      "elapsed_seconds": profile["launcher_elapsed_seconds"]}, sort_keys=True))
    return {"completed": 0, "timeout": 124, "interrupted": 130}.get(profile["status"], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--n-obs", type=int, required=True)
    parser.add_argument("--n-rct", type=int, required=True)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2642)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        output = _guard(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return _worker(args, output) if args.worker else _parent(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
