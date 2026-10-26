#!/usr/bin/env python3
"""Prospective native admission checks; run numerical stages on a server only.

Each --stage writes a new attempt directory and keeps full logs on success or
failure. Integrative R fit fidelity is deliberately a separate driver,
integrative_native_admission.py. This driver never changes the author checkouts.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback


ICM_REVISION = "0f0c6ca424b7ca44b84649ca072384b9933b2afe"
ICM_NOTEBOOK = "rebuttals/extrapolation.ipynb"
ICM_NOTEBOOK_SHA256 = "0326f655bac0993fd845412ce59fb0cde05e04c6f70890f42ef7e6750fcd855c"
ICM_CELL_INDEX = 3
ICM_CELL_SHA256 = "6c85f01d85087e3e69f75c1dbaf926bc8868b6b1be0521a699e00be643f7ca85"
INTR_REVISION = "a46ce2c1cc814865898b863d6a8e130bbc2c802b"
INTR_FILES = {
    "R/intRlearner.R": "98563e2423f41153fcbb2548f3b54ac779c72bcb8d154608bf51f778086fdce8",
    "R/utils.R": "2b85e2e56db83e33427f6f43aa1b59d995bafb6d46bfc1aec8e5f2e6a215525b",
}
NUMERICAL_VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.12.0", "GPy": "1.13.2", "paramz": "0.9.6",
}

# These observed arrays and all numerical rules are fixed before any fit output.
OBSERVED_FIXTURE = {
    "rct_X": [[-1.5, -.8], [-1.2, .3], [-.9, 1.1], [-.6, -1.2],
              [-.3, .7], [0., -.4], [.3, 1.4], [.6, -.9], [.9, .1],
              [1.2, 1.], [1.5, -.6], [1.8, .5]],
    "rct_T": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "rct_Y": [-.7, .4, 1.2, .2, .5, 1.6, 1.5, 1.1, .7, 2.8, 1.9, 3.1],
    "obs_X": [[-1.8, .2], [-1.5, -.7], [-1.2, 1.2], [-.9, -.3],
              [-.6, .8], [-.3, -1.1], [0., .4], [.3, 1.3], [.6, -.5],
              [.9, .9], [1.2, -1.3], [1.5, .1], [1.8, .7],
              [2.1, -.8], [2.4, 1.4], [2.7, -.1]],
    "obs_T": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "obs_Y": [-.5, 1.8, .2, 2., -.8, 2.4, .7, 1., 1.3, 3.5,
              1.1, 3., 2., 3.6, 1.4, 4.1],
    "query_X": [[-1.1, .2], [0., 0.], [.8, -.7], [1.6, 1.]],
    "tune_X": [[7., 8.], [9., 10.]], "tune_T": [0, 1],
    "tune_Y": [50., -30.], "reference_X": [[-7., 5.], [4., -6.]],
}
PLAN = {
    "schema_version": 1,
    "scope": "Independent native-subroutine checks and fixed-rho Causal-ICM fidelity; no campaign",
    "analytic_atol": 1e-12, "analytic_rtol": 0.,
    "fidelity_atol": 1e-10, "fidelity_rtol": 1e-8,
    "native_fidelity_seed": 2601026, "native_fidelity_rho": .5,
    "native_prediction": "treated task0 mean minus control task0 mean",
    "native_source": {"revision": ICM_REVISION, "notebook": ICM_NOTEBOOK,
                      "notebook_sha256": ICM_NOTEBOOK_SHA256,
                      "cell_index": ICM_CELL_INDEX, "cell_sha256": ICM_CELL_SHA256},
    "numerical_versions": NUMERICAL_VERSIONS,
    "gp_analytic": {
        "x": [-1., 0., .5, 1.5], "task": [0, 1, 0, 1],
        "y": [.5, -1., 2., .25], "query_x": [-.25, .75, -.25, .75],
        "query_task": [0, 0, 1, 1], "rho": .6, "rbf_variance": 1.3,
        "rbf_lengthscale": .8, "noise_variance": .25,
        "kappa": 1e-9, "training_diagonal_jitter": 1e-8,
    },
    "intr_analytic": {
        "preprocess_rows": [[-1., -6.], [1., -2.], [3., 2.]],
        "center": [1., -2.], "scale": [2., 4.],
        "tau_beta": [3., 5., -7., 11., 13.],
        "query": [[1., -2.], [3., 2.], [-1., -6.]], "expected": [3., 1., 5.],
        "changed_confounding_coefficients": [1000000., -1000000.],
    },
    "fixture": OBSERVED_FIXTURE,
    "adapter_repeat2_mutation": "add 1e6 to tuning Y; replace outcome-free reference X; fitting rows unchanged",
    "mixed_source_diagnostic": "must differ from trial contrast at atol1e-12 rtol0",
    "fit_limits": "Tiny arrays; untouched native optimize defaults; scheduler bounds total runtime",
    "intR_fit_fidelity": "Separate integrative_native_admission.py; linear and quadratic each need direct plus two fresh adapter fits",
}
STAGES = ("analytic-gp", "analytic-intr", "icm-direct", "icm-adapter-1",
          "icm-adapter-2", "icm-compare", "plan-only")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.PIPE, timeout=15).strip()


def source_root(variable, revision, files):
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"missing {variable}")
    root = Path(raw).expanduser().resolve(strict=True)
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError("source path is not the checkout top-level")
    if git(root, "rev-parse", "HEAD") != revision:
        raise RuntimeError("source revision differs from the prospective pin")
    status = git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError("author checkout has tracked changes")
    measured = {}
    for relative, expected in files.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError("author source must be a regular in-checkout file")
        measured[relative] = file_hash(path)
        if measured[relative] != expected:
            raise RuntimeError(f"source hash mismatch: {relative}")
    return root, {"revision": revision, "file_sha256": measured,
                  "checkout_path_sha256": hashlib.sha256(str(root).encode()).hexdigest()}


def environment(require_gpy):
    import numpy as np
    result = {"python": platform.python_version(), "executable_sha256": file_hash(sys.executable),
              "numpy": np.__version__, "platform": platform.platform()}
    if require_gpy:
        for name, expected in NUMERICAL_VERSIONS.items():
            version = importlib.metadata.version(name)
            if version != expected:
                raise RuntimeError(f"{name} must be {expected}, found {version}")
            result[name] = version
        import GPy
        import GPy.inference.latent_function_inference.exact_gaussian_inference as exact
        result["GPy_init_sha256"] = file_hash(GPy.__file__)
        result["GPy_exact_inference_sha256"] = file_hash(exact.__file__)
        # A source-level prerequisite, rather than inferring jitter from output.
        if "diag.add(Ky, variance+1e-8)" not in Path(exact.__file__).read_text():
            raise RuntimeError("GPy training-diagonal rule differs from the frozen analytic formula")
    return result


def check_close(actual, expected, *, atol, rtol, name):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{name}: shape or finiteness failure {a.shape} vs {b.shape}")
    maximum = float(np.max(np.abs(a - b))) if a.size else 0.
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max absolute discrepancy {maximum}; fixed atol={atol}, rtol={rtol}")
    return maximum


def analytic_gp(attempt, args):
    import numpy as np
    env = environment(True)
    import GPy
    spec = PLAN["gp_analytic"]
    x, task = np.array(spec["x"]), np.array(spec["task"], dtype=int)
    y = np.array(spec["y"], dtype=np.float64)[:, None]
    q, qt = np.array(spec["query_x"]), np.array(spec["query_task"], dtype=int)
    rho, kappa = spec["rho"], spec["kappa"]
    # This covariance and solve use no GPy or adapter prediction/fitting helper.
    B = np.array([[1. + kappa, rho], [rho, 1. + kappa]])

    def covariance(a, at, b, bt):
        return (spec["rbf_variance"] * np.exp(-((a[:, None] - b[None, :]) ** 2)
                / (2. * spec["rbf_lengthscale"] ** 2)) * B[at[:, None], bt[None, :]])

    K = covariance(x, task, x, task)
    Kstar = covariance(q, qt, x, task)
    Kquery = covariance(q, qt, q, qt)
    C = K + np.eye(len(x)) * (spec["noise_variance"] + spec["training_diagonal_jitter"])
    manual_mean = Kstar @ np.linalg.solve(C, y)
    manual_variance = np.diag(Kquery - Kstar @ np.linalg.solve(C, Kstar.T))[:, None]
    W = np.array([[1., 0.], [rho, np.sqrt(1. - rho ** 2)]])
    kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                          variance=spec["rbf_variance"], lengthscale=spec["rbf_lengthscale"])
              ** GPy.kern.Coregionalize(input_dim=1, output_dim=2, rank=2,
                                       W=W, kappa=np.full(2, kappa)))
    X, Q = np.column_stack((x, task)), np.column_stack((q, qt))
    model = GPy.models.GPRegression(X, y, kernel, noise_var=spec["noise_variance"])
    native_mean, native_variance = model.predict_noiseless(Q)
    maximum = {}
    for name, actual, expected in [
        ("coregionalization", model.kern.coregion.B, B),
        ("training_covariance", model.kern.K(X), K),
        ("cross_covariance", model.kern.K(Q, X), Kstar),
        ("posterior_mean", native_mean, manual_mean),
        ("posterior_latent_variance", native_variance, manual_variance),
    ]:
        maximum[name] = check_close(actual, expected, atol=1e-12, rtol=0., name=name)
    return {"environment": env, "max_absolute_discrepancy": maximum,
            "manual_mean": manual_mean.tolist(), "native_mean": native_mean.tolist(),
            "manual_latent_variance": manual_variance.tolist(),
            "native_latent_variance": native_variance.tolist(), "optimization_performed": False}


R_ANALYTIC = r'''
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[[1]], mustWork=TRUE)
out <- normalizePath(args[[2]], mustWork=TRUE)
for (pkg in c("caret", "stringr")) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop("missing analytic-map R dependency")
}
native <- new.env(parent=globalenv())
sys.source(file.path(root, "R/utils.R"), envir=native)
sys.source(file.path(root, "R/intRlearner.R"), envir=native)
preprocess_X <- matrix(c(-1,-6, 1,-2, 3,2), ncol=2, byrow=TRUE)
colnames(preprocess_X) <- c("covariate_1", "covariate_2")
standardization <- caret::preProcess(preprocess_X, method=c("center", "scale"))
fit <- list(standardization=standardization, rs=TRUE,
            tau_beta=c(3,5,-7,11,13))
class(fit) <- "rlasso"
query <- matrix(c(1,-2, 3,2, -1,-6), ncol=2, byrow=TRUE)
first <- as.numeric(native$predict.rlasso(fit, query))
fit$tau_beta[4:5] <- c(1000000, -1000000)
second <- as.numeric(native$predict.rlasso(fit, query))
writeBin(c(first, second), file.path(out, "native_prediction.bin"), size=8L, endian="little")
writeBin(as.numeric(standardization$mean), file.path(out, "center.bin"), size=8L, endian="little")
writeBin(as.numeric(standardization$std), file.path(out, "scale.bin"), size=8L, endian="little")
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
'''


def analytic_intr(attempt, args):
    import numpy as np
    root, source = source_root("INTEGRATIVE_R_SOURCE", INTR_REVISION, INTR_FILES)
    executable = shutil.which(os.path.expanduser(os.environ.get("INTEGRATIVE_R_RSCRIPT", "Rscript")))
    if executable is None:
        raise RuntimeError("Rscript is unavailable")
    script = attempt / "analytic_map.R"
    script.write_text(R_ANALYTIC)
    with (attempt / "native_r.stdout.log").open("wb") as stdout, (attempt / "native_r.stderr.log").open("wb") as stderr:
        completed = subprocess.run([executable, "--vanilla", str(script), str(root), str(attempt)],
                                   cwd=attempt, stdout=stdout, stderr=stderr,
                                   timeout=args.timeout_seconds, check=False)
    if completed.returncode:
        raise RuntimeError(f"native R analytic map failed: exit {completed.returncode}; full logs retained")
    values = np.fromfile(attempt / "native_prediction.bin", dtype="<f8")
    center = np.fromfile(attempt / "center.bin", dtype="<f8")
    scale = np.fromfile(attempt / "scale.bin", dtype="<f8")
    spec = PLAN["intr_analytic"]
    maximum = {
        "native_predictions": check_close(values, spec["expected"] * 2, atol=1e-12, rtol=0., name="native prediction map"),
        "native_center": check_close(center, spec["center"], atol=1e-12, rtol=0., name="native center"),
        "native_scale": check_close(scale, spec["scale"], atol=1e-12, rtol=0., name="native scale"),
    }
    # Adapter helper is the subject of this extra check, never the oracle.
    from obs_rct_benchmark.integrative_r import _native_prediction_map
    mapped = _native_prediction_map(spec["query"], spec["tau_beta"], spec["center"],
                                    spec["scale"], basis="linear")
    maximum["adapter_prediction_map"] = check_close(mapped, spec["expected"],
                                                     atol=1e-12, rtol=0., name="adapter prediction map")
    return {"source": source, "environment": environment(False),
            "rscript_sha256": file_hash(executable), "max_absolute_discrepancy": maximum,
            "native_predictions": values.tolist(), "adapter_predictions": mapped.tolist(),
            "r_runtime_session_info_sha256": file_hash(attempt / "sessionInfo.txt"),
            "native_fitting_performed": False}


def direct_icm_function():
    # Independent source loader: no adapter loader/provenance/fitting helper.
    root, metadata = source_root("CAUSAL_ICM_SOURCE", ICM_REVISION,
                                  {ICM_NOTEBOOK: ICM_NOTEBOOK_SHA256})
    notebook = json.loads((root / ICM_NOTEBOOK).read_bytes())
    cell = notebook["cells"][ICM_CELL_INDEX]
    source = "".join(cell["source"])
    if cell["cell_type"] != "code" or hashlib.sha256(source.encode()).hexdigest() != ICM_CELL_SHA256:
        raise RuntimeError("native ICM cell differs from the prospective identity")
    tree = ast.parse(source)
    if ([type(node).__name__ for node in tree.body] != ["Import", "Import", "FunctionDef"]
            or tree.body[-1].name != "ICM"):
        raise RuntimeError("native ICM cell contains unexpected statements")
    metadata.update(cell_index=ICM_CELL_INDEX, cell_sha256=ICM_CELL_SHA256)
    helper = root / "functions/model_training.py"
    try:
        ast.parse(helper.read_text())
    except SyntaxError as exc:
        metadata["nominal_helper"] = {"status": "SyntaxError_preserved",
                                      "line": exc.lineno, "message": exc.msg,
                                      "sha256": file_hash(helper)}
    else:
        raise RuntimeError("expected broken nominal helper no longer matches inspected upstream state")
    namespace = {"__name__": "_direct_author_icm_admission"}
    exec(compile(tree, f"{ICM_NOTEBOOK}:cell{ICM_CELL_INDEX}", "exec"), namespace)
    return namespace["ICM"], metadata


def icm_direct(attempt, args):
    import numpy as np
    env = environment(True)
    native, source = direct_icm_function()
    f = OBSERVED_FIXTURE
    np.random.seed(PLAN["native_fidelity_seed"])
    random.seed(PLAN["native_fidelity_seed"])
    m0, m1 = native(X_E=np.array(f["rct_X"], dtype=np.float64),
                    X_O=np.array(f["obs_X"], dtype=np.float64),
                    T_E=np.array(f["rct_T"], dtype=np.float64),
                    T_O=np.array(f["obs_T"], dtype=np.float64),
                    Y_E=np.array(f["rct_Y"], dtype=np.float64),
                    Y_O=np.array(f["obs_Y"], dtype=np.float64),
                    r=2, ID=2, AD=[0, 1], rho=PLAN["native_fidelity_rho"])
    q = np.array(f["query_X"], dtype=np.float64)
    q0, q1 = np.column_stack((q, np.zeros(len(q)))), np.column_stack((q, np.ones(len(q))))
    c0, v0 = m0.predict_noiseless(q0)
    t0, v1 = m1.predict_noiseless(q0)
    c1, _ = m0.predict_noiseless(q1)
    t1, _ = m1.predict_noiseless(q1)
    for name, value in [("control", c0), ("treated", t0), ("control variance", v0),
                        ("treated variance", v1), ("OBS control", c1), ("OBS treated", t1)]:
        if np.shape(value) != (len(q), 1) or not np.isfinite(value).all():
            raise AssertionError(f"native {name} shape/finiteness failure")
    trial, mixed = (t0 - c0)[:, 0], (t1 - c0)[:, 0]
    if np.allclose(trial, mixed, atol=1e-12, rtol=0.):
        raise AssertionError("frozen fixture did not distinguish mixed-source and trial contrasts")
    return {"source": source, "environment": env, "fixture_sha256": digest(f),
            "prediction": trial.tolist(), "trial_control_mean": c0[:, 0].tolist(),
            "trial_treated_mean": t0[:, 0].tolist(),
            "trial_latent_variance": (v0 + v1)[:, 0].tolist(),
            "observational_contrast": (t1 - c1)[:, 0].tolist(),
            "wrong_mixed_source_contrast": mixed.tolist(),
            "mixed_source_max_difference": float(np.max(np.abs(trial - mixed))),
            "native_arm_parameters": [m0.param_array.tolist(), m1.param_array.tolist()]}


def training_view(mutate=False):
    import numpy as np
    from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView
    f = OBSERVED_FIXTURE
    obs = SourceSample(f["obs_X"], f["obs_T"], f["obs_Y"], "obs", "obs_train")
    rct = SourceSample(f["rct_X"], f["rct_T"], f["rct_Y"], "rct", "rct_train")
    tune_y = np.array(f["tune_Y"]) + (1e6 if mutate else 0.)
    tune = SourceSample(f["tune_X"], f["tune_T"], tune_y, "rct", "rct_tune")
    reference = [[10000., -10000.], [-20000., 20000.]] if mutate else f["reference_X"]
    return TrainingView(obs, rct, tune, TargetCovariateSample(reference))


def icm_adapter(attempt, args, repeat):
    import numpy as np
    env = environment(True)
    from obs_rct_benchmark.causal_icm import OfficialCausalICM
    method = OfficialCausalICM(rho=PLAN["native_fidelity_rho"])
    try:
        prediction = method.fit_predict(training_view(mutate=repeat == 2),
                                        np.array(OBSERVED_FIXTURE["query_X"], dtype=np.float64),
                                        seed=PLAN["native_fidelity_seed"])
    except BaseException:
        diagnostics = getattr(method, "last_run_diagnostics_", None)
        if diagnostics is not None:
            write_json(attempt / "adapter_failure_diagnostics.json", diagnostics)
        raise
    if np.shape(prediction) != (len(OBSERVED_FIXTURE["query_X"]),) or not np.isfinite(prediction).all():
        raise AssertionError("adapter CATE shape/finiteness failure")
    return {"environment": env, "fixture_sha256": digest(OBSERVED_FIXTURE),
            "prediction": prediction.tolist(), "adapter_provenance": method.provenance_,
            "repeat": repeat, "tuning_reference_mutated": repeat == 2,
            "evaluation_outcomes_accessible": False}


def latest_stage(output, stage):
    paths = sorted((output / stage).glob("*/record.json"))
    if not paths:
        raise RuntimeError(f"required stage has not run: {stage}")
    record = json.loads(paths[-1].read_text())
    if record["status"] != "passed":
        raise RuntimeError(f"latest required stage failed: {stage}; retain its evidence")
    if record["plan_sha256"] != digest(PLAN):
        raise RuntimeError(f"required stage used a different prospective plan: {stage}")
    return record, str(paths[-1].relative_to(output))


def icm_compare(attempt, args):
    required = ("analytic-gp", "icm-direct", "icm-adapter-1", "icm-adapter-2")
    records, locations = {}, {}
    for name in required:
        records[name], locations[name] = latest_stage(args.output_dir, name)
    data = {name: record["result"] for name, record in records.items()}
    for name in required[1:]:
        if data[name]["fixture_sha256"] != digest(OBSERVED_FIXTURE):
            raise AssertionError("native/adapter fixtures differ")
        if data[name]["environment"] != data["icm-direct"]["environment"]:
            raise AssertionError("native/adapter numerical environment identity differs")
    maximum = {}
    for name in ("icm-adapter-1", "icm-adapter-2"):
        maximum[name] = check_close(data[name]["prediction"], data["icm-direct"]["prediction"],
                                    atol=1e-10, rtol=1e-8, name=name + " vs direct native")
    maximum["fresh_repeat_and_information_mutation"] = check_close(
        data["icm-adapter-1"]["prediction"], data["icm-adapter-2"]["prediction"],
        atol=1e-10, rtol=1e-8, name="fresh repeats with tuning/reference mutation")
    producer_hashes = [records[name]["producer_sha256"] for name in required[1:]]
    if any(value != producer_hashes[0] for value in producer_hashes[1:]):
        raise AssertionError("producer source changed between native/adapter fidelity stages")
    return {"required_stage_records": locations, "max_absolute_discrepancy": maximum,
            "admitted_scope": "fixed-rho native two-arm trial-response mean component on this checked environment",
            "not_established": ["paper table reproduction", "adaptive-rho recipe", "performance superiority",
                                "unrestricted transport", "campaign readiness", "intR fit fidelity"]}


def producer_identity(repo):
    return {name: file_hash(repo / "src/obs_rct_benchmark" / name)
            for name in ("causal_icm.py", "integrative_r.py", "data.py")}


def run_stage(stage, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = args.output_dir / stage / f"{stamp}-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / "prospective_plan.json", PLAN)
    record = {"stage": stage, "status": "running", "plan_sha256": digest(PLAN),
              "driver_sha256": file_hash(__file__), "producer_sha256": producer_identity(args.repo),
              "started_utc": stamp, "attempt": str(attempt.relative_to(args.output_dir))}
    write_json(attempt / "record.json", record)
    started = time.monotonic()
    print(f"Starting {stage}; evidence {attempt}", flush=True)
    with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                if stage == "analytic-gp":
                    result = analytic_gp(attempt, args)
                elif stage == "analytic-intr":
                    result = analytic_intr(attempt, args)
                elif stage == "icm-direct":
                    result = icm_direct(attempt, args)
                elif stage.startswith("icm-adapter-"):
                    result = icm_adapter(attempt, args, int(stage[-1]))
                elif stage == "icm-compare":
                    result = icm_compare(attempt, args)
                elif stage == "plan-only":
                    result = {"plan_sha256": digest(PLAN), "numerical_execution": False}
                else:
                    raise ValueError(stage)
                record.update(status="passed", result=result)
            except BaseException as exc:
                traceback.print_exc(file=stderr)
                record.update(status="failed", exception_type=type(exc).__name__,
                              exception_detail_server_only="stderr.log")
    record["wall_seconds"] = time.monotonic() - started
    record["self_maxrss_platform_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["children_maxrss_platform_units"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record["maxrss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
    record["log_sha256"] = {name: file_hash(attempt / name) for name in ("stdout.log", "stderr.log")}
    write_json(attempt / "record.json", record)
    print(f"{stage}: {record['status']}; wall {record['wall_seconds']:.3f}s", flush=True)
    return record["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="benchmark producer checkout")
    parser.add_argument("--output-dir", required=True, type=Path, help="server evidence root")
    parser.add_argument("--stage", required=True, choices=STAGES, action="append")
    parser.add_argument("--timeout-seconds", type=int, default=120,
                        help="R analytic-map subprocess limit; native fitting retains optimizer defaults")
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    # The same driver can travel with a public source tree. The actual shared
    # work root is supplied by the destination job rather than a personal path.
    work_raw = os.environ.get("GSDS_WORK_ROOT")
    if not work_raw or not os.environ.get("SLURM_JOB_ID"):
        parser.error("stages require GSDS_WORK_ROOT and a Slurm job")
    work_root = Path(work_raw).expanduser().resolve(strict=True)
    if not args.output_dir.is_relative_to(work_root):
        parser.error("output-dir must be inside GSDS_WORK_ROOT")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            parser.error(f"job must set {variable}=1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo / "src"))
    sys.dont_write_bytecode = True
    for stage in args.stage:
        if not run_stage(stage, args):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
