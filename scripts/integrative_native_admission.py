"""GSDS-only fixed native fidelity example; this is not a performance study.

Run under the newly provisioned v0.3 Slurm environment. Inputs, raw parameters
and complete diagnostics stay on the server. ``--direct-only --basis linear``
bounds an initial cost probe to one native fit without claiming admission.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView
from obs_rct_benchmark import integrative_r as adapter_module
from obs_rct_benchmark.runner import source_provenance
from obs_rct_benchmark.integrative_r import (
    OfficialIntegrativeR, _rscript, _source_provenance, _stratified_foldid,
    integrative_r_provenance,
)


ATOL, RTOL = 1e-10, 1e-8
ANALYTIC_ATOL = 1e-12
FIT_SEED, K_FOLDS = 41729, 5

# Independently assembled direct R call. It does not import the adapter's
# bridge or its prediction/basis helpers. The R implementation stays external.
DIRECT_R = r'''
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[[1]], mustWork=TRUE)
work <- normalizePath(args[[2]], mustWork=TRUE)
basis <- args[[3]]
seed <- as.integer(args[[4]])
k <- as.integer(args[[5]])
nr <- as.integer(args[[6]])
no <- as.integer(args[[7]])
nq <- as.integer(args[[8]])
d <- as.integer(args[[9]])
for (pkg in c("caret", "glmnet", "stringr", "ncvreg")) {
    if (!requireNamespace(pkg, quietly=TRUE)) stop("required native package unavailable")
}
native <- new.env(parent=globalenv())
sys.source(file.path(root, "R/utils.R"), envir=native)
sys.source(file.path(root, "R/intRlearner.R"), envir=native)
read_numeric <- function(name, n) {
    values <- readBin(file.path(work, name), "double", n=n, size=8L, endian="little")
    if (length(values) != n || any(!is.finite(values))) stop("invalid direct-call input")
    values
}
rct_x <- matrix(read_numeric("rct_X.bin", nr*d), ncol=d, byrow=TRUE)
obs_x <- matrix(read_numeric("obs_X.bin", no*d), ncol=d, byrow=TRUE)
query <- matrix(read_numeric("query_X.bin", nq*d), ncol=d, byrow=TRUE)
if (basis == "quadratic") {
    rct_x <- native$poly_interaction(rct_x, d=2, interact=TRUE)
    obs_x <- native$poly_interaction(obs_x, d=2, interact=TRUE)
    # The source routine drops dimensions on a one-row query. Native monomials
    # are defined on two duplicate rows, then the original row is retained.
    if (nq == 1L) query <- rbind(query, query)
    query <- native$poly_interaction(query, d=2, interact=TRUE)[seq_len(nq), , drop=FALSE]
}
x <- rbind(rct_x, obs_x)
w <- c(read_numeric("rct_T.bin", nr), read_numeric("obs_T.bin", no))
y <- c(read_numeric("rct_Y.bin", nr), read_numeric("obs_Y.bin", no))
s <- c(rep(1, nr), rep(0, no))
foldid <- as.integer(read_numeric("foldid.bin", nr+no))
set.seed(seed)
fit <- native$intrlearner(x=x, w=w, y=y, s=s, p=d+1L, k_folds=k, foldid=foldid)
prediction <- as.numeric(native$predict.rlasso(fit, query))
outputs <- list(prediction=prediction, tau_beta=as.numeric(fit$tau_beta),
    center=as.numeric(fit$standardization$mean), scale=as.numeric(fit$standardization$std),
    p_hat=as.numeric(fit$p_hat), m_hat=as.numeric(fit$m_hat),
    native_basis=as.numeric(t(x)), prediction_basis=as.numeric(t(query)))
for (name in names(outputs)) {
    writeBin(outputs[[name]], file.path(work, paste0(name, ".bin")), size=8L, endian="little")
}
writeLines(capture.output(utils::sessionInfo()), file.path(work, "session_info.txt"))
writeLines(capture.output(warnings()), file.path(work, "warnings.txt"))
saveRDS(fit, file.path(work, "native_fit.rds"))
'''


def _hash_array(array):
    array = np.asarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _observations():
    """Frozen observed arrays with deterministic residual variation, no oracle."""
    def sample(n, source, role):
        index = np.arange(n, dtype=np.float64)
        if source == "rct":
            x1 = np.linspace(-1.2, 1.2, n) + 0.07*np.sin(index)
            x2 = np.cos(0.31*index) + 0.2*np.sin(0.77*index)
            treatment = (index.astype(int) % 2).astype(np.float64)
            offset = 0.0
        else:
            x1 = np.linspace(-1.8, 1.8, n) + 0.1*np.cos(0.4*index)
            x2 = np.sin(0.49*index) + 0.2*np.cos(0.13*index)
            treatment = ((index.astype(int)*7) % 11 < 5).astype(np.float64)
            offset = 0.35*treatment*np.sin(x1)
        outcome = (1 + 0.3*x1 - 0.2*x2
                   + treatment*(0.7 + 0.4*x1 - 0.1*x2 + 0.2*x1*x2)
                   + offset + 0.15*np.sin(1.7*index) + 0.1*np.cos(0.9*index))
        return SourceSample(X=np.column_stack((x1, x2)), T=treatment, Y=outcome,
                            source=source, role=role)

    return TrainingView(obs_train=sample(120, "obs", "obs_train"),
                        rct_train=sample(80, "rct", "rct_train"),
                        rct_tune=sample(20, "rct", "rct_tune"))


def _independent_basis_d2(x, basis):
    if basis == "linear":
        return np.column_stack((x[:, 0], x[:, 1]))
    return np.column_stack((x[:, 0], x[:, 1], x[:, 0]*x[:, 0],
                            x[:, 1]*x[:, 1], x[:, 0]*x[:, 1]))


def _read(path, size):
    if path.stat().st_size != size*8:
        raise AssertionError(f"invalid direct float64 length: {path.name}")
    values = np.fromfile(path, dtype="<f8")
    if values.shape != (size,) or not np.isfinite(values).all():
        raise AssertionError(f"nonfinite direct result: {path.name}")
    return values


def _run_basis(output, training, basis, timeout_seconds, direct_only):
    work = output / basis
    work.mkdir()
    query = np.array([[0.2, 0.9]]) if basis == "quadratic" else np.array([
        [-1.0, 0.2], [-0.4, -0.7], [0.0, 0.0], [0.2, 0.9],
        [0.7, -0.3], [1.0, 1.1], [1.5, -1.2],
    ])
    foldid = _stratified_foldid(training, K_FOLDS, FIT_SEED)
    inputs = {"query_X": query, "foldid": foldid}
    for prefix, sample in (("rct", training.rct_train), ("obs", training.obs_train)):
        for name in ("X", "T", "Y"):
            inputs[f"{prefix}_{name}"] = getattr(sample, name)
    for name, values in inputs.items():
        np.asarray(values, dtype="<f8").tofile(work / f"{name}.bin")
    np.savez_compressed(work / "samples.npz", **inputs)
    record = {
        "basis": basis, "status": "started", "fit_seed": FIT_SEED,
        "native_seed": FIT_SEED % (2**31-1), "k_folds": K_FOLDS,
        "n_rct": 80, "n_obs": 120, "n_query": len(query), "d": 2,
        "direct_prediction_atol": ATOL, "direct_prediction_rtol": RTOL,
        "analytic_atol": ANALYTIC_ATOL, "direct_only": direct_only,
        "input_array_sha256": {name: _hash_array(x) for name, x in inputs.items()},
        "wrappers": [], "direct_script_sha256": hashlib.sha256(DIRECT_R.encode()).hexdigest(),
        "scope": "fixed observed native fidelity example; no estimation performance claim",
    }
    summary_path = work / "summary.json"
    _write_json(summary_path, record)
    method = None
    try:
        record["environment"] = integrative_r_provenance()
        if not record["environment"]["verified"]:
            raise RuntimeError("native source verification failed")
        if record["environment"]["runtime"]["availability"] != "available":
            raise RuntimeError("native R dependency inspection failed")
        source = Path(os.environ["INTEGRATIVE_R_SOURCE"]).expanduser().resolve()
        rscript = _rscript()
        executable_hash = hashlib.sha256(rscript.read_bytes()).hexdigest()
        direct_script = work / "direct.R"
        direct_script.write_text(DIRECT_R)
        command = [str(rscript), "--vanilla", str(direct_script), str(source), str(work),
                   basis, str(record["native_seed"]), str(K_FOLDS), "80", "120",
                   str(len(query)), "2"]
        # This exact command is private server evidence and may contain paths.
        _write_json(work / "direct_command_server_only.json", command)
        start = time.perf_counter()
        try:
            with (work / "direct.stdout.log").open("wb") as stdout, (work / "direct.stderr.log").open("wb") as stderr:
                completed = subprocess.run(command, cwd=work, stdout=stdout, stderr=stderr,
                                           timeout=timeout_seconds, check=False)
            record["direct_returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError("direct native R call failed")
        finally:
            record["direct_wall_seconds"] = time.perf_counter() - start
        if _source_provenance() != {k: v for k, v in record["environment"].items() if k != "runtime"}:
            raise RuntimeError("source changed during direct native fit")
        if hashlib.sha256(rscript.read_bytes()).hexdigest() != executable_hash:
            raise RuntimeError("Rscript changed during direct native fit")
        p = 2 if basis == "linear" else 5
        native = {"prediction": _read(work / "prediction.bin", len(query)),
                  "tau_beta": _read(work / "tau_beta.bin", 1+2*p),
                  "center": _read(work / "center.bin", p),
                  "scale": _read(work / "scale.bin", p)}
        pooled = np.concatenate((training.rct_train.X, training.obs_train.X))
        declared_basis = _independent_basis_d2(pooled, basis)
        declared_query = _independent_basis_d2(query, basis)
        r_basis = _read(work / "native_basis.bin", 200*p).reshape(200, p)
        r_query = _read(work / "prediction_basis.bin", len(query)*p).reshape(len(query), p)
        np.testing.assert_allclose(r_basis, declared_basis, atol=ANALYTIC_ATOL, rtol=0)
        np.testing.assert_allclose(r_query, declared_query, atol=ANALYTIC_ATOL, rtol=0)
        center = declared_basis.sum(axis=0) / 200
        scale = np.sqrt(((declared_basis-center)**2).sum(axis=0) / 199)
        np.testing.assert_allclose(native["center"], center, atol=ANALYTIC_ATOL, rtol=0)
        np.testing.assert_allclose(native["scale"], scale, atol=ANALYTIC_ATOL, rtol=0)
        independent_map = native["tau_beta"][0] + ((declared_query-center)/scale) @ native["tau_beta"][1:p+1]
        np.testing.assert_allclose(native["prediction"], independent_map, atol=ANALYTIC_ATOL, rtol=0)
        record["independent_basis_and_pooled_standardization_passed"] = True
        record["independent_prediction_map_max_abs"] = float(np.max(np.abs(native["prediction"]-independent_map)))
        if direct_only:
            record["status"] = "native_cost_probe_not_admitted"
            return record
        predictions = []
        for repetition in (1, 2):
            method = OfficialIntegrativeR(basis=basis, k_folds=K_FOLDS,
                                          timeout_seconds=timeout_seconds)
            fitting_view = training if repetition == 1 else replace(
                training,
                rct_tune=replace(training.rct_tune, Y=training.rct_tune.Y + 1e6),
                target_reference=TargetCovariateSample([[10000., -10000.], [-20000., 20000.]]),
            )
            start = time.perf_counter()
            try:
                prediction = method.fit_predict(fitting_view, query, seed=FIT_SEED)
            finally:
                elapsed = time.perf_counter() - start
                diagnostics = method.last_run_diagnostics_ or {}
                for stream, contents in diagnostics.items():
                    (work / f"wrapper_{repetition}.{stream}.log").write_text(contents)
                record["wrappers"].append({
                    "repetition": repetition, "wall_seconds": elapsed,
                    "tuning_reference_mutated": repetition == 2,
                    "attempt_provenance": method.last_run_provenance_,
                    "provenance": method.provenance_,
                    "warning_text_present": "warning" in diagnostics.get("stderr", "").lower(),
                })
                _write_json(summary_path, record)
            if prediction.shape != (len(query),):
                raise AssertionError("wrapper prediction shape changed")
            np.testing.assert_allclose(prediction, native["prediction"], atol=ATOL, rtol=RTOL)
            for name in ("tau_beta", "center", "scale"):
                np.testing.assert_allclose(method.native_parameters_[name], native[name], atol=ATOL, rtol=RTOL)
            predictions.append(prediction)
            np.savez_compressed(work / f"wrapper_{repetition}.npz", prediction=prediction,
                                **method.native_parameters_)
            record["wrappers"][-1]["direct_max_abs"] = float(np.max(np.abs(prediction-native["prediction"])))
        np.testing.assert_allclose(predictions[0], predictions[1], atol=ATOL, rtol=RTOL)
        record["repeat_max_abs"] = float(np.max(np.abs(predictions[0]-predictions[1])))
        record["status"] = "passed"
        return record
    except BaseException as exc:
        record["status"] = "failed"
        record["exception_type"] = type(exc).__name__
        (work / "failure_server_only.txt").write_text(traceback.format_exc())
        raise
    finally:
        for stream in ("stdout", "stderr"):
            log = work / f"direct.{stream}.log"
            if log.is_file():
                record[f"direct_{stream}_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
                if stream == "stderr":
                    record["direct_warning_text_present"] = "warning" in log.read_text(errors="replace").lower()
        _write_json(summary_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--basis", choices=("linear", "quadratic", "both"), default="both")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    raw_work = os.environ.get("GSDS_WORK_ROOT")
    if not raw_work or not Path(raw_work).is_absolute() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("native fitting requires a GSDS Slurm job and an absolute GSDS_WORK_ROOT")
    shared_work = Path(raw_work).resolve()
    if not output.is_relative_to(shared_work):
        raise SystemExit("native fitting output must be under GSDS_WORK_ROOT")
    if args.timeout_seconds < 1:
        raise SystemExit("timeout-seconds must be positive")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise SystemExit(f"the job must supply {variable}=1")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    if Path(adapter_module.__file__).resolve() != repo / "src/obs_rct_benchmark/integrative_r.py":
        raise RuntimeError("imported adapter does not belong to this driver producer checkout")
    producer = {
        "source": source_provenance(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "import_resolution": "verified_producer_checkout",
    }
    _write_json(output / "producer.json", producer)
    training = _observations()
    bases = ("linear", "quadratic") if args.basis == "both" else (args.basis,)
    records = [_run_basis(output, training, basis, args.timeout_seconds, args.direct_only)
               for basis in bases]
    _write_json(output / "integrative_native_admission.json", {
        "status": "native_cost_probe_not_admitted" if args.direct_only else "passed",
        "atol": ATOL, "rtol": RTOL, "analytic_atol": ANALYTIC_ATOL,
        "producer": producer,
        "records": records,
    })


if __name__ == "__main__":
    main()
