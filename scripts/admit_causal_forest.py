"""Allocated-server admission of the pinned RCT-only causal-forest recipe.

This checks source-call fidelity and a prediction map, not causal accuracy.
Observed arrays, native fitted forest, full process logs and failed attempts
stay in the chosen server output directory. No evaluation truth is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from obs_rct_benchmark.data import SourceSample, TrainingView
from obs_rct_benchmark.runner import source_provenance


FIT_SEED, DATA_SEED, TREES = 41729, 20261027, 2000
ATOL, RTOL, MAP_ATOL = 1e-10, 1e-8, 1e-12
PROFILE_SIZES, PROFILE_QUERIES = (128, 512), 2048

# Independently assembled direct call; neither the adapter's bridge nor its
# serialization/prediction helpers enter this reference.
DIRECT_R = r'''
args <- commandArgs(trailingOnly=TRUE)
lib <- normalizePath(args[[1]], mustWork=TRUE)
work <- normalizePath(args[[2]], mustWork=TRUE)
seed <- as.integer(args[[3]])
n <- as.integer(args[[4]])
d <- as.integer(args[[5]])
nq <- as.integer(args[[6]])
.libPaths(c(lib, .libPaths()))
loadNamespace("grf", lib.loc=lib)
if (normalizePath(getNamespaceInfo(asNamespace("grf"), "path")) !=
    normalizePath(file.path(lib, "grf"))) stop("unexpected native package")
read_values <- function(name, count) {
  file <- file.path(work, name)
  if (file.info(file)$size != 8*count) stop("invalid direct input length")
  value <- readBin(file, "double", n=count, size=8L, endian="little")
  if (length(value) != count || any(!is.finite(value))) stop("invalid direct input")
  value
}
x <- matrix(read_values("X.bin", n*d), ncol=d, byrow=TRUE)
a <- read_values("T.bin", n)
y <- read_values("Y.bin", n)
e <- read_values("propensity.bin", n)
query <- matrix(read_values("query.bin", nq*d), ncol=d, byrow=TRUE)
options(grf.legacy.seed=FALSE, grf.verbose=FALSE)
set.seed(seed)
started <- proc.time()[["elapsed"]]
fit <- grf::causal_forest(X=x, Y=y, W=a, Y.hat=NULL, W.hat=e,
  num.trees=2000, num.threads=1, seed=seed,
  sample.weights=NULL, clusters=NULL, equalize.cluster.weights=FALSE,
  sample.fraction=0.5, mtry=min(ceiling(sqrt(d)+20), d), min.node.size=5,
  honesty=TRUE, honesty.fraction=0.5, honesty.prune.leaves=TRUE,
  alpha=0.05, imbalance.penalty=0, stabilize.splits=TRUE,
  ci.group.size=2, tune.parameters="none", compute.oob.predictions=TRUE)
fit_seconds <- proc.time()[["elapsed"]]-started
started <- proc.time()[["elapsed"]]
prediction <- predict(fit, newdata=query, num.threads=1,
  estimate.variance=FALSE, linear.correction.variables=NULL)$predictions
prediction_seconds <- proc.time()[["elapsed"]]-started
weights <- as.matrix(grf::get_forest_weights(fit, query, num.threads=1))
if (!identical(dim(weights), c(nq, n))) stop("invalid native weight dimensions")
outputs <- list(prediction=as.numeric(prediction), Y_hat=as.numeric(fit$Y.hat),
  W_hat=as.numeric(fit$W.hat), weights=as.numeric(t(weights)),
  times=c(fit_seconds, prediction_seconds))
for (name in names(outputs)) {
  if (any(!is.finite(outputs[[name]]))) stop("nonfinite direct output")
  writeBin(outputs[[name]], file.path(work, paste0(name, ".bin")),
    size=8L, endian="little")
}
saveRDS(fit, file.path(work, "native_fit.rds"))
writeLines(capture.output(sessionInfo()), file.path(work, "session_info.txt"))
writeLines(capture.output(warnings()), file.path(work, "warnings.txt"))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def weighted_intercept_slopes(weights, treatment, outcome, y_hat, propensity):
    """Independent scalar arithmetic for weighted residual regression with intercept.

    Uses supplied neighborhoods/nuisances; it does not independently check how
    native trees were trained, or establish causal identification or coverage.
    """
    weights = np.asarray(weights, dtype=np.float64)
    vectors = [np.asarray(v, dtype=np.float64) for v in
               (treatment, outcome, y_hat, propensity)]
    if (weights.ndim != 2 or not weights.shape[0]
            or any(v.ndim != 1 or len(v) != weights.shape[1] for v in vectors)
            or not np.isfinite(weights).all()
            or any(not np.isfinite(v).all() for v in vectors)
            or np.any(weights < 0)):
        raise ValueError("invalid weighted-regression inputs")
    a, y, m, e = vectors
    result, denominators = [], []
    for row in weights:
        if not math.isclose(math.fsum(map(float, row)), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("native neighborhood weights must sum to one")
        sw = math.fsum(float(w)*(float(t)-float(p)) for w, t, p in zip(row, a, e))
        sy = math.fsum(float(w)*(float(v)-float(h)) for w, v, h in zip(row, y, m))
        swy = math.fsum(float(w)*(float(t)-float(p))*(float(v)-float(h))
                        for w, t, p, v, h in zip(row, a, e, y, m))
        sww = math.fsum(float(w)*(float(t)-float(p))**2 for w, t, p in zip(row, a, e))
        denominator = sww-sw*sw
        if not denominator > 0:
            raise ValueError("nonpositive local treatment variance")
        result.append((swy-sy*sw)/denominator)
        denominators.append(denominator)
    return np.asarray(result), np.asarray(denominators)


def observations(n: int, nq: int):
    """Fixed observed numerical fixture; no truth/evaluator objects are returned."""
    rng = np.random.default_rng(DATA_SEED)
    x = rng.normal(size=(512, 5))
    e = 0.25+0.5/(1+np.exp(-x[:, 0]))
    a = (rng.random(512) < e).astype(float)
    y = (np.sin(x[:, 1])+0.5*x[:, 2]
         +a*(1+0.4*x[:, 0])+0.3*rng.normal(size=512))
    query = np.random.default_rng(DATA_SEED+1).normal(size=(nq, 5))
    sample = SourceSample(x[:n], a[:n], y[:n], "rct", "rct_train", e[:n])
    return sample, query


def read_doubles(path: Path, size: int):
    if path.stat().st_size != size*8:
        raise AssertionError("invalid retained native output size")
    values = np.fromfile(path, dtype="<f8")
    if values.shape != (size,) or not np.isfinite(values).all():
        raise AssertionError("invalid retained native output")
    return values


def direct_call(sample, query, output: Path):
    output.mkdir()
    for name, value in (("X", sample.X), ("T", sample.T), ("Y", sample.Y),
                        ("propensity", sample.known_propensity), ("query", query)):
        np.asarray(value, dtype="<f8").tofile(output/(name+".bin"))
    script = output/"direct.R"
    script.write_text(DIRECT_R)
    command = [os.environ["GRF_RSCRIPT"], "--vanilla", str(script),
               os.environ["GRF_LIBRARY"], str(output), str(FIT_SEED),
               str(sample.n), str(sample.d), str(len(query))]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    (output/"stdout.log").write_text(completed.stdout)
    (output/"stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("direct native call failed; full process logs retained")
    prediction = read_doubles(output/"prediction.bin", len(query))
    return prediction, {"wall_seconds": time.monotonic()-started,
                        "R_fit_seconds": float(read_doubles(output/"times.bin", 2)[0]),
                        "R_prediction_seconds": float(read_doubles(output/"times.bin", 2)[1])}


def worker(input_path: Path, output_path: Path, metadata_path: Path, mode: str):
    from obs_rct_benchmark.causal_forest import RCTCausalForest

    with np.load(input_path) as arrays:
        sample = SourceSample(arrays["X"], arrays["T"], arrays["Y"], "rct", "rct_train", arrays["e"])
        query = arrays["query"]
    estimator = RCTCausalForest()
    started = time.monotonic()
    if mode == "hostile":
        class RCTOnlyView(TrainingView):
            def __getattribute__(self, name):
                if name in {"obs_train", "rct_tune", "target_reference"}:
                    raise AssertionError("excluded training field was accessed: " + name)
                return object.__getattribute__(self, name)

        # A typed adversarial view: even reading excluded roles must fail.
        view = object.__new__(RCTOnlyView)
        object.__setattr__(view, "rct_train", sample)
        prediction = estimator.fit_predict(view, query, seed=FIT_SEED)
    else:
        prediction = estimator.fit_predict_sample(sample, query, seed=FIT_SEED)
    elapsed = time.monotonic()-started
    np.save(output_path, prediction, allow_pickle=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_json(metadata_path, {
        "adapter_wall_seconds": elapsed,
        "children_cpu_seconds": usage.ru_utime+usage.ru_stime,
        "largest_child_RSS_KiB": usage.ru_maxrss,
        "memory_scope": "largest child-process RSS mark, not simultaneous whole-job peak",
        "mode": mode, "requested_threads": 1, "requested_causal_trees": TREES,
        "rct_outcomes_consumed": sample.n, "obs_outcomes_consumed": 0,
        "tuning_outcomes_consumed": 0,
    })


def adapter_call(sample, query, output: Path, mode="sample"):
    output.mkdir()
    input_path = output/"input.npz"
    np.savez(input_path, X=sample.X, T=sample.T, Y=sample.Y,
             e=sample.known_propensity, query=query)
    command = [sys.executable, str(Path(__file__).resolve()),
               "--worker-input", str(input_path), "--worker-output", str(output/"prediction.npy"),
               "--worker-metadata", str(output/"metadata.json"), "--worker-mode", mode]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=660)
    (output/"stdout.log").write_text(completed.stdout)
    (output/"stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("adapter worker failed; full process logs retained")
    return np.load(output/"prediction.npy", allow_pickle=False), json.loads((output/"metadata.json").read_text())


def admit(output: Path):
    from obs_rct_benchmark.causal_forest import causal_forest_provenance

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run admission in an allocated server job")
    output.mkdir(parents=True, exist_ok=False)
    sample, query = observations(128, 8)
    fixture = output/"fixture.npz"
    np.savez(fixture, X=sample.X, T=sample.T, Y=sample.Y,
             e=sample.known_propensity, query=query)
    plan = {
        "package_source": source_provenance(), "driver_sha256": sha256(Path(__file__)),
        "fixture_sha256": sha256(fixture), "fit_seed": FIT_SEED, "data_seed": DATA_SEED,
        "native_seed_rule": "uint32 seed modulo (2**31-1)",
        "causal_trees": TREES, "native_outcome_nuisance_trees": 500,
        "threads": 1, "fidelity_atol": ATOL, "fidelity_rtol": RTOL,
        "prediction_map_absolute_tolerance": MAP_ATOL,
        "query_extension_rule": "append two fixed distant rows, compare original eight predictions",
        "profile_rct_rows": list(PROFILE_SIZES), "profile_query_rows": PROFILE_QUERIES,
        "scope": "native-call/prediction-map admission and bounded costs; no accuracy ranking",
        "written_before_first_fit": True,
    }
    write_json(output/"prospective_plan.json", plan)
    try:
        provenance = causal_forest_provenance()
        write_json(output/"provenance_before.json", provenance)
        if not provenance.get("verified") or provenance.get("availability") != "available":
            raise RuntimeError("native source/build/runtime validation failed before fitting")
        # Exact hand oracle: local intercept gives 8/3, not 3.2 or 43/12.
        toy, _ = weighted_intercept_slopes([[.1, .2, .3, .4]],
                    [0, 1, 0, 1], [0, 3, 1, 5], [0, 1, 1, 2], [.5]*4)
        if abs(float(toy[0])-8/3) > MAP_ATOL:
            raise AssertionError("independent hand oracle failed")
        direct, direct_cost = direct_call(sample, query, output/"direct")
        repeats = []
        for label, mode in (("adapter_1", "sample"), ("adapter_2", "hostile")):
            prediction, cost = adapter_call(sample, query, output/label, mode)
            np.testing.assert_allclose(prediction, direct, atol=ATOL, rtol=RTOL)
            repeats.append({"call": label, "mode": mode,
                            "max_absolute_difference": float(np.max(np.abs(prediction-direct))),
                            "cost": cost})
        np.testing.assert_allclose(read_doubles(output/"direct/W_hat.bin", sample.n),
                                   sample.known_propensity, atol=0, rtol=0)
        weights = read_doubles(output/"direct/weights.bin", len(query)*sample.n).reshape(len(query), sample.n)
        reconstructed, denominators = weighted_intercept_slopes(
            weights, sample.T, sample.Y, read_doubles(output/"direct/Y_hat.bin", sample.n),
            sample.known_propensity)
        map_error = float(np.max(np.abs(reconstructed-direct)))
        if map_error > MAP_ATOL:
            raise AssertionError("independent weighted-intercept prediction map failed")
        np.save(output/"independent_prediction.npy", reconstructed, allow_pickle=False)
        extended = np.vstack((query, np.full((1, sample.d), -100), np.full((1, sample.d), 100)))
        extended_prediction, extension_cost = adapter_call(sample, extended, output/"query_extension")
        np.testing.assert_allclose(extended_prediction[:len(query)], direct, atol=ATOL, rtol=RTOL)
        profiles = []
        for n in PROFILE_SIZES:
            profile_sample, profile_query = observations(n, PROFILE_QUERIES)
            prediction, cost = adapter_call(profile_sample, profile_query, output/f"profile_{n}")
            if prediction.shape != (PROFILE_QUERIES,) or not np.isfinite(prediction).all():
                raise AssertionError("profile predictions malformed")
            profiles.append(dict(rct_rows=n, dimensions=sample.d, query_rows=PROFILE_QUERIES, **cost))
        after = causal_forest_provenance()
        write_json(output/"provenance_after.json", after)
        if provenance != after:
            raise AssertionError("native source/runtime provenance changed during admission")
        summary = {
            "status": "passed", "plan": plan, "provenance": provenance,
            "fidelity": repeats, "direct_cost": direct_cost,
            "independent_prediction_map": {"max_absolute_difference": map_error,
                "queries": len(query), "minimum_local_treatment_variance": float(denominators.min()),
                "hand_oracle": "8/3", "tolerance_absolute": MAP_ATOL,
                "scope": "arithmetic conditional on native neighborhoods/nuisance, not splitting or accuracy"},
            "query_extension": {"max_absolute_difference": float(np.max(np.abs(extended_prediction[:len(query)]-direct))),
                                 "cost": extension_cost},
            "profiles": profiles, "native_fitting_calls": 6,
            "outcomes": {"rct_only": True, "obs_consumed": 0, "tune_consumed": 0},
        }
        write_json(output/"summary.json", summary)
        return summary
    except Exception as exc:
        (output/"failure_traceback.txt").write_text(traceback.format_exc())
        write_json(output/"failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                           "plan": plan})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-metadata", type=Path)
    parser.add_argument("--worker-mode", choices=("sample", "hostile"), default="sample")
    args = parser.parse_args()
    if args.worker_input is not None:
        if not args.worker_output or not args.worker_metadata or not os.environ.get("SLURM_JOB_ID"):
            parser.error("worker requires outputs and an allocated server job")
        worker(args.worker_input, args.worker_output, args.worker_metadata, args.worker_mode)
    elif args.output is not None:
        result = admit(args.output)
        print(json.dumps({"status": result["status"], "native_fitting_calls": result["native_fitting_calls"],
                          "map_error": result["independent_prediction_map"]["max_absolute_difference"]}))
    else:
        parser.error("choose --output or the complete worker arguments")
