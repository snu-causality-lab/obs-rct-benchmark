"""Prospective source-access and mathematical checks for the GP source pilot.

Numerical stages require an allocated GSDS Slurm job. Raw predictions and full
diagnostics remain at a new server output directory. This checks fidelity and
input roles; no estimator is required to improve CATE risk to pass.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


REVISION = "0f0c6ca424b7ca44b84649ca072384b9933b2afe"
NOTEBOOK = "rebuttals/extrapolation.ipynb"
NOTEBOOK_SHA256 = "0326f655bac0993fd845412ce59fb0cde05e04c6f70890f42ef7e6750fcd855c"
CELL_SHA256 = "6c85f01d85087e3e69f75c1dbaf926bc8868b6b1be0521a699e00be643f7ca85"
FIT_SEED = 2601026
RHO = .5
KAPPA = 1e-9
ATOL, RTOL, ANALYTIC_ATOL = 1e-10, 1e-8, 1e-12
PLAN = {
    "schema_version": 1,
    "scope": "native source-access controls and independently specified pilot identities",
    "fit_seed": FIT_SEED, "rho": RHO, "kappa": KAPPA,
    "fidelity_atol": ATOL, "fidelity_rtol": RTOL,
    "analytic_atol": ANALYTIC_ATOL, "analytic_rtol": 0.,
    "query_source": "trial task0 for both treatment-arm models",
    "native_calls": "full, RCT-only and OBS-start; excluded source arrays have zero rows",
    "fresh_wrappers": "two per recipe, second changes unused source/tuning/reference observations",
    "strict_sample_entry": "one fit per control through fit_predict_sample with only its included fitting SourceSample",
    "native_fit_count": 3,
    "adapter_fit_count": 10,
    "rct_only_rho_values": [.2, .8],
    "obs_start_identity": "mu_task0 = rho/(1+kappa) * mu_task1 for each same fitted OBS-only GP",
    "rho_zero_substitution": False,
    "accuracy_or_superiority_required": False,
    "data_scope": "first fixed seed, all declared bias levels and both RCT budgets",
}
FIXTURE = {
    "rct_X": [[-1.5,-.8],[-1.2,.3],[-.9,1.1],[-.6,-1.2],[-.3,.7],[0.,-.4],
              [.3,1.4],[.6,-.9],[.9,.1],[1.2,1.],[1.5,-.6],[1.8,.5]],
    "rct_T": [0,1,0,1,0,1,0,1,0,1,0,1],
    "rct_Y": [-.7,.4,1.2,.2,.5,1.6,1.5,1.1,.7,2.8,1.9,3.1],
    "obs_X": [[-1.8,.2],[-1.5,-.7],[-1.2,1.2],[-.9,-.3],[-.6,.8],[-.3,-1.1],
              [0.,.4],[.3,1.3],[.6,-.5],[.9,.9],[1.2,-1.3],[1.5,.1],[1.8,.7],
              [2.1,-.8],[2.4,1.4],[2.7,-.1]],
    "obs_T": [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    "obs_Y": [-.5,1.8,.2,2.,-.8,2.4,.7,1.,1.3,3.5,1.1,3.,2.,3.6,1.4,4.1],
    "query_X": [[-1.1,.2],[0.,0.],[.8,-.7],[1.6,1.]],
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _close(actual, expected, *, analytic=False):
    import numpy as np
    np.testing.assert_allclose(actual, expected, atol=ANALYTIC_ATOL if analytic else ATOL,
                               rtol=0. if analytic else RTOL)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _direct_native():
    for name, expected in {"numpy": "1.26.4", "scipy": "1.12.0", "GPy": "1.13.2", "paramz": "0.9.6"}.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"direct native environment requires {name} {expected}")
    root = Path(os.environ["CAUSAL_ICM_SOURCE"]).expanduser().resolve(strict=True)
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE, timeout=15).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root or git("rev-parse", "HEAD") != REVISION:
        raise RuntimeError("direct native source revision/top-level mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("direct native source has tracked changes")
    path = root / NOTEBOOK
    if path.is_symlink() or not path.resolve().is_relative_to(root) or _sha(path) != NOTEBOOK_SHA256:
        raise RuntimeError("direct native notebook differs from the prospective source")
    notebook = json.loads(path.read_text())
    cell = notebook["cells"][3]
    source = "".join(cell["source"])
    if cell["cell_type"] != "code" or hashlib.sha256(source.encode()).hexdigest() != CELL_SHA256:
        raise RuntimeError("direct native definition cell mismatch")
    tree = ast.parse(source)
    if (len(tree.body) != 3 or ast.unparse(tree.body[0]) != "import GPy"
            or ast.unparse(tree.body[1]) != "import numpy as np"
            or not isinstance(tree.body[2], ast.FunctionDef) or tree.body[2].name != "ICM"):
        raise RuntimeError("direct native cell has unexpected executable statements")
    namespace = {"__name__": "_gp_pilot_direct_native"}
    exec(compile(tree, f"{NOTEBOOK}:cell3", "exec"), namespace)
    return namespace["ICM"]


def _fixture_view(*, excluded=None):
    import numpy as np
    from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView
    samples = {}
    for source in ("obs", "rct"):
        X = np.asarray(FIXTURE[source + "_X"], dtype=float)
        T = np.asarray(FIXTURE[source + "_T"], dtype=float)
        Y = np.asarray(FIXTURE[source + "_Y"], dtype=float)
        probability = None if source == "obs" else np.full(len(T), .5)
        if source == excluded:
            X = X * -37. + 91.
            T = np.zeros(len(T))
            Y = Y + 1e6
            if source == "rct":
                probability = np.linspace(.2, .8, len(T))
        samples[source] = SourceSample(X, T, Y, source, source + "_train", probability)
    mutated = excluded is not None
    tune = SourceSample(np.array([[7.,8.],[9.,10.]]) * (100. if mutated else 1.),
                        np.array([0.,1.]), np.array([50.,-30.]) + (1e6 if mutated else 0.),
                        "rct", "rct_tune")
    reference = TargetCovariateSample(np.array([[-7.,5.],[4.,-6.]]) * (100. if mutated else 1.))
    return TrainingView(samples["obs"], samples["rct"], tune, reference)


def _native_fit(native, view, query, source, rho):
    import numpy as np
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(FIT_SEED)
        random.seed(FIT_SEED)
        d = query.shape[1]
        def arrays(sample, included):
            return ((sample.X.copy(), sample.T.copy(), sample.Y.copy()) if included else
                    (np.empty((0, d)), np.empty(0), np.empty(0)))
        rct = arrays(view.rct_train, source != "obs")
        obs = arrays(view.obs_train, source != "rct")
        models = native(X_E=rct[0], X_O=obs[0], T_E=rct[1], T_O=obs[1],
                        Y_E=rct[2], Y_O=obs[2], r=2, ID=d, AD=list(range(d)), rho=rho)
        query0 = np.column_stack((query, np.zeros(len(query))))
        query1 = np.column_stack((query, np.ones(len(query))))
        means0 = [np.asarray(model.predict_noiseless(query0)[0])[:, 0] for model in models]
        means1 = [np.asarray(model.predict_noiseless(query1)[0])[:, 0] for model in models]
        return means0[1] - means0[0], means0, means1
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _native_checks(output):
    import numpy as np
    from obs_rct_benchmark.causal_icm import (
        CausalICMSourceControl, OfficialCausalICM, causal_icm_provenance,
    )
    native = _direct_native()
    view, query = _fixture_view(), np.asarray(FIXTURE["query_X"], dtype=float)
    maxima, native_values, wrapper_records = {}, {}, []
    for source in ("full", "rct", "obs"):
        prediction, task0, task1 = _native_fit(native, view, query, source, RHO)
        native_values[source] = prediction
        if source == "obs":
            for arm in (0, 1):
                maxima[f"obs_prior_mean_arm{arm}"] = _close(
                    task0[arm], RHO / (1 + KAPPA) * task1[arm], analytic=True)
        factory = (lambda: OfficialCausalICM(rho=RHO)) if source == "full" else (
            lambda source=source: CausalICMSourceControl(source=source, rho=RHO))
        predictions = []
        for repetition in (1, 2):
            estimator = factory()
            # 'unused' changes tune/reference only; its string does not match
            # an observed fitting source in _fixture_view.
            excluded = None if repetition == 1 else {"full": "unused", "rct": "obs", "obs": "rct"}[source]
            observed = view if repetition == 1 else _fixture_view(excluded=excluded)
            actual = np.asarray(estimator.fit_predict(observed, query, seed=FIT_SEED))
            if actual.shape != (len(query),) or not np.isfinite(actual).all():
                raise AssertionError("source-control prediction shape/finite failure")
            maxima[f"{source}_direct_wrapper{repetition}"] = _close(actual, prediction)
            predictions.append(actual)
            wrapper_records.append({"source": source, "repetition": repetition,
                                    "provenance": estimator.provenance_})
            np.save(output / f"{source}_wrapper{repetition}.npy", actual, allow_pickle=False)
        maxima[f"{source}_fresh_repeat_with_information_mutation"] = _close(*predictions)
        if source != "full":
            estimator = factory()
            included = view.rct_train if source == "rct" else view.obs_train
            actual = np.asarray(estimator.fit_predict_sample(included, query, seed=FIT_SEED))
            if actual.shape != (len(query),) or not np.isfinite(actual).all():
                raise AssertionError("single-source entry prediction shape/finite failure")
            maxima[f"{source}_direct_single_source_entry"] = _close(actual, prediction)
            wrapper_records.append({"source": source, "entry": "fit_predict_sample",
                                    "supplied_observed_roles": [included.role],
                                    "provenance": estimator.provenance_})
            np.save(output / f"{source}_single_source_entry.npy", actual, allow_pickle=False)
    for rho in (.2, .8):
        estimator = CausalICMSourceControl(source="rct", rho=rho)
        value = estimator.fit_predict(view, query, seed=FIT_SEED)
        maxima[f"rct_rho_{rho}_invariance"] = _close(value, native_values["rct"])
    return {"max_absolute_discrepancy": maxima, "native_fit_count": 3,
            "adapter_fit_count": 10, "source": causal_icm_provenance(),
            "wrapper_records": wrapper_records,
            "obs_start_meaning": "trial-task prior conditioned on OBS only; not naive OBS task1 contrast",
            "accuracy_test": False}


def _engine(repo):
    spec = importlib.util.spec_from_file_location("_gp_pilot_checked_engine", repo / "scripts/run_gp_source_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _data_checks(repo, config_path):
    import numpy as np
    from dataclasses import fields
    from obs_rct_benchmark.data import SourceSample, TrainingView
    engine = _engine(repo)
    config = engine.load_config(config_path)
    points = np.array([[0.,0.],[.5,.5],[-.5,.5],[0.,.5],[0.,-.5],[1.,.5],[-1.,-.5],[-.5,1.]])
    expected_m = np.array([0.,.375,-.125,.125,-.125,.625,-.625,0.])
    expected_tau = np.array([1.25,1.5,.5,1.,1.,1.,1.,.25])
    expected_b = np.array([0.,0.,0.,1.,-1.,-1.,1.,0.])
    maxima = {
        "hand_m": _close(engine.baseline_mean(points), expected_m, analytic=True),
        "hand_tau": _close(engine.true_effect(points), expected_tau, analytic=True),
        "hand_b": _close(engine.bias_function(points), expected_b, analytic=True),
    }
    m = lambda x: .5 * x[:, 0] + .25 * x[:, 1]
    tau = lambda x: 1. + .5 * np.sin(np.pi*x[:, 0]) + .25*np.cos(np.pi*x[:, 1])
    bias = lambda x: np.cos(np.pi*x[:, 0]) * np.sin(np.pi*x[:, 1])
    for delta in config["deltas"]:
        for u in (0., .5, 1.):
            potential0 = expected_m - .5*expected_tau + delta*(u-.5)*expected_b
            potential1 = expected_m + .5*expected_tau + delta*(u-.5)*expected_b
            maxima[f"potential_contrast_delta{delta}_u{u}"] = _close(
                potential1-potential0, expected_tau, analytic=True)
        obs1 = expected_m + .5*expected_tau + .5*delta*expected_b
        obs0 = expected_m - .5*expected_tau - .5*delta*expected_b
        maxima[f"observed_contrast_delta{delta}"] = _close(
            obs1-obs0, expected_tau+delta*expected_b, analytic=True)
    seed, budgets = config["seeds"][0], sorted(config["rct_budgets"])
    cases = {(delta, budget): engine.generate_case(config, seed=seed, delta=delta, n_rct=budget)
             for delta in config["deltas"] for budget in budgets}
    first_delta = config["deltas"][0]
    for (delta, budget), arrays in cases.items():
        base = cases[(first_delta, budget)]
        for prefix, rows in (("obs", config["n_obs"]), ("rct", budget)):
            if (arrays[prefix + "_X"].shape != (rows, 2)
                    or arrays[prefix + "_T"].shape != (rows,)
                    or arrays[prefix + "_Y"].shape != (rows,)
                    or not np.isin(arrays[prefix + "_T"], (0., 1.)).all()):
                raise AssertionError("pilot fitting rows/arms differ from the declared case")
        if arrays["eval_X"].shape != (config["n_eval"], 2) or arrays["eval_tau"].shape != (config["n_eval"],):
            raise AssertionError("pilot evaluation rows differ from the declared case")
        for name in ("rct_X", "rct_T", "rct_Y", "rct_known_propensity", "eval_X", "eval_tau", "obs_X", "obs_T"):
            np.testing.assert_array_equal(arrays[name], base[name])
        np.testing.assert_array_equal(arrays["rct_known_propensity"], np.full(budget, .5))
        maxima[f"eval_tau_delta{delta}_B{budget}"] = _close(arrays["eval_tau"], tau(arrays["eval_X"]), analytic=True)
        expected_delta = ((delta-first_delta)*(arrays["obs_T"]-.5)*bias(arrays["obs_X"]))
        maxima[f"obs_bias_shift_delta{delta}_B{budget}"] = _close(
            arrays["obs_Y"]-base["obs_Y"], expected_delta, analytic=True)
        obs_noise = arrays["obs_Y"] - m(arrays["obs_X"]) - (arrays["obs_T"]-.5)*(
            tau(arrays["obs_X"]) + delta*bias(arrays["obs_X"]))
        base_noise = base["obs_Y"]-m(base["obs_X"])-(base["obs_T"]-.5)*(
            tau(base["obs_X"])+first_delta*bias(base["obs_X"]))
        maxima[f"common_obs_noise_delta{delta}_B{budget}"] = _close(obs_noise, base_noise, analytic=True)
        for name in ("obs_X", "rct_X", "eval_X"):
            if (not np.isfinite(arrays[name]).all() or arrays[name].shape[1] != 2
                    or np.any(arrays[name] < -1) or np.any(arrays[name] > 1)):
                raise AssertionError("pilot covariates leave the declared target support/dimension")
        training = engine.make_training_view(arrays)
        if {f.name for f in fields(training)} != {"obs_train", "rct_train", "rct_tune", "target_reference"}:
            raise AssertionError("unexpected observed-data interface")
        if not isinstance(training, TrainingView) or hasattr(training, "eval_tau") or hasattr(training, "rct_test"):
            raise AssertionError("evaluator target/test data entered TrainingView")
        if training.target_reference is not None or training.obs_train.known_propensity is not None:
            raise AssertionError("undeclared target/observational propensity information")
        for source in ("obs", "rct"):
            sample = getattr(training, source + "_train")
            for name in ("X", "T", "Y"):
                np.testing.assert_array_equal(getattr(sample, name), arrays[source + "_" + name])
            single = engine.make_source_sample(arrays, source)
            if (not isinstance(single, SourceSample) or single.source != source
                    or single.role != source + "_train"
                    or {f.name for f in fields(single)} != {
                        "X", "T", "Y", "source", "role", "known_propensity"}):
                raise AssertionError("single-source entry contains undeclared observed roles")
            for name in ("X", "T", "Y"):
                np.testing.assert_array_equal(getattr(single, name), arrays[source + "_" + name])
            if source == "obs":
                if single.known_propensity is not None:
                    raise AssertionError("OBS-only entry contains undeclared propensity information")
            else:
                np.testing.assert_array_equal(single.known_propensity, arrays["rct_known_propensity"])
        np.testing.assert_array_equal(training.rct_train.known_propensity, arrays["rct_known_propensity"])
        for name in ("X", "T", "Y"):
            np.testing.assert_array_equal(getattr(training.rct_tune, name), getattr(training.rct_train, name)[:2])
        np.testing.assert_array_equal(training.rct_tune.known_propensity, arrays["rct_known_propensity"][:2])
    for delta in config["deltas"]:
        large = cases[(delta, budgets[-1])]
        for budget in budgets:
            small = cases[(delta, budget)]
            for name in ("rct_X", "rct_T", "rct_Y", "rct_known_propensity"):
                np.testing.assert_array_equal(small[name], large[name][:budget])
            for name in ("obs_X", "obs_T", "obs_Y", "eval_X", "eval_tau"):
                np.testing.assert_array_equal(small[name], large[name])
    grid = -1. + (np.arange(128)+.5)*2/128
    a, b = np.meshgrid(grid, grid, indexing="ij")
    quadrature = np.column_stack((a.ravel(), b.ravel()))
    values, discrepancies = tau(quadrature), bias(quadrature)
    for name, actual, expected in (
        ("mean_tau", np.mean(values), 1.), ("mean_b", np.mean(discrepancies), 0.),
        ("mean_b_squared", np.mean(discrepancies**2), .25),
        ("mean_tau_b", np.mean(values*discrepancies), 0.),
        ("mean_tau_squared", np.mean(values**2), 37/32),
    ):
        maxima[name] = _close(actual, expected, analytic=True)
    return {"config_sha256": _sha(config_path), "seed": seed,
            "case_count": len(cases), "max_absolute_discrepancy": maxima,
            "shared_observations_nested_RCT_and_query_checks": "passed",
            "control_method_input": "one declared fitting SourceSample; excluded outcomes and tune roles absent",
            "unused_tune_role": "full method only: first two RCT fitting rows; zero additional unique outcome labels",
            "assumptions": ["development-designed controlled SCM, not unseen-mechanism validation",
                            "observed treatment positivity holds; latent OBS positivity is not assumed",
                            "latent prognostic distribution differs by source; common causal CATE remains tau",
                            "trial observed noise variance and evaluation law are fixed across delta"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=("native", "data", "all"), default="all")
    args = parser.parse_args()
    raw_root = os.environ.get("GSDS_WORK_ROOT")
    if not raw_root or not Path(raw_root).is_absolute() or not os.environ.get("SLURM_JOB_ID"):
        parser.error("verification requires an allocated GSDS Slurm job and absolute GSDS_WORK_ROOT")
    output, repo = args.output.expanduser().resolve(), args.repo.expanduser().resolve(strict=True)
    if not output.is_relative_to(Path(raw_root).resolve()):
        parser.error("output must be below GSDS_WORK_ROOT")
    if any(os.environ.get(name) != "1" for name in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        parser.error("four numerical thread limits must all equal one")
    if args.stage != "native" and args.config is None:
        parser.error("data verification requires --config")
    os.umask(0o077)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "prospective_plan.json", {**PLAN, "observed_fixture": FIXTURE})
    record = {"schema_version": 1, "status": "running", "stage": args.stage,
              "plan_sha256": _digest({**PLAN, "observed_fixture": FIXTURE}),
              "driver_sha256": _sha(__file__),
              "pilot_engine_sha256": _sha(repo / "scripts/run_gp_source_pilot.py") if args.stage != "native" else None,
              "config_sha256": _sha(args.config) if args.config else None,
              "failure_details_server_only": "stderr.log"}
    _json(output / "verification.json", record)
    started = time.perf_counter()
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                from obs_rct_benchmark.runner import source_provenance
                record["producer"] = source_provenance()
                expected = {p.name: _sha(p) for p in sorted((repo / "src/obs_rct_benchmark").glob("*.py"))}
                if record["producer"]["package_file_sha256"] != expected:
                    raise RuntimeError("imported verification producer differs from the supplied checkout")
                if args.stage in {"native", "all"}:
                    record["native"] = _native_checks(output)
                if args.stage in {"data", "all"}:
                    record["data"] = _data_checks(repo, args.config.resolve(strict=True))
            record["status"] = "passed"
        except BaseException as exc:
            record.update(status="failed", exception_type=type(exc).__name__)
            traceback.print_exc(file=stderr)
    record["wall_seconds"] = time.perf_counter()-started
    record["log_sha256"] = {name: _sha(output / name) for name in ("stdout.log", "stderr.log")}
    _json(output / "verification.json", record)
    print(json.dumps({"status": record["status"], "stage": args.stage,
                      "wall_seconds": record["wall_seconds"]}, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
