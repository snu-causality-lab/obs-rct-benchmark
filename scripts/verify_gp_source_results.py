#!/usr/bin/env python3
"""Independently reconstruct the frozen GP pilot from retained GSDS artifacts.

The producer's generator, scorer and analysis functions are never imported.
Only optional first-seed replay imports estimator/data classes. All arithmetic
for the scientific target, losses and Monte Carlo summaries is specified here.

Run under Python 3.11+ in the frozen native environment and a GSDS allocation:
  python scripts/verify_gp_source_results.py --repo PRODUCER --config PILOT.toml
    --input FIT_ROOT --analysis ANALYSIS_ROOT --output NEW_VERIFICATION_ROOT
    [--tables-dir RENDER_ROOT] [--replay-first-seed]
All paths must be absolute shared-work paths. CAUSAL_ICM_SOURCE names the pinned
Git checkout; GSDS_WORK_ROOT and four numerical thread limits must be set. The
replay uses a fresh process and a 180-second limit per fit. Its scalar MSE bound
is deliberately tighter than its pointwise prediction bound. This checks saved
data identities, not statistical independence or normality of realized draws.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import tomllib
import traceback


SLOTS = ("full", "rct_only", "obs_start")
METHODS = dict(zip(SLOTS, ("causal_icm_fixed", "causal_icm_rct_only", "causal_icm_obs_start")))
OBSERVED = ("obs_X", "obs_T", "obs_Y", "rct_X", "rct_T", "rct_Y",
            "rct_known_propensity", "eval_X")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
REVISION = "0f0c6ca424b7ca44b84649ca072384b9933b2afe"
NOTEBOOK = "rebuttals/extrapolation.ipynb"
NOTEBOOK_SHA = "0326f655bac0993fd845412ce59fb0cde05e04c6f70890f42ef7e6750fcd855c"
CELL_SHA = "6c85f01d85087e3e69f75c1dbaf926bc8868b6b1be0521a699e00be643f7ca85"
VERSIONS = {"numpy": "1.26.4", "scipy": "1.12.0", "GPy": "1.13.2", "paramz": "0.9.6"}
PLAN = {"schema_version": 1, "expected_records": 576, "declared_independent_seed_blocks": 32,
        "reconstruction_atol": 1e-12, "reconstruction_rtol": 1e-11,
        "replay_atol": 1e-10, "replay_rtol": 1e-8,
        "replay_seed": 30000, "replay_fits": 18,
        "replay_per_fit_timeout_seconds": 180,
        "control_invariance_groups": 160, "minimum_successes_per_invariance_group": 2,
        "score": "Python math.sin/cos target and math.fsum squared errors",
        "inference": "condition-specific complete-pair descriptive t31 intervals only",
        "performance_threshold": None}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def close(actual, expected, *, replay=False):
    if actual is None or expected is None:
        require(actual is expected, "withheld numeric field changed")
    else:
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)), "nonfinite numeric field")
        require(math.isclose(float(actual), float(expected),
                             abs_tol=1e-10 if replay else 1e-12,
                             rel_tol=1e-8 if replay else 1e-11), "independent numeric reconstruction differs")


def identity(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def tau_rows(X):
    return [1 + .5*math.sin(math.pi*float(x)) + .25*math.cos(math.pi*float(y)) for x, y in X]


def mse(prediction, truth):
    require(len(prediction) == len(truth) > 0, "loss vector length changed")
    return math.fsum((float(p)-float(t))**2 for p, t in zip(prediction, truth))/len(truth)


def paired(values):
    n = len(values)
    mean = math.fsum(values)/n if n else None
    result = {"expected_pairs": 32, "successful_pairs": n, "complete": n == 32,
              "mean": mean, "mcse": None, "ci95": None}
    if n == 32:
        from scipy.special import stdtrit
        se = math.sqrt(math.fsum((x-mean)**2 for x in values)/(n-1)/n)
        width = float(stdtrit(31, .975))*se
        result.update(mcse=se, ci95=[mean-width, mean+width])
    return result


def configuration(path):
    cfg = tomllib.loads(path.read_text()) if path.suffix == ".toml" else read(path)
    for key, expected in {"schema_version": 1, "generator": "paired_gp_source_v1", "d": 2,
                          "n_obs": 512, "n_eval": 1024, "rct_budgets": [128, 512],
                          "deltas": [0, 1, 2], "seeds": list(range(30000, 30032)),
                          "rho": .5, "fit_timeout_seconds": 180, "methods": METHODS,
                          "tune_role": "first_two_training_rows_alias",
                          "inference": "pointwise_paired_t_descriptive"}.items():
        require(cfg.get(key) == expected, "configuration differs from the frozen pilot")
    cases = [{"id": f"d{di:02d}_r{budget}_s{seed}", "delta_index": di,
              "delta": delta, "n_rct": budget, "seed": seed}
             for di, delta in enumerate((0, 1, 2)) for budget in (128, 512)
             for seed in range(30000, 30032)]
    return cfg, cases


def provenance(repo, bound):
    source = bound["source"]
    files = {p.name: sha(p) for p in sorted((repo/"src/obs_rct_benchmark").glob("*.py"))}
    package = source["package_source"]
    require(package["package_file_sha256"] == files and package["aggregate_sha256"] == digest(files),
            "producer source fingerprint changed")
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"
    require(package["distribution"] == "obs-rct-benchmark"
            and package["version"] == version("obs-rct-benchmark"), "producer distribution identity changed")
    require(source["driver_sha256"] == sha(repo/"scripts/run_gp_source_pilot.py"), "producer driver changed")
    tree = ast.parse((repo/"src/obs_rct_benchmark/__init__.py").read_text())
    module_version = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    require(module_version == [source["module_version"]], "producer module version changed")
    environment = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                   "system": platform.system(), "machine": platform.machine(),
                   "packages": {name: version(name) for name in
                                ("obs-rct-benchmark", "numpy", "scikit-learn", "scipy", "dfppi", "torch", "GPy", "paramz")}}
    require(source["environment"] == environment, "reconstruction numerical environment differs")
    native = source["native_source"]
    for key, expected in {"availability": "available", "expected_revision": REVISION,
                          "declared_git_revision": REVISION, "tracked_changes": False,
                          "source_file": NOTEBOOK, "source_file_sha256": NOTEBOOK_SHA,
                          "expected_source_file_sha256": NOTEBOOK_SHA, "cell_index": 3,
                          "cell_sha256": CELL_SHA, "expected_cell_sha256": CELL_SHA,
                          "entrypoint": "ICM", "source_verified": True, "verified": True}.items():
        require(native.get(key) == expected, "native provenance differs from pinned author source")
    root = Path(os.environ["CAUSAL_ICM_SOURCE"]).resolve(strict=True)
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                               stderr=subprocess.PIPE, timeout=15).strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == root
            and git("rev-parse", "HEAD") == REVISION
            and not git("status", "--porcelain", "--untracked-files=no"), "native checkout changed")
    notebook = root/NOTEBOOK
    require(not notebook.is_symlink() and notebook.resolve().is_relative_to(root)
            and sha(notebook) == NOTEBOOK_SHA, "native notebook changed")
    cell = read(notebook)["cells"][3]
    require(cell["cell_type"] == "code"
            and hashlib.sha256("".join(cell["source"]).encode()).hexdigest() == CELL_SHA, "native cell changed")
    for name, expected in VERSIONS.items():
        entry = native["numerical_environment"][name]
        distribution = importlib.metadata.distribution(name)
        initializer = Path(distribution.locate_file(f"{name}/__init__.py")).resolve()
        spec = importlib.util.find_spec(name)
        require(distribution.version == expected and entry["expected_version"] == expected
                and entry["installed_version"] == expected and entry["init_sha256"] == sha(initializer)
                and entry["verified"] is True and entry["import_resolution"] == "verified_distribution"
                and spec is not None and Path(spec.origin).resolve() == initializer,
                "native numerical package identity changed")
    return native


def load_arrays(attempt, record, case):
    import numpy as np
    with np.load(attempt/"observed_inputs.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(OBSERVED), "saved observed arrays contain undeclared roles")
        arrays = {name: archive[name] for name in OBSERVED}
    expected_shapes = {f"{source}_{name}": (rows, 2) if name == "X" else (rows,)
                       for source, rows in (("obs", 512), ("rct", case["n_rct"])) for name in ("X", "T", "Y")}
    expected_shapes.update(rct_known_propensity=(case["n_rct"],), eval_X=(1024, 2))
    for name, shape in expected_shapes.items():
        require(arrays[name].shape == shape and arrays[name].dtype == np.dtype("float64")
                and np.isfinite(arrays[name]).all(), "saved observation shape/dtype/value changed")
    for source in ("obs", "rct"):
        require(np.isin(arrays[source+"_T"], (0., 1.)).all(), "nonbinary saved treatment")
    for name in ("obs_X", "rct_X", "eval_X"):
        require(np.all((arrays[name] >= -1) & (arrays[name] <= 1)), "saved covariates outside declared support")
    np.testing.assert_array_equal(arrays["rct_known_propensity"], np.full(case["n_rct"], .5))
    require(record["observed_array_identity"] == {k: identity(v) for k, v in arrays.items()}, "observed array hash mismatch")
    with np.load(attempt/"evaluation.npz", allow_pickle=False) as archive:
        require(archive.files == ["eval_tau"], "undeclared evaluator array")
        truth = archive["eval_tau"]
    require(truth.shape == (1024,) and truth.dtype == np.dtype("float64")
            and np.isfinite(truth).all(), "invalid saved evaluator truth")
    require(record["evaluation_array_identity"] == {"eval_tau": identity(truth)}, "saved evaluator hash mismatch")
    independent = tau_rows(arrays["eval_X"])
    for actual, expected in zip(truth, independent):
        close(actual, expected)
    return arrays, independent


def budget_inputs(record, arrays, case, slot):
    expected = {"supplied_unique_rct_labels": case["n_rct"] if slot != "obs_start" else 0,
                "supplied_unique_obs_labels": 512 if slot != "rct_only" else 0,
                "consumed_rct_labels": case["n_rct"] if slot != "obs_start" else 0,
                "consumed_obs_labels": 512 if slot != "rct_only" else 0,
                "additional_tuning_labels": 0, "tune_alias_train_indices": [0, 1] if slot == "full" else []}
    require(record["label_budget"] == expected, "observed label budget changed")
    names = OBSERVED if slot == "full" else tuple(k for k in OBSERVED
            if k.startswith("rct_" if slot == "rct_only" else "obs_") or k == "eval_X")
    require(record["method_argument_array_identity"] == {k: identity(arrays[k]) for k in names},
            "actual method argument array identity changed")
    roles = (["obs_train", "rct_train", "rct_tune_alias", "eval_X"] if slot == "full" else
             ["rct_train" if slot == "rct_only" else "obs_train", "eval_X"])
    require(record["method_argument_roles"] == roles, "actual method argument roles changed")


def fit_provenance(record, native, arrays, case, slot):
    import numpy as np
    p = record.get("estimator_provenance")
    require(isinstance(p, dict) and p.get("native_source") == native and p["rho"] == .5
            and p["rank"] == 2 and p["effect_modifier_columns"] == [0, 1]
            and p["prediction_source"] == 0 and p["source_encoding"] == {"rct": 0, "obs": 1}
            and p["prediction"] == "treated_mean_minus_control_mean"
            and p["noise"] == "native_GPy_Gaussian_noise_optimized"
            and p["optimization"] == "native_GPy_optimize_defaults_no_restarts", "estimator native recipe identity changed")
    if slot == "full":
        require(p["recipe"] == "fixed_rho_native_two_arm_gp" and p["rho_zero_is_rct_only"] is False,
                "full recipe changed")
    else:
        source = "rct" if slot == "rct_only" else "obs"
        require(p["recipe"] == "source_restricted_native_trial_task_gp" and p["source"] == source
                and p["supplied_seed"] == case["seed"] and p["status"] == "completed"
                and p["excluded_source"] == "empty_native_arrays"
                and p["rct_only_by_row_exclusion"] == (source == "rct")
                and p["known_propensity"] == "not_used"
                and p["rct_tune"] == p["target_reference"] == "not_accessed", "control recipe changed")
        for other in ("rct", "obs"):
            wanted = {"availability": "not_accessed", "fit_rows": 0, "arm_rows": {"0": 0, "1": 0}}
            if other == source:
                wanted = {"availability": "included", "fit_rows": len(arrays[other+"_T"]),
                          "arm_rows": {str(arm): int(np.count_nonzero(arrays[other+"_T"] == arm)) for arm in (0, 1)}}
            require(p["fitting_sources"][other] == wanted, "included/excluded native fitting rows changed")


def terminal_records(root, cfg, cases, manifest, native, output):
    import numpy as np
    wanted = {f"records/{case['id']}/{slot}/record.json" for case in cases for slot in SLOTS}
    actual = {str(p.relative_to(root)) for p in (root/"records").rglob("record.json")}
    require(actual == wanted and len(actual) == 576, "terminal record union has missing/extra cells")
    records, data, risks, attempts = {}, {}, {}, []
    for case in cases:
        for slot in SLOTS:
            relative = f"records/{case['id']}/{slot}/record.json"
            record = read(root/relative)
            request = {"case": case, "slot": slot, "method_id": METHODS[slot], "bindings": manifest["bindings"],
                       "manifest_sha256": manifest["manifest_sha256"], "rho": .5}
            require(all(record.get(k) == v for k, v in request.items()) and record["schema_version"] == 1,
                    "terminal case/producer identity changed")
            require(record["status"] in {"success", "failed", "timeout"} and record["process_group_reaped"] is True,
                    "record is not a reaped terminal result")
            parent = root/"records"/case["id"]/slot
            require(not (parent/"running.lock").exists(), "terminal cell still has an unresolved running lock")
            attempt = (root/record["attempt"]).resolve(strict=True)
            require(attempt.parent == (parent/"attempts").resolve() and attempt.is_relative_to(root),
                    "attempt path outside its declared record")
            files = {p.name: sha(p) for p in attempt.iterdir() if p.is_file()}
            require(files == record["artifact_sha256"] and read(attempt/"request.json") == request,
                    "attempt artifact hashes/request changed")
            for historic in sorted((parent/"attempts").iterdir()):
                require(historic.is_dir() and not historic.is_symlink(), "unexpected attempt inventory item")
                attempts.append({"case": case["id"], "slot": slot, "attempt": historic.name,
                                 "terminal": historic == attempt,
                                 "artifact_sha256": {p.name: sha(p) for p in sorted(historic.iterdir()) if p.is_file()}})
            records[(case["id"], slot)] = record
            if "observed_array_identity" not in record:
                require(record["status"] != "success", "successful fit lacks retained observations")
                continue
            arrays, truth = load_arrays(attempt, record, case)
            budget_inputs(record, arrays, case, slot)
            initial = read(attempt/"inputs.json")
            for key in (*request, "observed_array_identity", "evaluation_array_identity", "label_budget",
                        "method_argument_array_identity", "method_argument_roles"):
                require(initial.get(key) == record.get(key), "pre-fit argument evidence differs from terminal record")
            if case["id"] in data:
                for name in OBSERVED:
                    np.testing.assert_array_equal(arrays[name], data[case["id"]][name])
            else:
                data[case["id"]] = arrays
            if record["status"] == "success":
                require(record["process_returncode"] == 0, "successful worker return code changed")
                worker = read(attempt/"worker.json")
                required_worker = set(request) | {"schema_version", "status", "mse", "prediction_array_identity",
                    "observed_array_identity", "evaluation_array_identity", "label_budget",
                    "method_argument_array_identity", "method_argument_roles", "estimator_provenance"}
                require(required_worker <= set(worker) and worker.get("status") == "success"
                        and all(record.get(k) == v for k, v in worker.items()),
                        "successful terminal record disagrees with retained worker evidence")
                fit_provenance(record, native, arrays, case, slot)
                pred = np.load(attempt/"prediction.npy", allow_pickle=False)
                require(pred.shape == (1024,) and pred.dtype == np.dtype("float64")
                        and np.isfinite(pred).all() and identity(pred) == record["prediction_array_identity"],
                        "prediction shape/value/hash changed")
                value = mse(pred, truth)
                close(value, record["mse"])
                risks[(case["id"], slot)] = value
    require(len(data) == 192, "some declared case has no retained data for independent checks")
    write(output/"retained_attempt_inventory.json", attempts)
    return records, data, risks, {name: sha(root/name) for name in sorted(wanted)}


def paired_data(cases, data, records):
    import numpy as np
    lookup = {(c["seed"], c["delta"], c["n_rct"]): c["id"] for c in cases}
    for name in ("obs_X", "rct_X", "eval_X"):
        require(len({identity(data[lookup[(seed, 0, 512)]][name])["sha256"] for seed in range(30000, 30032)}) == 32,
                "different seed labels contain a duplicated covariate block")
    for seed in range(30000, 30032):
        base = data[lookup[(seed, 0, 512)]]
        for delta in (0, 1, 2):
            for budget in (128, 512):
                arrays = data[lookup[(seed, delta, budget)]]
                for name in ("rct_X", "rct_T", "rct_Y", "rct_known_propensity"):
                    np.testing.assert_array_equal(arrays[name], base[name][:budget])
                for name in ("obs_X", "obs_T", "eval_X"):
                    np.testing.assert_array_equal(arrays[name], base[name])
                for x, treatment, actual, reference in zip(arrays["obs_X"], arrays["obs_T"], arrays["obs_Y"], base["obs_Y"]):
                    expected = delta*(float(treatment)-.5)*math.cos(math.pi*float(x[0]))*math.sin(math.pi*float(x[1]))
                    close(float(actual)-float(reference), expected)
                np.testing.assert_array_equal(arrays["obs_Y"], data[lookup[(seed, delta, 512)]]["obs_Y"])
    groups = {"rct_only_delta_invariant": [], "obs_start_budget_invariant": []}
    for seed in range(30000, 30032):
        for budget in (128, 512):
            groups["rct_only_delta_invariant"].append([records[(lookup[(seed, d, budget)], "rct_only")] for d in (0, 1, 2)])
        for delta in (0, 1, 2):
            groups["obs_start_budget_invariant"].append([records[(lookup[(seed, delta, b)], "obs_start")] for b in (128, 512)])
    result = {}
    for name, collection in groups.items():
        for group in collection:
            successful = [r["prediction_array_identity"]["sha256"] for r in group if r["status"] == "success"]
            require(len(successful) >= 2, "source-exclusion invariant is vacuous in a declared group")
            require(len(set(successful)) == 1, "successful control prediction hashes differ across excluded conditions")
        complete = sum(all(r["status"] == "success" for r in group) for group in collection)
        result[name] = {"expected_groups": len(collection), "complete_groups": complete,
                        "all_groups_checked": complete == len(collection), "discrepant_successful_groups": 0}
    require(sum(x["expected_groups"] for x in result.values()) == 160, "control invariance group count changed")
    return result


def compare_summary(summary, cfg, cases, records, risks, invariants, record_hashes):
    require(summary["schema_version"] == 1 and summary["scope"] == cfg["scope"]
            and summary["expected_records"] == 576 and summary["missing"] == 0
            and summary["record_sha256"] == record_hashes and summary["invariances"] == invariants,
            "analysis coverage/source-invariance binding differs")
    success = len(risks)
    require(summary["success"] == success and summary["failed"] == 576-success, "failure counts changed")
    require(len(summary["conditions"]) == 6, "analysis contains extra/missing conditions")
    rebuilt = []
    for di, delta in enumerate((0, 1, 2)):
        for budget in (128, 512):
            subset = [c for c in cases if c["delta_index"] == di and c["n_rct"] == budget]
            row = summary["conditions"][len(rebuilt)]
            require(row["delta"] == delta and row["n_rct"] == budget
                    and set(row["recipes"]) == set(SLOTS)
                    and set(row["paired"]) == {"full_minus_rct_only", "full_minus_obs_start"}, "analysis condition/contrast grid changed")
            independent = {"delta": delta, "n_rct": budget, "recipes": {}, "paired": {}}
            for slot in SLOTS:
                values = [risks[(c["id"], slot)] for c in subset if (c["id"], slot) in risks]
                target = {"successful": len(values), "failed": 32-len(values),
                          "mean_mse_success_only": math.fsum(values)/len(values) if values else None}
                actual = row["recipes"][slot]
                require(set(actual) == set(target) and actual["successful"] == target["successful"]
                        and actual["failed"] == target["failed"], "recipe success/failure counts changed")
                close(actual["mean_mse_success_only"], target["mean_mse_success_only"])
                independent["recipes"][slot] = target
            for control in ("rct_only", "obs_start"):
                differences = [risks[(c["id"], "full")]-risks[(c["id"], control)] for c in subset
                               if (c["id"], "full") in risks and (c["id"], control) in risks]
                target, name = paired(differences), "full_minus_"+control
                actual = row["paired"][name]
                require(set(actual) == set(target) and all(actual[k] == target[k] for k in
                        ("expected_pairs", "successful_pairs", "complete")), "paired coverage/withholding changed")
                for key in ("mean", "mcse"):
                    close(actual[key], target[key])
                if target["ci95"] is None:
                    require(actual["ci95"] is None, "incomplete-pair interval was not withheld")
                else:
                    require(len(actual["ci95"]) == 2, "invalid paired interval")
                    for a, b in zip(actual["ci95"], target["ci95"]):
                        close(a, b)
                independent["paired"][name] = target
            rebuilt.append(independent)
    return rebuilt


def verify_tables(directory, cases, records, risks, conditions, record_hashes):
    """Check the renderer's scalar CSVs against independent arithmetic/evidence."""
    def table(filename, columns, count):
        with (directory/filename).open(newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == columns.split(","), "rendered CSV schema changed")
            rows = list(reader)
        require(len(rows) == count and all(None not in row for row in rows), "rendered CSV row count/shape changed")
        return rows
    def check(actual, expected):
        if isinstance(expected, bool):
            require(actual == str(expected).lower(), "rendered boolean changed")
        elif expected is None:
            require(actual == "", "rendered missing value was imputed")
        elif isinstance(expected, (int, float)):
            require(actual != "", "rendered numeric value missing")
            close(float(actual), expected)
        else:
            require(actual == expected, "rendered categorical/hash value changed")
    scored = table("condition_scores.csv", "delta,n_rct,slot,method_id,expected_seeds,successful,failed,mean_mse_success_only,mcse,complete", 18)
    paired_rows = table("paired_contrasts.csv", "delta,n_rct,contrast,expected_pairs,successful_pairs,complete,mean,mcse,ci95_low,ci95_high", 12)
    scalars = table("scalar_records.csv", "case_id,seed,delta_index,delta,n_rct,slot,method_id,status,rct_labels,obs_labels,additional_tuning_labels,mse,fit_predict_seconds,launcher_seconds,worker_self_maxrss_kib,worker_children_maxrss_kib,input_sha256,evaluator_inputs_sha256,eval_tau_sha256,prediction_sha256,record_sha256", 576)
    condition_lookup = {(x["delta"], x["n_rct"]): x for x in conditions}
    seen = set()
    for row in scored:
        key = (float(row["delta"]), int(row["n_rct"]), row["slot"])
        require(key not in seen and key[:2] in condition_lookup and key[2] in SLOTS, "duplicate/unrequested rendered score")
        seen.add(key)
        delta, budget, slot = key
        target = condition_lookup[key[:2]]["recipes"][slot]
        values = [risks[(c["id"], slot)] for c in cases if c["delta"] == delta and c["n_rct"] == budget
                  and (c["id"], slot) in risks]
        mean = target["mean_mse_success_only"]
        se = math.sqrt(math.fsum((x-mean)**2 for x in values)/31/32) if len(values) == 32 else None
        expected = {"delta": delta, "n_rct": budget, "slot": slot, "method_id": METHODS[slot],
                    "expected_seeds": 32, "successful": target["successful"], "failed": target["failed"],
                    "mean_mse_success_only": mean, "mcse": se, "complete": len(values) == 32}
        for name, value in expected.items():
            check(row[name], value)
    seen = set()
    for row in paired_rows:
        key = (float(row["delta"]), int(row["n_rct"]), row["contrast"])
        require(key not in seen and key[:2] in condition_lookup
                and key[2] in {"full_minus_rct_only", "full_minus_obs_start"}, "duplicate/unrequested rendered contrast")
        seen.add(key)
        target = condition_lookup[key[:2]]["paired"][key[2]]
        expected = {"delta": key[0], "n_rct": key[1], "contrast": key[2],
                    **{k: v for k, v in target.items() if k != "ci95"},
                    "ci95_low": target["ci95"][0] if target["ci95"] is not None else None,
                    "ci95_high": target["ci95"][1] if target["ci95"] is not None else None}
        for name, value in expected.items():
            check(row[name], value)
    case_lookup = {c["id"]: c for c in cases}
    seen = set()
    for row in scalars:
        key = (row["case_id"], row["slot"])
        require(key not in seen and key in records, "duplicate/unrequested rendered scalar record")
        seen.add(key)
        case, record = case_lookup[key[0]], records[key]
        expected = {"case_id": case["id"], **{k: case[k] for k in ("seed", "delta_index", "delta", "n_rct")},
                    "slot": key[1], "method_id": METHODS[key[1]], "status": record["status"],
                    "rct_labels": case["n_rct"] if key[1] != "obs_start" else 0,
                    "obs_labels": 512 if key[1] != "rct_only" else 0, "additional_tuning_labels": 0,
                    "mse": risks.get(key), "fit_predict_seconds": record.get("fit_predict_seconds"),
                    "launcher_seconds": record.get("launcher_seconds"),
                    "worker_self_maxrss_kib": record.get("worker_self_usage", {}).get("maxrss_kib"),
                    "worker_children_maxrss_kib": record.get("worker_children_usage", {}).get("maxrss_kib"),
                    "input_sha256": digest(record["method_argument_array_identity"]) if "method_argument_array_identity" in record else None,
                    "evaluator_inputs_sha256": digest(record["observed_array_identity"]) if "observed_array_identity" in record else None,
                    "eval_tau_sha256": record.get("evaluation_array_identity", {}).get("eval_tau", {}).get("sha256"),
                    "prediction_sha256": record.get("prediction_array_identity", {}).get("sha256"),
                    "record_sha256": record_hashes[f"records/{case['id']}/{key[1]}/record.json"]}
        for name, value in expected.items():
            check(row[name], value)
    return {name: {"rows": count, "sha256": sha(directory/name)} for name, count in
            (("condition_scores.csv", 18), ("paired_contrasts.csv", 12), ("scalar_records.csv", 576))}


def replay_worker(repo, arrays, case, slot, native, destination):
    """One fresh-process fit; full logs/provenance remain in its output folder."""
    sys.path.insert(0, str(repo/"src"))
    with (destination/"stdout.log").open("x") as stdout, (destination/"stderr.log").open("x") as stderr:
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                import numpy as np
                from obs_rct_benchmark.causal_icm import CausalICMSourceControl, OfficialCausalICM
                from obs_rct_benchmark.data import SourceSample, TrainingView
                for name, module in tuple(sys.modules.items()):
                    if name.startswith("obs_rct_benchmark") and getattr(module, "__file__", None):
                        require(Path(module.__file__).resolve().is_relative_to((repo/"src").resolve()),
                                "replay imported another producer")
                def sample(source):
                    return SourceSample(arrays[source+"_X"], arrays[source+"_T"], arrays[source+"_Y"], source,
                                        source+"_train", arrays["rct_known_propensity"] if source == "rct" else None)
                if slot == "full":
                    rct = sample("rct")
                    tune = SourceSample(rct.X[:2], rct.T[:2], rct.Y[:2], "rct", "rct_tune", rct.known_propensity[:2])
                    estimator = OfficialCausalICM(rho=.5)
                    prediction = estimator.fit_predict(TrainingView(sample("obs"), rct, tune), arrays["eval_X"], seed=30000)
                else:
                    source = "rct" if slot == "rct_only" else "obs"
                    estimator = CausalICMSourceControl(source=source, rho=.5)
                    prediction = estimator.fit_predict_sample(sample(source), arrays["eval_X"], seed=30000)
                fit_provenance({"estimator_provenance": estimator.provenance_}, native, arrays, case, slot)
                np.save(destination/"prediction.npy", np.asarray(prediction), allow_pickle=False)
                write(destination/"fit.json", {"status": "passed", "provenance": estimator.provenance_})
        except BaseException as exc:
            traceback.print_exc(file=stderr)
            write(destination/"fit.json", {"status": "failed", "exception_type": type(exc).__name__})
            raise


def replay(repo, root, cases, records, data, native, output):
    import numpy as np
    destination = output/"replay_seed30000"
    destination.mkdir()
    results = []
    for case in cases:
        if case["seed"] != 30000:
            continue
        arrays = data[case["id"]]
        for slot in SLOTS:
            record = records[(case["id"], slot)]
            require(record["status"] == "success", "fixed replay cell lacks successful saved prediction")
            attempt = destination/f"{case['id']}_{slot}"
            attempt.mkdir()
            process = multiprocessing.get_context("spawn").Process(
                target=replay_worker, args=(repo, arrays, case, slot, native, attempt))
            try:
                process.start()
                process.join(180)
                if process.is_alive():
                    write(attempt/"timeout.json", {"status": "timeout", "limit_seconds": 180})
                    raise TimeoutError("fixed first-seed replay fit exceeded its declared time limit")
                require(process.exitcode == 0 and read(attempt/"fit.json")["status"] == "passed",
                        "fixed first-seed replay fit failed; see retained per-fit logs")
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    if process.is_alive():
                        process.kill()
                        process.join(5)
                require(not process.is_alive(), "could not reap replay process")
            prediction = np.load(attempt/"prediction.npy", allow_pickle=False)
            require(prediction.shape == (1024,) and np.isfinite(prediction).all(), "invalid replay prediction")
            saved = np.load(root/record["attempt"]/"prediction.npy", allow_pickle=False)
            for actual, expected in zip(prediction, saved):
                close(actual, expected, replay=True)
            value = mse(prediction, tau_rows(arrays["eval_X"]))
            close(value, record["mse"])
            results.append({"case": case["id"], "slot": slot, "mse": value,
                            "max_absolute_prediction_difference": max(abs(float(a)-float(b)) for a, b in zip(prediction, saved)),
                            "prediction_identity": identity(prediction), "status": "passed",
                            "artifact_sha256": {p.name: sha(p) for p in sorted(attempt.iterdir()) if p.is_file()}})
    require(len(results) == 18, "fixed first-seed replay did not contain exactly 18 fits")
    write(destination/"replay.json", results)
    return {"seed": 30000, "fits": len(results), "status": "passed", "record_sha256": sha(destination/"replay.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "input", "analysis", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--replay-first-seed", action="store_true")
    args = parser.parse_args()
    raw = os.environ.get("GSDS_WORK_ROOT")
    if (sys.platform != "linux" or not raw or not Path(raw).is_absolute()
            or not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(n) != "1" for n in THREADS)):
        parser.error("requires an allocated GSDS Slurm job, absolute shared root and four thread limits equal one")
    work = Path(raw).resolve(strict=True)
    for name in ("repo", "config", "input", "analysis", "output", "tables_dir"):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve(strict=name != "output")
            if not value.is_relative_to(work) or value == work:
                parser.error("all paths must be below GSDS_WORK_ROOT")
            setattr(args, name, value)
    if any(args.output.is_relative_to(p) or p.is_relative_to(args.output) for p in (args.repo, args.input, args.analysis)):
        parser.error("verification output must be separate from source/input/analysis trees")
    os.umask(0o077)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"prospective_plan.json", PLAN)
    result = {"schema_version": 1, "status": "running", "driver_sha256": sha(__file__),
              "plan_sha256": digest(PLAN), "replay_requested": args.replay_first_seed,
              "failure_details_server_only": "stderr.log"}
    write(args.output/"verification.json", result)
    start = time.perf_counter()
    with (args.output/"stdout.log").open("x") as stdout, (args.output/"stderr.log").open("x") as stderr:
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                cfg, cases = configuration(args.config)
                manifest = read(args.input/"manifest.json")
                require(manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
                        and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
                        and type(manifest["shard_count"]) is int and 1 <= manifest["shard_count"] <= 192
                        and manifest["config"] == cfg and manifest["cases"] == cases and manifest["slots"] == list(SLOTS)
                        and manifest["expected_records"] == 576 and manifest["bindings"]["config_sha256"] == sha(args.config),
                        "frozen manifest/config/case binding changed")
                native = provenance(args.repo, manifest["bindings"])
                records, data, risks, hashes = terminal_records(args.input, cfg, cases, manifest, native, args.output)
                invariants = paired_data(cases, data, records)
                summary = read(args.analysis/"summary.json")
                require(summary["bindings"] == manifest["bindings"], "analysis producer binding changed")
                conditions = compare_summary(summary, cfg, cases, records, risks, invariants, hashes)
                write(args.output/"independent_conditions.json", conditions)
                result.update(expected_records=576, success=len(risks), failed=576-len(risks), missing=0,
                              source_fingerprint=manifest["bindings"]["source"]["package_source"]["aggregate_sha256"],
                              config_sha256=sha(args.config), manifest_sha256=sha(args.input/"manifest.json"),
                              analysis_sha256=sha(args.analysis/"summary.json"), invariant_groups=160,
                              retained_attempt_inventory_sha256=sha(args.output/"retained_attempt_inventory.json"),
                              checked_conditions=6, checked_recipe_means=18, checked_paired_summaries=12)
                if args.tables_dir is not None:
                    result["tables"] = verify_tables(args.tables_dir, cases, records, risks, conditions, hashes)
                if args.replay_first_seed:
                    result["replay"] = replay(args.repo, args.input, cases, records, data, native, args.output)
            result["status"] = "passed"
        except BaseException as exc:
            result.update(status="failed", exception_type=type(exc).__name__)
            traceback.print_exc(file=stderr)
    result["wall_seconds"] = time.perf_counter()-start
    result["log_sha256"] = {name: sha(args.output/name) for name in ("stdout.log", "stderr.log")}
    write(args.output/"verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "wall_seconds", "replay_requested")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
