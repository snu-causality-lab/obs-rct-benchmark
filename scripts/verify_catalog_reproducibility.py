"""Reproduce the first-seed, small-budget slice of every catalog preset.

The slice and two refits are fixed before result inspection. Saved successes
and terminal failures are both checked; unsupported methods remain N/A.
Repeating the producer is a reproducibility check, not an independent learner
implementation or an independent scientific replication.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from obs_rct_benchmark.evaluation import cate_metrics
from obs_rct_benchmark.methods import make_method
from obs_rct_benchmark.runner import _generate, load_config, method_seed


_SPEC = importlib.util.spec_from_file_location(
    "catalog_analysis_dependency", Path(__file__).with_name("analyze_catalog_study.py"))
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)

ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12
FIT_REPEATS = 2


def array_hash(array) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(values.shape), "dtype": values.dtype.str}, sort_keys=True).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def problem_hash(problem) -> str:
    """Hash the generated roles without retaining or publishing their arrays."""
    hashes = {}
    for name in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        sample = getattr(problem, name)
        hashes[name] = {field: array_hash(getattr(sample, field)) for field in ("X", "T", "Y")}
        hashes[name]["known_propensity"] = (None if sample.known_propensity is None
                                             else array_hash(sample.known_propensity))
    hashes["eval_X"], hashes["eval_tau"] = array_hash(problem.eval_X), array_hash(problem.eval_tau)
    hashes["target_reference"] = (None if problem.target_reference is None
                                   else array_hash(problem.target_reference.X))
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def selected_scenarios(config) -> tuple:
    """One small-budget condition for each preset; never choose on accuracy."""
    selected = tuple(scenario for scenario in config.scenarios
                     if scenario.parameters["n_obs"] == 512
                     and scenario.parameters["n_rct_train"] == 128)
    identities = [(scenario.kind, scenario.id) for scenario in selected]
    if not selected or len(set(identities)) != len(identities):
        raise ValueError("the replay slice must contain exactly one small-budget condition per preset")
    all_presets = {(scenario.kind, scenario.id) for scenario in config.scenarios}
    if set(identities) != all_presets:
        raise ValueError("small-budget replay slice does not cover every declared preset")
    return selected


def _fit_once(problem, estimator: str, options: dict, seed: int) -> dict:
    try:
        model = make_method(estimator, **options)
        raw = model.fit_predict(problem.training_view(), problem.eval_X, seed=seed)
        # Use the producer validator only to reproduce its failure type. The
        # successful score below is independently computed as a dot product.
        cate_metrics(raw, problem.eval_tau)
        prediction = np.asarray(raw, dtype=np.float64)
        difference = prediction - problem.eval_tau
        mse = float(np.dot(difference, difference) / difference.size)
        if not math.isfinite(mse):
            raise ValueError("direct dot-product MSE is nonfinite")
        return {"status": "ok", "prediction": prediction, "mse": mse,
                "prediction_sha256": array_hash(prediction)}
    except Exception as exc:
        return {"status": "failed", "stage": "method_or_metric", "error_type": type(exc).__name__}


def verify(config, runs: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValueError("reproducibility output must be new")
    records, inputs = analysis.collect_records(config, runs, require_full=False)
    scenarios = selected_scenarios(config)
    seed = config.seeds[0]
    wanted = {(scenario.key, seed, method) for scenario in scenarios
              for method in analysis.applicable_methods(config, scenario)}
    if not wanted <= records.keys():
        raise ValueError("saved terminal records do not cover the entire prespecified replay slice")
    generated, checks = [], []
    for scenario in scenarios:
        generated_problems, generation_errors = [], []
        for _ in range(2):
            try:
                generated_problems.append(_generate(scenario, seed))
                generation_errors.append(None)
            except Exception as exc:
                generated_problems.append(None)
                generation_errors.append(type(exc).__name__)
        hashes = [None if problem is None else problem_hash(problem) for problem in generated_problems]
        data_match = (hashes[0] == hashes[1] and generation_errors[0] == generation_errors[1])
        generated.append({"scenario": scenario.key, "seed": seed,
                          "repeat_data_sha256": hashes, "repeat_error_types": generation_errors,
                          "match": data_match})
        for method in analysis.applicable_methods(config, scenario):
            row = records[(scenario.key, seed, method)]
            entry = {"scenario": scenario.key, "seed": seed, "method": method,
                     "record_id": row["record_id"], "recorded_status": row["status"],
                     "recorded_stage": row.get("stage"), "recorded_error_type": row.get("error_type"),
                     "data_match": data_match, "match": False, "refit_attempts": 0,
                     "recorded_mse": None if row["status"] != "ok" else row["metrics"]["cate_mse"]}
            if generation_errors[0] is not None or generation_errors[1] is not None:
                entry["repeat_statuses"] = ["not_attempted"] * FIT_REPEATS
                entry["repeat_error_types"] = generation_errors
                entry["match"] = (data_match and row["status"] == "failed" and row.get("stage") == "simulation"
                                   and row.get("error_type") == generation_errors[0])
                checks.append(entry)
                continue
            fits = [_fit_once(problem, method, dict(config.method_options.get(method, {})),
                              method_seed(scenario.key, seed, method)) for problem in generated_problems]
            entry["refit_attempts"] = FIT_REPEATS
            entry["repeat_statuses"] = [fit["status"] for fit in fits]
            if row["status"] == "failed":
                entry["repeat_error_types"] = [fit.get("error_type") for fit in fits]
                entry["match"] = (data_match and all(
                    fit["status"] == "failed" and fit["stage"] == row.get("stage")
                    and fit["error_type"] == row.get("error_type") for fit in fits))
            elif all(fit["status"] == "ok" for fit in fits):
                entry["repeat_mse"] = [fit["mse"] for fit in fits]
                entry["absolute_mse_differences"] = [abs(fit["mse"] - row["metrics"]["cate_mse"]) for fit in fits]
                entry["repeat_prediction_sha256"] = [fit["prediction_sha256"] for fit in fits]
                entry["predictions_byte_identical"] = fits[0]["prediction_sha256"] == fits[1]["prediction_sha256"]
                entry["predictions_within_tolerance"] = bool(np.allclose(
                    fits[0]["prediction"], fits[1]["prediction"],
                    rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE))
                entry["match"] = (data_match and entry["predictions_within_tolerance"] and all(
                    math.isclose(fit["mse"], row["metrics"]["cate_mse"],
                                 rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE) for fit in fits))
            else:
                entry["repeat_error_types"] = [fit.get("error_type") for fit in fits]
            checks.append(entry)
    differences = [difference for entry in checks for difference in entry.get("absolute_mse_differences", [])]
    result = {
        "scope": "first declared seed; OBS512/RCTtraining128; every declared preset and applicable method; two regenerated datasets and two refits",
        "normalized_config_sha256": config.config_hash,
        "seed": seed, "preset_count": len(scenarios), "selected_cells": len(wanted),
        "fit_repeats": FIT_REPEATS,
        "planned_refit_attempts": FIT_REPEATS * len(wanted),
        "actual_refit_attempts": sum(entry["refit_attempts"] for entry in checks),
        "matched_cells": sum(entry["match"] for entry in checks),
        "mismatched_cells": sum(not entry["match"] for entry in checks),
        "selected_recorded_failures": sum(entry["recorded_status"] == "failed" for entry in checks),
        "consistent_terminal_failures": sum(entry["recorded_status"] == "failed" and entry["match"] for entry in checks),
        "successful_refit_attempts": sum(status == "ok" for entry in checks for status in entry.get("repeat_statuses", [])),
        "max_absolute_mse_difference": max(differences, default=None),
        "tolerance": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE},
        "original_data_hash_scope": "producer did not save data-array hashes; frozen generation is repeated twice here, while direct MSE is compared to the saved original fit",
        "original_prediction_hash_scope": "producer did not save predictions; hashes compare the two new refits, while both scores are compared to the saved original metric",
        "inference_scope": "reproducibility, not independent algorithm verification, a new replication, or evidence of method superiority",
        "checker_source_sha256": analysis.file_hash(Path(__file__)),
        "collector_source_sha256": analysis.file_hash(Path(__file__).with_name("analyze_catalog_study.py")),
        "design_validator_sha256": analysis.file_hash(Path(__file__).with_name("make_catalog_configs.py")),
        "inputs": inputs, "data_checks": generated, "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    analysis.validate_study(config)
    result = verify(config, args.runs, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"checks", "data_checks", "inputs"}}, sort_keys=True))
    raise SystemExit(int(result["mismatched_cells"] != 0))


if __name__ == "__main__":
    main()
