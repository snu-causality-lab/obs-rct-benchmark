"""Refit a complete static shard and score with a separate dot-product MSE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from obs_rct_benchmark.methods import custom_provenance, make_method
from obs_rct_benchmark.runner import (
    _generate, _hash, _load_json, _read_records, _validate_resume_records,
    environment_provenance, external_method_provenance, load_config,
    method_seed, record_id, resolved_scenarios, scenario_input_provenance,
    source_provenance,
)


def replay(config_path: Path, run: Path, output: Path) -> dict:
    config = load_config(config_path)
    manifest = _load_json(run / "run_manifest.json")
    actual = {
        "source": source_provenance(),
        "custom_methods": {name: value for name, value in custom_provenance().items()
                           if name in config.methods},
        "environment": environment_provenance(),
        "external_methods": external_method_provenance(config.methods),
        "resolved_scenarios": resolved_scenarios(config),
        "empirical_inputs": scenario_input_provenance(config),
    }
    if (manifest["config"] != config.payload()
            or manifest["config_hash"] != config.config_hash
            or manifest["source_fingerprint"] != _hash(actual)):
        raise ValueError("replay configuration or execution provenance differs")
    records_path = run / "records.jsonl"
    records = _read_records(records_path)
    expected = {
        record_id(config, scenario.key, seed, method): (scenario.key, seed, method)
        for scenario in config.scenarios for seed in config.seeds for method in config.methods
    }
    seen = _validate_resume_records(config, records, expected)
    if len(seen) != len(expected) or any(row["status"] != "ok" for row in records):
        raise ValueError("refit replay requires a complete successful shard")
    lookup = {(row["scenario"], row["seed"], row["method"]): row for row in records}
    checks = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            problem = _generate(scenario, seed)
            for method in config.methods:
                row = lookup[(scenario.key, seed, method)]
                model = make_method(method, **dict(config.method_options.get(method, {})))
                prediction = np.asarray(model.fit_predict(
                    problem.training_view(), problem.eval_X,
                    seed=method_seed(scenario.key, seed, method)), dtype=float)
                if prediction.shape != problem.eval_tau.shape or not np.isfinite(prediction).all():
                    raise ValueError("replayed predictions are malformed")
                difference = prediction - problem.eval_tau
                mse = float(np.dot(difference, difference) / difference.size)
                saved = row["metrics"]["cate_mse"]
                if not math.isclose(mse, saved, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("refitted dot-product score differs from saved MSE")
                checks.append({"scenario": scenario.key, "seed": seed, "method": method,
                               "absolute_difference": abs(mse - saved)})
    result = {
        "scope": "all cells of the supplied shard; deterministic refit, not a new replication",
        "config_hash": config.config_hash,
        "source_fingerprint": manifest["source_fingerprint"],
        "manifest_sha256": hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "checker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records_checked": len(checks),
        "max_absolute_metric_difference": max(item["absolute_difference"] for item in checks),
        "tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "checks": checks,
    }
    if output.exists():
        raise ValueError("replay output must be new")
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.config, args.run, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))


if __name__ == "__main__":
    main()
