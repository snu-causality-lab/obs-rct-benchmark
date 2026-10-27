"""Independently check retained native-panel trial CATE and score arithmetic.

No generator, producer truth, scorer, or native-panel data helper is imported.
CorNet's stated NumPy initialization convention is reconstructed here and its
network uses explicit einsum contractions. This checks that convention, not
identity with unseeded upstream PyTorch parameter draws. C08 uses adaptive
normal integration, independently of the producer's Hermite quadrature.

Query support checks are exact; moment checks are empirical sanity checks,
not proof of a sampling distribution, randomness, or independence. Full arrays
remain server evidence. Run numerical checks on an allocated compute node.
Analytic target agreement uses absolute tolerance only. Score replay also uses
a relative tolerance to allow floating reduction error at large finite losses.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ABSOLUTE_TOLERANCE = 1e-12
SCORE_ABSOLUTE_TOLERANCE = 1e-12
SCORE_RELATIVE_TOLERANCE = 1e-12
SCENARIOS = {
    "structural:M0": (5, "normal"),
    "structural:M2": (5, "normal"),
    "structural:M10": (5, "normal"),
    "addendum:C08@trial_target": (5, "normal"),
    "paper:integrative_r_learner_kps_code": (1, "uniform"),
    "paper:cornet_official_n_conf": (25, "normal"),
}


def _query(X: Any, scenario_key: str) -> np.ndarray:
    if scenario_key not in SCENARIOS:
        raise ValueError(f"unknown native-panel scenario: {scenario_key}")
    values = np.asarray(X, dtype=np.float64)
    dimension = SCENARIOS[scenario_key][0]
    if (values.ndim != 2 or values.shape[1] != dimension
            or values.shape[0] == 0 or not np.isfinite(values).all()):
        raise ValueError(f"eval_X must be a nonempty finite matrix with {dimension} columns")
    return values


@lru_cache(maxsize=1)
def c08_selected_mean() -> float:
    """E[Z expit(2Z)] / E[expit(2Z)], Z standard normal."""
    from scipy.integrate import quad

    def weighted_density(z: float) -> float:
        # A stable sigmoid avoids an overflow on infinite integration tails.
        exponential = math.exp(-2.0 * abs(z))
        acceptance = 1.0 / (1.0 + exponential) if z >= 0 else exponential / (1.0 + exponential)
        return acceptance * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    rate, rate_error = quad(weighted_density, -np.inf, np.inf,
                            epsabs=2e-14, epsrel=2e-14, limit=200)
    numerator, numerator_error = quad(lambda z: z * weighted_density(z),
                                      -np.inf, np.inf,
                                      epsabs=2e-14, epsrel=2e-14, limit=200)
    if (abs(rate - .5) > 1e-13 or rate_error > 1e-13
            or numerator_error > 1e-13):
        raise ValueError("adaptive C08 integration failed its error check")
    return numerator / rate


def _representation(X: np.ndarray, parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    value = X
    for layer in range(3):
        value = np.einsum("ij,kj->ik", value, parameters[f"rep_weight_{layer}"])
        value = value + parameters[f"rep_bias_{layer}"]
        if layer < 2:
            value = np.tanh(value)
    return value


def _head(representation: np.ndarray, parameters: Mapping[str, np.ndarray],
          label: str, arm: int) -> np.ndarray:
    return (np.einsum("ij,j->i", representation, parameters[f"{label}_weight_{arm}"])
            + float(parameters[f"{label}_bias_{arm}"][0]))


@lru_cache(maxsize=1)
def _cornet_parameters() -> dict[str, np.ndarray]:
    """Reconstruct d25/seed2025 without calling the producer initializer."""
    d, mechanism_seed = 25, 2025
    rng = np.random.Generator(np.random.PCG64(mechanism_seed))
    bound = 1.0 / math.sqrt(d)
    parameters = {}
    for layer in range(3):
        parameters[f"rep_weight_{layer}"] = rng.uniform(-bound, bound, (d, d))
        parameters[f"rep_bias_{layer}"] = rng.uniform(-bound, bound, d)
    for label in ("base", "correction"):
        for arm in (0, 1):
            weights = rng.uniform(-bound, bound, d)
            if label == "base":
                weights *= 10.0 / math.sqrt(math.fsum(float(w) ** 2 for w in weights))
            parameters[f"{label}_weight_{arm}"] = weights
            parameters[f"{label}_bias_{arm}"] = rng.uniform(-bound, bound, 1)
    for arm in (1, 0):
        zero_indices = rng.choice(np.arange(1, d), int(.666 * d), replace=False)
        parameters[f"correction_weight_{arm}"][zero_indices] = 0.0
    reference_rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(mechanism_seed, spawn_key=(0x434F524E,))))
    reference = reference_rng.standard_normal((10_000, d))
    representation = _representation(reference, parameters)
    correction = (_head(representation, parameters, "correction", 1)
                  - _head(representation, parameters, "correction", 0))
    rms = math.sqrt(math.fsum(float(value) ** 2 for value in correction) / len(correction))
    for arm in (0, 1):
        for kind in ("weight", "bias"):
            parameters[f"correction_{kind}_{arm}"] *= 2.0 / rms
    for values in parameters.values():
        values.setflags(write=False)
    return parameters


def independent_truth(X: Any, scenario_key: str) -> np.ndarray:
    """Return the panel's native trial-CATE at X using independent equations."""
    values = _query(X, scenario_key)
    if scenario_key == "paper:integrative_r_learner_kps_code":
        x = values[:, 0]
        return 1.0 + 2.0 * x + .75 * x ** 2
    if scenario_key == "paper:cornet_official_n_conf":
        parameters = _cornet_parameters()
        representation = _representation(values, parameters)
        means = [(_head(representation, parameters, "base", arm)
                  + _head(representation, parameters, "correction", arm)) for arm in (0, 1)]
        return means[1] - means[0]
    # Axis-aligned d5 Gaussian structural presets share the smooth base CATE.
    result = np.array([1.0 + .5 * math.tanh(float(row[1]))
                       + .25 * math.sin(float(row[2])) for row in values])
    if scenario_key == "addendum:C08@trial_target":
        # Arm loadings -1 and +1; participation depends on a distinct Z only.
        result += 2.0 * c08_selected_mean()
    return result


def score(prediction: Any, truth: Any) -> dict[str, float]:
    """Scalar differences plus math.fsum, independent of NumPy's mean scorer."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if (prediction.ndim != 1 or prediction.shape != truth.shape
            or prediction.size == 0 or not np.isfinite(prediction).all()
            or not np.isfinite(truth).all()):
        raise ValueError("prediction and truth must be matching nonempty finite vectors")
    try:
        mse = math.fsum((float(p) - float(t)) ** 2
                        for p, t in zip(prediction, truth)) / len(truth)
    except OverflowError as exc:
        raise ValueError("nonfinite independent score") from exc
    if not math.isfinite(mse):
        raise ValueError("nonfinite independent score")
    return {"cate_mse": mse, "root_pehe": math.sqrt(mse)}


def _query_law(X: np.ndarray, scenario_key: str) -> dict[str, Any]:
    law = SCENARIOS[scenario_key][1]
    n = len(X)
    if law == "uniform" and np.any(np.abs(X) > 1.0):
        raise ValueError("native KPS query lies outside Uniform[-1,1] support")
    means = np.mean(X, axis=0)
    second_moments = np.mean(X * X, axis=0)
    expected_second = 1.0 / 3.0 if law == "uniform" else 1.0
    mean_bound = 8.0 * math.sqrt(expected_second / n)
    variance_square = 4.0 / 45.0 if law == "uniform" else 2.0
    second_bound = 8.0 * math.sqrt(variance_square / n)
    checked = n >= 128
    if checked and (np.max(np.abs(means)) > mean_bound
                    or np.max(np.abs(second_moments - expected_second)) > second_bound):
        raise ValueError("query empirical moments fail the prespecified eight-standard-error sanity bounds")
    return {"dimension": X.shape[1], "n_eval": n,
            "law": "Uniform[-1,1]" if law == "uniform" else "N(0,I)",
            "support_passed": True, "moment_check_performed": checked,
            "coordinate_means": means.tolist(), "coordinate_second_moments": second_moments.tolist(),
            "mean_absolute_bound": mean_bound, "second_moment_absolute_bound": second_bound,
            "scope": "Empirical coordinate-moment sanity only; does not prove distributional form, randomness, or independence."}


def verify_case(arrays: Mapping[str, Any], scenario_key: str) -> dict[str, Any]:
    """Return passed/errors/checks; optional prediction and claimed scores replay loss."""
    checks: dict[str, Any] = {}
    errors = []
    try:
        X = _query(arrays["eval_X"], scenario_key)
        checks["query"] = _query_law(X, scenario_key)
        expected = independent_truth(X, scenario_key)
        actual = np.asarray(arrays["eval_tau"], dtype=np.float64)
        if actual.shape != (len(X),) or not np.isfinite(actual).all():
            raise ValueError("eval_tau must be a finite vector matching eval_X rows")
        maximum = float(np.max(np.abs(actual - expected)))
        checks["target"] = {"max_absolute_error": maximum,
                            "absolute_tolerance": ABSOLUTE_TOLERANCE,
                            "meaning": "native trial-CATE; C08 includes selected-participant shift"}
        if maximum > ABSOLUTE_TOLERANCE:
            raise ValueError(f"independent trial target differs: max error {maximum:.17g}")
        if "prediction" in arrays:
            checks["score"] = score(arrays["prediction"], expected)
            checks["score_tolerance"] = {
                "absolute": SCORE_ABSOLUTE_TOLERANCE,
                "relative": SCORE_RELATIVE_TOLERANCE,
                "comparison": "math.isclose: max(absolute, relative * max(abs(actual), abs(expected)))",
            }
            for metric, value in checks["score"].items():
                if metric in arrays:
                    claimed = np.asarray(arrays[metric], dtype=np.float64)
                    if claimed.size != 1 or not np.isfinite(claimed).all():
                        raise ValueError(f"invalid claimed {metric}")
                    claimed_value = float(claimed.item())
                    difference = abs(claimed_value - value)
                    checks[f"{metric}_absolute_error"] = difference
                    if not math.isclose(claimed_value, value,
                                        rel_tol=SCORE_RELATIVE_TOLERANCE,
                                        abs_tol=SCORE_ABSOLUTE_TOLERANCE):
                        raise ValueError(f"retained {metric} differs from independently replayed score")
        elif "cate_mse" in arrays or "root_pehe" in arrays:
            raise ValueError("claimed score requires a retained prediction array")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        errors.append(str(exc))
    return {"scenario_key": scenario_key, "passed": not errors,
            "errors": errors, "checks": checks}


def _retained_case(case: Mapping[str, Any], base: Path) -> dict[str, Any]:
    with np.load(base / case["arrays"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "evaluation" in case:
        with np.load(base / case["evaluation"], allow_pickle=False) as archive:
            if set(archive.files) != {"eval_tau"}:
                raise ValueError("evaluation archive must contain only eval_tau")
            if "eval_tau" in arrays:
                raise ValueError("eval_tau occurs in both observed and evaluation archives")
            arrays["eval_tau"] = archive["eval_tau"]
    if "prediction" in case:
        arrays["prediction"] = np.load(base / case["prediction"], allow_pickle=False)
    if "record" in case:
        record = json.loads((base / case["record"]).read_text())
        metrics = record.get("metrics", record)
        if not isinstance(metrics, dict):
            raise ValueError("retained record has no successful metric mapping")
        for metric in ("cate_mse", "root_pehe"):
            if metric in metrics:
                arrays[metric] = metrics[metric]
        if "mse" in metrics and "cate_mse" not in arrays:
            arrays["cate_mse"] = metrics["mse"]
    result = verify_case(arrays, case["scenario_key"])
    result["retained_case"] = dict(case)
    result["input_sha256"] = {key: hashlib.sha256((base / case[key]).read_bytes()).hexdigest()
                              for key in ("arrays", "evaluation", "prediction", "record") if key in case}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path,
                        help="JSON {cases:[{scenario_key,arrays,evaluation?,prediction?,record?}]}; paths relative to manifest")
    inputs.add_argument("--arrays", type=Path, help="retained NPZ containing eval_X and eval_tau")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--evaluation", type=Path, help="separate evaluator-only NPZ containing eval_tau")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("verification output must be new")
    if args.manifest:
        if args.scenario or args.evaluation or args.prediction or args.record:
            parser.error("--scenario/--evaluation/--prediction/--record apply only with --arrays")
        manifest = json.loads(args.manifest.read_text())
        cases = manifest["cases"]
        base = args.manifest.parent
    else:
        if args.scenario is None:
            parser.error("--arrays requires --scenario")
        cases = [{"scenario_key": args.scenario, "arrays": str(args.arrays.resolve())}]
        for key in ("evaluation", "prediction", "record"):
            if getattr(args, key) is not None:
                cases[0][key] = str(getattr(args, key).resolve())
        base = Path.cwd()
    if not isinstance(cases, list) or not cases:
        parser.error("manifest cases must be a nonempty list")
    rows = [_retained_case(case, base) for case in cases]
    result = {"schema_version": 1, "passed": all(row["passed"] for row in rows),
              "absolute_tolerance": ABSOLUTE_TOLERANCE,
              "score_absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
              "score_relative_tolerance": SCORE_RELATIVE_TOLERANCE,
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": len(rows)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
