"""CATE metrics and seed-block summaries, kept outside estimator code.

CATE MSE = mean((estimate - truth)**2). Root-PEHE = sqrt(CATE MSE).
These are distinct from mean root-PEHE over replicates and from the square root
of mean replicate MSE. Pairing is within simulation seed, and an overall
comparison averages scenarios within a seed before computing uncertainty.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike

# Two-sided Student t 95% critical values. For df > 30, the next smaller
# tabulated df is used, conservatively; df > 120 continues to use df=120.
_T95 = {
    1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053,
    4: 2.7764451052, 5: 2.5705818356, 6: 2.4469118488,
    7: 2.3646242516, 8: 2.3060041352, 9: 2.2621571629,
    10: 2.2281388520, 11: 2.2009851601, 12: 2.1788128297,
    13: 2.1603686565, 14: 2.1447866879, 15: 2.1314495456,
    16: 2.1199052992, 17: 2.1098155778, 18: 2.1009220402,
    19: 2.0930240544, 20: 2.0859634473, 21: 2.0796138447,
    22: 2.0738730679, 23: 2.0686576104, 24: 2.0638985616,
    25: 2.0595385528, 26: 2.0555294386, 27: 2.0518305165,
    28: 2.0484071418, 29: 2.0452296421, 30: 2.0422724563,
    40: 2.0210753903, 60: 2.0002978211, 120: 1.9799304051,
}
METRIC_DEFINITIONS = {
    "cate_mse": "mean((predicted_cate - true_cate)**2) over evaluation rows",
    "root_pehe": "sqrt(cate_mse) for one replicate",
    "mean_root_pehe": "arithmetic mean of per-replicate root_pehe",
    "root_mean_mse": "sqrt(arithmetic mean of per-replicate cate_mse)",
}


def cate_metrics(prediction: ArrayLike, truth: ArrayLike) -> dict[str, float]:
    """Score one evaluation sample; reject malformed or nonfinite values."""
    estimate = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if estimate.ndim != 1 or target.ndim != 1 or estimate.shape != target.shape:
        raise ValueError("prediction and truth must be matching one-dimensional arrays")
    if target.size == 0 or not np.isfinite(estimate).all() or not np.isfinite(target).all():
        raise ValueError("prediction and truth must be nonempty and finite")
    with np.errstate(over="raise", invalid="raise"):
        mse = float(np.mean(np.square(estimate - target)))
    if not math.isfinite(mse):
        raise ValueError("CATE MSE is not finite")
    return {"cate_mse": mse, "root_pehe": math.sqrt(mse)}


def mean_ci(values: Iterable[float]) -> dict[str, Any]:
    """Mean and paired-seed Student t interval; one block has no interval.

    Coverage is approximate and assumes independent, identically distributed
    seed blocks. This is not an interval over evaluation rows or methods.
    """
    numbers = np.asarray(list(values), dtype=np.float64)
    if numbers.ndim != 1 or not np.isfinite(numbers).all():
        raise ValueError("summary values must be finite scalars")
    n = int(numbers.size)
    result: dict[str, Any] = {
        "n_seed_blocks": n, "mean": None, "standard_error": None,
        "ci95": None, "ci_method": "student_t_conservative_table",
    }
    if n == 0:
        return result
    result["mean"] = float(numbers.mean())
    if n < 2:
        return result
    standard_error = float(numbers.std(ddof=1) / math.sqrt(n))
    df = max(value for value in _T95 if value <= n - 1)
    half = _T95[df] * standard_error
    result.update(standard_error=standard_error,
                  ci95=[result["mean"] - half, result["mean"] + half])
    return result


def validate_inference_policy(policy: Mapping[str, Any], *, resolved: bool = False) -> None:
    """Validate declarations, not their statistical truth or empirical coverage."""
    if not isinstance(policy, Mapping) or set(policy) - {"cate_mse", "root_pehe"}:
        raise ValueError("inference policy must map known metrics to declarations")
    for declaration in policy.values():
        allowed = {"eligible", "reason", "source"} if resolved else {"eligible", "reason"}
        if (not isinstance(declaration, Mapping)
                or set(declaration) - allowed
                or not {"eligible", "reason"} <= declaration.keys()
                or not isinstance(declaration["eligible"], bool)
                or not isinstance(declaration["reason"], str)
                or not declaration["reason"].strip()
                or any(c in declaration["reason"] for c in "\n\r")):
            raise ValueError("inference declaration needs a boolean eligible and one-line reason")
        if "source" in declaration and (
                not isinstance(declaration["source"], str) or not declaration["source"].strip()):
            raise ValueError("inference policy source must be nonempty")


def summarize_metric(values: Iterable[float], *, policy: Mapping[str, Any] | None = None,
                     complete: bool = True) -> dict[str, Any]:
    """Keep descriptive arithmetic while applying a declared uncertainty policy.

    Permission is an assumption fixed before scoring, not a finite-moment proof.
    Missing declarations fail closed. The raw ``mean_ci`` helper remains policy-free.
    """
    if policy is not None:
        validate_inference_policy({"cate_mse": policy}, resolved=True)
    permission = policy or {"eligible": False, "reason": "inference policy not declared"}
    numbers = list(values)
    # Do not calculate unsupported sample uncertainty even as a discarded result.
    result = mean_ci(numbers if complete and permission["eligible"] else ())
    if not complete or not permission["eligible"]:
        array = np.asarray(numbers, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("summary values must be finite scalars")
        result.update(n_seed_blocks=len(numbers), mean=float(array.mean()) if numbers else None,
                      ci_method=None)
    reason = ("failed or missing requested cells" if not complete else
              permission["reason"] if not permission["eligible"] else
              "fewer than two seed blocks" if len(numbers) < 2 else None)
    if reason is not None:
        result.update(standard_error=None, ci95=None, ci_method=None)
    result.update(eligible_for_inference=reason is None, withheld_reason=reason)
    return result


def summarize_records(records: Iterable[Mapping[str, Any]], *,
                      scenarios: Iterable[str], seeds: Iterable[int],
                      methods: Iterable[str], reference_method: str,
                      scenario_inference: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize an explicitly requested Cartesian grid, failing closed.

    Missing or failed cells are counted, never converted to zero or discarded
    from inferential comparisons. Complete paired means remain descriptive
    when a metric's uncertainty permission is absent or withheld. Incomplete
    pairs retain no paired mean; successful-cell descriptions remain available.
    """
    scenario_ids, seed_ids, method_ids = tuple(scenarios), tuple(seeds), tuple(methods)
    if (not scenario_ids or not seed_ids or not method_ids
            or len(set(scenario_ids)) != len(scenario_ids)
            or len(set(seed_ids)) != len(seed_ids)
            or len(set(method_ids)) != len(method_ids)):
        raise ValueError("summary axes must be nonempty and unique")
    if reference_method not in method_ids:
        raise ValueError("reference_method must be requested")
    if scenario_inference is not None and not isinstance(scenario_inference, Mapping):
        raise ValueError("scenario inference must be a mapping")
    policies = {}
    for scenario in scenario_ids:
        declared = (scenario_inference or {}).get(scenario, {})
        validate_inference_policy(declared, resolved=True)
        policies[scenario] = {
            metric: dict(declared.get(metric, {
                "eligible": False, "reason": f"inference policy not declared for {metric}",
                "source": "unknown_default",
            })) for metric in ("cate_mse", "root_pehe")
        }
    expected = {(scenario, seed, method) for scenario in scenario_ids
                for seed in seed_ids for method in method_ids}
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record in records:
        key = (record["scenario"], record["seed"], record["method"])
        if key not in expected:
            raise ValueError("records contain an unrequested cell")
        if key in by_key:
            raise ValueError("duplicate scenario/seed/method record")
        _validate_record(record)
        by_key[key] = record
    failed = sum(record["status"] != "ok" for record in by_key.values())
    missing = len(expected - by_key.keys())
    success = len(by_key) - failed
    result: dict[str, Any] = {
        "status": "complete" if failed == missing == 0 else "incomplete",
        "coverage": {"expected": len(expected), "recorded": len(by_key),
                     "successful": success, "failed": failed, "missing": missing},
        "metric_definitions": METRIC_DEFINITIONS,
        "reference_method": reference_method,
        "scenario_inference": policies,
        "uncertainty_unit": "independent seed block; scenarios averaged within each seed",
        "methods": {}, "scenarios": {}, "paired_comparisons": {},
    }
    for method in method_ids:
        rows = [by_key[key] for key in sorted(expected) if key[2] == method and key in by_key]
        result["methods"][method] = _describe(rows, len(scenario_ids) * len(seed_ids))
    for scenario in scenario_ids:
        result["scenarios"][scenario] = {}
        for method in method_ids:
            rows = [by_key[(scenario, seed, method)] for seed in seed_ids
                    if (scenario, seed, method) in by_key]
            result["scenarios"][scenario][method] = _describe(rows, len(seed_ids))
    for method in method_ids:
        if method == reference_method:
            continue
        result["paired_comparisons"][method] = {
            "overall": _paired(by_key, scenario_ids, seed_ids, method, reference_method, policies),
            "by_scenario": {
                scenario: _paired(by_key, (scenario,), seed_ids, method, reference_method, policies)
                for scenario in scenario_ids
            },
        }
    return result


def _validate_record(record: Mapping[str, Any]) -> None:
    if record["status"] not in {"ok", "failed"}:
        raise ValueError("unknown record status")
    metrics = record.get("metrics")
    if record["status"] == "failed":
        if metrics is not None:
            raise ValueError("failed records must not contain metrics")
        return
    if not isinstance(metrics, Mapping):
        raise ValueError("successful record requires metrics")
    mse, root = metrics.get("cate_mse"), metrics.get("root_pehe")
    for value in (mse, root):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0):
            raise ValueError("successful metrics must be finite nonnegative numbers")
    if not math.isclose(root * root, mse, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("root_pehe must be the square root of cate_mse")


def _describe(rows: list[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "ok"]
    mse = [row["metrics"]["cate_mse"] for row in good]
    roots = [row["metrics"]["root_pehe"] for row in good]
    return {
        "expected": expected, "successful": len(good),
        "failed": len(rows) - len(good), "missing": expected - len(rows),
        "complete_coverage": len(good) == expected,
        "mean_scope": "successful cells only; descriptive when coverage is incomplete",
        "mean_cate_mse": float(np.mean(mse)) if mse else None,
        "mean_root_pehe": float(np.mean(roots)) if roots else None,
        "root_mean_mse": math.sqrt(float(np.mean(mse))) if mse else None,
    }


def _paired(by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
            scenarios: tuple[str, ...], seeds: tuple[int, ...],
            method: str, reference: str, policies: Mapping[str, Any]) -> dict[str, Any]:
    blocks: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        pairs = [(by_key.get((scenario, seed, method)),
                  by_key.get((scenario, seed, reference))) for scenario in scenarios]
        if any(a is None or b is None or a["status"] != "ok" or b["status"] != "ok"
               for a, b in pairs):
            continue
        for metric in ("cate_mse", "root_pehe"):
            blocks[metric].append(float(np.mean([
                a["metrics"][metric] - b["metrics"][metric] for a, b in pairs
            ])))
    n_complete = len(blocks["cate_mse"])
    complete = n_complete == len(seeds)
    metrics = {}
    for metric in ("cate_mse", "root_pehe"):
        restrictions = [f"{scenario}: {policies[scenario][metric]['reason']}"
                        for scenario in scenarios if not policies[scenario][metric]["eligible"]]
        permission = {"eligible": not restrictions,
                      "reason": "; ".join(restrictions) if restrictions else
                      "all contributing scenarios declare uncertainty permission"}
        metrics[metric] = summarize_metric(blocks[metric] if complete else (),
                                           policy=permission, complete=complete)
    return {
        "difference": "method minus reference; negative favors method",
        "expected_seed_blocks": len(seeds), "complete_seed_blocks": n_complete,
        "complete_coverage": complete,
        # Compatibility aliases describe CATE MSE only; inspect each metric.
        "eligible_for_comparison": metrics["cate_mse"]["eligible_for_inference"],
        "metrics": metrics,
        "withheld_reason": metrics["cate_mse"]["withheld_reason"],
    }
