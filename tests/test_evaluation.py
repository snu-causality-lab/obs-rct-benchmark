"""Independent metric examples and failure-sensitive paired comparisons."""

import math

import numpy as np
import pytest

from obs_rct_benchmark.evaluation import cate_metrics, mean_ci, summarize_records


def row(scenario, seed, method, mse=None):
    return {"scenario": scenario, "seed": seed, "method": method,
            "status": "failed" if mse is None else "ok",
            "metrics": None if mse is None else {"cate_mse": mse, "root_pehe": math.sqrt(mse)}}


def test_metrics_match_hand_computed_errors_and_reject_column_vectors():
    # Errors 0, 2, -2 give squared error 8/3, not RMSE 8/3.
    metrics = cate_metrics([1, 4, 1], [1, 2, 3])
    assert metrics["cate_mse"] == pytest.approx(8 / 3)
    assert metrics["root_pehe"] == pytest.approx(math.sqrt(8 / 3))
    for prediction in ([np.nan, 0, 0], [[1], [4], [1]], [1, 2]):
        with pytest.raises(ValueError):
            cate_metrics(prediction, [1, 2, 3])


def test_mean_root_error_differs_from_root_mean_squared_error():
    summary = summarize_records([row("s", 0, "r", 1), row("s", 1, "r", 9)],
                                scenarios=["s"], seeds=[0, 1], methods=["r"],
                                reference_method="r")
    description = summary["methods"]["r"]
    assert description["mean_root_pehe"] == 2
    assert description["root_mean_mse"] == pytest.approx(math.sqrt(5))


def test_seed_blocks_prevent_pseudoreplication_across_scenarios():
    records = []
    # Overall seed-level MSE differences are (1+3)/2=2 and (3+5)/2=4.
    # n is 2, not 4, even though there are four scenario/seed pairs.
    for scenario, differences in (("a", (1, 3)), ("b", (3, 5))):
        for seed, difference in enumerate(differences):
            records.extend([row(scenario, seed, "r", 1),
                            row(scenario, seed, "m", 1 + difference)])
    result = summarize_records(records, scenarios=["a", "b"], seeds=[0, 1],
                               methods=["r", "m"], reference_method="r",
                               scenario_inference={s: {"cate_mse": {
                                   "eligible": True, "reason": "finite fixture variance assumed",
                               }} for s in ("a", "b")})
    estimate = result["paired_comparisons"]["m"]["overall"]["metrics"]["cate_mse"]
    assert estimate["n_seed_blocks"] == 2
    assert estimate["mean"] == 3
    assert estimate["standard_error"] == 1
    assert estimate["ci95"] == pytest.approx([3 - 12.7062047364, 3 + 12.7062047364])


def test_failing_or_missing_method_cannot_win_by_successful_subset():
    records = [row("s", 0, "r", 100), row("s", 1, "r", 100),
               row("s", 0, "m", 0), row("s", 1, "m")]
    for rows in (records, records[:-1]):
        result = summarize_records(rows, scenarios=["s"], seeds=[0, 1],
                                   methods=["r", "m"], reference_method="r")
        assert result["status"] == "incomplete"
        comparison = result["paired_comparisons"]["m"]["overall"]
        assert comparison["complete_seed_blocks"] == 1
        assert not comparison["eligible_for_comparison"]
        assert comparison["metrics"]["cate_mse"]["mean"] is None
        assert comparison["metrics"]["cate_mse"]["ci95"] is None


def test_no_interval_for_one_seed_and_duplicate_records_rejected():
    assert mean_ci([2])["ci95"] is None
    records = [row("s", 0, "m", 2)]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_records(records * 2, scenarios=["s"], seeds=[0], methods=["m"],
                          reference_method="m")
    records[0]["metrics"]["root_pehe"] = 2
    with pytest.raises(ValueError, match="square root"):
        summarize_records(records, scenarios=["s"], seeds=[0], methods=["m"],
                          reference_method="m")
