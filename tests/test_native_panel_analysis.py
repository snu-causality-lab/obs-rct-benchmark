"""Independent small arithmetic checks for the prospective win endpoint."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "scripts" / "analyze_native_panel.py"
_spec = importlib.util.spec_from_file_location("native_panel_analysis", _path)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


def fixture():
    conditions = [{"scenario_key": f"condition{i}"} for i in range(6)]
    methods = list(panel.FUSION) + ["rct_causal_forest", "causal_icm_rct_only"]
    config = {"conditions": conditions, "rct_budgets": [128, 512],
              "seeds": list(range(50000, 50064)), "methods": methods}
    records = [dict(case=dict(scenario_key=c["scenario_key"], n_rct=b, seed=s),
                    method=m, status="success",
                    metrics=dict(cate_mse=1. if m in panel.FUSION else 2.,
                                 root_pehe=1. if m in panel.FUSION else math.sqrt(2.)))
               for c in conditions for b in config["rct_budgets"]
               for s in config["seeds"] for m in methods]
    return config, records


def test_exact_tail_and_whole_family_adjustment():
    config, records = fixture()
    result = panel.summarize(records, config)
    assert result["family_complete"]
    assert len(result["paired_comparisons"]) == 72
    row = result["paired_comparisons"][0]
    # Independent enumeration of the exact all-wins binomial tail.
    assert row["one_sided_pvalue"] == math.comb(64, 64) / 2**64
    assert row["holm_pvalue"] == 72 / 2**64
    assert row["exact_pointwise_95_interval"][0] == pytest.approx(.025**(1/64))
    assert row["exact_pointwise_95_interval"][1] == 1.
    assert row["simultaneous_95_interval"] == [.75, 1.]
    assert result["precision"]["union_failure_bound"] < .05
    assert row["paired_mean_differences"]["cate_mse"] == -1
    assert not row["mean_risk_inference_eligible"]


def test_ties_count_as_nonwins_without_harm_claim():
    config, records = fixture()
    for record in records:
        record["metrics"] = {"cate_mse": 1., "root_pehe": 1.}
    row = panel.summarize(records, config)["paired_comparisons"][0]
    assert (row["wins"], row["losses"], row["ties"]) == (0, 0, 64)
    assert row["one_sided_pvalue"] == 1.
    assert row["reject_majority_nonwin"] is False
    assert not any("harm" in key for key in row)


def test_one_failure_withholds_entire_family_but_retains_descriptions():
    config, records = fixture()
    bad = next(r for r in records if r["method"] == "cornet_official")
    bad["status"] = "failure"
    bad.pop("metrics")
    result = panel.summarize(records, config)
    assert not result["family_complete"]
    for row in result["paired_comparisons"]:
        assert row["holm_pvalue"] is None
        assert row["reject_majority_nonwin"] is None
        assert row["simultaneous_95_interval"] is None
    incomplete = next(r for r in result["paired_comparisons"] if not r["complete"])
    assert incomplete["paired_mean_differences"]["cate_mse"] is None
    assert incomplete["strict_win_probability"] is None
    assert any(r["conditional_on_success"] for r in result["method_summaries"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "nonfinite"])
def test_invalid_union_fails_closed(mutation):
    config, records = fixture()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "extra":
        records[0]["case"]["seed"] = 100
    else:
        records[0]["metrics"]["cate_mse"] = float("inf")
    with pytest.raises(ValueError):
        panel.summarize(records, config)
