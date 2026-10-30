"""Protect exact incomplete-union disclosure and published arithmetic checking."""
import copy
import importlib.util
import math
import os
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("delivery_check", ROOT/"scripts/verify_native_panel_delivery.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def tiny_rows():
    config = {"conditions":[{"scenario_key":"simple"}], "rct_budgets":[8],
              "seeds":[1,2], "methods":["a","b"]}
    rows = [dict(scenario_key="simple",n_rct="8",seed=str(s),method=m,status="success",
                 cate_mse="4",root_pehe="2",record_path=f"case{s}/{m}/record.json",
                 record_sha256="a"*64) for s in [1,2] for m in ["a","b"]]
    return config, rows


@pytest.mark.parametrize("mutation", ["duplicate","unrequested","missing_row","bad_score","bad_root","bad_status"])
def test_scalar_union_and_scores_fail_closed(mutation):
    config, rows = tiny_rows()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "unrequested":
        rows[0]["seed"] = "3"
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "bad_score":
        rows[0]["cate_mse"] = "nan"
    elif mutation == "bad_root":
        rows[0]["root_pehe"] = "3"
    else:
        rows[0]["status"] = "failure"
    with pytest.raises(ValueError):
        check.parse_records(rows,config,set())


def test_missing_attempt_has_no_invented_record_or_score():
    config, rows = tiny_rows()
    row = rows[0]
    row.update(status="missing",cate_mse="",root_pehe="",record_path="",record_sha256="")
    missing = {("simple",8,1,"a")}
    parsed = check.parse_records(rows,config,missing)
    assert parsed[("simple",8,1,"a")]["cate_mse"] is None
    row["record_sha256"] = "a"*64
    with pytest.raises(ValueError,match="invented missing record"):
        check.parse_records(rows,config,missing)


@pytest.fixture(scope="module")
def public_inputs():
    analysis = Path(os.environ.get("NATIVE_DELIVERY_ROOT",ROOT/"results/native_panel"))
    config_path = ROOT/"configs/native_panel.toml"
    config = tomllib.loads(config_path.read_text())
    rows = check.read_scalar_records(analysis)
    return analysis,config_path,config,check.parse_records(rows,config,check.EXPECTED_MISSING),check.read_json(analysis/"summary.json")


def test_real_public_delivery_checks_all_requested_cells(public_inputs):
    analysis,config_path,_,_,_ = public_inputs
    result = check.run(config_path,analysis)
    assert result["requested_fits"] == 8448
    assert result["status_counts"] == {"success":8350,"failure":64,"timeout":31,"missing":3}
    assert result["method_rows"] == 132 and result["paired_rows"] == 72
    assert result["all_joint_decisions_withheld"] and result["simultaneous_precision_withheld"]


def test_manifest_pin_rejects_rebound_delivery(public_inputs):
    analysis,config_path,_,_,_ = public_inputs
    with pytest.raises(ValueError,match="pinned delivery manifest differs"):
        check.run(config_path,analysis,"0"*64)


def test_export_cannot_write_inside_scientific_inputs(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    spec=importlib.util.spec_from_file_location("delivery_export",ROOT/"scripts/export_native_panel_delivery.py")
    exporter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    monkeypatch.setenv("SLURM_JOB_ID","fixture")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK","1")
    monkeypatch.setenv("SLURM_MEM_PER_NODE","4096")
    campaign=tmp_path/"campaign"
    with pytest.raises(ValueError,match="destination overlaps scientific inputs"):
        exporter.export(campaign,tmp_path/"verification",tmp_path/"summary/data.json",
                        tmp_path/"code",tmp_path/"configs/input.toml",campaign/"output")


def test_changed_scalar_is_detected_independently_of_file_hashes(public_inputs):
    _,_,config,indexed,summary = public_inputs
    changed = copy.deepcopy(indexed)
    row = next(r for r in changed.values() if r["status"] == "success")
    row["cate_mse"] += .1
    row["root_pehe"] = math.sqrt(row["cate_mse"])
    with pytest.raises(ValueError,match="method summary"):
        check.check_arithmetic(changed,config,summary)


@pytest.mark.parametrize("mutation", ["partial_mean","joint_decision","simultaneous_interval","conditional_label"])
def test_incomplete_collection_cannot_silently_gain_inference(public_inputs,mutation):
    _,_,config,indexed,original = public_inputs
    summary = copy.deepcopy(original)
    if mutation == "conditional_label":
        row = next(r for r in summary["method_summaries"] if r["conditional_on_success"])
        row["conditional_on_success"] = False
    else:
        row = next(r for r in summary["paired_comparisons"] if not r["complete"])
        if mutation == "partial_mean":
            row["paired_mean_differences"]["cate_mse"] = 0.
        elif mutation == "joint_decision":
            row["reject_majority_nonwin"] = False
        else:
            row["simultaneous_95_interval"] = [0.,1.]
    with pytest.raises(ValueError):
        check.check_arithmetic(indexed,config,summary)
