"""Synthetic-record tests for the v1 core analyzer and its independent checker."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load("analyze_v1_core")
V = _load("verify_v1_core")
M = _load("make_v1_core_configs")
FAIL_SEED = 70014
ELASTIC = "paper:elastic_integrative_gen_data_overlap"


def _mse(scenario: str, seed: int, method: str) -> float:
    """Deterministic synthetic scores with known verdicts."""
    preset, label = scenario.split("@")
    rct = label.split("_")[1]
    noise = np.random.default_rng(abs(hash((preset, rct, seed))) % 2**32).normal(scale=0.05)
    ref = 1.0 + noise
    # RCT-only fits do not depend on the OBS size, as in the real grid.
    cell = (preset, rct) if method in A.RCT_ONLY else (scenario,)
    own = np.random.default_rng(abs(hash((*cell, seed, method))) % 2**32).normal(scale=0.05)
    return {
        "rct_selected": ref,
        "rct_t_ridge": ref,                  # identical: zero variance
        "cornet_official": ref - 0.5 + own,  # better everywhere
        "obs_t_ridge": ref + 0.5 + own,      # worse everywhere
    }.get(method, ref + own)                 # otherwise unresolved


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Synthetic records written as fake shard runs, the way the runner writes them."""
    from obs_rct_benchmark.runner import load_config, method_seed, record_id

    config = load_config(ROOT / "configs" / "v1_core_study.toml")
    M.validate(config)
    records, runs = {}, []
    base = tmp_path_factory.mktemp("runs")
    for name, shard in M.shards(config):
        lines = []
        for s in shard.scenarios:
            for seed in shard.seeds:
                for m in shard.methods:
                    key = (s.key, seed, m)
                    row = {"record_id": record_id(shard, *key), "config_hash": shard.config_hash,
                           "scenario": s.key, "seed": seed, "method": m, "method_seed": method_seed(*key)}
                    if m == "datafusionppi" and s.key.startswith(ELASTIC + "@o512") and seed == FAIL_SEED:
                        row.update(status="failed", metrics=None, stage="method_or_metric", error_type="RuntimeError")
                    else:
                        value = _mse(*key)
                        row.update(status="ok", metrics={"cate_mse": value, "root_pehe": value ** 0.5})
                        if m == "rct_selected":
                            row["fit_info"] = {"selected": "rct_t_ridge", "max_weight": 4.0}
                    records[key] = row
                    lines.append(json.dumps(row))
        run = base / name.removesuffix(".toml")
        run.mkdir()
        (run / "records.jsonl").write_text("\n".join(lines) + "\n")
        (run / "run_manifest.json").write_text(json.dumps({
            "config": shard.payload(), "config_hash": shard.config_hash,
            "source": {"fake": 1}, "environment": {"fake": 1},
            "external_methods": {m: "fake" for m in shard.methods if m.startswith(("cornet", "datafusion"))}}))
        runs.append(run)
    return config, records, runs


def test_collect_accepts_both_shard_groups_and_rejects_bad_sets(study):
    config, records, runs = study
    collected, requested = A.collect(config, runs, check_source=False)
    assert collected == records and len(requested) == 128
    with pytest.raises(A.AnalysisError, match="missing shards"):
        A.collect(config, runs[:-1], check_source=False)
    with pytest.raises(A.AnalysisError, match="given twice"):
        A.collect(config, runs + runs[:1], check_source=False)


def test_verdicts_follow_the_declared_rule(study, tmp_path):
    config, records, runs = study
    out = tmp_path / "analysis"
    result = A.analyze(config, records, out)
    f = result["families"]
    assert f["cornet_official"]["size"] == 120 and f["datafusionppi"]["size"] == 116
    assert f["rct_t_extra_trees"]["size"] == 60
    assert f["cornet_official"]["other_better"] == 96 and f["cornet_official"]["transport_better"] == 24
    assert f["obs_t_ridge"]["other_worse"] == 96 and f["obs_t_ridge"]["transport_worse"] == 24
    assert f["rct_t_ridge"]["other_unresolved"] == 48 and f["rct_t_ridge"]["transport_unresolved"] == 12
    assert f["datafusionppi"]["transport_untested"] == 2
    assert result["identity_passed"] and result["ridge_refit_mismatches"] == 0
    rows = list(csv.DictReader((out / "primary_contrasts.csv").open()))
    ridge = [r for r in rows if r["family"] == "rct_t_ridge"]
    assert all(r["zero_variance"] == "True" and float(r["p_two_sided"]) == 1.0 for r in ridge)
    untested = [r for r in rows if r["verdict"] == "untested"]
    assert {(r["family"], r["recipe_failures"], r["pairs"], r["wins_denominator"]) for r in untested} == {
        ("datafusionppi", "1", "63", "64")}
    assert not any(r["scenario"].startswith("structural:M11@") for r in rows)
    selection = list(csv.DictReader((out / "selection.csv").open()))
    assert all(r["chosen_rct_t_ridge"] == "64" for r in selection)


def test_independent_arithmetic_agrees(study, tmp_path):
    config, records, runs = study
    out = tmp_path / "analysis"
    A.analyze(config, records, out)
    report = V.arithmetic(out, config, runs)
    assert report["mismatches"] == []
    assert report["families"]["datafusionppi"]["size"] == 116


def test_independent_arithmetic_catches_a_changed_verdict(study, tmp_path):
    config, records, runs = study
    out = tmp_path / "analysis"
    A.analyze(config, records, out)
    path = out / "primary_contrasts.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["verdict"] = "better" if rows[0]["verdict"] != "better" else "worse"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert V.arithmetic(out, config, runs)["mismatches"]


def test_rct_only_identity_failure_is_reported(study, tmp_path):
    config, records, runs = study
    changed = dict(records)
    key = ("structural:M0@o2048_r128", 70000, "rct_t_extra_trees")
    changed[key] = {"status": "ok", "metrics": {"cate_mse": 9.0, "root_pehe": 3.0}}
    result = A.analyze(config, changed, tmp_path / "analysis")
    assert not result["identity_passed"]
    check = json.loads((tmp_path / "analysis" / "identity_check.json").read_text())
    assert check["recipes"]["rct_t_extra_trees"]["mismatched"] == 1


def test_independent_arithmetic_catches_a_score_that_differs_from_its_record(study, tmp_path):
    config, records, runs = study
    out = tmp_path / "analysis"
    A.analyze(config, records, out)
    path = out / "replicate_scores.csv"
    rows = list(csv.DictReader(path.open()))
    rows[5]["cate_mse"] = str(float(rows[5]["cate_mse"]) + 1e-9)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reasons = [m.get("reason") for m in V.arithmetic(out, config, runs)["mismatches"]]
    assert "replicate score differs from its record" in reasons


def test_obs2048_contrasts_use_the_obs512_reference(study, tmp_path):
    config, records, runs = study
    changed = dict(records)
    for seed in config.seeds:  # spoil only the OBS 2048 reference copy
        key = ("structural:M0@o2048_r128", seed, "rct_selected")
        changed[key] = {**records[key], "metrics": {"cate_mse": 50.0, "root_pehe": 50.0 ** 0.5}}
    A.analyze(config, changed, tmp_path / "analysis")
    rows = {(r["family"], r["scenario"]): r for r in csv.DictReader((tmp_path / "analysis" / "primary_contrasts.csv").open())}
    assert rows[("cornet_official", "structural:M0@o2048_r128")]["verdict"] == "better"
    assert float(rows[("cornet_official", "structural:M0@o2048_r128")]["mean_difference"]) > -1


def test_ridge_refit_mismatch_is_reported(study, tmp_path):
    config, records, runs = study
    changed = dict(records)
    key = ("structural:M0@o512_r128", 70001, "rct_t_ridge")
    value = records[key]["metrics"]["cate_mse"] + 1e-15
    changed[key] = {**records[key], "metrics": {"cate_mse": value, "root_pehe": value ** 0.5}}
    result = A.analyze(config, changed, tmp_path / "analysis")
    assert result["ridge_refit_mismatches"] == 1
