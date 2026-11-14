"""Wave-2 recipes on stress study 1: shard writer, analyzer and independent checker."""
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


A = _load("analyze_stress_wave2")
V = _load("verify_stress_wave2")
M = _load("make_stress_wave2_configs")


@pytest.fixture(scope="module")
def configs():
    from obs_rct_benchmark.runner import load_config

    out = {}
    for group in M.GROUPS:
        config = load_config(ROOT / "configs" / f"v1_stress_wave2_{group}.toml")
        assert M.validate(config) == group
        out[group] = config
    return out


def test_declared_configs_are_the_written_files(configs, tmp_path):
    for group in M.GROUPS:
        M.write_toml(M.declared_payload(group), tmp_path / f"{group}.toml")
        assert (tmp_path / f"{group}.toml").read_bytes() == (ROOT / "configs" / f"v1_stress_wave2_{group}.toml").read_bytes()


def test_shards_cover_each_group_once_per_seed_and_dial(configs):
    for group, config in configs.items():
        main = list(M.shards(config))
        pre = list(M.shards(config, preflight=True))
        assert len(main) == 64 * 8 and len(pre) == 8
        assert all(s.seeds == (M.PREFLIGHT_SEED,) and len(s.scenarios) == 8 for _, s in pre)
        assert len({name for name, _ in main}) == len(main)
        cells = sum(len(s.scenarios) * len(s.methods) for _, s in main)
        assert cells == 64 * 64 * (2 + len(M.GROUPS[group]))


def test_families_match_the_independent_checker():
    fams = A.families()
    assert len(fams) == len(V.FAMILIES) == 20
    assert {f["family"] for f in fams} == {f[0] for f in V.FAMILIES}
    assert sum(f["kind"] == "primary" for f in fams) == 11
    assert not any(f["recipe"] in A.RCT_ONLY for f in fams if f["kind"] == "primary")


def _mse(group, scenario, seed, method):
    common = method in ("rct_selected", "rct_t_ridge")
    rng = np.random.default_rng(abs(hash(((None if common else group), scenario, seed, method))) % 2**32)
    base = 1.0 + np.random.default_rng(abs(hash((scenario, seed))) % 2**32).normal(scale=0.05)
    shift = {"causal_icm_fixed": -0.4, "tsr_tabpfn_v25": 0.4}.get(method, 0.0)
    return base + shift + (0.0 if method == "rct_selected" else rng.normal(scale=0.05))


@pytest.fixture(scope="module")
def study(configs, tmp_path_factory):
    records, base = {}, tmp_path_factory.mktemp("sw2runs")
    for group, config in configs.items():
        lines = []
        for s in config.scenarios:
            for seed in config.seeds:
                for m in config.methods:
                    if m == "integrative_r_quadratic" and s.id == "obs_size" and s.label == "l0_r512":
                        row = {"status": "failed", "metrics": None, "stage": "method_or_metric", "error_type": "TimeoutError"}
                    else:
                        row = {"status": "ok", "metrics": {"cate_mse": _mse(group, s.key, seed, m)}}
                    records[(group, s.key, seed, m)] = row
                    lines.append(json.dumps({**row, "scenario": s.key, "seed": seed, "method": m}))
        run = base / group
        run.mkdir()
        (run / "records.jsonl").write_text("\n".join(lines) + "\n")
        (run / "run_manifest.json").write_text(json.dumps({"config_hash": config.config_hash, "config": config.payload()}))
    stress1 = base / "stress1"
    (stress1 / "replicate_scores").mkdir(parents=True)
    config = next(iter(configs.values()))
    with (stress1 / "replicate_scores" / "all.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scenario", "seed", "method", "status", "cate_mse"])
        for s in config.scenarios:
            for seed in config.seeds:
                for m in ("rct_selected", "rct_t_ridge"):
                    writer.writerow([s.key, seed, m, "ok", repr(_mse(None, s.key, seed, m))])
    return records, [base / g for g in configs], stress1


def test_analyzer_and_independent_arithmetic(configs, study, tmp_path):
    records, runs, stress1 = study
    out = tmp_path / "analysis"
    result = A.analyze(configs, records, A.stress1_scores(stress1), out)
    fam = result["families"]
    assert all(f["size"] == 64 for f in fam.values())
    match = fam["gp/causal_icm_fixed vs causal_icm_rct_only"]
    assert match["other_better"] + match["extrapolation_better"] == 64 and match["extrapolation_better"] == 16
    assert fam["tabpfn/tsr_tabpfn_v25 vs tsr_tabpfn_v2"]["other_worse"] == 48
    assert fam["native_r/integrative_r_quadratic"]["other_untested"] == 1
    identity = json.loads((out / "identity_check.json").read_text())["stress1_reproduction"]
    assert all(v["within_1e-9"] for v in identity.values())
    readout = list(csv.DictReader((out / "readout.csv").open()))
    assert len(readout) == 11 * 8 * 2
    assert {r["state"] for r in readout if r["family"] == "gp/causal_icm_fixed"} == {"kept"}
    report = V.arithmetic(out, runs, stress1)
    assert report["mismatches"] == [] and report["readout_states"] == 176


def _rewrite_csv(path, mutate):
    rows = list(csv.DictReader(path.open()))
    mutate(rows)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flip(row, key, a, b):
    row[key] = a if row[key] != a else b


@pytest.mark.parametrize("target", ["verdict", "count", "identity", "headline", "readout", "untested_flag", "duplicate"])
def test_checker_catches_each_altered_output(configs, study, tmp_path, target):
    records, runs, stress1 = study
    out = tmp_path / "analysis"
    A.analyze(configs, records, A.stress1_scores(stress1), out)
    if target == "verdict":
        _rewrite_csv(out / "contrasts.csv", lambda rows: _flip(rows[5], "verdict", "better", "worse"))
    elif target == "count":
        data = json.loads((out / "analysis.json").read_text())
        data["families"]["gp/causal_icm_fixed"]["other_worse"] = 1
        (out / "analysis.json").write_text(json.dumps(data))
    elif target == "identity":
        data = json.loads((out / "identity_check.json").read_text())
        data["stress1_reproduction"]["gp:rct_t_ridge"]["exact"] -= 1
        (out / "identity_check.json").write_text(json.dumps(data))
    elif target == "headline":
        path = out / "headline.md"
        lines = path.read_text().splitlines()
        cells = lines[2].split("|")
        cells[4] = f" {int(cells[4]) + 1} "
        lines[2] = "|".join(cells)
        path.write_text("\n".join(lines) + "\n")
    elif target == "readout":
        _rewrite_csv(out / "readout.csv", lambda rows: _flip(rows[0], "state", "broke", "kept"))
    elif target == "untested_flag":
        _rewrite_csv(out / "readout.csv", lambda rows: _flip(rows[0], "untested_at", "0", ""))
    else:
        _rewrite_csv(out / "readout.csv", lambda rows: rows.append(dict(rows[0])))
    assert V.arithmetic(out, runs, stress1)["mismatches"]


def test_job_limit_cells_are_filled_and_checked(configs, tmp_path):
    from obs_rct_benchmark.runner import environment_provenance, method_seed, record_id, source_provenance

    W = _load("analyze_v1_wave2")
    VW = _load("verify_v1_wave2")
    config = configs["gp"]
    name, shard = next(iter(M.shards(config)))
    cells = [(s.key, seed, m) for s in shard.scenarios for seed in shard.seeds for m in shard.methods]
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "config_hash": shard.config_hash, "config": shard.payload(), "source": source_provenance(),
        "environment": environment_provenance(), "external_methods": None}))
    lines = [json.dumps({"scenario": sc, "seed": seed, "method": m, "status": "ok", "metrics": {"cate_mse": 1.0},
                         "config_hash": shard.config_hash, "method_seed": method_seed(sc, seed, m),
                         "record_id": record_id(shard, sc, seed, m)}) for sc, seed, m in cells[:5]]
    (run / "records.jsonl").write_text("\n".join(lines) + "\n" + lines[0][:20])  # killed mid-write
    one = lambda c: [(name, shard)]
    with pytest.raises(W.AnalysisError, match="unreadable record line"):
        W.collect({"gp": config}, [run], check_source=False, declared=one)
    records = W.collect({"gp": config}, [run], check_source=False, declared=one,
                        job_limit={name: shard.config_hash})
    assert len(records) == len(cells)
    filled = [k for k, r in records.items() if r.get("error_type") == "JobLimit"]
    assert len(filled) == len(cells) - 5 and all(records[k]["stage"] == "job_limit" for k in filled)
    got, unrecorded = VW.load_records([run], frozenset({shard.config_hash}))
    assert len(got) == 5 and len(unrecorded) == len(cells) - 5
    table = {("gp", *c): {"status": "ok", "cate_mse": "1.0", "stage": "", "error_type": ""} for c in cells[:5]}
    table.update({k: {"status": "failed", "cate_mse": "", "stage": "job_limit", "error_type": "JobLimit"} for k in unrecorded})
    assert VW.check_cells(table, got, unrecorded) == []
    table[next(iter(unrecorded))]["error_type"] = "TimeoutError"
    assert VW.check_cells(table, got, unrecorded)


def test_data_identity_dump_and_compare(configs, tmp_path):
    from dataclasses import replace

    D = _load("check_data_identity")
    config = replace(configs["gp"], scenarios=configs["gp"].scenarios[:2])
    D.dump(config, 80000, tmp_path / "a.npz")
    D.dump(config, 80000, tmp_path / "b.npz")
    assert D.compare(tmp_path / "a.npz", [tmp_path / "b.npz"])["all_same"]
    arrays = dict(np.load(tmp_path / "b.npz"))
    key = next(k for k in arrays if k.endswith("|obs_train|Y"))
    arrays[key] = arrays[key] + 1e-9
    np.savez(tmp_path / "c.npz", **arrays)
    report = D.compare(tmp_path / "a.npz", [tmp_path / "c.npz"])
    assert not report["all_same"] and report["environments"]["c.npz"]["roles"]["obs_train"]["max_abs_difference"] > 1e-12
