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
        (run / "run_manifest.json").write_text(json.dumps({"config": {"run": {"methods": list(config.methods)}}}))
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


def test_checker_catches_a_changed_readout_state(configs, study, tmp_path):
    records, runs, stress1 = study
    out = tmp_path / "analysis"
    A.analyze(configs, records, A.stress1_scores(stress1), out)
    path = out / "readout.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["state"] = "broke" if rows[0]["state"] != "broke" else "kept"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert V.arithmetic(out, runs, stress1)["mismatches"]
