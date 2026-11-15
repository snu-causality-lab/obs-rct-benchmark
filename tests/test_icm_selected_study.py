"""Selected-rho Causal-ICM addendum: shard writer, analyzer and independent checker."""
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


M = _load("make_icm_selected_configs")
A = _load("analyze_icm_selected")
V = _load("verify_icm_selected")


@pytest.fixture(scope="module")
def configs():
    from obs_rct_benchmark.runner import load_config

    out = {}
    for block in M.BLOCKS:
        config = load_config(ROOT / "configs" / f"v1_icm_selected_{block}.toml")
        assert M.validate(config) == block
        out[block] = config
    return out


def test_declared_configs_are_written_by_the_shard_writer(tmp_path):
    from obs_rct_benchmark.runner import load_config

    for block in M.BLOCKS:
        path = tmp_path / f"{block}.toml"
        M.write_toml(M.declared_payload(block), path)
        assert load_config(path).config_hash == load_config(ROOT / "configs" / f"v1_icm_selected_{block}.toml").config_hash


def test_blocks_repeat_the_wave2_conditions_seeds_and_options(configs):
    from obs_rct_benchmark.runner import load_config

    for block, wave2 in (("core", "v1_wave2_gp"), ("stress", "v1_stress_wave2_gp")):
        old, new = load_config(ROOT / "configs" / f"{wave2}.toml"), configs[block]
        assert [s.key for s in old.scenarios] == [s.key for s in new.scenarios]
        assert [dict(s.parameters) for s in old.scenarios] == [dict(s.parameters) for s in new.scenarios]
        assert old.seeds == new.seeds and old.methods == new.methods[:4]
        assert {m: dict(o) for m, o in old.method_options.items()} == {
            m: dict(o) for m, o in new.method_options.items()}


def test_shards_cover_each_block_once(configs):
    core, stress = configs["core"], configs["stress"]
    assert len(list(M.shards(core))) == 1984 and len(list(M.shards(core, preflight=True))) == 31
    assert len(list(M.shards(stress))) == 1024 and len(list(M.shards(stress, preflight=True))) == 16
    for config in configs.values():
        cells = [(s.key, seed) for _, shard in M.shards(config) for s in shard.scenarios for seed in shard.seeds]
        assert len(cells) == len(set(cells)) == len(config.scenarios) * len(config.seeds)


def test_selected_takes_no_options_and_changed_recipes_are_refused(configs):
    from dataclasses import replace

    assert "causal_icm_selected" not in configs["core"].method_options
    with pytest.raises(ValueError):
        M.validate(replace(configs["core"], seeds=configs["core"].seeds[:-1]))


def _mse(scenario, seed, method):
    preset, label = scenario.split("@")
    rct = label.split("_")[1]
    cell = (preset, rct) if method in A.RCT_ONLY else (scenario,)
    rng = np.random.default_rng(abs(hash((*cell, seed, method))) % 2**32)
    base = 1.0 + np.random.default_rng(abs(hash((preset, rct, seed))) % 2**32).normal(scale=0.05)
    shift = {"causal_icm_fixed": -0.2, "causal_icm_selected": -0.4}.get(method, 0.0)
    return base + shift + (0.0 if method == "rct_selected" else rng.normal(scale=0.05))


def _study(config, base):
    from obs_rct_benchmark.runner import method_seed

    records, wave2, lines = {}, [], []
    for s in config.scenarios:
        for seed in config.seeds:
            for m in config.methods:
                ms = method_seed(s.key, seed, m)
                v = _mse(s.key, seed, m)
                row = {"status": "ok", "metrics": {"cate_mse": v, "root_pehe": v ** 0.5}, "method_seed": ms}
                if m == "causal_icm_selected":
                    rho = 0.5 if seed % 4 == 0 else 0.9
                    if rho == 0.5:
                        v = records[("gp", s.key, seed, "causal_icm_fixed")]["metrics"]["cate_mse"]
                        row["metrics"] = {"cate_mse": v, "root_pehe": v ** 0.5}
                    row["fit_info"] = {"selector": A.SELECTOR, "status": "completed", "selected_rho": rho,
                                       "fold_seed": ms, "folds_used": 5,
                                       "participation_weights": {"converged": True}}
                records[("gp", s.key, seed, m)] = row
                lines.append(json.dumps({**row, "scenario": s.key, "seed": seed, "method": m}))
                if m in A.WAVE2:
                    wave2.append({"group": "gp", "scenario": s.key, "seed": seed, "method": m, "status": "ok",
                                  "cate_mse": repr(v)})
    run = base / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("\n".join(lines) + "\n")
    (run / "run_manifest.json").write_text(json.dumps({"config_hash": config.config_hash, "config": config.payload()}))
    packet = base / "wave2" / "replicate_scores"
    packet.mkdir(parents=True)
    with (packet / "gp__all.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(wave2[0]))
        writer.writeheader()
        writer.writerows(wave2)
    return records, [run], base / "wave2"


@pytest.mark.parametrize("block,size,split", [("core", 120, "transport"), ("stress", 64, "extrapolation")])
def test_analyzer_families_identities_and_independent_arithmetic(configs, tmp_path, block, size, split):
    records, runs, wave2 = _study(configs[block], tmp_path)
    out = tmp_path / "analysis"
    result = A.analyze(block, configs[block], records, A.wave2_scores(wave2), out)
    assert [f["family"] for f in A.families()] == [f[0] for f in V.FAMILIES]
    for fam in result["families"].values():
        assert fam["size"] == size
    sel = result["families"]["gp/causal_icm_selected vs causal_icm_fixed"]
    assert sel[f"other_better"] + sel[f"{split}_better"] == size
    identity = json.loads((out / "identity_check.json").read_text())
    assert all(r["within_1e-9"] and r["exact"] == r["expected"] for r in identity["wave2_reproduction"].values())
    assert identity["selected_at_0.5_vs_fixed"] == {"cells": 16 * len(configs[block].scenarios), "differ": 0}
    assert result["selected_rho"]["0.5"] == 16 * len(configs[block].scenarios)
    report = V.arithmetic(block, out, runs, wave2)
    assert report["mismatches"] == []
    assert report["readout_states"] == (16 if block == "stress" else 0)


def test_checker_catches_a_changed_verdict_and_rho_count(configs, tmp_path):
    records, runs, wave2 = _study(configs["stress"], tmp_path)
    out = tmp_path / "analysis"
    A.analyze("stress", configs["stress"], records, A.wave2_scores(wave2), out)
    for name, column, value in (("contrasts.csv", "verdict", "worse"), ("rho_selection.csv", "rho_0.9", "0")):
        path = out / name
        original = path.read_text()
        rows = list(csv.DictReader(path.open()))
        rows[3][column] = value
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert V.arithmetic("stress", out, runs, wave2)["mismatches"]
        path.write_text(original)
    assert V.arithmetic("stress", out, runs, wave2)["mismatches"] == []


def test_analyzer_refuses_a_selected_fit_without_a_completed_selection(configs, tmp_path):
    records, _, wave2 = _study(configs["stress"], tmp_path)
    key = next(k for k in records if k[3] == "causal_icm_selected")
    records[key] = {**records[key], "fit_info": {**records[key]["fit_info"], "status": "final_fit"}}
    with pytest.raises(A.AnalysisError):
        A.analyze("stress", configs["stress"], records, A.wave2_scores(wave2), tmp_path / "analysis")


def test_reproduction_flags_a_changed_wave2_score(configs, tmp_path):
    records, runs, wave2 = _study(configs["core"], tmp_path)
    scores = A.wave2_scores(wave2)
    key = next(k for k in scores if k[2] == "causal_icm_fixed")
    scores[key] += 1e-6
    A.analyze("core", configs["core"], records, scores, tmp_path / "analysis")
    identity = json.loads((tmp_path / "analysis" / "identity_check.json").read_text())
    assert not identity["wave2_reproduction"]["causal_icm_fixed"]["within_1e-9"]
    assert identity["wave2_reproduction"]["rct_selected"]["within_1e-9"]
    # The checker reads the published packet, which still agrees with the records.
    assert V.arithmetic("core", tmp_path / "analysis", runs, wave2)["mismatches"]
