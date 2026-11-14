"""Wave-2 addendum: TabPFN release recipes, shard writer, analyzer and independent checker."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from obs_rct_benchmark import methods

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load("analyze_v1_wave2")
V = _load("verify_v1_wave2")
M = _load("make_v1_wave2_configs")


@pytest.mark.parametrize("name,component,release", [
    ("tsr_tabpfn_v2", "tsr", "v2"), ("tsr_tabpfn_v25", "tsr", "v2.5"), ("tsr_tabpfn_v3", "tsr", "v3"),
    ("tabpfn_v2_obs_only", "obs", "v2"), ("tabpfn_v25_obs_only", "obs", "v2.5"), ("tabpfn_v3_obs_only", "obs", "v3"),
])
def test_release_recipes_fix_component_and_release(name, component, release):
    estimator = methods.make_method(name, device="cpu", model_sha256="a" * 64)
    assert (estimator.component, estimator.tabpfn_release) == (component, release)
    for key in ("component", "tabpfn_release"):
        with pytest.raises(ValueError, match="fixed by the method ID"):
            methods.make_method(name, **{key: "tsr"})


def test_release_regressor_requires_a_pinned_checkpoint(monkeypatch, tmp_path):
    from obs_rct_benchmark import tsr

    with pytest.raises(methods.MethodUnavailableError, match="require model_sha256"):
        tsr._release_regressor("v2.5", "cpu", None)
    with pytest.raises(ValueError, match="unknown TabPFN release"):
        tsr.TwoStageResidual("tsr", tabpfn_release="v9")


def test_registry_has_twenty_seven_recipes():
    assert len(set(methods.method_names()) - set(methods.custom_provenance())) == 27


@pytest.fixture(scope="module")
def configs():
    from obs_rct_benchmark.runner import load_config

    out = {}
    for group in M.GROUPS:
        config = load_config(ROOT / "configs" / f"v1_wave2_{group}.toml")
        assert M.validate(config) == group
        out[group] = config
    return out


def test_shards_cover_each_group_once_per_seed_and_preset(configs):
    for group, config in configs.items():
        main = list(M.shards(config))
        pre = list(M.shards(config, preflight=True))
        assert len(main) == 1984 and len(pre) == 31
        assert all(s.seeds == (M.PREFLIGHT_SEED,) for _, s in pre)
        cells = sum(len(s.scenarios) * len(s.methods) for _, s in main)
        assert cells == 124 * 64 * (2 + len(M.GROUPS[group]))


def _mse(group, scenario, seed, method):
    preset, label = scenario.split("@")
    rct = label.split("_")[1]
    cell = (preset, rct) if method in A.RCT_ONLY else (scenario,)
    rng = np.random.default_rng(abs(hash((group, *cell, seed, method))) % 2**32)
    base = 1.0 + np.random.default_rng(abs(hash((preset, rct, seed))) % 2**32).normal(scale=0.05)
    shift = {"causal_icm_fixed": -0.4, "tsr_tabpfn_v25": 0.4}.get(method, 0.0)
    return base + shift + (0.0 if method == "rct_selected" else rng.normal(scale=0.05))


@pytest.fixture(scope="module")
def study(configs, tmp_path_factory):
    records, base = {}, tmp_path_factory.mktemp("w2runs")
    for group, config in configs.items():
        lines = []
        for s in config.scenarios:
            for seed in config.seeds:
                for m in config.methods:
                    if m == "integrative_r_quadratic" and s.id == "cornet_official_n_conf" and s.label.endswith("r512"):
                        row = {"status": "failed", "metrics": None, "stage": "method_or_metric", "error_type": "TimeoutError"}
                    else:
                        v = _mse(group, s.key, seed, m)
                        row = {"status": "ok", "metrics": {"cate_mse": v, "root_pehe": v ** 0.5}}
                    records[(group, s.key, seed, m)] = row
                    lines.append(json.dumps({**row, "scenario": s.key, "seed": seed, "method": m}))
        run = base / group
        run.mkdir()
        (run / "records.jsonl").write_text("\n".join(lines) + "\n")
        (run / "run_manifest.json").write_text(json.dumps({"config_hash": config.config_hash, "config": config.payload()}))
    return records, [base / g for g in configs]


def test_analyzer_families_and_independent_arithmetic(configs, study, tmp_path):
    records, runs = study
    out = tmp_path / "analysis"
    result = A.analyze(configs, records, {}, out)
    fam = result["families"]
    assert len(fam) == len(A.families()) == len(V.FAMILIES) == 24
    assert {f["family"] for f in A.families()} == {f[0] for f in V.FAMILIES}
    assert fam["native_r/rct_causal_forest"]["size"] == 60 and fam["native_r/integrative_r_linear"]["size"] == 120
    assert fam["tsr_legacy/tsr_tabpfn_lmcgp vs lmcgp_rct_only"]["size"] == 120
    match = fam["gp/causal_icm_fixed vs causal_icm_rct_only"]
    assert match["size"] == 120 and match["other_better"] + match["transport_better"] == 120
    worse = fam["tabpfn/tsr_tabpfn_v25 vs tsr_tabpfn_v2"]
    assert worse["other_worse"] + worse["transport_worse"] == 120
    assert fam["native_r/integrative_r_quadratic"]["other_untested"] == 2
    report = V.arithmetic(out, runs)
    assert report["mismatches"] == []


def test_checker_catches_a_changed_verdict(configs, study, tmp_path):
    records, runs = study
    out = tmp_path / "analysis"
    A.analyze(configs, records, {}, out)
    path = out / "contrasts.csv"
    rows = list(csv.DictReader(path.open()))
    rows[3]["verdict"] = "better" if rows[3]["verdict"] != "better" else "worse"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert V.arithmetic(out, runs)["mismatches"]
