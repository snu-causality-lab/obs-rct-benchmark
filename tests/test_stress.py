"""Stress study 1: assumption dials, identification, shards, analyzer and independent checks."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from obs_rct_benchmark.stress_dgp import StressDGPConfig, generate_stress_problem, mu0, obs_bias, tau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, V, M = _load("analyze_stress"), _load("verify_stress"), _load("make_stress_configs")


def cfg(dial, level, **kw):
    return StressDGPConfig(name="t", dial=dial, level=level, n_obs=kw.pop("n_obs", 2048 if dial != "obs_size" else
                                                                          (8192, 2048, 512, 128)[level]), **kw)


def test_trial_data_do_not_depend_on_bias_levels():
    for dial in ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection", "obs_size"):
        first = generate_stress_problem(cfg(dial, 0), 5)
        for level in (1, 2, 3):
            other = generate_stress_problem(cfg(dial, level), 5)
            assert np.array_equal(first.rct_train.Y, other.rct_train.Y)
            assert np.array_equal(first.eval_tau, other.eval_tau)


def test_bias_variance_is_fixed_across_levels():
    X = np.random.default_rng(0).normal(size=(400_000, 6))
    for dial, stat in (("hidden_modifier", lambda b0, b1: np.mean((b1 - b0) ** 2)),
                       ("bias_complexity", lambda b0, b1: np.mean((b1 - b0) ** 2)),
                       ("common_shift", lambda b0, b1: np.mean(b1 ** 2))):
        values = [stat(*obs_bias(X, cfg(dial, level))) for level in range(4)]
        assert max(values) / min(values) < 1.03, (dial, values)
    b0, b1 = obs_bias(X[:50], cfg("common_shift", 2))
    assert np.array_equal(b0, b1)
    b0, b1 = obs_bias(X[:50], cfg("obs_selection", 3))
    assert not b0.any() and not b1.any()


def test_obs_size_samples_are_nested():
    big = generate_stress_problem(cfg("obs_size", 0), 3).obs_train
    small = generate_stress_problem(cfg("obs_size", 3), 3).obs_train
    assert small.n == 128
    for a, b in ((small.X, big.X[:128]), (small.T, big.T[:128]), (small.Y, big.Y[:128])):
        assert np.array_equal(a, b)


def test_hidden_modifier_removes_x4_from_obs_outcomes_at_level_three():
    X = np.random.default_rng(1).normal(size=(20, 6))
    Z = X.copy()
    Z[:, 3] += 3.0
    c = cfg("hidden_modifier", 3)
    obs_mean = lambda x: mu0(x) + tau(x) + obs_bias(x, c)[1]
    assert np.allclose(obs_mean(X), obs_mean(Z), atol=1e-12)
    assert not np.allclose(tau(X), tau(Z))


def test_obs_size_level_fixes_the_obs_rows():
    with pytest.raises(ValueError, match="obs_size dial"):
        StressDGPConfig(name="t", dial="obs_size", level=0, n_obs=2048)


def test_trial_shrink_targets_the_obs_population():
    p = generate_stress_problem(cfg("trial_shrink", 3), 2)
    assert np.std(p.rct_train.X) < 0.45 and abs(np.std(p.eval_X) - 1) < 0.1


@pytest.fixture(scope="module")
def config():
    from obs_rct_benchmark.runner import load_config

    c = load_config(ROOT / "configs" / "v1_stress_cornet.toml")
    M.validate(c)
    return c


def test_shards_and_targets(config):
    assert len(list(M.shards(config))) == 64
    report = V.targets(config)
    assert report["cases"] == 64 and report["mismatches"] == []


@pytest.fixture(scope="module")
def study(config, tmp_path_factory):
    records, lines = {}, []
    for s in config.scenarios:
        dial, level, rct = A.parse(s.key)
        for seed in config.seeds:
            base = 1 + np.random.default_rng(abs(hash((s.key, seed))) % 2**32).normal(scale=0.05)
            for m in config.methods:
                noise = np.random.default_rng(abs(hash((s.key, seed, m))) % 2**32).normal(scale=0.05)
                v = base if m == "rct_selected" else base + noise + (0.5 * level - 0.6 if m == "cornet_raw" else 0.0)
                row = {"status": "ok", "metrics": {"cate_mse": float(v), "root_pehe": float(v) ** 0.5}}
                records[(s.key, seed, m)] = row
                lines.append(json.dumps({**row, "scenario": s.key, "seed": seed, "method": m}))
    run = tmp_path_factory.mktemp("stress") / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("\n".join(lines) + "\n")
    return records, [run]


def test_readout_and_independent_arithmetic(config, study, tmp_path):
    records, runs = study
    out = tmp_path / "analysis"
    result = A.analyze(config, records, out)
    assert result["families"]["cornet_raw"]["size"] == 64
    assert result["families"]["cornet_raw"]["extrapolation_worse"] == 8
    readout = list(csv.DictReader((out / "readout.csv").open()))
    raw = [r for r in readout if r["recipe"] == "cornet_raw"]
    assert len(raw) == 16 and all(r["level_0"] == "better" and r["level_3"] == "worse" for r in raw)
    assert {r["state"] for r in raw} == {"broke"}
    assert {r["state"] for r in readout if r["recipe"] == "cornet_official"} == {"no_level0_win"}
    assert V.arithmetic(out, runs)["mismatches"] == []
