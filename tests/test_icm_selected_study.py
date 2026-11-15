"""Selected-rho Causal-ICM addendum: shard writer, collector, analyzer and independent checker."""
from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
import math
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


def test_selected_takes_no_options_and_changed_configurations_are_refused(configs):
    assert "causal_icm_selected" not in configs["core"].method_options
    with pytest.raises(ValueError):
        M.validate(replace(configs["core"], seeds=configs["core"].seeds[:-1]))


# ---------------------------------------------------------------- synthetic studies

def _base(scenario, seed):
    preset, label = scenario.split("@")
    rct = label.split("_")[1]
    return 1.0 + np.random.default_rng(abs(hash((preset, rct, seed))) % 2**32).normal(scale=0.05)


def _mse(scenario, seed, method):
    preset, label = scenario.split("@")
    rct = label.split("_")[1]
    cell = (preset, rct) if method in A.RCT_ONLY else (scenario,)
    noise = np.random.default_rng(abs(hash((*cell, seed, method))) % 2**32).normal(scale=0.05)
    shift = {"causal_icm_fixed": -0.2, "causal_icm_selected": -0.4}.get(method, 0.0)
    return _base(scenario, seed) + shift + (0.0 if method == "rct_selected" else noise)


def _ok(v, **extra):
    return {"status": "ok", "metrics": {"cate_mse": v, "root_pehe": math.sqrt(v)}, **extra}


def _failed(**extra):
    return {"status": "failed", "metrics": None, "stage": "method_or_metric", "error_type": "LinAlgError", **extra}


def _info(ms, rho, status="completed", converged=True):
    return {"selector": A.SELECTOR, "status": status, "selected_rho": rho, "fold_seed": ms, "folds_used": 5,
            "participation_weights": {"converged": converged}}


def _records(config, special=None):
    """Synthetic records; `special(key, row, records)` may replace a row."""
    from obs_rct_benchmark.runner import method_seed

    records = {}
    for s in config.scenarios:
        for seed in config.seeds:
            for m in config.methods:
                ms = method_seed(s.key, seed, m)
                key = ("gp", s.key, seed, m)
                if m == "causal_icm_selected":
                    rho = 0.5 if seed % 4 == 0 else 0.9
                    fixed = records[("gp", s.key, seed, "causal_icm_fixed")]
                    v = fixed["metrics"]["cate_mse"] if rho == 0.5 and fixed["status"] == "ok" else _mse(s.key, seed, m)
                    row = _ok(v, fit_info=_info(ms, rho))
                else:
                    row = _ok(_mse(s.key, seed, m))
                row["method_seed"] = ms
                if special:
                    row = special(key, row, records) or row
                records[key] = row
    return records


def _write(config, records, base, *, drop=(), truncate=None):
    """One run directory per declared shard, as the shard driver leaves them."""
    from obs_rct_benchmark.runner import record_id

    runs = []
    for name, shard in M.shards(config):
        run = base / "runs" / Path(name).stem
        run.mkdir(parents=True)
        lines = []
        for s in shard.scenarios:
            for seed in shard.seeds:
                for m in shard.methods:
                    key = ("gp", s.key, seed, m)
                    if key in drop:
                        continue
                    lines.append(json.dumps({**records[key], "scenario": s.key, "seed": seed, "method": m,
                                             "config_hash": shard.config_hash,
                                             "record_id": record_id(shard, s.key, seed, m)}))
        text = "\n".join(lines) + "\n"
        if truncate == name:
            text += '{"scenario": "trunc'
        (run / "records.jsonl").write_text(text)
        (run / "run_manifest.json").write_text(json.dumps({
            "config_hash": shard.config_hash, "config": shard.payload(), "source": {"package": "test"},
            "environment": {"python": "test"}, "external_methods": {"causal_icm": "test"}}))
        runs.append(run)
    return runs


def _wave2(records, base):
    packet = base / "wave2" / "replicate_scores"
    packet.mkdir(parents=True)
    rows = [{"group": "gp", "scenario": k[1], "seed": k[2], "method": k[3], "status": r["status"],
             "cate_mse": repr(r["metrics"]["cate_mse"]) if r["status"] == "ok" else ""}
            for k, r in records.items() if k[3] in A.WAVE2]
    with (packet / "gp__all.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return base / "wave2"


def _run(block, config, records, base, **write):
    job_limit = write.pop("job_limit", None)
    runs = _write(config, records, base, **write)
    wave2 = _wave2(records, base)
    collected = A.collect(config, runs, check_source=False, job_limit=job_limit)
    out = base / "analysis"
    result = A.analyze(block, config, collected, A.wave2_scores(wave2), out)
    limited = frozenset((job_limit or {}).values())
    return result, out, runs, wave2, collected, limited


# ---------------------------------------------------------------- stress block with edge cases

LEVEL1 = ("gp", "stress:common_shift@l1_r128", 80003, "causal_icm_selected")
HALF_FIXED_FAILED = ("gp", "stress:noise_shape@l2_r512", 80004, "causal_icm_fixed")
ZERO = ("gp", "stress:obs_size@l1_r128", 80010, "causal_icm_selected")


def _stress_special(key, row, records):
    _, sc, seed, m = key
    if key == LEVEL1:
        return _failed(method_seed=row["method_seed"],
                       fit_info={**_info(row["method_seed"], None, status="selecting", converged=False)})
    if key == HALF_FIXED_FAILED:
        return _failed(method_seed=row["method_seed"])
    if (*key[:3], "causal_icm_fixed") == HALF_FIXED_FAILED and m == "causal_icm_selected":
        return _ok(0.77, method_seed=row["method_seed"], fit_info=_info(row["method_seed"], 0.5))
    if key == ZERO:
        return _ok(0.0, method_seed=row["method_seed"], fit_info=_info(row["method_seed"], 0.9))
    if sc == "stress:target_shift@l2_r512" and m in ("rct_selected", "causal_icm_selected"):
        # exactly identical differences against rct_selected: zero variance, p = 1
        if m == "rct_selected":
            return _ok(1.0, method_seed=row["method_seed"])
        return _ok(1.125, method_seed=row["method_seed"], fit_info=_info(row["method_seed"], 0.9))
    return None


@pytest.fixture(scope="module")
def stress(configs, tmp_path_factory):
    base = tmp_path_factory.mktemp("stress")
    records = _records(configs["stress"], _stress_special)
    return (records, *_run("stress", configs["stress"], records, base))


def test_stress_edge_cases_are_reported_and_independently_recomputed(configs, stress):
    records, result, out, runs, wave2, collected, _ = stress
    assert all(f["size"] == 64 for f in result["families"].values())
    identity = json.loads((out / "identity_check.json").read_text())
    half = identity["selected_at_0.5_vs_fixed"]
    assert half["fixed_not_successful"] == 1 and half["differ"] == 1
    assert all(r["within_1e-9"] for r in identity["wave2_reproduction"].values())
    contrasts = {(r["family"], r["scenario"]): r for r in csv.DictReader((out / "contrasts.csv").open())}
    flat = contrasts[("gp/causal_icm_selected", "stress:target_shift@l2_r512")]
    assert flat["zero_variance"] == "True" and float(flat["p_two_sided"]) == 1.0 and flat["verdict"] == "unresolved"
    readout = {(r["dial"], r["rct"]): r for r in csv.DictReader((out / "readout.csv").open())}
    assert readout[("common_shift", "128")]["level_1"] == "untested"
    assert readout[("common_shift", "128")]["untested_at"] == "1"
    rho = {r["scenario"]: r for r in csv.DictReader((out / "rho_selection.csv").open())}
    level1 = rho["stress:common_shift@l1_r128"]
    assert (level1["failed_selecting"], level1["weights_recorded"], level1["weights_not_converged"]) == ("1", "64", "1")
    dose = {(r["dial"], r["level"], r["rct"], r["method"]): r for r in csv.DictReader((out / "dose_response.csv").open())}
    assert dose[("obs_size", "1", "128", "causal_icm_selected")]["geometric_mean_ratio"] == ""
    report = V.arithmetic(configs["stress"], out, runs, wave2)
    assert report["mismatches"] == [] and report["readout_states"] == 16


@pytest.mark.parametrize("name,key_fields,field,value", [
    ("contrasts.csv", ("family", "scenario"), "ci95_lower", "-9.0"),
    ("contrasts.csv", ("family", "scenario"), "reference_scenario", "stress:x@l0_r128"),
    ("condition_scores.csv", ("scenario", "method"), "mean_mse", "9.0"),
    ("rho_selection.csv", ("scenario",), "mean_rho", "0.1"),
    ("rho_selection.csv", ("scenario",), "weights_not_converged", "7"),
    ("dose_response.csv", ("dial", "level", "rct", "method"), "gmr_ci95_upper", "9.0"),
    ("readout.csv", ("family", "dial", "rct"), "level_2", "worse"),
    ("replicate_scores.csv", ("group", "scenario", "seed", "method"), "cate_mse", "9.0"),
])
def test_checker_catches_a_changed_output_field(configs, stress, name, key_fields, field, value):
    _, _, out, runs, wave2, _, _ = stress
    path = out / name
    original = path.read_text()
    try:
        rows = list(csv.DictReader(path.open()))
        rows[5][field] = value
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert V.arithmetic(configs["stress"], out, runs, wave2)["mismatches"]
    finally:
        path.write_text(original)


@pytest.mark.parametrize("name,old,new", [
    ("headline.md", "| Successful fits |", "| Successful fits | 1"),
    ("headline.md", "twin: 1.", "twin: 0."),
    ("identity_check.json", '"fixed_not_successful": 1', '"fixed_not_successful": 0'),
    ("analysis.json", '"block": "stress"', '"block": "core"'),
])
def test_checker_catches_changed_text_outputs(configs, stress, name, old, new):
    _, _, out, runs, wave2, _, _ = stress
    path = out / name
    original = path.read_text()
    assert old in original
    try:
        path.write_text(original.replace(old, new, 1))
        assert V.arithmetic(configs["stress"], out, runs, wave2)["mismatches"]
    finally:
        path.write_text(original)


def test_checker_catches_a_duplicate_replicate_row(configs, stress):
    _, _, out, runs, wave2, _, _ = stress
    path = out / "replicate_scores.csv"
    original = path.read_text()
    try:
        lines = original.splitlines(keepends=True)
        path.write_text("".join(lines + [lines[1]]))
        assert V.arithmetic(configs["stress"], out, runs, wave2)["mismatches"]
    finally:
        path.write_text(original)


def test_checker_authenticates_records_without_the_analyzer(configs, stress, tmp_path):
    records, _, out, _, wave2, _, _ = stress
    # A run with one requested record missing and one selected fit with a foreign fold seed.
    key = next(k for k in records if k[3] == "causal_icm_selected" and records[k]["status"] == "ok")
    changed = dict(records)
    changed[key] = {**records[key], "fit_info": {**records[key]["fit_info"], "fold_seed": 1}}
    drop = {next(k for k in records if k[3] == "rct_t_ridge")}
    runs = _write(configs["stress"], changed, tmp_path, drop=drop)
    problems = V.arithmetic(configs["stress"], out, runs, wave2)["mismatches"]
    reasons = " ".join(str(p.get("reason")) for p in problems)
    assert "lack a record" in reasons and "completed selection" in reasons


# ---------------------------------------------------------------- collector

def test_collector_refuses_invalid_metrics_and_selections(configs, tmp_path):
    config = configs["stress"]
    key = ("gp", config.scenarios[0].key, config.seeds[0], "causal_icm_selected")
    for i, bad in enumerate((
            lambda r: {**r, "metrics": {"cate_mse": float("nan"), "root_pehe": float("nan")}},
            lambda r: {**r, "metrics": {"cate_mse": 0.25, "root_pehe": 0.4}},
            lambda r: {**r, "fit_info": {**r["fit_info"], "status": "final_fit"}},
            lambda r: {**r, "fit_info": {**r["fit_info"], "selected_rho": 0.55}})):
        records = _records(config)
        records[key] = bad(records[key])
        runs = _write(config, records, tmp_path / str(i))
        with pytest.raises(A.AnalysisError):
            A.collect(config, runs, check_source=False)


def test_collector_handles_partial_complete_and_truncated_job_limit_shards(configs, tmp_path):
    config = configs["stress"]
    records = _records(config)
    shards = list(M.shards(config))
    (partial, ps), (complete, cs), (cut, ts) = shards[3], shards[10], shards[20]
    drop = {("gp", s.key, ps.seeds[0], m) for s in ps.scenarios[2:] for m in ps.methods}
    job_limit = {partial: ps.config_hash, complete: cs.config_hash, cut: ts.config_hash}
    result, out, runs, wave2, collected, limited = _run("stress", config, records, tmp_path, drop=drop,
                                                        truncate=cut, job_limit=job_limit)
    assert sum(r.get("error_type") == "JobLimit" for r in collected.values()) == len(drop)
    assert result["coverage"] == {"ok": len(records) - len(drop), "failed": len(drop)}
    assert V.arithmetic(config, out, runs, wave2, limited)["mismatches"] == []
    with pytest.raises(A.AnalysisError):  # undeclared, the partial shard is refused
        A.collect(config, runs, check_source=False, job_limit={complete: cs.config_hash})


# ---------------------------------------------------------------- core block

def test_core_rct_only_reference_uses_the_obs512_record(configs, tmp_path):
    config = configs["core"]

    def sentinel(key, row, records):  # break the OBS-size identity of RCT-only recipes
        if key[3] in A.RCT_ONLY and "@o2048_" in key[1]:
            return _ok(row["metrics"]["cate_mse"] + 5.0, method_seed=row["method_seed"])
        return None

    records = _records(config, sentinel)
    result, out, runs, wave2, _, _ = _run("core", config, records, tmp_path)
    assert all(f["size"] == 120 for f in result["families"].values())
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    assert {r["scenario"].split("@")[0] for r in rows}.isdisjoint({"structural:M11"})
    assert sum(r["transport"] == "True" for r in rows if r["family"] == "gp/causal_icm_selected") == 24
    for r in rows:
        want = r["scenario"].replace("@o2048_", "@o512_") if r["reference"] in A.RCT_ONLY else r["scenario"]
        assert r["reference_scenario"] == want
    row = next(r for r in rows if r["family"] == "gp/causal_icm_selected" and "@o2048_" in r["scenario"])
    twin = row["scenario"].replace("@o2048_", "@o512_")
    diffs = [records[("gp", row["scenario"], s, "causal_icm_selected")]["metrics"]["cate_mse"]
             - records[("gp", twin, s, "rct_selected")]["metrics"]["cate_mse"] for s in config.seeds]
    assert math.isclose(float(row["mean_difference"]), math.fsum(diffs) / 64, rel_tol=1e-12)
    identity = json.loads((out / "identity_check.json").read_text())
    assert identity["rct_only_obs_size_mismatches"]["rct_selected"] == 62 * 64
    assert V.arithmetic(config, out, runs, wave2)["mismatches"] == []


def test_reproduction_flags_a_changed_wave2_score(configs, tmp_path):
    config = configs["stress"]
    records = _records(config)
    runs = _write(config, records, tmp_path)
    wave2 = _wave2(records, tmp_path)
    scores = A.wave2_scores(wave2)
    key = next(k for k in scores if k[2] == "causal_icm_fixed")
    scores[key] += 1e-6
    A.analyze("stress", config, A.collect(config, runs, check_source=False), scores, tmp_path / "analysis")
    identity = json.loads((tmp_path / "analysis" / "identity_check.json").read_text())
    assert not identity["wave2_reproduction"]["causal_icm_fixed"]["within_1e-9"]
    assert identity["wave2_reproduction"]["rct_selected"]["within_1e-9"]
    assert V.arithmetic(config, tmp_path / "analysis", runs, wave2)["mismatches"]


# ---------------------------------------------------------------- second review

def test_checker_authenticates_shards_record_ids_and_provenance(configs, stress, tmp_path):
    records, _, out, _, wave2, _, _ = stress
    config = configs["stress"]
    runs = _write(config, records, tmp_path / "a")
    path = runs[7] / "records.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["record_id"] = "0" * 64
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    manifest = runs[9] / "run_manifest.json"
    m = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**m, "environment": {"python": "other"}}))
    reasons = " ".join(str(p.get("reason")) for p in V.arithmetic(config, out, runs, wave2)["mismatches"])
    assert "record identity" in reasons and "provenance differs" in reasons
    # The whole block as one run is not a declared shard.
    one = tmp_path / "b" / "run"
    one.mkdir(parents=True)
    lines = [json.dumps({**r, "scenario": k[1], "seed": k[2], "method": k[3], "config_hash": config.config_hash})
             for k, r in records.items()]
    (one / "records.jsonl").write_text("\n".join(lines) + "\n")
    (one / "run_manifest.json").write_text(json.dumps({"config_hash": config.config_hash, "config": config.payload()}))
    reasons = " ".join(str(p.get("reason")) for p in V.arithmetic(config, out, [one], wave2)["mismatches"])
    assert "not a declared shard" in reasons and "declared shards are missing" in reasons


def test_integer_rho_is_accepted_and_boolean_rho_refused(configs, tmp_path):
    config = configs["stress"]

    def integer(key, row, records):
        if key[3] == "causal_icm_selected" and key[2] % 4 == 1:
            return {**row, "fit_info": {**row["fit_info"], "selected_rho": 0}}
        return None

    records = _records(config, integer)
    result, out, runs, wave2, _, _ = _run("stress", config, records, tmp_path / "int")
    assert result["selected_rho"]["0.0"] == 16 * 64
    assert V.arithmetic(config, out, runs, wave2)["mismatches"] == []
    key = next(k for k in records if k[3] == "causal_icm_selected")
    records[key] = {**records[key], "fit_info": {**records[key]["fit_info"], "selected_rho": True}}
    with pytest.raises(A.AnalysisError):
        A.collect(config, _write(config, records, tmp_path / "bool"), check_source=False)


def test_a_failure_only_reference_still_yields_untested_verdicts(configs, tmp_path):
    config = configs["stress"]
    records = _records(config, lambda key, row, records: _failed(method_seed=row["method_seed"])
                       if key[3] == "rct_selected" else None)
    result, out, runs, wave2, _, _ = _run("stress", config, records, tmp_path)
    assert result["families"]["gp/causal_icm_selected"]["other_untested"] == 48
    assert (out / "dose_response.csv").read_text().startswith("dial,level,rct,method,pairs")
    assert V.arithmetic(config, out, runs, wave2)["mismatches"] == []


def test_checker_compares_the_identity_rule_and_handles_underflow(configs, tmp_path):
    config = configs["stress"]

    def tiny(key, row, records):  # differences whose computed variance underflows to zero
        sc, seed, m = key[1:]
        if sc == "stress:noise_shape@l1_r128" and m in ("rct_selected", "causal_icm_selected"):
            base = 1.0 if m == "rct_selected" else 1.0 + (1e-200 if seed % 2 else 2e-200)
            return _ok(base, method_seed=row["method_seed"],
                       **({"fit_info": _info(row["method_seed"], 0.9)} if m == "causal_icm_selected" else {}))
        return None

    records = _records(config, tiny)
    _, out, runs, wave2, _, _ = _run("stress", config, records, tmp_path)
    assert V.arithmetic(config, out, runs, wave2)["mismatches"] == []
    path = out / "identity_check.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "rule": "the block ran on wave 2's data"}))
    assert V.arithmetic(config, out, runs, wave2)["mismatches"]
