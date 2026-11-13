"""Hand-checked and broken-input tests for scripts/analyze_tsr_study.py."""
from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_tsr_study", ROOT / "scripts" / "analyze_tsr_study.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
SHA = "a" * 64
SEEDS = (1, 2, 3)


def _config(tmp_path: Path):
    from obs_rct_benchmark.runner import load_config

    params = "n_obs = 40\nn_rct_train = 20\nn_rct_tune = 4\nn_rct_test = 4\nn_eval = 10\n"
    opts = "".join(f'[method_options.{m}]\ndevice = "cuda"\ngp_maxiter = 200\nmodel_sha256 = "{SHA}"\n'
                   for m in (A.TSR, A.RCT_GP, A.OBS))
    text = (f'schema_version = 1\n[run]\npurpose = "reference"\nseeds = [1, 2, 3]\n'
            f'methods = {json.dumps(list(A.METHODS))}\nreference_method = "{A.RCT_GP}"\nscope = "test"\n{opts}'
            f'[[scenarios]]\nkind = "structural"\nid = "M0"\nlabel = "a"\n[scenarios.parameters]\n{params}'
            f'[[scenarios]]\nkind = "structural"\nid = "M11"\nlabel = "a"\n[scenarios.parameters]\n{params}')
    path = tmp_path / "study.toml"
    path.write_text(text)
    return load_config(path)


VALUES = {  # per method, per scenario: MSE for seeds 1..3
    A.TSR: {"structural:M0@a": [1.0, 2.0, 3.0], "structural:M11@a": [1.0, 1.0, 1.0]},
    A.RCT_GP: {"structural:M0@a": [2.0, 4.0, 5.0], "structural:M11@a": [2.0, 2.0, 2.0]},
    A.OBS: {"structural:M0@a": [3.0, 3.0, 3.0], "structural:M11@a": [3.0, 3.0, 3.0]},
    A.TREES: {"structural:M0@a": [1.5, 2.5, 2.5], "structural:M11@a": [2.0, 2.0, 2.0]},
    A.RIDGE: {"structural:M0@a": [0.5, 1.0, 4.0], "structural:M11@a": [2.0, 2.0, 2.0]},
}


def _records(drop=()):
    out = {}
    for m, by in VALUES.items():
        for sc, vals in by.items():
            for s, v in zip(SEEDS, vals):
                if (sc, s, m) in drop:
                    continue
                out[(sc, s, m)] = {"status": "ok", "metrics": {"cate_mse": v}}
    return out


def _catalog(tmp_path: Path, perturb: float = 0.0) -> Path:
    cat = tmp_path / "catalog"
    (cat / "designs" / "x").mkdir(parents=True, exist_ok=True)
    with (cat / "designs" / "x" / "replicate_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "seed", "method", "status", "cate_mse"])
        for m in (A.TREES, A.RIDGE):
            for sc, vals in VALUES[m].items():
                for s, v in zip(SEEDS, vals):
                    w.writerow([sc, s, m, "ok", v + (perturb if (m, s) == (A.TREES, 1) else 0.0)])
    with (cat / "condition_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "method", "applicable", "expected", "successful", "mean"])
        for sc in VALUES[A.TREES]:
            for m in A.CATALOG_RCT_ONLY:
                vals = VALUES.get(m, {}).get(sc, [9.0, 9.0, 9.0])
                w.writerow([sc, m, "True", 3, 3, sum(vals) / 3])
    return cat


def test_hand_example_families_holm_and_heavy_tail(tmp_path):
    from scipy import stats

    cfg = _config(tmp_path)
    res = A.analyze(cfg, _records(), _catalog(tmp_path), tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out" / "primary_contrasts.csv").open()))
    a = next(r for r in rows if r["family"] == "A_obs_offset" and r["scenario"] == "structural:M0@a")
    p = 2 * stats.t.sf(5.0, 2)  # diffs -1,-2,-2: mean -5/3, se 1/3, t = -5
    assert float(a["mean"]) == pytest.approx(-5 / 3) and float(a["p_two_sided"]) == pytest.approx(p)
    assert float(a["holm_p"]) == pytest.approx(p)  # one inferential test in family A
    m11 = next(r for r in rows if r["family"] == "A_obs_offset" and r["scenario"] == "structural:M11@a")
    assert m11["decision"] == "descriptive_heavy_tail" and m11["p_two_sided"] == ""
    b = {r["second"]: r for r in rows if r["family"] == "B_complete_procedure" and r["scenario"] == "structural:M0@a"}
    # Family B has two inferential tests (M0 vs each catalog control). TSR - trees: -0.5,-0.5,+0.5.
    p_trees = 2 * stats.t.sf(abs((-0.5 / 3) / (stats.tstd([-0.5, -0.5, 0.5]) / 3 ** 0.5)), 2)
    p_ridge = float(b[A.RIDGE]["p_two_sided"])
    assert float(b[A.TREES]["p_two_sided"]) == pytest.approx(p_trees)
    small, large = sorted([p_trees, p_ridge])
    holm = {small: min(2 * small, 1.0), large: min(max(2 * small, large), 1.0)}
    assert float(b[A.TREES]["holm_p"]) == pytest.approx(holm[p_trees])
    assert float(b[A.RIDGE]["holm_p"]) == pytest.approx(holm[p_ridge])
    assert res["identity_passed"] and res["families"]["A_obs_offset"]["complete"]
    counts = (tmp_path / "out" / "budget_counts.md").read_text()
    assert "| matched RCT-only GP | 2/2 |" in counts  # TSR mean below GP mean in both conditions


def test_missing_pair_withholds_only_when_inferential(tmp_path):
    cfg = _config(tmp_path)
    res = A.analyze(cfg, _records(drop={("structural:M11@a", 2, A.TSR)}), _catalog(tmp_path), tmp_path / "o1")
    assert res["families"]["A_obs_offset"]["complete"]  # M11 is descriptive, not a required pair
    res = A.analyze(cfg, _records(drop={("structural:M0@a", 2, A.TSR)}), _catalog(tmp_path), tmp_path / "o2")
    assert not res["families"]["A_obs_offset"]["complete"]
    rows = list(csv.DictReader((tmp_path / "o2" / "primary_contrasts.csv").open()))
    assert {r["decision"] for r in rows if r["inference_eligible"] == "True"} == {"withheld_incomplete_family"}
    assert res["coverage"]["missing"] == 1


def test_identity_failure_drops_cross_study_counts(tmp_path):
    res = A.analyze(_config(tmp_path), _records(), _catalog(tmp_path, perturb=1e-6), tmp_path / "out")
    assert not res["identity_passed"]
    assert "not reported" in (tmp_path / "out" / "budget_counts.md").read_text()


def _shard(tmp_path: Path, cfg, *, device="cuda", sha=SHA, options_override=None) -> Path:
    from obs_rct_benchmark.runner import method_seed, record_id

    shard = replace(cfg, seeds=SEEDS)
    payload = shard.payload()
    if options_override:
        payload = json.loads(json.dumps(payload))
        payload["method_options"][A.TSR].update(options_override)
    run = tmp_path / f"shard-{device}-{sha[:2]}-{bool(options_override)}"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"config": payload, "config_hash": shard.config_hash,
                                                       "source": {"x": 1}, "environment": {"y": 2}}))
    lines = []
    for (sc, s, m), row in _records().items():
        rec = {"scenario": sc, "seed": s, "method": m, "config_hash": shard.config_hash,
               "method_seed": method_seed(sc, s, m), "record_id": record_id(shard, sc, s, m), **row}
        if m in A.STAGE1_METHODS:
            rec["fit_info"] = {"stage1_device": device, "stage1_model_sha256": sha}
        lines.append(json.dumps(rec))
    (run / "records.jsonl").write_text("\n".join(lines) + "\n")
    return run


def test_collect_accepts_declared_shard_and_rejects_drift(tmp_path):
    cfg = _config(tmp_path)
    assert len(A.collect(cfg, [_shard(tmp_path, cfg)], SHA, check_source=False)) == 30
    with pytest.raises(A.AnalysisError, match="CUDA"):
        A.collect(cfg, [_shard(tmp_path, cfg, device="cpu")], SHA, check_source=False)
    with pytest.raises(A.AnalysisError, match="CUDA"):
        A.collect(cfg, [_shard(tmp_path, cfg, sha="b" * 64)], SHA, check_source=False)
    with pytest.raises(A.AnalysisError, match="settings differ"):
        A.collect(cfg, [_shard(tmp_path, cfg, options_override={"gp_maxiter": 50})], SHA, check_source=False)
