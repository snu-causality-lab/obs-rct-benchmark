"""Independent synthetic terminal records for adaptive shard aggregation."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path

import pytest

from obs_rct_benchmark.adaptive_runner import AdaptiveConfig, _identity, _provenance
from obs_rct_benchmark.runner import ScenarioConfig


spec = importlib.util.spec_from_file_location(
    "adaptive_analysis", Path(__file__).parents[1]/"scripts/analyze_adaptive_study.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def config():
    return AdaptiveConfig((ScenarioConfig("structural", "M0", {"n_obs": 8, "n_eval": 8}),),
        seeds=(1, 2), policies=("uniform", "geometry"), estimators=("rct_t_ridge",),
        budgets=(16,), pilot_size=8, tune_size=4, pool_size=24, purpose="reference")


def shard(path, config, *, fail_geometry=False):
    """Write observed-contract records without sampling a DGP or fitting."""
    path.mkdir()
    provenance, fingerprint = _provenance(config)
    manifest = dict(provenance, config=config.payload(), config_hash=config.config_hash,
                    source_fingerprint=fingerprint)
    (path/"run_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for seed in config.seeds:
        for policy in config.policies:
            row = _identity(config, config.scenarios[0].key, seed, policy, 16, "rct_t_ridge")
            mse = float(seed+1 if policy == "uniform" else seed)
            row.update(query_indices=list(range(16)), query_cost=16, pilot_cost=8, tune_cost=4, train_cost=12,
                       status="ok", metrics={"cate_mse": mse, "root_pehe": math.sqrt(mse)})
            if fail_geometry and policy == "geometry":
                row.update(status="failed", stage="method_or_metric", error_type="ValueError", metrics=None)
            rows.append(row)
    (path/"records.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return rows


def test_complete_disjoint_collection_and_hand_calculated_policy_contrast(tmp_path):
    c = config()
    paths = [tmp_path/"one", tmp_path/"two"]
    for path, seed in zip(paths, c.seeds):
        shard(path, replace(c, seeds=(seed,)))
    records, inputs = analysis.collect_records(c, paths)
    assert len(records) == 4 and len(inputs) == 2
    summary, scores, contrasts = analysis.analyze_records(c, records)
    assert summary["coverage"] == {"expected": 4, "recorded": 4, "successful": 4, "failed": 0, "missing": 0}
    assert next(x for x in scores if x["policy"] == "uniform")["mean_mse"] == 2.5
    assert contrasts[0]["mean_difference"] == -1.
    assert contrasts[0]["mcse_difference"] == 0.
    assert contrasts[0]["ci95_lower"] == contrasts[0]["ci95_upper"] == -1.


def test_duplicate_and_missing_seed_shards_fail_closed(tmp_path):
    c = config()
    first, second = tmp_path/"first", tmp_path/"second"
    shard(first, replace(c, seeds=(1,)))
    shard(second, replace(c, seeds=(1,)))
    with pytest.raises(ValueError, match="overlapping"):
        analysis.collect_records(c, [first, second])
    with pytest.raises(ValueError, match="every declared"):
        analysis.collect_records(c, [first])


def test_missing_cell_and_tampered_method_cost_are_rejected(tmp_path):
    c = replace(config(), seeds=(1,))
    path = tmp_path/"shard"
    rows = shard(path, c)
    (path/"records.jsonl").write_text(json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError, match="missing"):
        analysis.collect_records(c, [path])
    rows[1]["query_cost"] = 15
    (path/"records.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="cost"):
        analysis.collect_records(c, [path])


def test_failed_cells_survive_and_withhold_policy_comparison(tmp_path):
    c = config()
    path = tmp_path/"failed"
    shard(path, c, fail_geometry=True)
    records, _ = analysis.collect_records(c, [path])
    summary, scores, contrasts = analysis.analyze_records(c, records)
    assert summary["coverage"]["failed"] == 2
    assert len(summary["terminal_failures"]) == 2
    assert next(x for x in scores if x["policy"] == "geometry")["failed"] == 2
    assert not contrasts[0]["complete"] and contrasts[0]["mean_difference"] is None
