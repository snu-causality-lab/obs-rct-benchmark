"""Write one shard per seed for stress study 1 (docs/V1_STRESS_CORNET.md).

    python scripts/make_stress_configs.py --config configs/v1_stress_cornet.toml --output shards
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from obs_rct_benchmark.runner import BenchmarkConfig, load_config
from obs_rct_benchmark.stress_dgp import OBS_SIZES, STRESS_DIALS

SEEDS = tuple(range(80000, 80064))
RECIPES = (
    "rct_selected",
    "rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
    "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
    "rct_dr_ridge", "rct_dr_extra_trees",
    "cornet_official", "cornet_raw", "datafusionppi",
)
TREES = {"n_estimators": 100, "min_samples_leaf": 5}
CORNET = {"source_epochs": 500, "correction_epochs": 200, "width": 25,
          "learning_rate": 0.1, "outcome_penalty": 0.1, "correction_penalty": 0.0}
OPTIONS = {
    "rct_selected": {"ridge_alpha": 1.0, **TREES},
    **{m: ({"ridge_alpha": 1.0} if m.endswith("ridge") else dict(TREES)) for m in RECIPES[1:9]},
    "cornet_official": {**CORNET, "standardize": True},
    "cornet_raw": dict(CORNET),
    "datafusionppi": {"learner": "DR", "covariate_shift": True},
}


def expected_scenarios() -> list[tuple]:
    out = []
    for dial in STRESS_DIALS:
        for level in range(4):
            for rct in (128, 512):
                n_obs = OBS_SIZES[level] if dial == "obs_size" else 2048
                out.append((dial, f"l{level}_r{rct}", {"level": level, "n_obs": n_obs, "n_rct_train": rct,
                                                       "n_rct_tune": 32, "n_rct_test": 64, "n_eval": 2048,
                                                       "n_target_reference": 0}))
    return out


def validate(config: BenchmarkConfig) -> None:
    if config.seeds != SEEDS or config.methods != RECIPES or config.reference_method != "rct_selected":
        raise ValueError("seeds, recipes or reference differ from the declared stress design")
    got = [(s.id, s.label, dict(s.parameters)) for s in config.scenarios]
    if any(s.kind != "stress" for s in config.scenarios) or got != expected_scenarios():
        raise ValueError("scenarios differ from the declared 8 dials x 4 levels x 2 trial sizes")
    if {k: dict(v) for k, v in config.method_options.items()} != OPTIONS:
        raise ValueError("recipe options differ from the declared design")


def shards(config: BenchmarkConfig):
    for seed in config.seeds:
        yield f"stress-seed-{seed:05d}.toml", replace(config, seeds=(seed,))


def main() -> None:
    from make_reference_configs import write_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = load_config(args.config)
    validate(config)
    args.output.mkdir(parents=False, exist_ok=False)
    manifest = {"study_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "shards": []}
    for filename, shard in shards(config):
        path = args.output / filename
        write_config(shard.payload(), path)
        if load_config(path).config_hash != shard.config_hash:
            raise SystemExit(f"{filename}: written shard does not reload identically")
        manifest["shards"].append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "config_hash": shard.config_hash,
                                   "cells": len(shard.scenarios) * len(shard.methods)})
    manifest["requested_cells"] = sum(s["cells"] for s in manifest["shards"])
    (args.output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest['shards'])} shards, {manifest['requested_cells']} requested cells")


if __name__ == "__main__":
    main()
