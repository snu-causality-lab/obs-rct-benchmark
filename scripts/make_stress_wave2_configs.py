"""Write shards for the wave-2 recipes on stress study 1 (docs/V1_STRESS_WAVE2.md).

One configuration per wave-2 recipe group, with the group's wave-2 recipes and options
and the 64 stress conditions. One shard per seed and dial (eight conditions). Every
shard also refits `rct_selected` and `rct_t_ridge`, so contrasts pair within one
process and environment.

    python scripts/make_stress_wave2_configs.py --config configs/v1_stress_wave2_gp.toml --output shards-gp
    python scripts/make_stress_wave2_configs.py --config configs/v1_stress_wave2_gp.toml --output preflight-gp --preflight

`--preflight` writes the same shards for seed 79999 only, outside the study grid.
`--write-configs DIR` writes the four declared configurations instead.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from make_stress_configs import SEEDS, expected_scenarios
from make_v1_wave2_configs import COMMON, COMMON_OPTIONS, GROUPS, write_toml
from obs_rct_benchmark.runner import BenchmarkConfig, load_config
from obs_rct_benchmark.stress_dgp import STRESS_DIALS

PREFLIGHT_SEED = 79999


def group_of(config: BenchmarkConfig) -> str:
    for name, recipes in GROUPS.items():
        if config.methods == (*COMMON, *recipes):
            return name
    raise ValueError("recipes do not match a declared wave-2 group")


def validate(config: BenchmarkConfig) -> str:
    group = group_of(config)
    if config.seeds != SEEDS or config.reference_method != "rct_selected":
        raise ValueError("seeds or reference differ from the declared stress design")
    got = [(s.id, s.label, dict(s.parameters)) for s in config.scenarios]
    if any(s.kind != "stress" for s in config.scenarios) or got != expected_scenarios():
        raise ValueError("scenarios differ from the declared 8 dials x 4 levels x 2 trial sizes")
    if {k: dict(v) for k, v in config.method_options.items()} != {**COMMON_OPTIONS, **GROUPS[group]}:
        raise ValueError("recipe options differ from the declared wave-2 options")
    return group


def shards(config: BenchmarkConfig, *, preflight: bool = False):
    group = group_of(config)
    for seed in ((PREFLIGHT_SEED,) if preflight else config.seeds):
        for dial in STRESS_DIALS:
            scenarios = tuple(s for s in config.scenarios if s.id == dial)
            yield (f"{group}-stress-seed-{seed:05d}-{dial}.toml",
                   replace(config, scenarios=scenarios, seeds=(seed,)))


def declared_payload(group: str) -> dict:
    """The declared configuration of one group, as written to configs/."""
    return {
        "schema_version": 1,
        "run": {"purpose": "reference", "seeds": list(SEEDS), "methods": [*COMMON, *GROUPS[group]],
                "reference_method": "rct_selected",
                "scope": (f"stress study 1 wave-2 addendum (docs/V1_STRESS_WAVE2.md), group {group}: the wave-2 "
                          "recipes and options on the 64 stress conditions and seeds; each shard refits "
                          "rct_selected and rct_t_ridge")},
        "method_options": {**COMMON_OPTIONS, **GROUPS[group]},
        "scenarios": [{"kind": "stress", "id": dial, "label": label, "parameters": params}
                      for dial, label, params in expected_scenarios()],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--write-configs", type=Path)
    args = ap.parse_args()
    if args.write_configs:
        for group in GROUPS:
            path = args.write_configs / f"v1_stress_wave2_{group}.toml"
            write_toml(declared_payload(group), path)
            if validate(load_config(path)) != group:
                raise SystemExit(f"{path}: written configuration does not validate")
            print(path)
        return
    if args.config is None or args.output is None:
        ap.error("--config and --output are required unless --write-configs is given")
    config = load_config(args.config)
    group = validate(config)
    args.output.mkdir(parents=False, exist_ok=False)
    manifest = {"group": group, "preflight": args.preflight,
                "study_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "shards": []}
    for filename, shard in shards(config, preflight=args.preflight):
        path = args.output / filename
        write_toml(shard.payload(), path)
        if load_config(path).config_hash != shard.config_hash:
            raise SystemExit(f"{filename}: written shard does not reload identically")
        manifest["shards"].append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "config_hash": shard.config_hash,
                                   "cells": len(shard.scenarios) * len(shard.methods) * len(shard.seeds)})
    manifest["requested_cells"] = sum(s["cells"] for s in manifest["shards"])
    (args.output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{group}: {len(manifest['shards'])} shards, {manifest['requested_cells']} requested cells")


if __name__ == "__main__":
    main()
