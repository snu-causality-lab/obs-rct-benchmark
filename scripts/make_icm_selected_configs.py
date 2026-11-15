"""Write shards for the selected-rho Causal-ICM addendum (docs/V1_ICM_SELECTED.md).

Two blocks, each one configuration in the `gp` environment:
- `core`: the 124 v1 core conditions on seeds 70000-70063, one shard per seed and
  preset (four budget conditions), as in wave 2;
- `stress`: the 64 stress conditions on seeds 80000-80063, one shard per seed, dial and
  trial size (four levels). The stress addendum held both trial sizes in one shard; the
  split halves the longest shard, whose 8,192 OBS rows make each selection slow.

Every shard fits `rct_selected`, `rct_t_ridge`, `causal_icm_fixed`,
`causal_icm_rct_only` and `causal_icm_selected`, so every contrast pairs within one
process and environment, and the four wave-2 recipes can be compared with wave 2's
`gp` records.

    python scripts/make_icm_selected_configs.py --write-configs configs
    python scripts/make_icm_selected_configs.py --config configs/v1_icm_selected_core.toml --output shards-core
    python scripts/make_icm_selected_configs.py --config configs/v1_icm_selected_core.toml --output pre-core --preflight

`--preflight` writes the same shards for one seed outside the grid: 69999 (core) or
79999 (stress).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from make_stress_configs import SEEDS as STRESS_SEEDS, expected_scenarios as stress_scenarios
from make_v1_wave2_configs import (COMMON, COMMON_OPTIONS, PREFLIGHT_SEED as CORE_PREFLIGHT,
                                   SEEDS as CORE_SEEDS, presets, slug, write_toml)
from obs_rct_benchmark.runner import BenchmarkConfig, load_config
from obs_rct_benchmark.stress_dgp import STRESS_DIALS

GROUP = "gp"  # the wave-2 GPy environment
RECIPES = ("causal_icm_fixed", "causal_icm_rct_only", "causal_icm_selected")
# causal_icm_selected takes no options: its grid, folds and covariates are its method ID.
OPTIONS = {**COMMON_OPTIONS, "causal_icm_fixed": {"rho": 0.5}, "causal_icm_rct_only": {"rho": 0.5}}
METHODS = (*COMMON, *RECIPES)
STRESS_PREFLIGHT = 79999
BLOCKS = ("core", "stress")


def core_scenarios() -> list[dict]:
    out = []
    for kind, ident in presets():
        for o in (512, 2048):
            for r in (128, 512):
                params = {"n_obs": o, "n_rct_train": r, "n_rct_tune": 32, "n_rct_test": 64, "n_eval": 2048}
                if kind != "paper":  # paper presets have no target-reference sample
                    params["n_target_reference"] = 0
                out.append({"kind": kind, "id": ident, "label": f"o{o}_r{r}", "parameters": params})
    return out


def declared_payload(block: str) -> dict:
    """The declared configuration of one block, as written to configs/."""
    if block == "core":
        seeds, scenarios = CORE_SEEDS, core_scenarios()
        scope = ("selected-rho Causal-ICM addendum (docs/V1_ICM_SELECTED.md), v1 core block: the 124 wave-2 "
                 "conditions and seeds in the gp environment; each shard refits rct_selected, rct_t_ridge, "
                 "causal_icm_fixed and causal_icm_rct_only")
    elif block == "stress":
        seeds = STRESS_SEEDS
        scenarios = [{"kind": "stress", "id": d, "label": lab, "parameters": p} for d, lab, p in stress_scenarios()]
        scope = ("selected-rho Causal-ICM addendum (docs/V1_ICM_SELECTED.md), stress block: the 64 stress "
                 "conditions and seeds in the gp environment; each shard refits rct_selected, rct_t_ridge, "
                 "causal_icm_fixed and causal_icm_rct_only")
    else:
        raise ValueError(f"unknown block {block!r}")
    return {"schema_version": 1,
            "run": {"purpose": "reference", "seeds": list(seeds), "methods": list(METHODS),
                    "reference_method": "rct_selected", "scope": scope},
            "method_options": OPTIONS, "scenarios": scenarios}


def validate(config: BenchmarkConfig) -> str:
    """The block a configuration declares; anything else is refused."""
    if config.methods != METHODS or config.reference_method != "rct_selected":
        raise ValueError("recipes or reference differ from the declared addendum")
    if {k: dict(v) for k, v in config.method_options.items()} != OPTIONS:
        raise ValueError("recipe options differ from the declared addendum")
    got = [(s.kind, s.id, s.label, dict(s.parameters)) for s in config.scenarios]
    for block in BLOCKS:
        want = declared_payload(block)
        if config.seeds == tuple(want["run"]["seeds"]) and got == [
                (s["kind"], s["id"], s["label"], s["parameters"]) for s in want["scenarios"]]:
            return block
    raise ValueError("seeds or scenarios differ from both declared blocks")


def shards(config: BenchmarkConfig, *, preflight: bool = False):
    """(file name, shard) pairs of a validated configuration."""
    block = validate(config)
    if block == "core":
        for seed in ((CORE_PREFLIGHT,) if preflight else config.seeds):
            for kind, ident in presets():
                scenarios = tuple(s for s in config.scenarios if (s.kind, s.id) == (kind, ident))
                yield (f"icmsel-core-seed-{seed:05d}-{slug(kind, ident)}.toml",
                       replace(config, scenarios=scenarios, seeds=(seed,)))
    else:
        for seed in ((STRESS_PREFLIGHT,) if preflight else config.seeds):
            for dial in STRESS_DIALS:
                for rct in (128, 512):
                    scenarios = tuple(s for s in config.scenarios
                                      if s.id == dial and dict(s.parameters)["n_rct_train"] == rct)
                    yield (f"icmsel-stress-seed-{seed:05d}-{dial}-r{rct}.toml",
                           replace(config, scenarios=scenarios, seeds=(seed,)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--write-configs", type=Path)
    args = ap.parse_args()
    if args.write_configs:
        for block in BLOCKS:
            path = args.write_configs / f"v1_icm_selected_{block}.toml"
            write_toml(declared_payload(block), path)
            if validate(load_config(path)) != block:
                raise SystemExit(f"{path}: written configuration does not validate")
            print(path)
        return
    if args.config is None or args.output is None:
        ap.error("--config and --output are required unless --write-configs is given")
    config = load_config(args.config)
    block = validate(config)
    args.output.mkdir(parents=False, exist_ok=False)
    manifest = {"group": GROUP, "block": block, "preflight": args.preflight,
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
    print(f"{block}: {len(manifest['shards'])} shards, {manifest['requested_cells']} requested cells")


if __name__ == "__main__":
    main()
