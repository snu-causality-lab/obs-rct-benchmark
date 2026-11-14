"""Write shards for the v1 core wave-2 addendum (docs/V1_CORE_WAVE2.md).

One configuration per recipe group; one shard per sample seed and preset (four
budget conditions). Every shard runs the group's recipes plus `rct_selected` and
`rct_t_ridge`, so contrasts pair within one process and environment.

    python scripts/make_v1_wave2_configs.py --config configs/v1_wave2_native_r.toml --output shards-native_r
    python scripts/make_v1_wave2_configs.py --config configs/v1_wave2_native_r.toml --output preflight-native_r --preflight

`--preflight` writes the same shards for seed 69999 only, outside the study grid.
Groups: native_r, gp, tabpfn (TabPFN releases under tabpfn 10.0.0) and tsr_legacy
(the registered TSR recipes under tabpfn 2.0.9).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from obs_rct_benchmark.paper_dgp import PAPER_DGPS
from obs_rct_benchmark.runner import BenchmarkConfig, load_config
from obs_rct_benchmark.structural_dgp import STRUCTURAL_ADDENDUM_CELLS, STRUCTURAL_MECHANISMS

SEEDS = tuple(range(70000, 70064))
PREFLIGHT_SEED = 69999
COMMON = ("rct_selected", "rct_t_ridge")
TABPFN_SHA256 = {
    "v2": "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    "v2.5": "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8",
    "v3": "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
}
GPU = {"device": "cuda", "gp_maxiter": 1000}
GROUPS = {
    "native_r": {
        "integrative_r_linear": {"k_folds": 5, "timeout_seconds": 1200},
        "integrative_r_quadratic": {"k_folds": 5, "timeout_seconds": 1200},
        "rct_causal_forest": {"num_trees": 2000, "num_threads": 1, "timeout_seconds": 600},
    },
    "gp": {
        "causal_icm_fixed": {"rho": 0.5},
        "causal_icm_rct_only": {"rho": 0.5},
    },
    "tabpfn": {
        "lmcgp_rct_only": dict(GPU),
        **{f"tsr_tabpfn_{tag}": {**GPU, "model_sha256": TABPFN_SHA256[release]}
           for tag, release in (("v2", "v2"), ("v25", "v2.5"), ("v3", "v3"))},
        **{f"tabpfn_{tag}_obs_only": {**GPU, "model_sha256": TABPFN_SHA256[release]}
           for tag, release in (("v2", "v2"), ("v25", "v2.5"), ("v3", "v3"))},
    },
    # The registered TSR trio under tabpfn==2.0.9, as in configs/tsr_study.toml: the
    # replication on fresh seeds that the v1 plan promised.
    "tsr_legacy": {m: {**GPU, "model_sha256": TABPFN_SHA256["v2"]}
                   for m in ("tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only")},
}
COMMON_OPTIONS = {"rct_selected": {"ridge_alpha": 1.0, "n_estimators": 100, "min_samples_leaf": 5},
                  "rct_t_ridge": {"ridge_alpha": 1.0}}
ROLE_FIELDS = {"n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval", "n_target_reference"}


def presets() -> tuple[tuple[str, str], ...]:
    return (tuple(("structural", m) for m in STRUCTURAL_MECHANISMS)
            + tuple(("addendum", c) for c in STRUCTURAL_ADDENDUM_CELLS)
            + tuple(("paper", p) for p in PAPER_DGPS))


def group_of(config: BenchmarkConfig) -> str:
    for name, recipes in GROUPS.items():
        if config.methods == (*COMMON, *recipes):
            return name
    raise ValueError("recipes do not match a declared wave-2 group")


def validate(config: BenchmarkConfig) -> str:
    group = group_of(config)
    if config.seeds != SEEDS or config.reference_method != "rct_selected":
        raise ValueError("seeds or reference differ from the declared wave-2 design")
    items = presets()
    expected = [(k, i, f"o{o}_r{r}") for k, i in items for o in (512, 2048) for r in (128, 512)]
    if len(items) != 31 or [(s.kind, s.id, s.label) for s in config.scenarios] != expected:
        raise ValueError("expected all 31 presets x 4 budgets in canonical order")
    for s in config.scenarios:
        p = dict(s.parameters)
        fields = ROLE_FIELDS - ({"n_target_reference"} if s.kind == "paper" else set())
        if (set(p) != fields or (p["n_obs"], p["n_rct_train"]) not in
                {(512, 128), (512, 512), (2048, 128), (2048, 512)}
                or p["n_rct_tune"] != 32 or p["n_rct_test"] != 64 or p["n_eval"] != 2048
                or p.get("n_target_reference", 0) != 0
                or s.label != f"o{p['n_obs']}_r{p['n_rct_train']}"):
            raise ValueError(f"{s.key}: only the declared sample-role sizes may be set")
    if {k: dict(v) for k, v in config.method_options.items()} != {**COMMON_OPTIONS, **GROUPS[group]}:
        raise ValueError("recipe options differ from the declared design")
    return group


def slug(kind: str, ident: str) -> str:
    return f"{kind}__{ident}"


def shards(config: BenchmarkConfig, *, preflight: bool = False):
    group = group_of(config)
    seeds = (PREFLIGHT_SEED,) if preflight else config.seeds
    for seed in seeds:
        for kind, ident in presets():
            scenarios = tuple(s for s in config.scenarios if (s.kind, s.id) == (kind, ident))
            yield (f"{group}-seed-{seed:05d}-{slug(kind, ident)}.toml",
                   replace(config, scenarios=scenarios, seeds=(seed,)))


def write_toml(payload: dict, path: Path) -> None:
    def value(item):
        return json.dumps(item, ensure_ascii=False, allow_nan=False)
    lines = [f"schema_version = {payload['schema_version']}", "", "[run]"]
    lines += [f"{k} = {value(v)}" for k, v in payload["run"].items()]
    for name, options in payload.get("method_options", {}).items():
        lines += ["", f"[method_options.{name}]"] + [f"{k} = {value(v)}" for k, v in options.items()]
    for scenario in payload["scenarios"]:
        lines += ["", "[[scenarios]]"]
        lines += [f"{k} = {value(v)}" for k, v in scenario.items() if k != "parameters"]
        lines += ["[scenarios.parameters]"]
        lines += [f"{k} = {value(v)}" for k, v in scenario["parameters"].items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
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
