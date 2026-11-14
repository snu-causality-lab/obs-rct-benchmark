"""Write disjoint shards of the v1 core study (docs/V1_CORE_STUDY.md).

One shard per sample seed and applicability group. The pinned DataFusionPPI
adapter accepts only a constant trial probability, so the multivariate
Integrative R preset forms its own group without DataFusionPPI recipes. The
nominal configuration is validated against the declared design first.

    python scripts/make_v1_core_configs.py --config configs/v1_core_study.toml --output shards
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
WAVE1 = (
    "rct_selected",
    "rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
    "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
    "rct_dr_ridge", "rct_dr_extra_trees",
    "cornet_official", "cornet_raw", "datafusionppi",
)
VARIABLE_PROPENSITY = ("paper", "integrative_r_learner_multivariate_paper")
CONSTANT_PROPENSITY_ONLY = ("datafusionppi", "datafusionppi_r")
ROLE_FIELDS = {"n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval", "n_target_reference"}
TREES = {"n_estimators": 100, "min_samples_leaf": 5}
CORNET = {"source_epochs": 500, "correction_epochs": 200, "width": 25,
          "learning_rate": 0.1, "outcome_penalty": 0.1, "correction_penalty": 0.0}
EXPECTED_OPTIONS = {
    "rct_selected": {"ridge_alpha": 1.0, **TREES},
    **{m: ({"ridge_alpha": 1.0} if m.endswith("ridge") else dict(TREES))
       for m in WAVE1[1:9]},
    "cornet_official": {**CORNET, "standardize": True},
    "cornet_raw": dict(CORNET),
    "datafusionppi": {"learner": "DR", "covariate_shift": True},
}


def presets() -> tuple[tuple[str, str], ...]:
    return (tuple(("structural", m) for m in STRUCTURAL_MECHANISMS)
            + tuple(("addendum", c) for c in STRUCTURAL_ADDENDUM_CELLS)
            + tuple(("paper", p) for p in PAPER_DGPS))


def validate(config: BenchmarkConfig) -> None:
    items = presets()
    if len(items) != 31:
        raise ValueError("expected the 31 canonical presets")
    if config.seeds != SEEDS or config.methods != WAVE1 or config.reference_method != "rct_selected":
        raise ValueError("seeds, recipes or reference differ from the declared wave-1 design")
    expected = [(k, i, f"o{o}_r{r}") for k, i in items for o in (512, 2048) for r in (128, 512)]
    if [(s.kind, s.id, s.label) for s in config.scenarios] != expected:
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
    if {k: dict(v) for k, v in config.method_options.items()} != EXPECTED_OPTIONS:
        raise ValueError("recipe options differ from the declared design")


def group(scenario) -> str:
    return "variable_propensity" if (scenario.kind, scenario.id) == VARIABLE_PROPENSITY else "common"


def shards(config: BenchmarkConfig):
    for seed in config.seeds:
        for name in ("common", "variable_propensity"):
            scenarios = tuple(s for s in config.scenarios if group(s) == name)
            methods = tuple(m for m in config.methods
                            if name == "common" or m not in CONSTANT_PROPENSITY_ONLY)
            options = {k: dict(v) for k, v in config.method_options.items() if k in methods}
            yield (f"group-{name}-seed-{seed:05d}.toml",
                   replace(config, scenarios=scenarios, seeds=(seed,), methods=methods,
                           method_options=options))


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
    args = ap.parse_args()
    config = load_config(args.config)
    validate(config)
    args.output.mkdir(parents=False, exist_ok=False)
    manifest = {"study_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "shards": []}
    for filename, shard in shards(config):
        path = args.output / filename
        write_toml(shard.payload(), path)
        if load_config(path).config_hash != shard.config_hash:
            raise SystemExit(f"{filename}: written shard does not reload identically")
        cells = len(shard.scenarios) * len(shard.methods)
        manifest["shards"].append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "config_hash": shard.config_hash, "cells": cells})
    manifest["requested_cells"] = sum(s["cells"] for s in manifest["shards"])
    (args.output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest['shards'])} shards, {manifest['requested_cells']} requested cells")


if __name__ == "__main__":
    main()
