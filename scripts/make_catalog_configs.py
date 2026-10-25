"""Write disjoint, applicability-aware shards of the fixed full-catalog study.

The nominal configuration is loadable by the ordinary benchmark runner, but
must be split before execution: one paper preset is outside the current
DataFusionPPI adapter's constant-trial-propensity contract.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Iterator

from obs_rct_benchmark.paper_dgp import PAPER_DGPS
from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, load_config
from obs_rct_benchmark.structural_dgp import (
    STRUCTURAL_ADDENDUM_CELLS, STRUCTURAL_MECHANISMS,
)


GROUPS = ("common", "variable_propensity")
FROZEN_CATALOG_ITEMS = (
    *(("structural", f"M{number}") for number in range(12)),
    *(("addendum", f"C{number:02d}") for number in range(1, 11)),
    *(("paper", name) for name in (
        "cornet_official_n_conf", "experimental_grounding_kps",
        "integrative_r_learner_kps_normal_target_extension",
        "integrative_r_learner_kps_code", "integrative_r_learner_multivariate_paper",
        "elastic_integrative_gen_data_overlap", "power_likelihood_paper_text",
        "power_likelihood_supplement_code", "cvci_official_linear_bias",
    )),
)
CATALOG_METHODS = (
    "rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
    "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
    "rct_dr_ridge", "rct_dr_extra_trees", "datafusionppi", "cornet_official",
)
NOT_APPLICABLE = {
    ("paper", "integrative_r_learner_multivariate_paper"): {
        "datafusionppi": (
            "The canonical preset has covariate-dependent known RCT treatment "
            "probabilities; the locked official adapter requires a constant probability."
        ),
    },
}
MEAN_INFERENCE_EXCLUSIONS = {
    ("structural", "M11"): (
        "Canonical Student-t noise has three degrees of freedom: finite second "
        "moment but infinite fourth moment. Squared-loss variance need not be "
        "finite; mean MCSE, t intervals and mean-t p-values are withheld. "
        "All fits, raw errors, means, medians and IQRs remain descriptive."
    ),
}
FUSION_METHODS = ("cornet_official", "datafusionppi")
PRIMARY_REFERENCES = ("rct_t_extra_trees", "rct_dr_extra_trees")
ROLE_FIELDS = frozenset({
    "n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval",
    "n_target_reference",
})
PAPER_ROLE_FIELDS = ROLE_FIELDS - {"n_target_reference"}
EXPECTED_SCOPE = (
    "prespecified full catalog study; 31 canonical presets, 4 OBS/RCT budget "
    "pairs, 64 independent sample seeds, fixed public recipes; explicit "
    "adapter applicability and heavy-tail inference limits"
)


def catalog_items() -> tuple[tuple[str, str], ...]:
    return FROZEN_CATALOG_ITEMS


def scenario_group(scenario: ScenarioConfig) -> str:
    return ("variable_propensity" if (scenario.kind, scenario.id) in NOT_APPLICABLE
            else "common")


def applicable_methods(config: BenchmarkConfig, scenario: ScenarioConfig) -> tuple[str, ...]:
    unavailable = NOT_APPLICABLE.get((scenario.kind, scenario.id), {})
    return tuple(method for method in config.methods if method not in unavailable)


def mean_inference_reason(scenario: ScenarioConfig) -> str | None:
    return MEAN_INFERENCE_EXCLUSIONS.get((scenario.kind, scenario.id))


def expected_cells(config: BenchmarkConfig) -> set[tuple[str, int, str]]:
    return {(scenario.key, seed, method)
            for scenario in config.scenarios for seed in config.seeds
            for method in applicable_methods(config, scenario)}


def expected_not_applicable(config: BenchmarkConfig) -> dict[tuple[str, int, str], str]:
    return {(scenario.key, seed, method): reason
            for scenario in config.scenarios for seed in config.seeds
            for method, reason in NOT_APPLICABLE.get((scenario.kind, scenario.id), {}).items()
            if method in config.methods}


def primary_contrasts(config: BenchmarkConfig) -> set[tuple[str, str, str]]:
    return {(scenario.key, method, reference)
            for scenario in config.scenarios if mean_inference_reason(scenario) is None
            for method in FUSION_METHODS if method in applicable_methods(config, scenario)
            for reference in PRIMARY_REFERENCES if reference in config.methods}


def shard_config(config: BenchmarkConfig, seed: int, group: str) -> BenchmarkConfig:
    if isinstance(seed, bool) or seed not in config.seeds or group not in GROUPS:
        raise ValueError("shard requires one declared sample seed and a known group")
    scenarios = tuple(s for s in config.scenarios if scenario_group(s) == group)
    if not scenarios:
        raise ValueError("shard group has no declared scenarios")
    methods = applicable_methods(config, scenarios[0])
    if any(applicable_methods(config, scenario) != methods for scenario in scenarios):
        raise ValueError("a shard group must share one applicability contract")
    options = {name: dict(value) for name, value in config.method_options.items() if name in methods}
    return replace(config, scenarios=scenarios, seeds=(seed,), methods=methods,
                   method_options=options)


def iter_shards(config: BenchmarkConfig) -> Iterator[tuple[str, BenchmarkConfig]]:
    present = {scenario_group(scenario) for scenario in config.scenarios}
    for seed in config.seeds:
        for group in GROUPS:
            if group in present:
                yield f"group-{group}-seed-{seed:05d}.toml", shard_config(config, seed, group)


def validate_catalog_study(config: BenchmarkConfig) -> None:
    """Reject accidental changes to the declared design before splitting it."""
    registered = tuple(("structural", name) for name in STRUCTURAL_MECHANISMS) + tuple(
        ("addendum", name) for name in STRUCTURAL_ADDENDUM_CELLS
    ) + tuple(("paper", name) for name in PAPER_DGPS)
    if registered != FROZEN_CATALOG_ITEMS or len(FROZEN_CATALOG_ITEMS) != 31:
        raise ValueError("canonical registry differs from the frozen 31 preset identities")
    if (config.seeds != tuple(range(20000, 20064))
            or config.methods != CATALOG_METHODS
            or config.reference_method != "rct_dr_extra_trees"
            or config.purpose != "reference" or config.scope != EXPECTED_SCOPE):
        raise ValueError("expected the fixed 64-seed, ten-recipe catalog protocol")
    expected = [(kind, name, f"o{obs}_r{trial}") for kind, name in catalog_items()
                for obs in (512, 2048) for trial in (128, 512)]
    actual = [(s.kind, s.id, s.label) for s in config.scenarios]
    if actual != expected:
        raise ValueError("expected all 31 presets in canonical order and all four budget labels")
    for scenario in config.scenarios:
        p = scenario.parameters
        fields = PAPER_ROLE_FIELDS if scenario.kind == "paper" else ROLE_FIELDS
        if (set(p) != fields or p["n_obs"] not in (512, 2048)
                or p["n_rct_train"] not in (128, 512)
                or p["n_rct_tune"] != 32 or p["n_rct_test"] != 64
                or p["n_eval"] != 2048 or p.get("n_target_reference", 0) != 0
                or scenario.label != f"o{p['n_obs']}_r{p['n_rct_train']}"):
            raise ValueError("only the declared sample-role budgets may override a preset")
    expected_options = {
        name: ({"ridge_alpha": 1.0} if name.endswith("ridge") else
               {"n_estimators": 100, "min_samples_leaf": 5})
        for name in CATALOG_METHODS if name not in FUSION_METHODS
    }
    expected_options.update({
        "datafusionppi": {"learner": "DR", "covariate_shift": True},
        "cornet_official": {
            "source_epochs": 500, "correction_epochs": 200, "width": 25,
            "standardize": True, "learning_rate": .1,
            "outcome_penalty": .1, "correction_penalty": 0.,
        },
    })
    if dict(config.method_options) != expected_options:
        raise ValueError("method recipes differ from the fixed catalog protocol")


def _write_config(payload: dict, path: Path) -> None:
    def value(item):
        return json.dumps(item, ensure_ascii=False, allow_nan=False)
    lines = [f"schema_version = {payload['schema_version']}", "", "[run]"]
    lines.extend(f"{key} = {value(item)}" for key, item in payload["run"].items())
    for name, options in payload.get("method_options", {}).items():
        lines.extend(["", f"[method_options.{name}]"])
        lines.extend(f"{key} = {value(item)}" for key, item in options.items())
    for scenario in payload["scenarios"]:
        lines.extend(["", "[[scenarios]]"])
        lines.extend(f"{key} = {value(item)}" for key, item in scenario.items() if key != "parameters")
        lines.append("[scenarios.parameters]")
        lines.extend(f"{key} = {value(item)}" for key, item in scenario["parameters"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_shards(config: BenchmarkConfig, output: Path, *, study_path: Path | None = None) -> dict:
    """Write only a validated instance of the frozen full-catalog study."""
    validate_catalog_study(config)
    return _write_shards_unchecked(config, output, study_path=study_path)


def _write_shards_unchecked(config: BenchmarkConfig, output: Path, *, study_path: Path | None = None) -> dict:
    """Internal serializer, also used by small hand-calculated contract tests."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("shard output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for group in GROUPS:
        scenarios = tuple(s for s in config.scenarios if scenario_group(s) == group)
        if scenarios:
            groups[group] = {"scenario_keys": [s.key for s in scenarios],
                             "methods": list(applicable_methods(config, scenarios[0]))}
    shards = []
    for index, (name, part) in enumerate(iter_shards(config)):
        path = output / name
        _write_config(part.payload(), path)
        shards.append({"index": index, "path": name, "seed": part.seeds[0],
                       "group": scenario_group(part.scenarios[0]),
                       "scenario_count": len(part.scenarios),
                       "expected_cells": len(part.scenarios) * len(part.methods),
                       "config_hash": part.config_hash, "config_sha256": _file_hash(path)})
    exclusions = [{"scenario": scenario.key, "method": method,
                   "sample_seed_count": len(config.seeds), "reason": reason}
                  for scenario in config.scenarios
                  for method, reason in NOT_APPLICABLE.get((scenario.kind, scenario.id), {}).items()
                  if method in config.methods]
    inference = [{"scenario": scenario.key, "reason": reason}
                 for scenario in config.scenarios
                 if (reason := mean_inference_reason(scenario)) is not None]
    n_valid = sum(len(applicable_methods(config, scenario)) for scenario in config.scenarios) * len(config.seeds)
    n_nominal = len(config.scenarios) * len(config.seeds) * len(config.methods)
    manifest = {
        "schema_version": 1, "study_config_hash": config.config_hash,
        "study_config_sha256": None if study_path is None else _file_hash(study_path),
        "sharding_source_sha256": _file_hash(Path(__file__)),
        "catalog_order": [f"{kind}:{name}" for kind, name in dict.fromkeys(
            (s.kind, s.id) for s in config.scenarios)],
        "counts": {"conditions": len(config.scenarios), "sample_seeds": len(config.seeds),
                   "nominal_fit_cells": n_nominal, "not_applicable_fit_cells": n_nominal-n_valid,
                   "applicable_fit_cells": n_valid, "mean_t_primary_contrasts": len(primary_contrasts(config))},
        "groups": groups, "not_applicable": exclusions,
        "mean_inference_exclusions": inference, "shards": shards,
        "interpretation": "Disjoint requested cell rectangles; shared sample seeds across compatibility groups are intentional.",
    }
    (output / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/catalog_study.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_catalog_study(config)
    manifest = write_shards(config, args.output, study_path=args.config)
    print(json.dumps({"shards": len(manifest["shards"]), **manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
