"""Resumable configuration runner for the two-stage recruitment track."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .adaptive import (
    POLICY_NAMES, _indices, _positive_integer, replay_adaptive_records,
    run_adaptive_problem,
)
from .evaluation import summarize_records
from .methods import custom_provenance, make_method, method_names
from .runner import (
    ScenarioConfig, _append_record, _generate, _hash, _load_json, _read_records,
    _result_readme, _run_lock, _validate_public_values, _write_json, _write_text,
    environment_provenance, external_method_provenance, method_seed,
    resolve_inference_policy, resolved_scenarios, source_provenance,
)


@dataclass(frozen=True)
class AdaptiveConfig:
    scenarios: tuple[ScenarioConfig, ...]
    seeds: tuple[int, ...] = (0, 1)
    policies: tuple[str, ...] = POLICY_NAMES
    estimators: tuple[str, ...] = ("rct_t_extra_trees", "naive_pooled_t_extra_trees")
    budgets: tuple[int, ...] = (128, 512)
    pilot_size: int = 32
    tune_size: int = 16
    pool_size: int = 2048
    method_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    purpose: str = "smoke"
    scope: str = "two-stage recruitment; development-inspected mechanisms"

    def __post_init__(self) -> None:
        if not self.scenarios or len({x.key for x in self.scenarios}) != len(self.scenarios):
            raise ValueError("adaptive scenarios must be nonempty with unique keys")
        forbidden = {"n_rct_train", "n_rct_tune", "n_rct_test", "n_target_reference"}
        for scenario in self.scenarios:
            if scenario.kind not in {"structural", "addendum"}:
                raise ValueError("adaptive track currently supports structural and addendum mechanisms")
            if forbidden & scenario.parameters.keys():
                raise ValueError("adaptive pool and trial-role sizes are controlled by the track")
        for values, name in ((self.seeds, "seeds"), (self.policies, "policies"),
                             (self.estimators, "estimators"), (self.budgets, "budgets")):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in self.seeds):
            raise ValueError("seeds must be integers in [0,2**32)")
        if any(x not in POLICY_NAMES for x in self.policies) or "uniform" not in self.policies:
            raise ValueError("registered policies and the uniform reference are required")
        if any(x not in method_names() for x in self.estimators):
            raise ValueError("estimators must be registered public methods")
        for value, name in ((self.pilot_size, "pilot_size"), (self.tune_size, "tune_size"),
                            (self.pool_size, "pool_size")):
            _positive_integer(value, name)
        for value in self.budgets:
            _positive_integer(value, "budget")
        if not self.tune_size < self.pilot_size < min(self.budgets) or max(self.budgets) > self.pool_size:
            raise ValueError("require tune_size < pilot_size < every budget <= pool_size")
        if self.purpose not in {"smoke", "reference"} or not self.scope:
            raise ValueError("purpose must be smoke/reference and scope must be nonempty")
        if set(self.method_options) - set(self.estimators):
            raise ValueError("method options contain an unrequested estimator")
        _validate_public_values(self.payload())
        for name in self.estimators:
            make_method(name, **dict(self.method_options.get(name, {})))

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(policy+"__"+estimator for policy in self.policies for estimator in self.estimators)

    @property
    def scenario_keys(self) -> tuple[str, ...]:
        return tuple(s.key+f"/budget_{budget}" for s in self.scenarios for budget in self.budgets)

    def pool_scenario(self, scenario: ScenarioConfig) -> ScenarioConfig:
        # Extra fixed trial roles are evaluator-owned and are never supplied to
        # policy/final estimator. They satisfy the existing generator contract.
        parameters = dict(scenario.parameters, n_rct_train=self.pool_size,
                          n_rct_tune=4, n_rct_test=4, n_target_reference=0)
        return replace(scenario, parameters=parameters)

    def payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "track": "adaptive_fixed_pool_v1",
                "run": {"seeds": list(self.seeds), "policies": list(self.policies),
                        "estimators": list(self.estimators), "budgets": list(self.budgets),
                        "pilot_size": self.pilot_size, "tune_size": self.tune_size,
                        "pool_size": self.pool_size, "purpose": self.purpose, "scope": self.scope},
                "scenarios": [dict(kind=x.kind, id=x.id, parameters=dict(x.parameters),
                                   **({"label": x.label} if x.label is not None else {}),
                                   **({"inference_policy": dict(x.inference_policy)}
                                      if x.inference_policy else {}))
                              for x in self.scenarios],
                "method_options": {k: dict(v) for k, v in self.method_options.items()}}

    @property
    def config_hash(self) -> str:
        return _hash(self.payload())


def load_adaptive_config(path: str | Path) -> AdaptiveConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) - {"schema_version", "run", "scenarios", "method_options"} or raw.get("schema_version") != 1:
        raise ValueError("invalid adaptive schema or top-level field")
    run = dict(raw.get("run", {}))
    allowed = {"seeds", "policies", "estimators", "budgets", "pilot_size", "tune_size",
               "pool_size", "purpose", "scope"}
    if set(run)-allowed:
        raise ValueError("unknown adaptive run field")
    for name in ("seeds", "policies", "estimators", "budgets"):
        if name in run:
            run[name] = tuple(run[name])
    scenarios = []
    for item in raw.get("scenarios", []):
        if set(item)-{"kind", "id", "parameters", "label", "inference_policy"}:
            raise ValueError("unknown adaptive scenario field")
        scenarios.append(ScenarioConfig(**item))
    return AdaptiveConfig(tuple(scenarios), method_options=raw.get("method_options", {}), **run)


def adaptive_inference_policy(config: AdaptiveConfig) -> dict[str, Any]:
    """Resolve metric declarations for each original scenario and budget.

    Estimator names identify the fitting recipes. Recruitment-policy prefixes
    and pool-only sample sizes do not change the declarations being resolved.
    """
    policies = {}
    for scenario in config.scenarios:
        resolved = resolve_inference_policy(scenario, config.estimators, config.method_options)
        for budget in config.budgets:
            policies[scenario.key+f"/budget_{budget}"] = resolved
    return policies


def _identity(config: AdaptiveConfig, scenario: str, seed: int, policy: str,
              budget: int, estimator: str) -> dict[str, Any]:
    full = scenario+f"/budget_{budget}"
    return {"schema_version": 1, "config_hash": config.config_hash,
            "record_id": _hash(["adaptive_record_v1", config.config_hash, full, seed, policy, estimator]),
            "scenario": full, "seed": seed, "policy": policy, "budget": budget,
            "estimator": estimator, "method": policy+"__"+estimator,
            "policy_seed": method_seed(scenario, seed, "recruitment_"+policy),
            "fit_seed": method_seed(full, seed, "fit_"+estimator),
            "planned_pilot_size": config.pilot_size, "planned_tune_size": config.tune_size}


def _expected(config: AdaptiveConfig) -> dict[str, dict[str, Any]]:
    rows = [_identity(config, s.key, seed, p, b, e) for s in config.scenarios
            for seed in config.seeds for p in config.policies
            for b in config.budgets for e in config.estimators]
    return {x["record_id"]: x for x in rows}


def _validate_records(config: AdaptiveConfig, records: list[dict[str, Any]]) -> set[str]:
    expected = _expected(config)
    seen = set()
    for row in records:
        if row.get("stage") == "recruitment" and row.get("status") != "failed":
            raise ValueError("recruitment-stage records must be failed cells")
        key = row.get("record_id")
        if key in seen or key not in expected:
            raise ValueError("duplicate or unrequested adaptive record")
        if any(row.get(k) != value for k, value in expected[key].items()):
            raise ValueError("adaptive record identity or budget changed")
        ids = _indices(row.get("query_indices"), size=config.pool_size)
        spent = len(ids)
        cost_names = ("query_cost", "pilot_cost", "tune_cost", "train_cost")
        if any(isinstance(row.get(k), bool) or not isinstance(row.get(k), int) or row[k] < 0 for k in cost_names):
            raise ValueError("adaptive outcome costs must be nonnegative integers")
        if (spent > row["budget"] or row["query_cost"] != spent
                or row["pilot_cost"] != min(spent, config.pilot_size)
                or row["tune_cost"] != min(spent, config.tune_size)
                or row["train_cost"]+row["tune_cost"] != spent):
            raise ValueError("adaptive record query cost changed")
        if row.get("stage") == "simulation":
            if spent != 0 or row["status"] != "failed":
                raise ValueError("simulation failure cannot contain acquired outcomes")
        elif row.get("stage") != "recruitment" and spent != row["budget"]:
            raise ValueError("completed acquisition must spend its budget")
        seen.add(key)
    summarize_records(records, scenarios=config.scenario_keys, seeds=config.seeds,
                      methods=config.methods, reference_method="uniform__"+config.estimators[0],
                      scenario_inference=adaptive_inference_policy(config))
    return seen


def _provenance(config: AdaptiveConfig) -> tuple[dict[str, Any], str]:
    source = source_provenance()
    pools = tuple(config.pool_scenario(s) for s in config.scenarios)
    # resolved_scenarios only needs a config-like object with .scenarios.
    class PoolCollection:
        scenarios = pools
    provenance = {"source": source, "environment": environment_provenance(),
                  "external_methods": external_method_provenance(config.estimators),
                  "custom_methods": {k: v for k, v in custom_provenance().items() if k in config.estimators},
                  "resolved_pools": resolved_scenarios(PoolCollection())}
    _validate_public_values(provenance["custom_methods"])
    return provenance, _hash(provenance)


def run_adaptive_benchmark(config: AdaptiveConfig, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("adaptive output must be empty unless resuming")
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        inference = adaptive_inference_policy(config)
        provenance, fingerprint = _provenance(config)
        path = output/"run_manifest.json"
        if resume:
            manifest = _load_json(path)
            if (manifest.get("config") != config.payload()
                    or manifest.get("source_fingerprint") != fingerprint
                    or manifest.get("scenario_inference") != inference):
                raise ValueError("adaptive resume config or source provenance does not match")
        else:
            manifest = dict(provenance, config=config.payload(), config_hash=config.config_hash,
                            source_fingerprint=fingerprint, created_utc=datetime.now(timezone.utc).isoformat(),
                            scenario_inference=inference,
                            purpose=config.purpose, scope=config.scope,
                            information_contract="fixed-pool two-stage recruitment; one outcome per queried row",
                            interpretation="Declared reference comparison; development-inspected mechanisms; no automatic confirmatory claim"
                            if config.purpose == "reference" else "Development smoke; not method-ranking evidence")
            _write_json(path, manifest)
        records = _read_records(output/"records.jsonl")
        seen = _validate_records(config, records)
        expected = _expected(config)
        for scenario in config.scenarios:
            for seed in config.seeds:
                pending = {(p, b, e) for p in config.policies for b in config.budgets for e in config.estimators
                           if _identity(config, scenario.key, seed, p, b, e)["record_id"] not in seen}
                if not pending:
                    continue
                try:
                    problem = _generate(config.pool_scenario(scenario), seed)
                except Exception as exc:
                    generated = [dict(_identity(config, scenario.key, seed, p, b, e), status="failed",
                                      stage="simulation", error_type=type(exc).__name__, metrics=None,
                                      query_indices=[], query_cost=0, pilot_cost=0, tune_cost=0,
                                      train_cost=0) for p, b, e in sorted(pending)]
                else:
                    generated = run_adaptive_problem(problem, budgets=config.budgets,
                        pilot_size=config.pilot_size, tune_size=config.tune_size,
                        policies=config.policies, estimators=config.estimators, seed=seed,
                        method_options=config.method_options, scenario=scenario.key, requested_cells=pending)
                for row in generated:
                    base = _identity(config, scenario.key, seed, row["policy"], row["budget"], row["estimator"])
                    row = dict(row, **base)
                    if row["record_id"] not in expected:
                        raise ValueError("adaptive evaluator returned an unrequested cell")
                    _append_record(output/"records.jsonl", row)
                    records.append(row)
        _validate_records(config, records)
        summary = summarize_records(records, scenarios=config.scenario_keys, seeds=config.seeds,
                                    methods=config.methods, reference_method="uniform__"+config.estimators[0],
                                    scenario_inference=inference)
        # Recruitment effects use a same-estimator uniform reference, rather
        # than conflating policy changes with OBS use in a different estimator.
        summary["within_estimator_policy_comparisons"] = {}
        for estimator in config.estimators:
            subset = [row for row in records if row["estimator"] == estimator]
            comparison = summarize_records(subset, scenarios=config.scenario_keys, seeds=config.seeds,
                            methods=tuple(p+"__"+estimator for p in config.policies),
                            reference_method="uniform__"+estimator,
                            scenario_inference=inference)
            summary["within_estimator_policy_comparisons"][estimator] = comparison["paired_comparisons"]
        summary.update(scope=config.scope, purpose=config.purpose, config_hash=config.config_hash,
                       run_id=config.config_hash[:16], source_fingerprint=fingerprint)
        summary["scenario_definitions"] = {}
        for scenario in config.scenarios:
            pool = provenance["resolved_pools"][scenario.key]["config"]
            for budget in config.budgets:
                summary["scenario_definitions"][scenario.key+f"/budget_{budget}"] = {
                    "sample_sizes": {"n_obs": pool["n_obs"], "n_rct_train": budget-config.tune_size,
                                     "n_rct_tune": config.tune_size, "n_rct_test": 4,
                                     "n_target_reference": 0, "n_eval": pool["n_eval"]},
                    "pool_size": config.pool_size,
                }
        _write_json(output/"summary.json", summary)
        report = _result_readme(summary)
        report += ("\nRecruitment uses one common uniform pilot and fixed policy prefixes. "
                   "Every queried outcome, including pilot and tuning labels, is charged. "
                   "`query_indices` records the exact no-replacement acquisition order. "
                   "The JSON also gives policy-versus-uniform contrasts separately for each final estimator. "
                   "Pooling is naive; no acquisition-optimality claim is made.\n")
        report += _policy_report(summary)
        _write_text(output/"README.md", report)
        return summary


def _policy_report(summary: Mapping[str, Any]) -> str:
    lines = ["\n## Recruitment effects with the final estimator held fixed\n",
             "Differences are policy minus uniform recruitment for the same final estimator. "
             "Eligible intervals are pointwise paired Monte Carlo summaries without multiplicity adjustment.\n",
             "| Estimator | Scenario | Policy | Mean MSE difference | 95% interval | Complete / expected seeds | MSE inference |",
             "| --- | --- | --- | ---: | --- | ---: | --- |"]
    for estimator, methods in summary["within_estimator_policy_comparisons"].items():
        for method, comparisons in methods.items():
            for scenario, comparison in comparisons["by_scenario"].items():
                metric = comparison["metrics"]["cate_mse"]
                mean = "withheld" if metric["mean"] is None else f"{metric['mean']:.6g}"
                ci = metric["ci95"]
                interval = "unavailable" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
                state = ("eligible" if metric["eligible_for_inference"] else
                         f"withheld: {metric['withheld_reason']}")
                policy = method.split("__", 1)[0]
                lines.append(f"| `{estimator}` | `{scenario}` | `{policy}` | {mean} | {interval} | "
                             f"{comparison['complete_seed_blocks']} / {comparison['expected_seed_blocks']} | {state} |")
    return "\n".join(lines)+"\n"


def verify_adaptive_run(config: AdaptiveConfig, output: str | Path) -> dict[str, int]:
    """Regenerate all pools and replay saved policy decisions/final predictions."""
    output = Path(output)
    manifest = _load_json(output/"run_manifest.json")
    _, fingerprint = _provenance(config)
    if (manifest.get("config") != config.payload()
            or manifest.get("source_fingerprint") != fingerprint
            or manifest.get("scenario_inference") != adaptive_inference_policy(config)):
        raise ValueError("adaptive replay config or provenance does not match")
    records = _read_records(output/"records.jsonl")
    seen = _validate_records(config, records)
    if len(seen) != len(_expected(config)):
        raise ValueError("adaptive replay requires all requested terminal records")
    for scenario in config.scenarios:
        for seed in config.seeds:
            prefix = scenario.key+"/budget_"
            selected = [row for row in records if row["seed"] == seed and row["scenario"].startswith(prefix)]
            problem = _generate(config.pool_scenario(scenario), seed)
            replay_adaptive_records(problem, selected, pilot_size=config.pilot_size,
                                    tune_size=config.tune_size, method_options=config.method_options)
    return {"records": len(records), "successful_replayed": sum(row["status"] == "ok" for row in records),
            "terminal_failures": sum(row["status"] != "ok" for row in records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="replay all saved queries, fits and metrics")
    args = parser.parse_args(argv)
    config = load_adaptive_config(args.config)
    if args.verify:
        result = verify_adaptive_run(config, args.output)
        print(json.dumps(result, sort_keys=True))
        return int(result["terminal_failures"] > 0)
    summary = run_adaptive_benchmark(config, args.output, resume=args.resume)
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"]}, sort_keys=True))
    return int(summary["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
