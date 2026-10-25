"""Authenticate disjoint recruitment-study shards and retain every condition.

This aggregates saved metrics and query costs; deterministic prediction replay
is performed separately with adaptive_runner --verify. No ranking or selected
scenario determines which rows, contrasts or plots are included.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from obs_rct_benchmark.adaptive_runner import (
    _expected, _validate_records, load_adaptive_config,
)
from obs_rct_benchmark.evaluation import mean_ci, summarize_records
from obs_rct_benchmark.runner import (
    _hash, _load_json, _read_records, resolved_scenarios, source_provenance,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_records(config, runs: list[Path]) -> tuple[list[dict], list[dict]]:
    if not runs:
        raise ValueError("at least one shard is required")
    expected_pools = resolved_scenarios(SimpleNamespace(
        scenarios=tuple(config.pool_scenario(x) for x in config.scenarios)))
    actual_source = source_provenance()
    shared_fingerprint = None
    seen_seeds: set[int] = set()
    records, inputs = [], []
    for ordinal, run in enumerate(runs):
        run = Path(run)
        manifest_path, records_path = run/"run_manifest.json", run/"records.jsonl"
        manifest = _load_json(manifest_path)
        seeds = manifest["config"]["run"]["seeds"]
        if (not seeds or len(set(seeds)) != len(seeds)
                or not set(seeds) <= set(config.seeds)):
            raise ValueError("shard seeds must be unique members of the declared study")
        if set(seeds) & seen_seeds:
            raise ValueError("adaptive shards have overlapping seed sets")
        seen_seeds.update(seeds)
        shard = replace(config, seeds=tuple(seeds))
        if manifest["config"] != shard.payload() or manifest["config_hash"] != shard.config_hash:
            raise ValueError("adaptive shard configuration differs from the declared study")
        if manifest["resolved_pools"] != expected_pools:
            raise ValueError("adaptive shard pool controls differ from the declared study")
        source = manifest["source"]
        if (source.get("aggregate_sha256") != _hash(source["package_file_sha256"])
                or source["package_file_sha256"] != actual_source["package_file_sha256"]):
            raise ValueError("adaptive shard source does not match the analysis package")
        bound = {key: manifest[key] for key in
                 ("source", "environment", "external_methods", "custom_methods", "resolved_pools")}
        fingerprint = _hash(bound)
        if fingerprint != manifest["source_fingerprint"]:
            raise ValueError("invalid adaptive source/environment fingerprint")
        if shared_fingerprint is not None and fingerprint != shared_fingerprint:
            raise ValueError("adaptive shards use different source or numerical environments")
        shared_fingerprint = fingerprint
        part = _read_records(records_path)
        seen = _validate_records(shard, part)
        if len(seen) != len(_expected(shard)):
            raise ValueError("adaptive shard has missing requested terminal records")
        records.extend(part)
        inputs.append({"input_ordinal": ordinal, "sample_seeds": seeds,
                       "manifest_sha256": file_hash(manifest_path),
                       "records_sha256": file_hash(records_path),
                       "config_hash": shard.config_hash, "source_fingerprint": fingerprint})
    if seen_seeds != set(config.seeds):
        raise ValueError("adaptive shards do not cover every declared sample seed")
    expected = {(s, seed, method) for s in config.scenario_keys
                for seed in config.seeds for method in config.methods}
    actual = [(row["scenario"], row["seed"], row["method"]) for row in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("adaptive records do not form the exact declared study grid")
    return records, inputs


def analyze_records(config, records: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    summary = summarize_records(records, scenarios=config.scenario_keys, seeds=config.seeds,
                                methods=config.methods, reference_method="uniform__"+config.estimators[0])
    comparisons = {}
    for estimator in config.estimators:
        subset = [row for row in records if row["estimator"] == estimator]
        partial = summarize_records(subset, scenarios=config.scenario_keys, seeds=config.seeds,
                    methods=tuple(p+"__"+estimator for p in config.policies),
                    reference_method="uniform__"+estimator)
        comparisons[estimator] = partial["paired_comparisons"]
    summary["within_estimator_policy_comparisons"] = comparisons
    summary["config"] = config.payload()
    by_cell = {(row["scenario"], row["seed"], row["method"]): row for row in records}
    scores, contrasts = [], []
    for scenario in config.scenarios:
        for budget in config.budgets:
            key = scenario.key+f"/budget_{budget}"
            for estimator in config.estimators:
                for policy in config.policies:
                    method = policy+"__"+estimator
                    rows = [by_cell[(key, seed, method)] for seed in config.seeds]
                    good = [row for row in rows if row["status"] == "ok"]
                    estimate = mean_ci(row["metrics"]["cate_mse"] for row in good)
                    interval = estimate["ci95"] or [None, None]
                    scores.append({"scenario": scenario.key, "budget": budget,
                                   "estimator": estimator, "policy": policy,
                                   "expected_seeds": len(config.seeds), "successful": len(good),
                                   "failed": len(rows)-len(good), "missing": 0,
                                   "mean_mse": estimate["mean"], "mcse_mse": estimate["standard_error"],
                                   "ci95_lower": interval[0], "ci95_upper": interval[1],
                                   "mean_scope": "successful cells; descriptive if incomplete"})
                    if policy == "uniform":
                        continue
                    comparison = comparisons[estimator][method]["by_scenario"][key]
                    metric = comparison["metrics"]["cate_mse"]
                    interval = metric["ci95"] or [None, None]
                    contrasts.append({"scenario": scenario.key, "budget": budget,
                                      "estimator": estimator, "policy": policy, "reference": "uniform",
                                      "complete": comparison["eligible_for_comparison"],
                                      "expected_pairs": comparison["expected_seed_blocks"],
                                      "successful_pairs": comparison["complete_seed_blocks"],
                                      "mean_difference": metric["mean"], "mcse_difference": metric["standard_error"],
                                      "ci95_lower": interval[0], "ci95_upper": interval[1],
                                      "withheld_reason": comparison["withheld_reason"]})
    summary["scores"] = scores
    summary["policy_contrasts"] = contrasts
    summary["terminal_failures"] = [dict(row) for row in records if row["status"] != "ok"]
    summary["interpretation"] = ("All predeclared conditions retained. Paired differences are policy minus uniform "
                                 "for the same estimator. Pointwise Monte Carlo intervals have no multiplicity "
                                 "adjustment and do not establish unseen-mechanism generalization.")
    return summary, scores, contrasts


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(config, scores: list[dict], contrasts: list[dict], output: Path, *, paper_assets: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    policy_colors = {"uniform": "#333333", "geometry": "#0072B2", "outcome_residual": "#D55E00"}
    policy_labels = {"uniform": "Uniform", "geometry": "Geometry", "outcome_residual": "Outcome residual"}
    estimator_labels = {"rct_t_extra_trees": "RCT-only T-learner",
                        "naive_pooled_t_extra_trees": "Naive pooled T-learner"}
    fig, axes = plt.subplots(len(config.estimators), len(config.scenarios),
                             figsize=(4*len(config.scenarios), 3.3*len(config.estimators)), squeeze=False)
    for row, estimator in enumerate(config.estimators):
        for column, scenario in enumerate(config.scenarios):
            ax = axes[row, column]
            for policy in config.policies:
                selected = sorted([x for x in scores if x["scenario"] == scenario.key
                                   and x["estimator"] == estimator and x["policy"] == policy], key=lambda x:x["budget"])
                valid = [x for x in selected if x["mean_mse"] is not None]
                if not valid:
                    ax.text(.02, .96-.07*config.policies.index(policy),
                            policy+": no successful cells", transform=ax.transAxes,
                            va="top", fontsize=7, color=policy_colors[policy])
                    continue
                ax.plot([x["budget"] for x in valid], [x["mean_mse"] for x in valid],
                        marker="o", label=policy_labels[policy], color=policy_colors[policy])
                for x in valid:
                    if x["ci95_lower"] is not None:
                        ax.vlines(x["budget"], x["ci95_lower"], x["ci95_upper"], color=policy_colors[policy], linewidth=1.3)
                    if x["failed"]:
                        ax.annotate(f"{x['failed']} failed", (x["budget"], x["mean_mse"]), fontsize=7)
            ax.set_title(scenario.id)
            ax.set_xlabel("Charged RCT outcomes (pilot included)")
            ax.set_ylabel(estimator_labels.get(estimator, estimator)+"\nCATE MSE")
            ax.set_xticks(config.budgets)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-budget recruitment: every mechanism and final estimator\nPointwise 95% seed-block Monte Carlo intervals", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output/"adaptive_budget_curves.svg")
    if paper_assets:
        fig.savefig(output/"adaptive_budget_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(config.estimators), figsize=(6.5*len(config.estimators), 6.5), squeeze=False)
    for column, estimator in enumerate(config.estimators):
        ax = axes[0, column]
        selected = [x for x in contrasts if x["estimator"] == estimator]
        for index, x in enumerate(selected):
            if x["mean_difference"] is None:
                ax.text(.02, index, "withheld: incomplete coverage", transform=ax.get_yaxis_transform(), fontsize=8)
                continue
            color = policy_colors[x["policy"]]
            ax.plot(x["mean_difference"], index, "o", color=color)
            if x["ci95_lower"] is not None:
                ax.hlines(index, x["ci95_lower"], x["ci95_upper"], color=color, linewidth=1.5)
        ax.set_yticks(np.arange(len(selected)))
        ax.set_yticklabels([x["scenario"].split(":")[-1]+f" B{x['budget']} {policy_labels[x['policy']]}" for x in selected], fontsize=8)
        ax.axvline(0, color="#444444", linewidth=.8)
        ax.set_title(estimator_labels.get(estimator, estimator))
        ax.set_xlabel("CATE MSE: policy minus uniform (negative favors policy)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Recruitment contrasts with the final estimator held fixed\nPointwise 95% intervals; no multiplicity-adjusted superiority claim", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output/"adaptive_policy_contrasts.svg")
    if paper_assets:
        fig.savefig(output/"adaptive_policy_contrasts.pdf")
    plt.close(fig)


def write_analysis(config, runs: list[Path], output: Path, *, paper_assets: bool = False) -> dict:
    records, inputs = collect_records(config, runs)
    summary, scores, contrasts = analyze_records(config, records)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("analysis output must be a fresh directory")
    summary["inputs"] = inputs
    summary["normalized_config_sha256"] = config.config_hash
    summary["protocol_sha256"] = file_hash(Path(__file__).parents[1]/"docs/ADAPTIVE.md")
    summary["analysis_source_sha256"] = file_hash(Path(__file__))
    summary["analysis_environment"] = {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}
    _write_csv(output/"scores.csv", scores)
    _write_csv(output/"policy_contrasts.csv", contrasts)
    plot_results(config, scores, contrasts, output, paper_assets=paper_assets)
    coverage = summary["coverage"]
    report = ("# Adaptive recruitment reference comparison\n\n"
              f"Terminal grid: {coverage['recorded']}/{coverage['expected']}; "
              f"successful {coverage['successful']}, failed {coverage['failed']}, missing {coverage['missing']}.\n\n"
              "[Budget curves](adaptive_budget_curves.svg) retain every mechanism, budget, policy and final estimator. "
              "[Policy contrasts](adaptive_policy_contrasts.svg) hold the final estimator fixed. "
              "The data are in [scores.csv](scores.csv), [policy_contrasts.csv](policy_contrasts.csv) and [summary.json](summary.json).\n\n"
              "Each queried outcome, including the common pilot and tuning sample, is charged. "
              "The source pool and independent evaluation rows are shared within each mechanism/seed. "
              "The outcome policy is a residual-based heuristic and pooling is naive. "
              "Pointwise paired Monte Carlo intervals have no multiplicity adjustment; favorable, adverse and unresolved directions are all retained. "
              "Fresh seeds are sampling replications on development-inspected mechanisms, not an independent mechanism challenge.\n\n"
              "Input manifests and records are authenticated by seed-grid/config/source/environment checks and file hashes in summary.json. "
              "This aggregation does not independently refit predictions; run the separate adaptive replay verifier for that check.\n")
    (output/"README.md").write_text(report, encoding="utf-8")
    summary["output_sha256"] = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paper-assets", action="store_true")
    args = parser.parse_args(argv)
    summary = write_analysis(load_adaptive_config(args.config), args.runs, args.output,
                             paper_assets=args.paper_assets)
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"]}, sort_keys=True))
    return int(summary["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
