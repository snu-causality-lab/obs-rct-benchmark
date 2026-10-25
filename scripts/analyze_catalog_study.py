"""Authenticate and summarize the complete, compatibility-declared catalog study.

This checks recorded-artifact consistency and aggregates saved metric values.
It does not independently establish the correctness of fitted predictions.
The separate reproducibility checker refits a prespecified subset.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from obs_rct_benchmark.runner import (
    BenchmarkConfig, _hash, _load_json, _read_records, environment_provenance,
    external_method_provenance, load_config, resolved_scenarios, source_provenance,
)


FUSION = ("cornet_official", "datafusionppi")
CONTROLS = ("rct_t_extra_trees", "rct_dr_extra_trees")
PPI_UNSUPPORTED = ("paper", "integrative_r_learner_multivariate_paper")
PPI_REASON = "The pinned official adapter requires constant trial propensity; this preset has covariate-dependent trial propensity."
HEAVY_TAIL_PRESET = ("structural", "M11")
HEAVY_TAIL_REASON = "Canonical Student-t(3) outcome noise need not give finite variance for squared-error contrasts; mean-t MCSE, intervals and tests are not reported."


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def applicable_methods(config: BenchmarkConfig, scenario) -> tuple[str, ...]:
    """Independent, explicit applicability rule fixed before study execution."""
    return tuple(method for method in config.methods
                 if not ((scenario.kind, scenario.id) == PPI_UNSUPPORTED
                         and method == "datafusionppi"))


def inference_reason(scenario) -> str | None:
    return HEAVY_TAIL_REASON if (scenario.kind, scenario.id) == HEAVY_TAIL_PRESET else None


def validate_study(config: BenchmarkConfig) -> None:
    """Use the published design validator, while checking inference independently."""
    path = Path(__file__).with_name("make_catalog_configs.py")
    spec = importlib.util.spec_from_file_location("catalog_plan_dependency", path)
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    plan.validate_catalog_study(config)
    if expected_cells(config) != plan.expected_cells(config):
        raise ValueError("analysis applicability rules disagree with the declared protocol")
    independent_primary = {(scenario.key, fusion, control) for scenario in config.scenarios
                           if inference_reason(scenario) is None
                           for fusion in FUSION if fusion in applicable_methods(config, scenario)
                           for control in CONTROLS if control in config.methods}
    if independent_primary != plan.primary_contrasts(config):
        raise ValueError("analysis inference exclusions disagree with the declared protocol")


def expected_cells(config: BenchmarkConfig) -> set[tuple[str, int, str]]:
    return {(scenario.key, seed, method) for scenario in config.scenarios
            for seed in config.seeds for method in applicable_methods(config, scenario)}


def not_applicable_rows(config: BenchmarkConfig) -> list[dict]:
    return [{"scenario": scenario.key, "method": method, "seed_count": len(config.seeds),
             "reason": PPI_REASON}
            for scenario in config.scenarios for method in config.methods
            if method not in applicable_methods(config, scenario)]


def summarize_values(values: Iterable[float], *, inferential: bool = True) -> dict:
    """Sample summaries; Student-t inference requires finite variance."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("summary values must be finite scalar observations")
    result = {"n": len(data), "mean": None, "mcse": None, "ci95": None,
              "p_two_sided_zero": None, "sample_median": None,
              "sample_q10": None, "sample_q25": None, "sample_q75": None, "sample_q90": None}
    if not len(data):
        return result
    result.update(mean=float(np.mean(data)), sample_median=float(np.median(data)),
                  sample_q10=float(np.quantile(data, .1)), sample_q25=float(np.quantile(data, .25)),
                  sample_q75=float(np.quantile(data, .75)), sample_q90=float(np.quantile(data, .9)))
    if len(data) < 2 or not inferential:
        return result
    se = float(np.std(data, ddof=1) / math.sqrt(len(data)))
    half = float(stats.t.ppf(.975, len(data) - 1)) * se
    p = (1.0 if result["mean"] == 0 else 0.0) if se == 0 else float(
        2 * stats.t.sf(abs(result["mean"] / se), len(data) - 1))
    result.update(mcse=se, ci95=[result["mean"] - half, result["mean"] + half],
                  p_two_sided_zero=p)
    return result


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = list(pvalues)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Holm inputs must be finite probabilities")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted, running = [0.] * len(values), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def apply_primary_family(contrasts: list[dict]) -> bool:
    """Only the prospectively eligible family is tested; missing pairs never shrink it."""
    eligible = [row for row in contrasts if row["inference_eligible"]]
    complete = bool(eligible) and all(row["complete"] and row["p_two_sided_zero"] is not None
                                      for row in eligible)
    values = holm_adjust(row["p_two_sided_zero"] for row in eligible) if complete else []
    for row in contrasts:
        row.update(holm_p=None, holm_reject_05=None)
        row["decision"] = ("not_assessed_heavy_tail" if not row["inference_eligible"]
                           else "withheld_incomplete_family")
    for index, row in enumerate(eligible):
        if complete:
            row["holm_p"] = values[index]
            row["holm_reject_05"] = values[index] <= .05
            row["decision"] = ("favorable" if row["mean"] < 0 else "adverse") if values[index] <= .05 else "unresolved"
    return complete


def _validate_metrics(row: dict) -> None:
    if row.get("status") not in {"ok", "failed"}:
        raise ValueError("unknown terminal status")
    if row["status"] == "failed":
        if row.get("metrics") is not None:
            raise ValueError("failed record must not contain metrics")
        return
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("successful record requires metrics")
    mse, root = metrics.get("cate_mse"), metrics.get("root_pehe")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in (mse, root))
            or not math.isclose(root * root, mse, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("invalid terminal metrics")


def _shard_config(config: BenchmarkConfig, payload: dict) -> BenchmarkConfig:
    """Reconstruct each permissible rectangular split from the canonical design."""
    raw_scenarios = payload.get("scenarios", [])
    raw_run = payload.get("run", {})
    seeds = raw_run.get("seeds", [])
    if (not seeds or len(set(seeds)) != len(seeds)
            or not set(seeds) <= set(config.seeds)):
        raise ValueError("shard contains duplicate or undeclared seeds")
    canonical = {scenario.key: scenario for scenario in config.scenarios}
    by_payload = {_hash(configured): key for key, configured in zip(
        canonical, config.payload()["scenarios"])}
    selected = []
    for raw in raw_scenarios:
        key = by_payload.get(_hash(raw))
        if key is None or key in {scenario.key for scenario in selected}:
            raise ValueError("shard scenarios differ from the canonical design")
        selected.append(canonical[key])
    if not selected:
        raise ValueError("shard scenarios must not be empty")
    methods = applicable_methods(config, selected[0])
    if any(applicable_methods(config, scenario) != methods for scenario in selected):
        raise ValueError("a shard must have one predeclared applicability group")
    options = {name: dict(options) for name, options in config.method_options.items()
               if name in methods}
    shard = replace(config, scenarios=tuple(selected), seeds=tuple(seeds), methods=methods,
                    method_options=options)
    if payload != shard.payload():
        raise ValueError("shard recipe, scope or configuration differs from the declared study")
    return shard


def collect_records(config: BenchmarkConfig, runs: list[Path], *, require_full: bool = True,
                    identity_sink: dict | None = None) -> tuple[dict, list[dict]]:
    if not runs:
        raise ValueError("at least one shard is required")
    expected = expected_cells(config)
    declared, collected, provenance = set(), {}, []
    source_now, environment_now = source_provenance(), environment_provenance()
    shared_source_environment = None
    for ordinal, run in enumerate(runs):
        run = Path(run)
        manifest_path, records_path = run / "run_manifest.json", run / "records.jsonl"
        manifest = _load_json(manifest_path)
        shard = _shard_config(config, manifest["config"])
        if manifest.get("config_hash") != shard.config_hash:
            raise ValueError("invalid shard configuration hash")
        cells = expected_cells(shard)
        if declared.intersection(cells):
            raise ValueError("input shard declarations overlap, even if their records differ")
        declared.update(cells)
        resolved = resolved_scenarios(shard)
        if manifest.get("resolved_scenarios") != resolved:
            raise ValueError("resolved generator controls differ from the declared study")
        source = manifest["source"]
        if (source.get("aggregate_sha256") != _hash(source["package_file_sha256"])
                or source != source_now):
            raise ValueError("recorded producer source differs from the analysis package")
        if manifest.get("environment") != environment_now:
            raise ValueError("recorded numerical environment differs from the analysis environment")
        external = external_method_provenance(shard.methods)
        if manifest.get("external_methods") != external:
            raise ValueError("recorded external implementations differ from the pinned analysis environment")
        if manifest.get("custom_methods") != {} or manifest.get("empirical_inputs") != {}:
            raise ValueError("this catalog study has no custom methods or empirical-file inputs")
        bound = {"source": source, "custom_methods": {}, "environment": manifest["environment"],
                 "external_methods": external, "resolved_scenarios": resolved,
                 "empirical_inputs": {}}
        fingerprint = _hash(bound)
        if manifest.get("source_fingerprint") != fingerprint:
            raise ValueError("invalid source/environment fingerprint")
        common = _hash({"source": source, "environment": manifest["environment"]})
        if shared_source_environment is not None and common != shared_source_environment:
            raise ValueError("shards use different producer sources or numerical environments")
        shared_source_environment = common
        seen = set()
        for row in _read_records(records_path):
            key = (row.get("scenario"), row.get("seed"), row.get("method"))
            if key not in cells or key not in expected:
                raise ValueError("record lies outside its declared shard or applicability grid")
            if key in collected or key in seen:
                raise ValueError("duplicate record in catalog inputs")
            identity = _hash(["record_v1", shard.config_hash, *key])
            seed = int(_hash(["method_seed_v1", key[0].split("@", 1)[0], key[1], key[2]])[:8], 16)
            if (row.get("record_id") != identity or row.get("config_hash") != shard.config_hash
                    or row.get("method_seed") != seed or row.get("schema_version") != 1):
                raise ValueError("record identity does not match its shard manifest")
            _validate_metrics(row)
            seen.add(key)
            collected[key] = row
        if require_full and seen != cells:
            raise ValueError("shard is missing requested terminal records")
        provenance.append({"input_ordinal": ordinal, "run_name": run.name,
                           "config_hash": shard.config_hash, "source_fingerprint": fingerprint,
                           "sample_seeds": list(shard.seeds), "scenario_count": len(shard.scenarios),
                           "methods": list(shard.methods), "requested_cells": len(cells),
                           "recorded_cells": len(seen), "manifest_sha256": file_hash(manifest_path),
                           "records_sha256": file_hash(records_path)})
        if identity_sink is not None:
            for name in ("source", "environment", "custom_methods", "empirical_inputs"):
                if name in identity_sink and identity_sink[name] != manifest[name]:
                    raise ValueError("authenticated producer identity differs across input manifests")
                identity_sink[name] = manifest[name]
            for name in ("external_methods", "resolved_scenarios"):
                merged = identity_sink.setdefault(name, {})
                for key, value in manifest[name].items():
                    if key in merged and merged[key] != value:
                        raise ValueError("authenticated producer controls differ across input manifests")
                    merged[key] = value
    if require_full and (declared != expected or set(collected) != expected):
        raise ValueError("shards do not form the complete declared applicable catalog grid")
    return collected, provenance


def analyze_records(config: BenchmarkConfig, records: dict) -> tuple[list[dict], list[dict], dict]:
    scores, contrasts = [], []
    for scenario in config.scenarios:
        reason = inference_reason(scenario)
        applicable = applicable_methods(config, scenario)
        for method in config.methods:
            if method not in applicable:
                scores.append({"scenario": scenario.key, "method": method, "applicable": False,
                               "reason": PPI_REASON, "expected": 0, "successful": 0,
                               "failed": 0, "missing": 0, **summarize_values([]),
                               "mean_root_pehe": None, "inference_eligible": False})
                continue
            rows = [records.get((scenario.key, seed, method)) for seed in config.seeds]
            good = [row for row in rows if row is not None and row["status"] == "ok"]
            summary = summarize_values((row["metrics"]["cate_mse"] for row in good), inferential=reason is None)
            scores.append({"scenario": scenario.key, "method": method, "applicable": True,
                           "reason": reason, "expected": len(config.seeds), "successful": len(good),
                           "failed": sum(row is not None and row["status"] == "failed" for row in rows),
                           "missing": sum(row is None for row in rows), **summary,
                           "mean_root_pehe": float(np.mean([row["metrics"]["root_pehe"] for row in good])) if good else None,
                           "inference_eligible": reason is None})
        for fusion in FUSION:
            if fusion not in applicable or fusion not in config.methods:
                continue
            for control in CONTROLS:
                if control not in applicable:
                    continue
                pairs = [(records.get((scenario.key, seed, fusion)),
                          records.get((scenario.key, seed, control))) for seed in config.seeds]
                complete = all(a is not None and b is not None and a["status"] == b["status"] == "ok"
                               for a, b in pairs)
                differences = [a["metrics"]["cate_mse"] - b["metrics"]["cate_mse"]
                               for a, b in pairs] if complete else []
                summary = summarize_values(differences, inferential=reason is None)
                contrasts.append({"scenario": scenario.key, "method": fusion, "reference": control,
                                  "complete": complete, "expected_pairs": len(config.seeds),
                                  "successful_pairs": sum(a is not None and b is not None
                                      and a["status"] == b["status"] == "ok" for a, b in pairs),
                                  "inference_eligible": reason is None, "reason": reason, **summary})
    family_complete = apply_primary_family(contrasts)
    expected = expected_cells(config)
    failures = sum(row["status"] == "failed" for row in records.values())
    nominal = len(config.scenarios) * len(config.seeds) * len(config.methods)
    coverage = {"nominal_cells": nominal, "not_applicable_cells": nominal - len(expected),
                "expected": len(expected), "recorded": len(records), "successful": len(records) - failures,
                "failed": failures, "missing": len(expected - records.keys()),
                "supported_primary_contrasts": len(contrasts),
                "nominal_primary_contrasts": len(config.scenarios) * sum(name in config.methods for name in FUSION)
                    * sum(name in config.methods for name in CONTROLS),
                "mean_t_family_size": sum(row["inference_eligible"] for row in contrasts),
                "descriptive_heavy_tail_contrasts": sum(not row["inference_eligible"] for row in contrasts),
                "mean_t_family_complete": family_complete}
    coverage["not_applicable_primary_contrasts"] = coverage["nominal_primary_contrasts"] - len(contrasts)
    return scores, contrasts, coverage


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                          for key, value in row.items()} for row in rows)


def _number(value: float | None, digits: int = 5) -> str:
    return "unavailable" if value is None else f"{value:.{digits}g}"


def _condition_report(scenarios: list, scores: list[dict], contrasts: list[dict], *,
                      family_status: dict | None = None) -> str:
    title = scenarios[0].kind + ":" + scenarios[0].id
    lines = ["# " + title, "", "All declared OBS/RCT budgets and methods are retained. "
             "Means describe successful runs with their coverage; N/A is a prospectively unsupported combination. "
             "For incomplete rows, the reported mean, MCSE and interval describe successful fits only; "
             "they are not full-grid performance estimates and assign no score to failed fits. "
             "Intervals are pointwise and Holm decisions require the complete eligible cross-catalog family. "
             "The optional budget plot focuses on the two nonlinear RCT controls and official fusion recipes; all recipes appear in the table.", ""]
    if family_status is not None and not family_status["complete"]:
        lines += [f"**All {family_status['size']} eligible family-level Holm decisions are withheld:** "
                  f"{family_status['incomplete']} eligible contrasts lack a successful pair at every requested seed. "
                  "Complete individual pointwise intervals do not restore the withheld family-level inference.", ""]
    if inference_reason(scenarios[0]):
        lines += ["**Heavy-tailed descriptive setting:** " + HEAVY_TAIL_REASON,
                  "Sample median and interquartile range describe the realized replicate scores; they are not intervals for population mean risk.", ""]
    lines += ["| Condition | Method | Mean MSE | MCSE | Pointwise 95% interval | Success / expected | State |",
              "| --- | --- | ---: | ---: | --- | ---: | --- |"]
    for row in scores:
        state = "N/A" if not row["applicable"] else ("descriptive heavy-tail" if not row["inference_eligible"] else "complete" if row["successful"] == row["expected"] else "incomplete")
        ci = row["ci95"]
        interval = "unavailable" if ci is None else f"[{ci[0]:.5g}, {ci[1]:.5g}]"
        lines.append(f"| `{row['scenario']}` | `{row['method']}` | {_number(row['mean'])} | {_number(row['mcse'])} | {interval} | {row['successful']} / {row['expected']} | {state} |")
    if inference_reason(scenarios[0]):
        lines += ["", "| Condition | Method | Sample median MSE | Sample interquartile range |",
                  "| --- | --- | ---: | --- |"]
        for row in scores:
            lines.append(f"| `{row['scenario']}` | `{row['method']}` | {_number(row['sample_median'])} | [{_number(row['sample_q25'])}, {_number(row['sample_q75'])}] |")
    lines += ["", "| Condition | Fusion | RCT reference | Paired mean MSE difference | Pointwise 95% interval | Holm p | Decision |",
              "| --- | --- | --- | ---: | --- | ---: | --- |"]
    for row in contrasts:
        ci = row["ci95"]
        interval = "unavailable" if ci is None else f"[{ci[0]:.5g}, {ci[1]:.5g}]"
        lines.append(f"| `{row['scenario']}` | `{row['method']}` | `{row['reference']}` | {_number(row['mean'])} | {interval} | {_number(row['holm_p'])} | {row['decision']} |")
    lines += ["", "Negative differences favor fusion. Unresolved is not equivalence. "
              "Source assumptions, overlap and transport stresses differ across presets; no raw-MSE global ranking is reported.", ""]
    return "\n".join(lines)


def make_figure(scenarios: list, scores: list[dict], output: Path, *, paper_assets: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "obsrct-catalog-study-v1"
    displayed = (*CONTROLS, *FUSION)
    labels = {"rct_t_extra_trees": "RCT-only T", "rct_dr_extra_trees": "RCT-only DR",
              "cornet_official": "CorNet", "datafusionppi": "DataFusionPPI"}
    colors = dict(zip(displayed, ("#4477AA", "#228833", "#CC6677", "#AA3377")))
    obs_sizes = sorted({scenario.parameters["n_obs"] for scenario in scenarios})
    fig, axes = plt.subplots(1, len(obs_sizes), figsize=(5.2 * len(obs_sizes), 3.8), squeeze=False)
    lookup = {(row["scenario"], row["method"]): row for row in scores}
    for column, n_obs in enumerate(obs_sizes):
        ax = axes[0, column]
        selected = sorted((scenario for scenario in scenarios if scenario.parameters["n_obs"] == n_obs),
                          key=lambda scenario: scenario.parameters["n_rct_train"])
        for method in displayed:
            if method not in {row["method"] for row in scores}:
                continue
            valid = [(scenario, lookup.get((scenario.key, method))) for scenario in selected]
            valid = [(scenario, row) for scenario, row in valid if row and row["applicable"] and row["mean"] is not None]
            if not valid:
                applicable = [row for _, row in [(scenario, lookup.get((scenario.key, method)))
                                                 for scenario in selected]
                              if row is not None and row["applicable"]]
                if applicable:
                    state = (f"0 successes; {sum(row['failed'] for row in applicable)} failed, "
                             f"{sum(row['missing'] for row in applicable)} missing")
                else:
                    state = "N/A"
                ax.text(.02, .96 - .06 * displayed.index(method), labels[method] + ": " + state,
                        transform=ax.transAxes, va="top", fontsize=8, color=colors[method])
                continue
            x = [scenario.parameters["n_rct_train"] + scenario.parameters["n_rct_tune"] for scenario, _ in valid]
            ax.plot(x, [row["mean"] for _, row in valid], marker="o", color=colors[method], label=labels[method])
            for xvalue, (_, row) in zip(x, valid):
                if row["ci95"] is not None:
                    ax.vlines(xvalue, *row["ci95"], color=colors[method])
                if row["successful"] != row["expected"]:
                    label = (f"{row['successful']}/{row['expected']}; "
                             f"{row['failed']} failed, {row['missing']} missing")
                    rightmost = xvalue == max(x)
                    ax.annotate(label, (xvalue, row["mean"]),
                                xytext=(-6 if rightmost else 6, 10), textcoords="offset points",
                                ha="right" if rightmost else "left", fontsize=7, color=colors[method])
        ax.set_title(f"OBS {n_obs:,}")
        ax.set_xlabel("Available RCT labels (training + 32 reserved)")
        ax.set_ylabel("Target CATE MSE")
        ax.set_xticks([scenario.parameters["n_rct_train"] + scenario.parameters["n_rct_tune"] for scenario in selected])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    suffix = "descriptive means; no mean-t intervals" if inference_reason(scenarios[0]) else "means and pointwise 95% seed intervals"
    fig.suptitle(scenarios[0].kind + ":" + scenarios[0].id + "\n" + suffix, fontsize=11)
    fig.text(.5, .015, "Incomplete points show successes/expected and failed/missing counts.\n"
             "Their means and intervals condition on fit success, not full-grid performance.",
             ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, .11, 1, .87))
    fig.savefig(output / "budget_curves.svg", metadata={"Date": None})
    if paper_assets:
        fig.savefig(output / "budget_curves.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def write_analysis(config: BenchmarkConfig, runs: list[Path], output: Path, *,
                   paper_assets: bool = False, no_plots: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("analysis output must be a fresh directory")
    producer_identity = {}
    records, inputs = collect_records(config, runs, identity_sink=producer_identity)
    if producer_identity["resolved_scenarios"] != resolved_scenarios(config):
        raise ValueError("producer identities do not cover all declared resolved controls")
    producer_identity.update(schema_version=1, normalized_configuration=config.payload(),
                             normalized_config_sha256=config.config_hash, authenticated_inputs=inputs,
                             scope="Exact identity fields merged from all authenticated input manifests; no data arrays or fitted models are included")
    scores, contrasts, coverage = analyze_records(config, records)
    family_status = {"size": coverage["mean_t_family_size"],
                     "complete": coverage["mean_t_family_complete"],
                     "incomplete": sum(row["inference_eligible"] and not row["complete"]
                                       for row in contrasts)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "producer_identity.json").write_text(json.dumps(
        producer_identity, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_csv(output / "condition_scores.csv", scores)
    _write_csv(output / "primary_contrasts.csv", contrasts)
    _write_csv(output / "not_applicable.csv", not_applicable_rows(config))
    presets = {}
    for scenario in config.scenarios:
        presets.setdefault((scenario.kind, scenario.id), []).append(scenario)
    per_preset = []
    for (kind, identity), scenarios in presets.items():
        relative = Path("designs") / (kind + "__" + identity)
        folder = output / relative
        folder.mkdir(parents=True, exist_ok=True)
        keys = {scenario.key for scenario in scenarios}
        selected_scores = [row for row in scores if row["scenario"] in keys]
        selected_contrasts = [row for row in contrasts if row["scenario"] in keys]
        detail = {"preset": kind + ":" + identity, "scope": "development-inspected preset; sampling replication only",
                  "conditions": [scenario.key for scenario in scenarios], "scores": selected_scores,
                  "primary_contrasts": selected_contrasts, "inference_note": inference_reason(scenarios[0])}
        (folder / "summary.json").write_text(json.dumps(detail, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (folder / "README.md").write_text(_condition_report(
            scenarios, selected_scores, selected_contrasts, family_status=family_status))
        _write_csv(folder / "replicate_scores.csv", [
            {"scenario": scenario.key, "seed": seed, "method": method,
             "status": "missing" if records.get((scenario.key, seed, method)) is None else records[(scenario.key, seed, method)]["status"],
             "cate_mse": None if records.get((scenario.key, seed, method)) is None or records[(scenario.key, seed, method)]["status"] != "ok" else records[(scenario.key, seed, method)]["metrics"]["cate_mse"],
             "root_pehe": None if records.get((scenario.key, seed, method)) is None or records[(scenario.key, seed, method)]["status"] != "ok" else records[(scenario.key, seed, method)]["metrics"]["root_pehe"],
             "error_type": None if records.get((scenario.key, seed, method)) is None else records[(scenario.key, seed, method)].get("error_type"),
             "stage": None if records.get((scenario.key, seed, method)) is None else records[(scenario.key, seed, method)].get("stage")}
            for scenario in scenarios for seed in config.seeds for method in applicable_methods(config, scenario)])
        if not no_plots:
            make_figure(scenarios, selected_scores, folder, paper_assets=paper_assets)
        per_preset.append({"preset": kind + ":" + identity, "path": relative.as_posix(),
                           "conditions": len(scenarios),
                           "inference": "descriptive heavy-tail" if inference_reason(scenarios[0]) else "mean-t family"})
    summary = {"schema_version": 1, "normalized_config_sha256": config.config_hash,
               "coverage": coverage, "inputs": inputs, "presets": per_preset,
               "not_applicable": not_applicable_rows(config),
               "inference_policy": {"heavy_tail_preset": ":".join(HEAVY_TAIL_PRESET),
                                    "reason": HEAVY_TAIL_REASON,
                                    "eligible_family": "two-sided paired mean-t tests; joint Holm at .05; incomplete eligible family withholds every decision"},
               "analysis_source_sha256": file_hash(Path(__file__)),
               "producer_identity_file": "producer_identity.json",
               "scope": "Complete declared catalog and budget grid; inspected mechanisms, fixed recipes, no universal ranking or unseen-mechanism claim",
               "prediction_replay": "Not performed by aggregation; see separate reproducibility checker"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = ["# Full catalog reference comparison", "",
             f"Applicable cells: {coverage['successful']}/{coverage['expected']} successful; {coverage['failed']} failed, {coverage['missing']} missing. "
             f"The nominal grid has {coverage['nominal_cells']} cells, including {coverage['not_applicable_cells']} prospectively unsupported cells.", "",
             "No cross-design raw-MSE ranking is reported. Every preset retains every budget, method and comparison direction. "
             "Known unsupported combinations are N/A, not zero-error successes or unexplained failures.", "",
             f"Supported primary contrasts: {coverage['supported_primary_contrasts']}; mean-t eligible family: {coverage['mean_t_family_size']}; "
             f"heavy-tail descriptive contrasts: {coverage['descriptive_heavy_tail_contrasts']}. "
             "The eligible family uses joint Holm adjustment at .05. M11's canonical Student-t(3) noise is retained, with no mean-t MCSE/CI/p-value. "
             "Its replicate means and empirical quantiles are descriptive, not a substitute for finite-variance inference.", "",
             "| Preset | Conditions | Inference | Complete report |", "| --- | ---: | --- | --- |"]
    if not family_status["complete"]:
        lines[2:2] = [f"**All {family_status['size']} eligible family-level Holm decisions are withheld.** "
                      f"{family_status['incomplete']} eligible contrasts lack complete successful paired coverage. "
                      "No smaller family is selected, and no failed case is assigned a favorable score.", ""]
    lines.extend(f"| `{item['preset']}` | {item['conditions']} | {item['inference']} | [Results]({item['path']}/README.md) |" for item in per_preset)
    lines.extend(["", "[All condition scores](condition_scores.csv), [all supported primary contrasts](primary_contrasts.csv), "
                  "[declared unsupported combinations](not_applicable.csv) and [input/config/source bindings](summary.json). "
                  "[Producer identity](producer_identity.json) includes the actual source-file hashes, numerical environment, "
                  "upstream implementations and resolved controls merged from the authenticated input manifests. "
                  "Aggregation authenticates recorded artifacts; it does not independently refit predictions. "
                  "The design is in `docs/CATALOG_STUDY.md`.", ""])
    (output / "README.md").write_text("\n".join(lines))
    hashes = {path.relative_to(output).as_posix(): file_hash(path)
              for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "analysis_source_sha256": file_hash(Path(__file__)),
                "design_validator_sha256": file_hash(Path(__file__).with_name("make_catalog_configs.py")),
                "normalized_config_sha256": config.config_hash, "coverage": coverage,
                "inputs": inputs, "output_sha256": hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if any(path.stat().st_size > 1024 * 1024 for path in output.rglob("*") if path.is_file()):
        raise ValueError("an analysis artifact exceeds the compact-publication one-MiB limit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-assets", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_study(config)
    summary = write_analysis(config, args.runs, args.output,
                             paper_assets=args.paper_assets, no_plots=args.no_plots)
    print(json.dumps(summary["coverage"], sort_keys=True))
    if summary["coverage"]["failed"] or summary["coverage"]["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
