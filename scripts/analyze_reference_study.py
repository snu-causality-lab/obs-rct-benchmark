"""Reconstruct a declared reference study from complete, disjoint seed shards.

This independently aggregates terminal metrics, not the underlying predictions.
No result is selected on the basis of its favorable direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from obs_rct_benchmark.runner import load_config, resolved_scenarios


FUSION = ("cornet_official", "datafusionppi")
CONTROLS = ("rct_t_extra_trees", "rct_dr_extra_trees")
MECHANISMS = ("clean", "confounded", "shift", "both")


def digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_values(values: Iterable[float]) -> dict[str, Any]:
    """Direct mean, MCSE and exact Student-t critical value over seed values."""
    data = np.asarray(list(values), dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("values must be finite one-dimensional seed observations")
    n = len(data)
    result = {"n": n, "mean": None, "mcse": None, "ci95": None,
              "p_two_sided_zero": None}
    if not n:
        return result
    mean = float(np.mean(data))
    result["mean"] = mean
    if n < 2:
        return result
    se = float(np.std(data, ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(.975, n - 1)) * se
    if se == 0:
        p = 1.0 if mean == 0 else 0.0
    else:
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    result.update(mcse=se, ci95=[mean - half, mean + half], p_two_sided_zero=p)
    return result


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    """Step-down Holm adjusted p-values, returned in the original order."""
    values = list(pvalues)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("p-values must be finite probabilities")
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = [0.] * len(values)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def apply_primary_family(rows: list[dict[str, Any]]) -> bool:
    """Never shrink the declared family because a comparison is incomplete."""
    complete = bool(rows) and all(row["complete"] and row["p_two_sided_zero"] is not None
                                  for row in rows)
    adjusted = holm_adjust(row["p_two_sided_zero"] for row in rows) if complete else []
    for index, row in enumerate(rows):
        row["holm_p"] = adjusted[index] if complete else None
        row["holm_reject_05"] = adjusted[index] <= .05 if complete else None
    return complete


def _load_json(path: Path) -> dict[str, Any]:
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    return value


def collect_records(config, runs: list[Path]) -> tuple[dict, list[dict]]:
    """Validate disjoint shard identities against the declared full configuration."""
    canonical = config.payload()
    expected_resolved = resolved_scenarios(config)
    expected_axes = {(scenario.key, seed, method) for scenario in config.scenarios
                     for seed in config.seeds for method in config.methods}
    collected: dict[tuple[str, int, str], dict] = {}
    provenance = []
    shared_fingerprint = None
    seen_shard_seeds: set[int] = set()
    for ordinal, run in enumerate(runs):
        manifest_path, record_path = run / "run_manifest.json", run / "records.jsonl"
        manifest = _load_json(manifest_path)
        part = manifest["config"]
        seeds = part["run"]["seeds"]
        if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config.seeds):
            raise ValueError("shard contains duplicate or undeclared seeds")
        if seen_shard_seeds.intersection(seeds):
            raise ValueError("input shards declare overlapping seed sets")
        seen_shard_seeds.update(seeds)
        expected = {**canonical, "run": {**canonical["run"], "seeds": seeds}}
        if part != expected or manifest["config_hash"] != digest(part):
            raise ValueError("shard configuration differs from the declared study")
        if manifest.get("resolved_scenarios") != expected_resolved:
            raise ValueError("shard generator controls differ from the declared study")
        source = manifest["source"]
        if source.get("aggregate_sha256") != digest(source["package_file_sha256"]):
            raise ValueError("invalid recorded package-source hash")
        fingerprint = digest({
            "source": source, "custom_methods": manifest["custom_methods"],
            "environment": manifest["environment"],
            "external_methods": manifest["external_methods"],
            "resolved_scenarios": manifest["resolved_scenarios"],
            "empirical_inputs": manifest["empirical_inputs"],
        })
        if fingerprint != manifest["source_fingerprint"]:
            raise ValueError("invalid recorded source/environment fingerprint")
        if shared_fingerprint is not None and fingerprint != shared_fingerprint:
            raise ValueError("shards use different source or numerical environments")
        shared_fingerprint = fingerprint
        raw = record_path.read_text(encoding="utf-8")
        if raw and not raw.endswith("\n"):
            raise ValueError("partial final record")
        for line in raw.splitlines():
            if not line.strip():
                raise ValueError("blank record line")
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")))
            key = (row["scenario"], row["seed"], row["method"])
            if key not in expected_axes or row["seed"] not in seeds:
                raise ValueError("record outside declared grid or shard seed set")
            if key in collected:
                raise ValueError("duplicate record across study inputs")
            expected_id = digest(["record_v1", manifest["config_hash"], *key])
            # Labels identify cells, while a preset's matched variants share
            # the method RNG stream. Reconstruct this independently of runner.
            preset = key[0].split("@", 1)[0]
            expected_method_seed = int(digest(["method_seed_v1", preset, key[1], key[2]])[:8], 16)
            if (row["record_id"] != expected_id or row["config_hash"] != manifest["config_hash"]
                    or row["method_seed"] != expected_method_seed
                    or row["schema_version"] != 1):
                raise ValueError("record identity does not match its shard manifest")
            if row["status"] not in {"ok", "failed"}:
                raise ValueError("unknown terminal status")
            if row["status"] == "ok":
                metrics = row["metrics"]
                mse, root = metrics["cate_mse"], metrics["root_pehe"]
                if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                        or not math.isfinite(x) or x < 0 for x in (mse, root))
                        or not math.isclose(root * root, mse, rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError("invalid terminal metrics")
            elif row["metrics"] is not None:
                raise ValueError("failed record contains metrics")
            collected[key] = row
        provenance.append({"input_ordinal": ordinal,
                           "manifest_sha256": file_hash(manifest_path),
                           "records_sha256": file_hash(record_path),
                           "config_hash": manifest["config_hash"],
                           "source_fingerprint": fingerprint,
                           "sample_seeds": seeds})
    return collected, provenance


def analyze_records(config, records: dict) -> tuple[list[dict], list[dict], dict]:
    scores, contrasts = [], []
    for scenario in config.scenarios:
        for method in config.methods:
            rows = [records.get((scenario.key, seed, method)) for seed in config.seeds]
            good = [row for row in rows if row is not None and row["status"] == "ok"]
            summary = summarize_values(row["metrics"]["cate_mse"] for row in good)
            scores.append({"scenario": scenario.key, "method": method,
                           "expected": len(config.seeds), "successful": len(good),
                           "failed": sum(row is not None and row["status"] == "failed" for row in rows),
                           "missing": sum(row is None for row in rows),
                           "mean_mse": summary["mean"], "mcse_mse": summary["mcse"],
                           "ci95_mse": summary["ci95"],
                           "mean_root_pehe": float(np.mean([row["metrics"]["root_pehe"]
                                                           for row in good])) if good else None})
        for fusion in FUSION:
            for control in CONTROLS:
                pairs = [(records.get((scenario.key, seed, fusion)),
                          records.get((scenario.key, seed, control))) for seed in config.seeds]
                complete = all(a is not None and b is not None
                               and a["status"] == b["status"] == "ok" for a, b in pairs)
                differences = [a["metrics"]["cate_mse"] - b["metrics"]["cate_mse"]
                               for a, b in pairs] if complete else []
                summary = summarize_values(differences)
                contrasts.append({"scenario": scenario.key, "method": fusion,
                                  "reference": control, "complete": complete,
                                  "expected_pairs": len(config.seeds),
                                  "successful_pairs": sum(a is not None and b is not None
                                                          and a["status"] == b["status"] == "ok"
                                                          for a, b in pairs),
                                  **summary})
    family_complete = apply_primary_family(contrasts)
    expected = len(config.scenarios) * len(config.seeds) * len(config.methods)
    failed = sum(row["status"] == "failed" for row in records.values())
    coverage = {"expected": expected, "recorded": len(records),
                "successful": len(records) - failed, "failed": failed,
                "missing": expected - len(records),
                "primary_family_size": len(contrasts),
                "primary_family_complete": family_complete}
    return scores, contrasts, coverage


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                          for key, value in row.items()} for row in rows)


def make_figures(config, scores: list[dict], contrasts: list[dict], output: Path, *, paper_assets: bool = False) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "obsrct-reference-v1"})
    lookup = {(row["scenario"], row["method"]): row for row in scores}
    conditions = {scenario.label: scenario for scenario in config.scenarios}
    plotted = CONTROLS + FUSION
    display_names = {"rct_t_extra_trees": "RCT-only T-learner", "rct_dr_extra_trees": "RCT-only DR-learner",
                     "cornet_official": "CorNet", "datafusionppi": "DataFusionPPI"}
    colors = {method: color for method, color in zip(plotted, ("#4477AA", "#228833", "#CC6677", "#AA3377"))}
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12), sharex=True, squeeze=False)
    for row, mechanism in enumerate(MECHANISMS):
        for col, n_obs in enumerate((512, 2048)):
            ax = axes[row, col]
            for method in plotted:
                entries = []
                for n_rct in (128, 512):
                    scenario = conditions[f"{mechanism}_o{n_obs}_r{n_rct}"]
                    score = lookup[(scenario.key, method)]
                    if score["successful"] != score["expected"]:
                        continue
                    mean, ci = score["mean_mse"], score["ci95_mse"]
                    entries.append((n_rct + 32, mean, mean - ci[0], ci[1] - mean))
                if entries:
                    ax.errorbar([x[0] for x in entries], [x[1] for x in entries],
                                yerr=np.asarray([[x[2] for x in entries], [x[3] for x in entries]]),
                                label=display_names[method], color=colors[method], marker="o", capsize=3)
            ax.set_title(f"{mechanism}; OBS {n_obs:,}")
            ax.set_xticks((160, 544))
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Target CATE MSE")
            if row == 3:
                ax.set_xlabel("Available RCT labels (training + 32 reserved)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .005))
    fig.suptitle("Fixed public recipes: mean MSE and pointwise 95% seed intervals")
    fig.tight_layout(rect=(0, .055, 1, .97))
    first = "reference_budget_curves.svg"
    fig.savefig(output / first, metadata={"Date": None})
    if paper_assets:
        fig.savefig(output / first.replace(".svg", ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)

    comparison_keys = [(fusion, control) for fusion in FUSION for control in CONTROLS]
    contrast_lookup = {(row["scenario"], row["method"], row["reference"]): row for row in contrasts}
    image = np.full((len(comparison_keys), len(config.scenarios)), np.nan)
    for i, (method, reference) in enumerate(comparison_keys):
        for j, scenario in enumerate(config.scenarios):
            item = contrast_lookup[(scenario.key, method, reference)]
            if item["complete"]:
                image[i, j] = item["mean"]
    limit = max(float(np.nanmax(np.abs(image))) if np.isfinite(image).any() else 1., 1e-12)
    fig, ax = plt.subplots(figsize=(15.5, 4.6))
    shown = ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i, (method, reference) in enumerate(comparison_keys):
        for j, scenario in enumerate(config.scenarios):
            item = contrast_lookup[(scenario.key, method, reference)]
            text = "missing" if not item["complete"] else f"{item['mean']:.3g}"
            if item["holm_reject_05"]:
                text += "*"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(image[i, j]) and abs(image[i, j]) > .55 * limit else "black")
    ax.set_xticks(range(len(config.scenarios)), [s.label.replace("_", "\n") for s in config.scenarios])
    ax.tick_params(axis="x", labelsize=7)
    short = {"cornet_official": "CorNet", "datafusionppi": "DataFusionPPI",
             "rct_t_extra_trees": "RCT T", "rct_dr_extra_trees": "RCT DR"}
    ax.set_yticks(range(4), [f"{short[m]} minus {short[r]}" for m, r in comparison_keys])
    ax.set_title("All 64 primary paired MSE contrasts; negative favors fusion; * Holm64 p <= .05")
    fig.colorbar(shown, ax=ax, label="Paired mean MSE difference", shrink=.8)
    fig.tight_layout()
    second = "reference_primary_contrasts.svg"
    fig.savefig(output / second, metadata={"Date": None})
    if paper_assets:
        fig.savefig(output / second.replace(".svg", ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    return [first, second]


def make_readme(config, scores: list[dict], contrasts: list[dict], coverage: dict) -> str:
    lines = ["# Static reference comparison", "",
             "Four matched M2 variants, two OBS sizes and two RCT training sizes; "
             "32 independent seeds per condition. Fixed recipes; no unseen-mechanism claim.", "",
             f"Coverage: {coverage['successful']}/{coverage['expected']} successful; "
             f"{coverage['failed']} failed; {coverage['missing']} missing.", "",
             "Available RCT labels are 160/544: 128/512 training and 32 reserved tuning labels. "
             "All fixed recipes ignore the reserved tuning labels. Evaluator-only factual test labels are separate.", "",
             "All intervals below are pointwise 95% Student-t intervals over independent seed blocks. "
             "MCSE describes precision conditional on these development-inspected mechanisms. "
             "There is no cross-mechanism raw-error ranking.", "",
             "## Condition scores", "",
             "| Condition | Method | Mean MSE | MCSE | Pointwise 95% interval | Successful / expected |",
             "| --- | --- | ---: | ---: | --- | ---: |"]
    for row in scores:
        mean = "withheld" if row["mean_mse"] is None else f"{row['mean_mse']:.6g}"
        se = "—" if row["mcse_mse"] is None else f"{row['mcse_mse']:.3g}"
        ci = "—" if row["ci95_mse"] is None else f"[{row['ci95_mse'][0]:.6g}, {row['ci95_mse'][1]:.6g}]"
        lines.append(f"| `{row['scenario']}` | `{row['method']}` | {mean} | {se} | {ci} | "
                     f"{row['successful']} / {row['expected']} |")
    lines.extend(["", "## All prespecified primary contrasts", "",
                  "Differences are fusion minus RCT reference; negative favors fusion. "
                  "Two-sided p-values are adjusted together over 64 tests. "
                  "Pointwise intervals do not have simultaneous family coverage.", "",
                  "| Condition | Fusion | RCT reference | Mean difference | MCSE | Pointwise 95% interval | Holm p |",
                  "| --- | --- | --- | ---: | ---: | --- | ---: |"])
    for row in contrasts:
        mean = "withheld" if row["mean"] is None else f"{row['mean']:.6g}"
        se = "—" if row["mcse"] is None else f"{row['mcse']:.3g}"
        ci = "—" if row["ci95"] is None else f"[{row['ci95'][0]:.6g}, {row['ci95'][1]:.6g}]"
        p = "withheld" if row["holm_p"] is None else f"{row['holm_p']:.4g}"
        lines.append(f"| `{row['scenario']}` | `{row['method']}` | `{row['reference']}` | "
                     f"{mean} | {se} | {ci} | {p} |")
    if not coverage["primary_family_complete"]:
        lines.extend(["", "**The declared primary family is incomplete: all Holm decisions are withheld.**"])
    lines.extend(["", "The JSON/CSV files preserve every requested condition and comparison. "
                  "This analysis reconstructs terminal metric summaries; it does not independently replay fitted predictions. "
                  "See `docs/REFERENCE_STUDY.md` for the frozen design and limitations.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reference_study.toml"))
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--paper-assets", action="store_true", help="also write vector PDFs for manuscript use")
    args = parser.parse_args()
    config = load_config(args.config)
    if (len(config.scenarios) != 16 or len(config.seeds) != 32
            or set(config.methods) != set(CONTROLS + FUSION + ("obs_t_extra_trees", "naive_pooled_t_extra_trees"))
            or {s.label for s in config.scenarios} != {
                f"{mechanism}_o{n_obs}_r{n_rct}" for mechanism in MECHANISMS
                for n_obs in (512, 2048) for n_rct in (128, 512)}):
        parser.error("expected the complete declared 16-condition, 32-seed, six-method study")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("analysis output must be empty")
    records, inputs = collect_records(config, args.runs)
    scores, contrasts, coverage = analyze_records(config, records)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "study_config_sha256": file_hash(args.config),
               "study_config_hash": config.config_hash, "inputs": inputs,
               "coverage": coverage, "scores": scores, "primary_contrasts": contrasts,
               "scope": "prespecified static comparison on development-inspected M2 variants",
               "prediction_replay": "not performed; reconstructs terminal metrics only"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "condition_scores.csv", scores)
    write_csv(args.output / "primary_contrasts.csv", contrasts)
    (args.output / "README.md").write_text(make_readme(config, scores, contrasts, coverage), encoding="utf-8")
    figures = [] if args.no_plots else make_figures(config, scores, contrasts, args.output, paper_assets=args.paper_assets)
    manifest = {"schema_version": 1, "study_config_sha256": file_hash(args.config),
                "analysis_source_sha256": file_hash(Path(__file__)),
                "inputs": inputs, "coverage": coverage,
                "output_sha256": {path.name: file_hash(path) for path in sorted(args.output.iterdir())
                                  if path.is_file()}, "figures": figures}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                         allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(coverage, sort_keys=True))
    if coverage["failed"] or coverage["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
