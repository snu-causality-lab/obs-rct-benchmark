#!/usr/bin/env python3
"""Render authenticated scalar GP-pilot records without importing its fitter.

All numerical formatting and plots run on an allocated GSDS node. This reads
terminal-record JSON and the completed analysis; no raw arrays are copied or
models fitted. The separate result verifier checks prediction/target arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys


SLOTS = ("full", "rct_only", "obs_start")
LABELS = {"full": "OBS + RCT", "rct_only": "RCT only", "obs_start": "OBS-conditioned start"}
COLORS = {"full": "#238b45", "rct_only": "#2166ac", "obs_start": "#d97706"}
CONTRASTS = ("full_minus_rct_only", "full_minus_obs_start")
SCORE_FIELDS = ["delta", "n_rct", "slot", "method_id", "expected_seeds", "successful", "failed",
                "mean_mse_success_only", "mcse", "complete"]
PAIR_FIELDS = ["delta", "n_rct", "contrast", "expected_pairs", "successful_pairs", "complete",
               "mean", "mcse", "ci95_low", "ci95_high"]
SCALAR_FIELDS = ["case_id", "seed", "delta_index", "delta", "n_rct", "slot", "method_id", "status",
                 "rct_labels", "obs_labels", "additional_tuning_labels", "mse", "fit_predict_seconds",
                 "launcher_seconds", "worker_self_maxrss_kib", "worker_children_maxrss_kib", "input_sha256",
                 "evaluator_inputs_sha256", "eval_tau_sha256", "prediction_sha256", "record_sha256"]
COST_FIELDS = ["delta", "n_rct", "slot", "method_id", "rct_labels", "obs_labels",
               "additional_tuning_labels", "requested", "successful", "failed",
               "mean_fit_predict_seconds_success_only", "median_fit_predict_seconds_success_only",
               "mean_launcher_seconds_all_attempts", "max_worker_self_rss_kib", "max_worker_child_rss_kib"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text())
    canonical(data)
    return data


def close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"missing-value mismatch: {label}")
    elif not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError(f"scalar mismatch: {label}")


def guard(path):
    raw = os.environ.get("GSDS_WORK_ROOT")
    if (sys.platform != "linux" or not raw or not os.environ.get("SLURM_JOB_ID")
            or not Path(raw).is_absolute()):
        raise RuntimeError("rendering requires a GSDS Slurm job and an absolute shared work root")
    work, target = Path(raw).resolve(strict=True), Path(path).expanduser().resolve()
    if target == work or not target.is_relative_to(work):
        raise ValueError("rendering paths must be below GSDS_WORK_ROOT")
    if any(os.environ.get(name) != "1" for name in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("all numerical thread limits must equal one")
    return target


def authenticate(fits, analysis):
    manifest = read_json(fits / "manifest.json")
    summary = read_json(analysis / "summary.json")
    if manifest.get("manifest_sha256") != digest({k:v for k,v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("input manifest canonical payload changed")
    if (manifest["schema_version"] != 1 or summary["schema_version"] != 1
            or manifest["bindings"] != summary["bindings"] or tuple(manifest["slots"]) != SLOTS):
        raise ValueError("analysis/producer bindings disagree")
    cfg, cases = manifest["config"], manifest["cases"]
    expected = {f"records/{case['id']}/{slot}/record.json": (case, slot) for case in cases for slot in SLOTS}
    actual = {str(path.relative_to(fits)) for path in (fits / "records").rglob("record.json")}
    if (actual != set(expected) or set(summary["record_sha256"]) != set(expected)
            or manifest["expected_records"] != len(expected) or summary["expected_records"] != len(expected)):
        raise ValueError("rendering requires the exact complete terminal record collection")
    records = {}
    for relative, (case, slot) in expected.items():
        path = fits / relative
        if file_hash(path) != summary["record_sha256"][relative]:
            raise ValueError("record changed after analysis")
        record = read_json(path)
        if (record["case"] != case or record["slot"] != slot
                or record["manifest_sha256"] != manifest["manifest_sha256"]
                or record["bindings"] != manifest["bindings"]
                or record["method_id"] != cfg["methods"][slot]
                or record["rho"] != cfg["rho"]
                or record["status"] not in {"success", "failed", "timeout"}):
            raise ValueError("terminal record disagrees with the declared pilot")
        if record["status"] == "success" and (not math.isfinite(record["mse"]) or record["mse"] < 0):
            raise ValueError("successful risk must be finite and nonnegative")
        records[(case["id"], slot)] = record
    successful = sum(record["status"] == "success" for record in records.values())
    if (summary["missing"] != 0 or summary["success"] != successful
            or summary["failed"] != len(expected) - successful):
        raise ValueError("analysis coverage differs from terminal records")
    conditions = {(row["delta"], row["n_rct"]): row for row in summary["conditions"]}
    if (len(conditions) != len(summary["conditions"])
            or set(conditions) != {(delta, budget) for delta in cfg["deltas"] for budget in cfg["rct_budgets"]}):
        raise ValueError("analysis condition collection differs from the configured grid")
    return manifest, summary, records, conditions


def optional_mean(values):
    return math.fsum(values) / len(values) if values else None


def optional_mcse(values, expected):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) == expected and len(values) >= 2 else None


def expected_labels(cfg, case, slot):
    return case["n_rct"] if slot != "obs_start" else 0, cfg["n_obs"] if slot != "rct_only" else 0


def scalar_rows(manifest, summary, records):
    cfg, rows = manifest["config"], []
    for case in manifest["cases"]:
        for slot in SLOTS:
            record = records[(case["id"], slot)]
            rct, obs = expected_labels(cfg, case, slot)
            budget = record.get("label_budget")
            if record["status"] == "success" and budget is None:
                raise ValueError("successful fit lacks source-label accounting")
            if budget is not None and any(budget.get(name) != expected for name, expected in
                                          (("supplied_unique_rct_labels", rct), ("supplied_unique_obs_labels", obs),
                                           ("consumed_rct_labels", rct), ("consumed_obs_labels", obs),
                                           ("additional_tuning_labels", 0))):
                raise ValueError("recorded source-label accounting changed")
            relative = f"records/{case['id']}/{slot}/record.json"
            rows.append({
                "case_id": case["id"], "seed": case["seed"], "delta_index": case["delta_index"],
                "delta": case["delta"], "n_rct": case["n_rct"], "slot": slot,
                "method_id": record["method_id"], "status": record["status"],
                "rct_labels": rct, "obs_labels": obs, "additional_tuning_labels": 0,
                "mse": record.get("mse") if record["status"] == "success" else None,
                "fit_predict_seconds": record.get("fit_predict_seconds"),
                "launcher_seconds": record.get("launcher_seconds"),
                "worker_self_maxrss_kib": record.get("worker_self_usage", {}).get("maxrss_kib"),
                "worker_children_maxrss_kib": record.get("worker_children_usage", {}).get("maxrss_kib"),
                "input_sha256": digest(record["method_argument_array_identity"]) if "method_argument_array_identity" in record else None,
                "evaluator_inputs_sha256": digest(record["observed_array_identity"]) if "observed_array_identity" in record else None,
                "eval_tau_sha256": record.get("evaluation_array_identity", {}).get("eval_tau", {}).get("sha256"),
                "prediction_sha256": record.get("prediction_array_identity", {}).get("sha256"),
                "record_sha256": summary["record_sha256"][relative],
            })
    return rows


def tables(manifest, summary, records, conditions):
    cfg, scores, pairs, costs, wide = manifest["config"], [], [], [], []
    for delta in cfg["deltas"]:
        for budget in cfg["rct_budgets"]:
            source = conditions[(delta, budget)]
            group = [case for case in manifest["cases"] if case["delta"] == delta and case["n_rct"] == budget]
            if len(group) != len(cfg["seeds"]):
                raise ValueError("condition does not contain each configured sample seed")
            row = {"delta": delta, "n_rct": budget, "expected_seeds": len(group)}
            for slot in SLOTS:
                subset = [records[(case["id"], slot)] for case in group]
                successes = [record for record in subset if record["status"] == "success"]
                values = [record["mse"] for record in successes]
                declared = source["recipes"][slot]
                if declared["successful"] != len(successes) or declared["failed"] != len(group)-len(successes):
                    raise ValueError("condition recipe coverage differs from terminal records")
                close(optional_mean(values), declared["mean_mse_success_only"], "recipe mean")
                score = {"delta": delta, "n_rct": budget, "slot": slot, "method_id": cfg["methods"][slot],
                         "expected_seeds": len(group), "successful": len(successes),
                         "failed": len(group)-len(successes), "mean_mse_success_only": declared["mean_mse_success_only"],
                         "mcse": optional_mcse(values, len(group)), "complete": len(successes) == len(group)}
                scores.append(score)
                row.update({slot + "_" + name: score[name] for name in
                            ("successful", "failed", "mean_mse_success_only", "mcse")})
                fit_times = [record["fit_predict_seconds"] for record in successes]
                launch_times = [record["launcher_seconds"] for record in subset]
                rct, obs = expected_labels(cfg, group[0], slot)
                costs.append({"delta": delta, "n_rct": budget, "slot": slot, "method_id": cfg["methods"][slot],
                              "rct_labels": rct, "obs_labels": obs, "additional_tuning_labels": 0,
                              "requested": len(group), "successful": len(successes), "failed": len(group)-len(successes),
                              "mean_fit_predict_seconds_success_only": optional_mean(fit_times),
                              "median_fit_predict_seconds_success_only": statistics.median(fit_times) if fit_times else None,
                              "mean_launcher_seconds_all_attempts": optional_mean(launch_times),
                              "max_worker_self_rss_kib": max((r["worker_self_usage"]["maxrss_kib"] for r in subset if "worker_self_usage" in r), default=None),
                              "max_worker_child_rss_kib": max((r["worker_children_usage"]["maxrss_kib"] for r in subset if "worker_children_usage" in r), default=None)})
            for contrast, control in zip(CONTRASTS, ("rct_only", "obs_start")):
                item = source["paired"][contrast]
                differences = [records[(case["id"], "full")]["mse"] - records[(case["id"], control)]["mse"]
                               for case in group if all(records[(case["id"], slot)]["status"] == "success"
                                                        for slot in ("full", control))]
                if (item["expected_pairs"] != len(group) or item["successful_pairs"] != len(differences)
                        or item["complete"] != (len(differences) == len(group))):
                    raise ValueError("paired coverage differs from terminal records")
                close(optional_mean(differences), item["mean"], "paired mean")
                close(optional_mcse(differences, len(group)), item["mcse"], "paired MCSE")
                ci = item["ci95"]
                if not item["complete"] and ci is not None:
                    raise ValueError("incomplete paired interval was not withheld")
                pair = {"delta": delta, "n_rct": budget, "contrast": contrast,
                        **{name:item[name] for name in ("expected_pairs", "successful_pairs", "complete", "mean", "mcse")},
                        "ci95_low": ci[0] if ci is not None else None, "ci95_high": ci[1] if ci is not None else None}
                pairs.append(pair)
                row.update({contrast + "_" + name: pair[name] for name in
                            ("successful_pairs", "complete", "mean", "mcse", "ci95_low", "ci95_high")})
            wide.append(row)
    return scores, pairs, costs, wide


def write_csv(path, rows, fields):
    def scalar(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite CSV scalar")
            return format(value, ".17g")
        return value
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows({name: scalar(row[name]) for name in fields} for row in rows)


def figures(cfg, scores, pairs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "obsrct-gp-source-pilot-v1", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    budgets, deltas, nseed = cfg["rct_budgets"], cfg["deltas"], len(cfg["seeds"])
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.45*len(budgets), 3.5), squeeze=False, sharey=True)
    for j, budget in enumerate(budgets):
        ax = axes[0, j]
        for slot in SLOTS:
            subset = [next(row for row in scores if row["n_rct"] == budget and row["delta"] == delta and row["slot"] == slot)
                      for delta in deltas]
            available = [row for row in subset if row["mean_mse_success_only"] is not None]
            ax.plot([row["delta"] for row in available], [row["mean_mse_success_only"] for row in available],
                    marker="o", color=COLORS[slot], label=LABELS[slot], linewidth=1.6, markersize=4.5)
            for row in available:
                if row["mcse"] is not None:
                    ax.errorbar(row["delta"], row["mean_mse_success_only"], yerr=row["mcse"],
                                color=COLORS[slot], capsize=3, linewidth=1)
                if not row["complete"]:
                    ax.annotate(f"* {row['successful']}/{nseed}",
                                (row["delta"], row["mean_mse_success_only"]), xytext=(4, 5),
                                textcoords="offset points", fontsize=8, color=COLORS[slot])
            if not available:
                ax.text(.5, .5, f"{LABELS[slot]}: no successful fits", transform=ax.transAxes, fontsize=8)
        ax.set(title=f"RCT fitting budget B={budget}", xlabel=r"Observational discrepancy $\delta$", xticks=deltas)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Trial-target CATE MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .975), ncol=3, frameon=False)
    fig.text(.5, .025, f"Development diagnostic; {nseed} shared seed blocks.\nBars: 1 MCSE for complete groups; *: conditional success mean.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .88))
    save_figure(fig, output, "mse_by_delta")
    plt.close(fig)

    fig, axes = plt.subplots(2, len(budgets), figsize=(3.45*len(budgets), 5.5), squeeze=False, sharex=True, sharey="row")
    for i, (contrast, control) in enumerate(zip(CONTRASTS, ("rct_only", "obs_start"))):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            subset = [next(row for row in pairs if row["n_rct"] == budget and row["delta"] == delta and row["contrast"] == contrast)
                      for delta in deltas]
            available = [row for row in subset if row["mean"] is not None]
            ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
            ax.plot([row["delta"] for row in available], [row["mean"] for row in available], marker="o",
                    color=COLORS[control], linewidth=1.5, markersize=4.5)
            for row in available:
                if row["ci95_low"] is not None:
                    ax.errorbar(row["delta"], row["mean"],
                                yerr=[[max(0., row["mean"]-row["ci95_low"])], [max(0., row["ci95_high"]-row["mean"])]],
                                color=COLORS[control], capsize=3, linewidth=1)
                if not row["complete"]:
                    ax.annotate(f"* {row['successful_pairs']}/{nseed}", (row["delta"], row["mean"]),
                                xytext=(4, 5), textcoords="offset points", fontsize=8)
            if i == 0:
                ax.set_title(f"RCT fitting budget B={budget}")
            if j == 0:
                ax.set_ylabel("Full − RCT only\nMSE difference" if control == "rct_only" else "Full − OBS-conditioned start\nMSE difference")
            if i == 1:
                ax.set_xlabel(r"Observational discrepancy $\delta$")
            ax.set_xticks(deltas)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Paired risk differences: negative favors OBS + RCT", y=.985, fontsize=12)
    fig.text(.5, .02, "Approximate pointwise 95% paired t intervals; separate row scales.\n*: incomplete pairs, interval withheld. No multiplicity decisions.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    save_figure(fig, output, "paired_differences")
    plt.close(fig)


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".svg"), metadata={"Date": None})
    fig.savefig(output / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fits, analysis, output = guard(args.input), guard(args.analysis), guard(args.output)
    if output.exists() or output.is_relative_to(fits) or output.is_relative_to(analysis):
        raise ValueError("output must be a fresh directory outside both inputs")
    manifest, summary, records, conditions = authenticate(fits, analysis)
    scalars = scalar_rows(manifest, summary, records)
    scores, pairs, costs, wide = tables(manifest, summary, records, conditions)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "scalar_records.csv", scalars, SCALAR_FIELDS)
    write_csv(output / "condition_scores.csv", scores, SCORE_FIELDS)
    write_csv(output / "paired_contrasts.csv", pairs, PAIR_FIELDS)
    write_csv(output / "cost_summary.csv", costs, COST_FIELDS)
    write_csv(output / "condition_summary.csv", wide, list(wide[0]))
    figures(manifest["config"], scores, pairs, output)
    cfg = manifest["config"]
    readme = ["# Three-source GP development pilot: figures and scalar tables", "",
              f"Complete terminal coverage: {summary['success']} successful, {summary['failed']} failed, zero missing fits.", "",
              f"The two figures show condition-specific trial-target CATE risk and paired full-minus-control differences. Negative differences favor the full OBS+RCT recipe. The {len(cfg['seeds'])}-seed pilot is a development diagnostic, not unseen-mechanism evaluation or a source-paper table reproduction.", "",
              f"OBS-conditioned start predicts the trial task after conditioning only on observational outcomes. Under the fixed zero-mean two-task GP, its cross-task conditional mean is shrunk by rho/(1+kappa), with rho={cfg['rho']} and kappa=1e-9. It is not the unshrunk OBS task contrast.", "",
              "MSE bars are one Monte Carlo standard error only for complete success groups. Paired bars are the analysis's approximate pointwise 95% t intervals over sample-seed blocks; incomplete intervals are withheld, and conditional means are marked. No p-values or multiplicity decisions are introduced. Separate paired-figure rows use separate vertical scales.", "",
              f"CSV floats retain 17 significant digits; blanks indicate unavailable values, and rounding appears only in figure/Markdown display. Label columns give the frozen procedure's allowed source outcomes; successful records are checked against actual supplied/consumed counts. Full uses B RCT and {cfg['n_obs']} OBS outcomes, RCT-only uses B and 0, OBS-conditioned start uses 0 and {cfg['n_obs']}. The full procedure's two tuning-role rows alias charged fitting rows and supply no extra outcomes.", "",
              "Fit-time summaries condition on successful fits; launcher-time means include all attempts and exclude queue delay. Worker self and child RSS columns are separate high-water marks, not simultaneous process-group or whole-job peaks. Concurrent jobs were not a controlled runtime-ratio study.", "",
              "The renderer authenticates the saved terminal-record collection and its scalar agreement with the analysis; it does not refit models or independently reconstruct target/prediction arithmetic. Raw fitting arrays, query truth, predictions and full logs remain server evidence. figure_source.json binds the input hashes and exported files.", "",
              "| Delta | RCT rows | Full MSE | RCT-only MSE | OBS start MSE | Full−RCT [pointwise 95% CI] | Full−OBS start [pointwise 95% CI] |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    display = lambda x: "withheld" if x is None else f"{x:.6g}"
    for row in wide:
        risks = [display(row[slot+"_mean_mse_success_only"]) + ("*" if row[slot+"_failed"] else "") for slot in SLOTS]
        intervals = []
        for contrast in CONTRASTS:
            mean = display(row[contrast+"_mean"])
            low, high = row[contrast+"_ci95_low"], row[contrast+"_ci95_high"]
            intervals.append(mean + (f" [{display(low)}, {display(high)}]" if low is not None else " [withheld]"))
        readme.append(f"| {row['delta']} | {row['n_rct']} | " + " | ".join(risks+intervals) + " |")
    readme.append("")
    (output / "README.md").write_text("\n".join(readme))
    source = {"schema_version": 1, "scope": manifest["config"]["scope"],
              "renderer_sha256": file_hash(__file__), "producer_bindings": manifest["bindings"],
              "input_manifest_file_sha256": file_hash(fits / "manifest.json"),
              "input_manifest_payload_sha256": manifest["manifest_sha256"],
              "input_analysis_sha256": file_hash(analysis / "summary.json"),
              "terminal_record_sha256": summary["record_sha256"],
              "source_config": manifest["config"], "coverage": {key:summary[key] for key in ("expected_records", "success", "failed", "missing")},
              "rows": {"scalar_records.csv": len(scalars), "condition_scores.csv": len(scores),
                       "paired_contrasts.csv": len(pairs), "condition_summary.csv": len(wide), "cost_summary.csv": len(costs)},
              "plot_data": {"scores": scores, "pairs": pairs},
              "environment": {"python": platform.python_version(),
                              **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}},
              "output_sha256": {p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "figure_source.json").write_text(json.dumps(source, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(canonical({"status": "rendered", "files": len(source["output_sha256"])+1,
                     "records": len(scalars), "conditions": len(wide)}))


if __name__ == "__main__":
    main()
