"""Analyze the TSR static study exactly as declared in docs/TSR_STUDY.md.

Usage:
  python scripts/analyze_tsr_study.py --config configs/tsr_study.toml \
      --runs outputs/tsr/shard-* --catalog results/catalog_study --output outputs/tsr-analysis

Every shard must be a seed split of the declared configuration, produced by the
same source and environment as the installed analysis package. Outputs:
condition_scores.csv, primary_contrasts.csv, descriptive_contrasts.csv,
identity_check.json, budget_counts.md and analysis.json. The script reports numbers;
interpretation belongs in the result packet.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from collections import defaultdict
from pathlib import Path

TSR, RCT_GP, OBS, TREES, RIDGE = "tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only", "rct_t_extra_trees", "rct_t_ridge"
METHODS = (TSR, RCT_GP, OBS, TREES, RIDGE)
STAGE1_METHODS = (TSR, OBS)
CATALOG_CONTROLS = (TREES, RIDGE)
CATALOG_RCT_ONLY = ("rct_t_ridge", "rct_t_extra_trees", "rct_dr_ridge", "rct_dr_extra_trees")
IDENTITY_TOL = 1e-9
HEAVY_TAIL_PRESET = "structural:M11"


class AnalysisError(ValueError):
    """The inputs do not match the declared study; nothing is reported."""


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def paired(diffs: list[float], inference: bool) -> dict:
    from scipy import stats

    n = len(diffs)
    mean = sum(diffs) / n
    out = {"n": n, "mean": mean}
    if not inference or n < 2:
        return {**out, "mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": None}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    q = float(stats.t.ppf(0.975, n - 1))
    p = float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {**out, "mcse": se, "ci95_lower": mean - q * se, "ci95_upper": mean + q * se, "p_two_sided": p}


def collect(config, runs: list[Path], expected_sha: str | None, *, check_source: bool = True) -> dict:
    """Validate shards against the declared configuration and return cell records."""
    from obs_rct_benchmark.runner import environment_provenance, method_seed, record_id, source_provenance

    records, seen_seeds, identities = {}, set(), set()
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        seeds = tuple(int(s) for s in manifest["config"]["run"]["seeds"])
        if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config.seeds):
            raise AnalysisError(f"{run}: duplicate or undeclared seeds")
        if seen_seeds.intersection(seeds):
            raise AnalysisError(f"{run}: seeds overlap another shard")
        seen_seeds.update(seeds)
        shard = replace(config, seeds=seeds)
        if manifest.get("config") != shard.payload() or manifest.get("config_hash") != shard.config_hash:
            raise AnalysisError(f"{run}: shard settings differ from the declared configuration")
        identities.add(json.dumps({"source": manifest["source"], "environment": manifest["environment"],
                                   "external_methods": manifest.get("external_methods")}, sort_keys=True))
        if check_source and (manifest["source"] != source_provenance() or manifest["environment"] != environment_provenance()):
            raise AnalysisError(f"{run}: producer source or environment differs from the analysis package")
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["scenario"], int(row["seed"]), row["method"])
            if key in records:
                raise AnalysisError(f"duplicate cell {key}")
            if (row["config_hash"] != shard.config_hash or row["method_seed"] != method_seed(*key)
                    or row["record_id"] != record_id(shard, *key)):
                raise AnalysisError(f"{run}: record identity mismatch for {key}")
            if row["status"] == "ok" and row["method"] in STAGE1_METHODS:
                info = row.get("fit_info") or {}
                if info.get("stage1_device") != "cuda" or info.get("stage1_model_sha256") != expected_sha:
                    raise AnalysisError(f"{key}: Stage 1 did not run on CUDA with the pinned checkpoint")
            records[key] = row
    if seen_seeds != set(config.seeds):
        raise AnalysisError("shards do not cover the declared seeds")
    if len(identities) != 1:
        raise AnalysisError("shards differ in source, environment or external-method provenance")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(config, records: dict, catalog: Path, output: Path) -> dict:
    seeds = list(config.seeds)
    scenarios = [s.key for s in config.scenarios]
    output.mkdir(parents=True, exist_ok=True)
    grid = {(sc, s, m) for sc in scenarios for s in seeds for m in METHODS}
    if set(records) - grid:
        raise AnalysisError("records outside the declared grid")

    mse, status, failures = {}, defaultdict(int), []
    for key in sorted(grid):
        row = records.get(key)
        if row is None:
            status["missing"] += 1
            failures.append({"cell": key, "status": "missing"})
        elif row["status"] != "ok":
            status["failed"] += 1
            failures.append({"cell": key, "status": row["status"], "stage": row.get("stage"), "error_type": row.get("error_type")})
        else:
            status["ok"] += 1
            mse[key] = float(row["metrics"]["cate_mse"])

    # Data identity: reruns of the catalog's controls must reproduce its per-seed scores.
    published = {}
    for path in sorted((catalog / "designs").glob("*/replicate_scores.csv")):
        for row in csv.DictReader(path.open()):
            if row["method"] in CATALOG_CONTROLS and row["status"] == "ok":
                published[(row["scenario"], int(row["seed"]), row["method"])] = float(row["cate_mse"])
    identity = {}
    for m in CATALOG_CONTROLS:
        d = [abs(mse[(sc, s, m)] - published[(sc, s, m)]) for sc in scenarios for s in seeds
             if (sc, s, m) in mse and (sc, s, m) in published]
        identity[m] = {"compared": len(d), "expected": len(scenarios) * len(seeds),
                       "max_abs_difference": max(d, default=None)}
        identity[m]["passed"] = identity[m]["compared"] == identity[m]["expected"] and max(d, default=math.inf) <= IDENTITY_TOL
    identity_ok = all(v["passed"] for v in identity.values())
    (output / "identity_check.json").write_text(json.dumps({"tolerance": IDENTITY_TOL, "recipes": identity,
                                                              "passed": identity_ok}, indent=2) + "\n")

    scores = []
    for sc in scenarios:
        for m in METHODS:
            vals = [mse[(sc, s, m)] for s in seeds if (sc, s, m) in mse]
            scores.append({"scenario": sc, "method": m, "successful": len(vals), "expected": len(seeds),
                           "mean_mse": sum(vals) / len(vals) if vals else None})
    write_csv(output / "condition_scores.csv", scores)

    def rows_for(first: str, second: str, family: str, inferential: bool) -> list[dict]:
        out = []
        for sc in scenarios:
            pairs = [(mse[(sc, s, first)], mse[(sc, s, second)]) for s in seeds if (sc, s, first) in mse and (sc, s, second) in mse]
            eligible = inferential and not sc.startswith(HEAVY_TAIL_PRESET + "@")
            empty = {"n": 0, "mean": None, "mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": None}
            out.append({"family": family, "scenario": sc, "first": first, "second": second,
                        "complete": len(pairs) == len(seeds), "inference_eligible": eligible,
                        **(paired([a - b for a, b in pairs], eligible) if pairs else empty)})
        return out

    primary, families = [], {}
    for family, seconds in (("A_obs_offset", (RCT_GP,)), ("B_complete_procedure", CATALOG_CONTROLS)):
        rows = [r for second in seconds for r in rows_for(TSR, second, family, True)]
        tests = [r for r in rows if r["inference_eligible"]]
        complete = all(r["complete"] for r in tests)  # required pairs: every inferential condition
        if complete:
            for r, p in zip(tests, holm([r["p_two_sided"] for r in tests])):
                r["holm_p"], r["decision"] = p, ("reject_05" if p < 0.05 else "not_rejected")
        for r in rows:
            r.setdefault("holm_p", None)
            r.setdefault("decision", "descriptive_heavy_tail" if not r["inference_eligible"] else "withheld_incomplete_family")
        families[family] = {
            "tests": len(tests), "complete": complete,
            **{f"{second}: {kind}": sum(1 for r in tests if r["second"] == second and r["decision"] == "reject_05" and (r["mean"] < 0) == (kind == "tsr_lower"))
               for second in seconds for kind in ("tsr_lower", "tsr_higher")}}
        primary += rows
    write_csv(output / "primary_contrasts.csv", primary)
    write_csv(output / "descriptive_contrasts.csv", rows_for(TSR, OBS, "descriptive_obs_only", False))

    lines = []
    complete_mean = {(r["scenario"], r["method"]): r["mean_mse"] for r in scores if r["successful"] == len(seeds)}
    if identity_ok:
        catalog_means = {}
        for row in csv.DictReader((catalog / "condition_scores.csv").open()):
            if row["applicable"] == "True" and row["successful"] == row["expected"]:
                catalog_means[(row["scenario"], row["method"])] = float(row["mean"])
        budgets = sorted({sc.split("@")[1] for sc in scenarios})
        lines = ["| TSR mean MSE lower than | " + " | ".join(budgets) + " |", "| --- |" + " ---: |" * len(budgets)]
        refs = (("matched RCT-only GP", lambda sc: complete_mean.get((sc, RCT_GP))),
                ("RCT-only ExtraTrees", lambda sc: catalog_means.get((sc, TREES))),
                ("RCT-only ridge", lambda sc: catalog_means.get((sc, RIDGE))),
                ("all four catalog RCT-only recipes",
                 lambda sc: min(catalog_means[(sc, m)] for m in CATALOG_RCT_ONLY) if all((sc, m) in catalog_means for m in CATALOG_RCT_ONLY) else None))
        for label, ref in refs:
            cells = []
            for b in budgets:
                pairs = [(complete_mean[(sc, TSR)], ref(sc)) for sc in scenarios
                         if sc.endswith("@" + b) and (sc, TSR) in complete_mean and ref(sc) is not None]
                cells.append(f"{sum(t < r for t, r in pairs)}/{len(pairs)}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    else:
        lines = ["Data identity failed; cross-study comparisons are not reported."]
    (output / "budget_counts.md").write_text("\n".join(lines) + "\n")

    analysis = {"protocol": "docs/TSR_STUDY.md", "coverage": dict(status), "failures": failures,
                "identity_passed": identity_ok, "families": families}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str) + "\n")
    return analysis


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    config = load_config(a.config)
    if tuple(config.methods) != METHODS:
        raise SystemExit("configuration recipes differ from the protocol")
    shas = {opts.get("model_sha256") for name, opts in config.method_options.items() if name in STAGE1_METHODS}
    if len(shas) != 1 or None in shas:
        raise SystemExit("Stage 1 recipes must pin one checkpoint")
    records = collect(config, a.runs, shas.pop())
    result = analyze(config, records, a.catalog, a.output)
    print(json.dumps({k: result[k] for k in ("coverage", "identity_passed", "families")}, indent=2))


if __name__ == "__main__":
    main()
