"""Analyze the TSR static study exactly as declared in docs/TSR_STUDY.md.

Usage:
  python scripts/analyze_tsr_study.py --config configs/tsr_study.toml \
      --runs outputs/tsr/shard-* --catalog results/catalog_study --output outputs/tsr-analysis

Outputs: condition_scores.csv, primary_contrasts.csv, descriptive_contrasts.csv,
identity_check.json, budget_counts.md and analysis.json. The script reports numbers;
interpretation belongs in the result packet.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import tomllib

TSR, RCT_GP, OBS, TREES = "tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only", "rct_t_extra_trees"
METHODS = (TSR, RCT_GP, OBS, TREES)
CATALOG_RCT_ONLY = ("rct_t_ridge", "rct_t_extra_trees", "rct_dr_ridge", "rct_dr_extra_trees")
IDENTITY_TOL = 1e-9
HEAVY_TAIL_PRESET = "structural:M11"


def t_two_sided(t: float, df: int) -> float:
    from scipy import stats

    return float(2 * stats.t.sf(abs(t), df))


def t_quantile(df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.975, df))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def paired(diffs: list[float], inference: bool) -> dict:
    n = len(diffs)
    mean = sum(diffs) / n
    out = {"n": n, "mean": mean}
    if not inference or n < 2:
        return {**out, "mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": None}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    q = t_quantile(n - 1)
    p = t_two_sided(mean / se, n - 1) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {**out, "mcse": se, "ci95_lower": mean - q * se, "ci95_upper": mean + q * se, "p_two_sided": p}


def load_runs(run_dirs: list[Path]) -> tuple[dict, dict]:
    records, identities = {}, set()
    for run in run_dirs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        identities.add(json.dumps({"source_fingerprint": manifest.get("source_fingerprint"),
                                   "environment": manifest.get("environment")}, sort_keys=True))
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["scenario"], int(row["seed"]), row["method"])
            if key in records:
                raise SystemExit(f"duplicate cell across shards: {key}")
            records[key] = row
    if len(identities) != 1:
        raise SystemExit(f"shards have {len(identities)} different source/environment identities")
    return records, json.loads(identities.pop())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    if tuple(cfg["run"]["methods"]) != METHODS:
        raise SystemExit("configuration recipes differ from the protocol")
    seeds = [int(s) for s in cfg["run"]["seeds"]]
    scenarios = [f"{s['kind']}:{s['id']}@{s['label']}" for s in cfg["scenarios"]]
    records, identity = load_runs(a.runs)
    a.output.mkdir(parents=True, exist_ok=True)

    # ---- coverage ----
    mse: dict[tuple, float] = {}
    status = defaultdict(int)
    failures = []
    for sc in scenarios:
        for seed in seeds:
            for m in METHODS:
                row = records.get((sc, seed, m))
                if row is None:
                    status["missing"] += 1
                    failures.append({"scenario": sc, "seed": seed, "method": m, "status": "missing"})
                elif row["status"] != "ok":
                    status["failed"] += 1
                    failures.append({"scenario": sc, "seed": seed, "method": m, "status": row["status"],
                                     "stage": row.get("stage"), "error_type": row.get("error_type")})
                else:
                    status["ok"] += 1
                    mse[(sc, seed, m)] = float(row["metrics"]["cate_mse"])
    extra = set(records) - {(sc, s, m) for sc in scenarios for s in seeds for m in METHODS}
    if extra:
        raise SystemExit(f"{len(extra)} records outside the declared grid")

    # ---- data identity against the catalog's published per-seed scores ----
    published = {}
    for path in sorted((a.catalog / "designs").glob("*/replicate_scores.csv")):
        for row in csv.DictReader(path.open()):
            if row["method"] == TREES and row["status"] == "ok":
                published[(row["scenario"], int(row["seed"]))] = float(row["cate_mse"])
    diffs = [abs(mse[(sc, s, TREES)] - published[(sc, s)]) for sc in scenarios for s in seeds
             if (sc, s, TREES) in mse and (sc, s) in published]
    identity_ok = len(diffs) == len(scenarios) * len(seeds) and max(diffs, default=math.inf) <= IDENTITY_TOL
    identity_report = {"compared_cells": len(diffs), "expected_cells": len(scenarios) * len(seeds),
                       "max_abs_difference": max(diffs, default=None), "tolerance": IDENTITY_TOL,
                       "passed": identity_ok}
    (a.output / "identity_check.json").write_text(json.dumps(identity_report, indent=2) + "\n")

    # ---- per-condition scores ----
    scores = []
    for sc in scenarios:
        for m in METHODS:
            vals = [mse[(sc, s, m)] for s in seeds if (sc, s, m) in mse]
            scores.append({"scenario": sc, "method": m, "successful": len(vals), "expected": len(seeds),
                           "mean_mse": sum(vals) / len(vals) if vals else None})
    write_csv(a.output / "condition_scores.csv", scores)

    # ---- primary families ----
    def contrast_rows(first: str, second: str, family: str, inferential: bool) -> list[dict]:
        rows = []
        for sc in scenarios:
            complete = all((sc, s, first) in mse and (sc, s, second) in mse for s in seeds)
            d = [mse[(sc, s, first)] - mse[(sc, s, second)] for s in seeds
                 if (sc, s, first) in mse and (sc, s, second) in mse]
            eligible = inferential and not sc.startswith(HEAVY_TAIL_PRESET + "@")
            summary = paired(d, eligible) if d else {"n": 0, "mean": None, "mcse": None, "ci95_lower": None,
                                                     "ci95_upper": None, "p_two_sided": None}
            rows.append({"family": family, "scenario": sc, "first": first, "second": second,
                         "complete": complete, "inference_eligible": eligible, **summary})
        return rows

    primary = []
    decisions = {}
    for family, second in (("A_obs_offset", RCT_GP), ("B_complete_procedure", TREES)):
        rows = contrast_rows(TSR, second, family, True)
        tests = [r for r in rows if r["inference_eligible"]]
        all_complete = all(r["complete"] for r in tests)
        if all_complete:
            adjusted = holm([r["p_two_sided"] for r in tests])
            for r, p in zip(tests, adjusted):
                r["holm_p"] = p
                r["decision"] = "reject_05" if p < 0.05 else "not_rejected"
        for r in rows:
            r.setdefault("holm_p", None)
            r.setdefault("decision", "descriptive_heavy_tail" if not r["inference_eligible"]
                         else "withheld_incomplete_family")
        decisions[family] = {"tests": len(tests), "complete": all_complete,
                             "rejected_favoring_tsr": sum(1 for r in tests if r["decision"] == "reject_05" and r["mean"] < 0),
                             "rejected_favoring_other": sum(1 for r in tests if r["decision"] == "reject_05" and r["mean"] > 0)}
        primary += rows
    write_csv(a.output / "primary_contrasts.csv", primary)
    write_csv(a.output / "descriptive_contrasts.csv", contrast_rows(TSR, OBS, "descriptive_obs_only", False))

    # ---- budget counts against catalog recipes (descriptive; requires identity) ----
    lines = []
    if identity_ok:
        catalog = {}
        for row in csv.DictReader((a.catalog / "condition_scores.csv").open()):
            if row["applicable"] == "True" and int(row["successful"]) > 0:
                catalog[(row["scenario"], row["method"])] = float(row["mean"])
        tsr_mean = {r["scenario"]: r["mean_mse"] for r in scores if r["method"] == TSR and r["mean_mse"] is not None}
        gp_mean = {r["scenario"]: r["mean_mse"] for r in scores if r["method"] == RCT_GP and r["mean_mse"] is not None}
        budgets = sorted({sc.split("@")[1] for sc in scenarios})
        lines = ["| Comparison | " + " | ".join(budgets) + " |", "| --- |" + " ---: |" * len(budgets)]
        for label, ref in (("TSR below RCT-only ExtraTrees", lambda sc: catalog.get((sc, TREES))),
                           ("TSR below all four catalog RCT-only recipes",
                            lambda sc: min(catalog[(sc, m)] for m in CATALOG_RCT_ONLY) if all((sc, m) in catalog for m in CATALOG_RCT_ONLY) else None),
                           ("TSR below matched RCT-only GP", lambda sc: gp_mean.get(sc))):
            cells = []
            for b in budgets:
                pairs = [(tsr_mean[sc], ref(sc)) for sc in scenarios if sc.endswith("@" + b) and sc in tsr_mean and ref(sc) is not None]
                cells.append(f"{sum(t < r for t, r in pairs)}/{len(pairs)}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    else:
        lines = ["Data identity failed; cross-study comparisons are not reported."]
    (a.output / "budget_counts.md").write_text("\n".join(lines) + "\n")

    analysis = {"protocol": "docs/TSR_STUDY.md", "coverage": dict(status), "failures": failures,
                "identity": identity_report, "families": decisions, "run_identity": identity}
    (a.output / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str) + "\n")
    print(json.dumps({"coverage": dict(status), "identity_passed": identity_ok, "families": decisions}, indent=2))


if __name__ == "__main__":
    main()
