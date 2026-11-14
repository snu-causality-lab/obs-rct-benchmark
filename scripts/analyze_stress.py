"""Analyze stress study 1 exactly as declared in docs/V1_STRESS_CORNET.md.

  python scripts/analyze_stress.py --config configs/v1_stress_cornet.toml --runs outputs/stress-seed-* --output stress-analysis

Outputs: replicate_scores.csv, condition_scores.csv, contrasts.csv (with Holm verdicts),
readout.csv (CorNet verdict sequences by dial and trial size), dose_response.csv and
analysis.json. Each requested cell must have a terminal record from one source and environment.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

from analyze_v1_core import ALPHA, holm, paired_t
from make_stress_configs import shards as declared_shards, validate
from obs_rct_benchmark.stress_dgp import EXTRAPOLATION_DIALS

REFERENCE = "rct_selected"
OBS_RECIPES = ("cornet_raw", "cornet_official", "datafusionppi", "obs_t_ridge", "obs_t_extra_trees",
               "naive_pooled_t_ridge", "naive_pooled_t_extra_trees")


class AnalysisError(ValueError):
    """The inputs do not match the declared study; nothing is reported."""


def parse(scenario: str) -> tuple[str, int, int]:
    kind_dial, label = scenario.split("@")
    level, rct = label.split("_")
    return kind_dial.split(":")[1], int(level[1:]), int(rct[1:])


def collect(config, runs: list[Path], *, check_source: bool = True) -> dict:
    from obs_rct_benchmark.runner import environment_provenance, method_seed, record_id, source_provenance

    expected = {s.config_hash: (n, s) for n, s in declared_shards(config)}
    seen, records, ident = set(), {}, set()
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        name, shard = expected.get(manifest.get("config_hash"), (None, None))
        if shard is None or manifest.get("config") != shard.payload() or name in seen:
            raise AnalysisError(f"{run}: not a declared shard, or given twice")
        seen.add(name)
        ident.add(json.dumps({k: manifest.get(k) for k in ("source", "environment", "external_methods")}, sort_keys=True))
        if check_source and (manifest["source"] != source_provenance() or manifest["environment"] != environment_provenance()):
            raise AnalysisError(f"{run}: producer source or environment differs from the analysis package")
        h = shard.config_hash
        cells = {(s.key, seed, m) for s in shard.scenarios for seed in shard.seeds for m in shard.methods}
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            cell = (row["scenario"], int(row["seed"]), row["method"])
            if cell not in cells or cell in records:
                raise AnalysisError(f"{run}: unrequested or duplicate cell {cell}")
            if row["config_hash"] != h or row["method_seed"] != method_seed(*cell) or row["record_id"] != record_id(shard, *cell):
                raise AnalysisError(f"{run}: record identity mismatch for {cell}")
            if row["status"] not in {"ok", "failed"}:
                raise AnalysisError(f"{cell}: status is not terminal")
            records[cell] = row
            cells.discard(cell)
        if cells:
            raise AnalysisError(f"{run}: {len(cells)} requested cells lack a terminal record")
    if seen != {n for n, _ in expected.values()}:
        raise AnalysisError("declared shards are missing")
    if len(ident) != 1:
        raise AnalysisError("shards differ in source, environment or external-method provenance")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(config, records: dict, output: Path) -> dict:
    from scipy import stats

    output.mkdir(parents=True, exist_ok=False)
    seeds = list(config.seeds)
    scenarios = [s.key for s in config.scenarios]
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}
    write_csv(output / "replicate_scores.csv", [
        {"scenario": k[0], "seed": k[1], "method": k[2], "status": r["status"],
         "cate_mse": r["metrics"]["cate_mse"] if r["status"] == "ok" else "",
         "error_type": r.get("error_type") or "", "stage": r.get("stage") or "",
         "selected": ((r.get("fit_info") or {}).get("selected") or "") if k[2] == REFERENCE else ""}
        for k, r in sorted(records.items())])
    conditions, dose = [], []
    for sc in scenarios:
        dial, level, rct = parse(sc)
        for m in config.methods:
            vals = [mse[(sc, s, m)] for s in seeds if (sc, s, m) in mse]
            conditions.append({"scenario": sc, "dial": dial, "level": level, "rct": rct, "method": m,
                               "successful": len(vals), "mean_mse": math.fsum(vals) / len(vals) if vals else None})
            pairs = [(mse[(sc, s, m)], mse[(sc, s, REFERENCE)]) for s in seeds
                     if (sc, s, m) in mse and (sc, s, REFERENCE) in mse]
            if m != REFERENCE and len(pairs) >= 2:
                logs = [math.log(a / b) for a, b in pairs]
                mean = math.fsum(logs) / len(logs)
                se = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (len(logs) - 1) / len(logs))
                q = float(stats.t.ppf(0.975, len(logs) - 1))
                dose.append({"dial": dial, "level": level, "rct": rct, "method": m, "pairs": len(pairs),
                             "ratio_of_means": math.fsum(a for a, _ in pairs) / math.fsum(b for _, b in pairs),
                             "geometric_mean_ratio": math.exp(mean),
                             "gmr_ci95_lower": math.exp(mean - q * se), "gmr_ci95_upper": math.exp(mean + q * se)})
    write_csv(output / "condition_scores.csv", conditions)
    write_csv(output / "dose_response.csv", dose)

    rows_all, summary = [], {}
    for recipe in OBS_RECIPES:
        rows = []
        for sc in scenarios:
            dial, level, rct = parse(sc)
            ok = [s for s in seeds if (sc, s, recipe) in mse and (sc, s, REFERENCE) in mse]
            diffs = [mse[(sc, s, recipe)] - mse[(sc, s, REFERENCE)] for s in ok]
            row = {"family": recipe, "scenario": sc, "dial": dial, "level": level, "rct": rct,
                   "extrapolation": dial in EXTRAPOLATION_DIALS, "pairs": len(ok),
                   "recipe_failures": sum((sc, s, recipe) not in mse for s in seeds),
                   "complete": len(ok) == len(seeds),
                   "mean_difference": math.fsum(diffs) / len(diffs) if diffs else None,
                   "wins_failure_as_loss": sum(1 for s in ok if mse[(sc, s, recipe)] < mse[(sc, s, REFERENCE)])}
            row.update(paired_t(diffs) if row["complete"] else
                       {"mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": 1.0, "zero_variance": None})
            rows.append(row)
        for row, p in zip(rows, holm([r["p_two_sided"] for r in rows])):
            row["holm_p"] = p
            row["verdict"] = ("untested" if not row["complete"] else
                              "better" if p <= ALPHA and row["mean_difference"] < 0 else
                              "worse" if p <= ALPHA and row["mean_difference"] > 0 else "unresolved")
        summary[recipe] = {"size": len(rows), **{
            f"{grp}_{v}": sum(1 for r in rows if r["verdict"] == v and r["extrapolation"] == (grp == "extrapolation"))
            for grp in ("other", "extrapolation") for v in ("better", "worse", "unresolved", "untested")}}
        rows_all += rows
    write_csv(output / "contrasts.csv", rows_all)

    readout = []
    verdict = {(r["family"], r["dial"], r["level"], r["rct"]): r["verdict"] for r in rows_all}
    for recipe in ("cornet_raw", "cornet_official"):
        for dial in dict.fromkeys(parse(sc)[0] for sc in scenarios):
            for rct in (128, 512):
                seq = [verdict[(recipe, dial, level, rct)] for level in range(4)]
                # Declared states; "untested" at level 3 counts as not better.
                state = ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
                         else "stopped" if seq[3] != "better" else "kept")
                readout.append({"recipe": recipe, "dial": dial, "rct": rct,
                                **{f"level_{i}": v for i, v in enumerate(seq)}, "state": state})
    write_csv(output / "readout.csv", readout)
    analysis = {"protocol": "docs/V1_STRESS_CORNET.md", "requested": len(records),
                "coverage": dict(Counter(r["status"] for r in records.values())),
                "failures": dict(Counter(f"{k[2]}:{r.get('error_type')}" for k, r in records.items() if r["status"] != "ok")),
                "families": summary}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    config = load_config(a.config)
    validate(config)
    result = analyze(config, collect(config, a.runs), a.output)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
