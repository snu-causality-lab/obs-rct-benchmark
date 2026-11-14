"""Analyze the v1 core study exactly as declared in docs/V1_CORE_STUDY.md.

Usage:
  python scripts/analyze_v1_core.py --config configs/v1_core_study.toml \
      --runs outputs/v1core/group-* --output outputs/v1core-analysis

Every requested shard must be present once, produced from the declared
configuration by one source and environment. Every requested cell must have a
terminal record; an incomplete shard is an infrastructure failure to rerun, not
a method failure. Outputs: replicate_scores.csv, condition_scores.csv,
primary_contrasts.csv, descriptive_contrasts.csv, selection.csv,
identity_check.json, headline.md and analysis.json. The script reports
numbers; interpretation belongs in the result packet.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from make_v1_core_configs import shards as declared_shards, validate

REFERENCE = "rct_selected"
CATALOG_REFERENCE = "rct_t_extra_trees"
CANDIDATES = ("rct_dr_extra_trees", "rct_dr_ridge", "rct_t_extra_trees", "rct_t_ridge")
RCT_ONLY = (REFERENCE, *CANDIDATES)
HEAVY_TAIL = "structural:M11"
ALPHA = 0.05
TRANSPORT = frozenset({
    "structural:M9", "addendum:C08", "paper:experimental_grounding_kps",
    "paper:integrative_r_learner_kps_normal_target_extension",
    "paper:integrative_r_learner_multivariate_paper",
    "paper:elastic_integrative_gen_data_overlap",
})


class AnalysisError(ValueError):
    """The inputs do not match the declared study; nothing is reported."""


def preset(scenario: str) -> str:
    return scenario.split("@", 1)[0]


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def paired_t(diffs: list[float]) -> dict:
    """Two-sided paired t; zero sample variance gives p = 1 (declared rule)."""
    from scipy import stats

    n = len(diffs)
    mean = math.fsum(diffs) / n
    variance = math.fsum((d - mean) ** 2 for d in diffs) / (n - 1)
    if variance == 0.0:
        return {"mcse": 0.0, "ci95_lower": mean, "ci95_upper": mean, "p_two_sided": 1.0,
                "zero_variance": True}
    se = math.sqrt(variance / n)
    q = float(stats.t.ppf(0.975, n - 1))
    p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    return {"mcse": se, "ci95_lower": mean - q * se, "ci95_upper": mean + q * se,
            "p_two_sided": p, "zero_variance": False}


def collect(config, runs: list[Path], *, check_source: bool = True) -> tuple[dict, dict]:
    """Authenticate shards and return (records, requested cells by shard)."""
    from obs_rct_benchmark.runner import (
        environment_provenance, external_method_provenance, method_seed, record_id,
        source_provenance,
    )

    expected = {shard.config_hash: (name, shard) for name, shard in declared_shards(config)}
    seen, records, identities, requested = set(), {}, set(), {}
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        name, shard = expected.get(manifest.get("config_hash"), (None, None))
        if shard is None or manifest.get("config") != shard.payload():
            raise AnalysisError(f"{run}: not a declared shard of this configuration")
        if name in seen:
            raise AnalysisError(f"{run}: shard {name} given twice")
        seen.add(name)
        # Shard groups request different recipes, so external provenance is checked per shard.
        identities.add(json.dumps({"source": manifest["source"], "environment": manifest["environment"]},
                                  sort_keys=True))
        if check_source:
            if (manifest["source"] != source_provenance()
                    or manifest["environment"] != environment_provenance()):
                raise AnalysisError(f"{run}: producer source or environment differs from the analysis package")
            if manifest.get("external_methods") != external_method_provenance(shard.methods):
                raise AnalysisError(f"{run}: external implementations differ from the analysis environment")
        cells = {(s.key, seed, m) for s in shard.scenarios for seed in shard.seeds for m in shard.methods}
        requested[name] = cells
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["scenario"], int(row["seed"]), row["method"])
            if key not in cells or key in records:
                raise AnalysisError(f"{run}: unrequested or duplicate cell {key}")
            if (row["config_hash"] != shard.config_hash or row["method_seed"] != method_seed(*key)
                    or row["record_id"] != record_id(shard, *key)):
                raise AnalysisError(f"{run}: record identity mismatch for {key}")
            if row["status"] not in {"ok", "failed"}:
                raise AnalysisError(f"{key}: status {row['status']!r} is not terminal")
            records[key] = row
    if seen != {name for name, _ in expected.values()}:
        raise AnalysisError(f"missing shards: {sorted({n for n, _ in expected.values()} - seen)}")
    if len(identities) != 1:
        raise AnalysisError("shards differ in source or environment")
    missing = set().union(*requested.values()) - set(records)
    if missing:
        raise AnalysisError(f"{len(missing)} requested cells lack a terminal record; rerun those shards")
    return records, requested


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family_conditions(config, recipe: str, applicable: set) -> list[str]:
    """Declared Holm family: conditions outside M11; RCT-only recipes at OBS 512 only."""
    out = []
    for s in config.scenarios:
        if preset(s.key) == HEAVY_TAIL or (s.key, recipe) not in applicable:
            continue
        if recipe in RCT_ONLY and s.parameters["n_obs"] != 512:
            continue
        out.append(s.key)
    return out


def analyze(config, records: dict, output: Path) -> dict:
    seeds = list(config.seeds)
    scenarios = [s.key for s in config.scenarios]
    applicable = {(sc, m) for (sc, _, m) in records}
    output.mkdir(parents=True, exist_ok=False)
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}

    replicate = []
    for key in sorted(records):
        row, info = records[key], records[key].get("fit_info") or {}
        replicate.append({
            "scenario": key[0], "seed": key[1], "method": key[2], "status": row["status"],
            "cate_mse": row["metrics"]["cate_mse"] if row["status"] == "ok" else "",
            "root_pehe": row["metrics"]["root_pehe"] if row["status"] == "ok" else "",
            "error_type": row.get("error_type") or "", "stage": row.get("stage") or "",
            "selected": (info.get("selected") or "") if key[2] == REFERENCE else "",
            "max_weight": info.get("max_weight", "") if key[2] == REFERENCE else "",
        })
    write_csv(output / "replicate_scores.csv", replicate)

    # RCT samples and method seeds do not depend on the OBS size: verify, then use OBS 512.
    identity = {}
    for m in RCT_ONLY:
        diffs, mismatched, compared = [], 0, 0
        for s in config.scenarios:
            if s.parameters["n_obs"] != 512:
                continue
            twin = s.key.replace("@o512_", "@o2048_")
            for seed in seeds:
                a, b = records[(s.key, seed, m)], records[(twin, seed, m)]
                compared += 1
                if a["status"] != b["status"]:
                    mismatched += 1
                elif a["status"] == "ok":
                    d = abs(mse[(s.key, seed, m)] - mse[(twin, seed, m)])
                    diffs.append(d)
                    mismatched += d != 0.0
        identity[m] = {"compared": compared, "mismatched": mismatched, "max_abs_difference": max(diffs, default=None)}
    identity_ok = all(v["mismatched"] == 0 for v in identity.values())
    # A chosen ridge refit must equal the standalone ridge fit; otherwise the
    # zero-variance rule would see rounding noise. Reported, never repaired.
    ridge = [(k, mse.get(k), mse.get((k[0], k[1], "rct_t_ridge"))) for k, r in records.items()
             if k[2] == REFERENCE and r["status"] == "ok"
             and (r.get("fit_info") or {}).get("selected") == "rct_t_ridge"]
    ridge_mismatch = [list(k) for k, a, b in ridge if a != b]
    (output / "identity_check.json").write_text(json.dumps(
        {"rule": "RCT-only records must be identical at OBS 512 and 2048; OBS 512 records are used",
         "recipes": identity, "passed": identity_ok,
         "ridge_refit": {"rule": "rct_selected records that chose rct_t_ridge equal the standalone record",
                         "compared": len(ridge), "mismatched": ridge_mismatch}}, indent=2) + "\n")

    conditions = []
    for sc in scenarios:
        for m in config.methods:
            if (sc, m) not in applicable:
                continue
            vals = [mse[(sc, s, m)] for s in seeds if (sc, s, m) in mse]
            q = statistics.quantiles(vals, n=4, method="inclusive") if len(vals) >= 2 else [None] * 3
            conditions.append({
                "scenario": sc, "method": m, "transport": preset(sc) in TRANSPORT,
                "expected": len(seeds), "successful": len(vals), "failed": len(seeds) - len(vals),
                "mean_mse": math.fsum(vals) / len(vals) if vals else None,
                "median_mse": statistics.median(vals) if vals else None,
                "q25_mse": q[0], "q75_mse": q[2],
                "mean_root_pehe": (math.fsum(float(records[(sc, s, m)]["metrics"]["root_pehe"]) for s in seeds
                                             if (sc, s, m) in mse) / len(vals)) if vals else None,
            })
    write_csv(output / "condition_scores.csv", conditions)
    mean = {(r["scenario"], r["method"]): r["mean_mse"] for r in conditions if r["successful"] == len(seeds)}

    primary, families = [], {}
    for recipe in config.methods:
        if recipe == REFERENCE:
            continue
        rows = []
        for sc in family_conditions(config, recipe, applicable):
            rk = sc.replace("@o2048_", "@o512_")  # the declared reference record
            ok = [s for s in seeds if (sc, s, recipe) in mse and (rk, s, REFERENCE) in mse]
            diffs = [mse[(sc, s, recipe)] - mse[(rk, s, REFERENCE)] for s in ok]
            wins = sum(1 for s in ok if mse[(sc, s, recipe)] < mse[(rk, s, REFERENCE)])
            row = {"family": recipe, "scenario": sc, "transport": preset(sc) in TRANSPORT,
                   "pairs": len(ok), "recipe_failures": sum((sc, s, recipe) not in mse for s in seeds),
                   "reference_failures": sum((rk, s, REFERENCE) not in mse for s in seeds),
                   "complete": len(ok) == len(seeds),
                   "mean_difference": math.fsum(diffs) / len(diffs) if diffs else None,
                   "wins_failure_as_loss": wins,
                   "wins_denominator": sum((rk, s, REFERENCE) in mse for s in seeds)}
            if row["complete"]:
                row.update(paired_t(diffs))
            else:
                row.update(mcse=None, ci95_lower=None, ci95_upper=None, p_two_sided=1.0, zero_variance=None)
            rows.append(row)
        for row, p in zip(rows, holm([r["p_two_sided"] for r in rows])):
            row["holm_p"] = p
            if not row["complete"]:
                row["verdict"] = "untested"
            elif p <= ALPHA and row["mean_difference"] < 0:
                row["verdict"] = "better"
            elif p <= ALPHA and row["mean_difference"] > 0:
                row["verdict"] = "worse"
            else:
                row["verdict"] = "unresolved"
        families[recipe] = {
            "size": len(rows),
            **{f"{group}_{verdict}": sum(1 for r in rows if r["verdict"] == verdict and r["transport"] == (group == "transport"))
               for group in ("other", "transport") for verdict in ("better", "worse", "unresolved", "untested")}}
        primary += rows
    write_csv(output / "primary_contrasts.csv", primary)

    descriptive, selection = [], []
    for sc in scenarios:
        best = None
        if all((sc, c) in mean for c in CANDIDATES):
            best = min(CANDIDATES, key=lambda c: (mean[(sc, c)], CANDIDATES.index(c)))
        ref = mean.get((sc, REFERENCE))
        for m in config.methods:
            if (sc, m) not in mean:
                continue
            descriptive.append({
                "scenario": sc, "method": m, "transport": preset(sc) in TRANSPORT,
                "mean_mse": mean[(sc, m)],
                "ratio_to_rct_selected": mean[(sc, m)] / ref if ref else None,
                "ratio_to_catalog_reference": (mean[(sc, m)] / mean[(sc, CATALOG_REFERENCE)]
                                               if (sc, CATALOG_REFERENCE) in mean else None),
                "ratio_to_best_of_four": mean[(sc, m)] / mean[(sc, best)] if best else None,
            })
        chosen = Counter((records[(sc, s, REFERENCE)].get("fit_info") or {}).get("selected") for s in seeds)
        weights = [(records[(sc, s, REFERENCE)].get("fit_info") or {}).get("max_weight") for s in seeds]
        selection.append({
            "scenario": sc, "transport": preset(sc) in TRANSPORT,
            **{f"chosen_{c}": chosen.get(c, 0) for c in CANDIDATES},
            "chosen_none": chosen.get(None, 0),
            "max_weight": max((w for w in weights if w is not None), default=None),
            "rct_selected_mean_mse": ref, "best_of_four": best,
            "best_of_four_mean_mse": mean[(sc, best)] if best else None,
            "regret_ratio": ref / mean[(sc, best)] if best and ref else None,
        })
    write_csv(output / "descriptive_contrasts.csv", descriptive)
    write_csv(output / "selection.csv", selection)

    lines = ["| Recipe | Family | Better | Worse | Unresolved | Untested | Transport: better | worse | unresolved | untested |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for recipe, f in families.items():
        lines.append(f"| `{recipe}` | {f['size']} | " + " | ".join(
            str(f[f"{g}_{v}"]) for g in ("other", "transport") for v in ("better", "worse", "unresolved", "untested")) + " |")
    (output / "headline.md").write_text("\n".join(lines) + "\n")

    failures = [{"scenario": k[0], "seed": k[1], "method": k[2], "stage": r.get("stage"),
                 "error_type": r.get("error_type")} for k, r in sorted(records.items()) if r["status"] != "ok"]
    status = Counter(r["status"] for r in records.values())
    analysis = {"protocol": "docs/V1_CORE_STUDY.md", "requested": len(records), "coverage": dict(status),
                "failures": failures, "identity_passed": identity_ok,
                "ridge_refit_mismatches": len(ridge_mismatch), "families": families}
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
    records, _ = collect(config, a.runs)
    result = analyze(config, records, a.output)
    print(json.dumps({k: result[k] for k in ("requested", "coverage", "identity_passed", "families")}, indent=2))


if __name__ == "__main__":
    main()
