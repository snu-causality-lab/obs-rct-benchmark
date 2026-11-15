"""Analyze the selected-rho Causal-ICM addendum exactly as declared in docs/V1_ICM_SELECTED.md.

    python scripts/analyze_icm_selected.py --config configs/v1_icm_selected_core.toml \
        --runs out-core/*/ --wave2 results/v1_wave2 --output icmsel-core-analysis
    python scripts/analyze_icm_selected.py --config configs/v1_icm_selected_stress.toml \
        --runs out-stress/*/ --wave2 results/stress_wave2 --output icmsel-stress-analysis

One block per call. Every declared shard must be present once with a terminal record
for every requested cell, all shards sharing one source, environment and external-method
provenance (the wave-2 collector checks this). Every successful selected fit must record
a completed selection with a rho on the grid. Each contrast pairs two recipes fitted in
the same shard. Outputs: replicate_scores.csv, condition_scores.csv, rho_selection.csv,
contrasts.csv, identity_check.json, headline.md and analysis.json; for the stress block
also dose_response.csv and readout.csv.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

from analyze_stress import parse
from analyze_v1_core import ALPHA, HEAVY_TAIL, TRANSPORT, holm, paired_t, preset
from analyze_v1_wave2 import IDENTITY_TOL, collect, load_job_limit, write_csv
from make_icm_selected_configs import GROUP, METHODS, shards as declared_shards, validate
from obs_rct_benchmark.causal_icm import CAUSAL_ICM_RHO_GRID
from obs_rct_benchmark.stress_dgp import EXTRAPOLATION_DIALS

REFERENCE = "rct_selected"
SELECTED = "causal_icm_selected"
FIXED = "causal_icm_fixed"
RCT_ONLY = frozenset({"rct_selected", "rct_t_ridge", "causal_icm_rct_only"})
# The recipes wave 2 fitted in the same environment, refitted here for pairing.
WAVE2 = ("rct_selected", "rct_t_ridge", "causal_icm_fixed", "causal_icm_rct_only")
SELECTOR = "causal_icm_weighted_rct_cv_v1"
OBS_ONLY_DIALS = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection", "obs_size")


class AnalysisError(ValueError):
    """The inputs do not match the declared addendum; nothing is reported."""


def families() -> list[dict]:
    """The three declared Holm families of each block."""
    def fam(kind, reference):
        name = f"{GROUP}/{SELECTED}" + ("" if reference == REFERENCE else f" vs {reference}")
        return {"family": name, "kind": kind, "recipe": SELECTED, "reference": reference}
    return [fam("primary", REFERENCE), fam("matched_obs", "causal_icm_rct_only"), fam("selection", FIXED)]


def selected_rho(key: tuple, row: dict) -> float:
    """The rho a successful selected fit recorded; anything else is refused."""
    info = row.get("fit_info") or {}
    rho = info.get("selected_rho")
    if (info.get("selector") != SELECTOR or info.get("status") != "completed"
            or rho not in CAUSAL_ICM_RHO_GRID or info.get("fold_seed") != row["method_seed"]):
        raise AnalysisError(f"{key[1]} seed {key[2]}: a successful selected fit lacks a completed "
                            "selection on the grid with its own fold seed")
    return float(rho)


def wave2_scores(packet: Path) -> dict:
    """Wave 2's `gp` scores of the four refitted recipes: {(scenario, seed, method): mse or None}."""
    out = {}
    for path in sorted((packet / "replicate_scores").glob(f"{GROUP}__*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["group"] == GROUP and r["method"] in WAVE2:
                out[(r["scenario"], int(r["seed"]), r["method"])] = float(r["cate_mse"]) if r["status"] == "ok" else None
    return out


def comparisons(block: str, scenarios: list[str], fam: dict) -> list[tuple[str, str]]:
    """(scenario, reference scenario) pairs of a family. In the core block the heavy-tailed
    preset is descriptive, and an RCT-only reference's OBS 512 record serves both OBS sizes."""
    if block == "stress":
        return [(sc, sc) for sc in scenarios]
    return [(sc, sc.replace("@o2048_", "@o512_") if fam["reference"] in RCT_ONLY else sc)
            for sc in scenarios if preset(sc) != HEAVY_TAIL]


def split_flag(block: str, scenario: str) -> bool:
    """True for conditions counted separately: transport (core) or extrapolation (stress)."""
    return parse(scenario)[0] in EXTRAPOLATION_DIALS if block == "stress" else preset(scenario) in TRANSPORT


def state(seq: list[str]) -> str:
    """The stress plan's readout; "untested" at level 3 counts as not better."""
    return ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
            else "stopped" if seq[3] != "better" else "kept")


def analyze(block: str, config, records: dict, wave2: dict, output: Path) -> dict:
    from scipy import stats

    output.mkdir(parents=True, exist_ok=False)
    seeds = list(config.seeds)
    scenarios = [s.key for s in config.scenarios]
    split = "extrapolation" if block == "stress" else "transport"
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}
    rho = {k: selected_rho(k, r) for k, r in records.items() if k[3] == SELECTED and r["status"] == "ok"}

    write_csv(output / "replicate_scores.csv", [
        {"group": k[0], "scenario": k[1], "seed": k[2], "method": k[3], "status": r["status"],
         "cate_mse": r["metrics"]["cate_mse"] if r["status"] == "ok" else "",
         "root_pehe": r["metrics"]["root_pehe"] if r["status"] == "ok" else "",
         "error_type": r.get("error_type") or "", "stage": r.get("stage") or "",
         "selected": ((r.get("fit_info") or {}).get("selected") or "") if k[3] == REFERENCE else "",
         "selected_rho": rho.get(k, "")}
        for k, r in sorted(records.items())])

    # Identities. Wave 2 fitted the four refitted recipes in this environment on the same seeds.
    reproduction = {}
    for m in WAVE2:
        keys = [(sc, s) for sc in scenarios for s in seeds]
        both = [(mse[(GROUP, sc, s, m)], wave2[(sc, s, m)]) for sc, s in keys
                if (GROUP, sc, s, m) in mse and wave2.get((sc, s, m)) is not None]
        status_differs = sum(((GROUP, sc, s, m) in mse) != (wave2.get((sc, s, m)) is not None)
                             or (sc, s, m) not in wave2 for sc, s in keys)
        worst = max((abs(a - b) for a, b in both), default=None)
        reproduction[m] = {"compared": len(both), "expected": len(keys), "status_differs": status_differs,
                           "exact": sum(a == b for a, b in both), "max_abs_difference": worst,
                           "within_1e-9": status_differs == 0 and worst is not None and worst <= IDENTITY_TOL}
    at_half = [k for k, v in rho.items() if v == 0.5 and (*k[:3], FIXED) in mse]
    selected_at_fixed = {"cells": len(at_half),
                         "differ": sum(mse[k] != mse[(*k[:3], FIXED)] for k in at_half)}
    rct_only = {}
    for m in sorted(RCT_ONLY):
        bad = 0
        if block == "core":
            for sc in (x for x in scenarios if "@o512_" in x):
                for s in seeds:
                    a, b = mse.get((GROUP, sc, s, m)), mse.get((GROUP, sc.replace("@o512_", "@o2048_"), s, m))
                    bad += (a is None) != (b is None) or (a is not None and a != b)
        else:
            for dial in OBS_ONLY_DIALS:
                for rct in (128, 512):
                    for s in seeds:
                        vals = [mse.get((GROUP, f"stress:{dial}@l{level}_r{rct}", s, m)) for level in range(4)]
                        bad += any(v != vals[0] for v in vals[1:])
        rct_only[m] = bad
    identity = {"wave2_reproduction": reproduction, "selected_at_0.5_vs_fixed": selected_at_fixed,
                ("rct_only_obs_size_mismatches" if block == "core" else "rct_only_level_mismatches"): rct_only,
                "rule": ("contrasts pair recipes fitted in the same shard. The block ran on wave 2's data and fits "
                         "only if all four refitted recipes reproduce wave 2's gp scores within 1e-9 with the same "
                         "statuses. A selected fit that chose 0.5 must equal causal_icm_fixed exactly")}
    (output / "identity_check.json").write_text(json.dumps(identity, indent=2) + "\n")

    conditions, rho_rows, dose = [], [], []
    for sc in scenarios:
        extra = dict(zip(("dial", "level", "rct"), parse(sc))) if block == "stress" else {}
        for m in METHODS:
            vals = [mse[(GROUP, sc, s, m)] for s in seeds if (GROUP, sc, s, m) in mse]
            conditions.append({"scenario": sc, **extra, "method": m, split: split_flag(block, sc),
                               "successful": len(vals), "failed": len(seeds) - len(vals),
                               "mean_mse": math.fsum(vals) / len(vals) if vals else None})
            pairs = [(mse[(GROUP, sc, s, m)], mse[(GROUP, sc, s, REFERENCE)]) for s in seeds
                     if (GROUP, sc, s, m) in mse and (GROUP, sc, s, REFERENCE) in mse]
            if block == "stress" and m != REFERENCE and len(pairs) >= 2:
                logs = [math.log(a / b) for a, b in pairs]
                mean = math.fsum(logs) / len(logs)
                se = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (len(logs) - 1) / len(logs))
                q = float(stats.t.ppf(0.975, len(logs) - 1))
                dose.append({**extra, "method": m, "pairs": len(pairs),
                             "ratio_of_means": math.fsum(a for a, _ in pairs) / math.fsum(b for _, b in pairs),
                             "geometric_mean_ratio": math.exp(mean),
                             "gmr_ci95_lower": math.exp(mean - q * se), "gmr_ci95_upper": math.exp(mean + q * se)})
        chosen = [rho[(GROUP, sc, s, SELECTED)] for s in seeds if (GROUP, sc, s, SELECTED) in rho]
        infos = [records[(GROUP, sc, s, SELECTED)].get("fit_info") or {} for s in seeds
                 if (GROUP, sc, s, SELECTED) in rho]
        rho_rows.append({"scenario": sc, **extra, "successful": len(chosen),
                         **{f"rho_{r:.1f}": chosen.count(r) for r in CAUSAL_ICM_RHO_GRID},
                         "mean_rho": math.fsum(chosen) / len(chosen) if chosen else None,
                         "fewer_than_5_folds": sum(i.get("folds_used") != 5 for i in infos),
                         "weights_not_converged": sum(
                             (i.get("participation_weights") or {}).get("converged") is not True for i in infos)})
    write_csv(output / "condition_scores.csv", conditions)
    write_csv(output / "rho_selection.csv", rho_rows)
    if block == "stress":
        write_csv(output / "dose_response.csv", dose)

    rows_all, summary = [], {}
    for fam in families():
        rec, ref = fam["recipe"], fam["reference"]
        rows = []
        for sc, rk in comparisons(block, scenarios, fam):
            ok = [s for s in seeds if (GROUP, sc, s, rec) in mse and (GROUP, rk, s, ref) in mse]
            diffs = [mse[(GROUP, sc, s, rec)] - mse[(GROUP, rk, s, ref)] for s in ok]
            row = {"family": fam["family"], "kind": fam["kind"], "recipe": rec, "reference": ref, "scenario": sc,
                   "reference_scenario": rk, split: split_flag(block, sc), "pairs": len(ok),
                   "recipe_failures": sum((GROUP, sc, s, rec) not in mse for s in seeds),
                   "reference_failures": sum((GROUP, rk, s, ref) not in mse for s in seeds),
                   "complete": len(ok) == len(seeds),
                   "mean_difference": math.fsum(diffs) / len(diffs) if diffs else None}
            row.update(paired_t(diffs) if row["complete"] else
                       {"mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": 1.0, "zero_variance": None})
            rows.append(row)
        for row, p in zip(rows, holm([r["p_two_sided"] for r in rows])):
            row["holm_p"] = p
            row["verdict"] = ("untested" if not row["complete"] else
                              "better" if p <= ALPHA and row["mean_difference"] < 0 else
                              "worse" if p <= ALPHA and row["mean_difference"] > 0 else "unresolved")
        summary[fam["family"]] = {"kind": fam["kind"], "size": len(rows), **{
            f"{grp}_{v}": sum(1 for r in rows if r["verdict"] == v and r[split] == (grp == split))
            for grp in ("other", split) for v in ("better", "worse", "unresolved", "untested")}}
        rows_all += rows
    write_csv(output / "contrasts.csv", rows_all)

    if block == "stress":
        verdict = {(r["family"], r["scenario"]): r["verdict"] for r in rows_all}
        readout = []
        primary = families()[0]["family"]
        for dial in dict.fromkeys(parse(sc)[0] for sc in scenarios):
            for rct in (128, 512):
                seq = [verdict[(primary, f"stress:{dial}@l{level}_r{rct}")] for level in range(4)]
                readout.append({"family": primary, "dial": dial, "rct": rct,
                                **{f"level_{i}": v for i, v in enumerate(seq)}, "state": state(seq),
                                "untested_at": ",".join(str(i) for i in (0, 3) if seq[i] == "untested")})
        write_csv(output / "readout.csv", readout)

    label = split.capitalize()
    lines = [f"| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | {label}: better / worse |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, f in summary.items():
        tot = {v: f[f"other_{v}"] + f[f"{split}_{v}"] for v in ("better", "worse", "unresolved", "untested")}
        lines.append(f"| `{name}` | {f['kind']} | {f['size']} | {tot['better']} | {tot['worse']} | {tot['unresolved']} "
                     f"| {tot['untested']} | {f[split + '_better']} / {f[split + '_worse']} |")
    all_rho = list(rho.values())
    lines += ["", "| Recipe refitted from wave 2 | Equal to wave 2 | Largest difference | Within 1e-9, same statuses |",
              "| --- | ---: | ---: | --- |"]
    for m, r in reproduction.items():
        lines.append(f"| `{m}` | {r['exact']} of {r['expected']} | {r['max_abs_difference']} | "
                     f"{'yes' if r['within_1e-9'] else 'no'} |")
    lines += ["", f"Selected fits that chose 0.5 and differ from `{FIXED}`: {selected_at_fixed['differ']} of "
              f"{selected_at_fixed['cells']}.", "",
              "| Selected rho | " + " | ".join(f"{r:.1f}" for r in CAUSAL_ICM_RHO_GRID) + " |",
              "| --- | " + " | ".join("---:" for _ in CAUSAL_ICM_RHO_GRID) + " |",
              "| Successful fits | " + " | ".join(str(all_rho.count(r)) for r in CAUSAL_ICM_RHO_GRID) + " |"]
    (output / "headline.md").write_text("\n".join(lines) + "\n")

    failures = Counter((k[3], r.get("stage"), r.get("error_type")) for k, r in records.items() if r["status"] != "ok")
    analysis = {"protocol": "docs/V1_ICM_SELECTED.md", "block": block, "requested": len(records),
                "coverage": dict(Counter(r["status"] for r in records.values())),
                "failures": [{"method": m, "stage": st, "error_type": e, "count": n}
                             for (m, st, e), n in sorted(failures.items(), key=str)],
                "families": summary,
                "selected_rho": {f"{r:.1f}": all_rho.count(r) for r in CAUSAL_ICM_RHO_GRID}}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--wave2", type=Path, required=True, help="results/v1_wave2 (core) or results/stress_wave2")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--job-limit-shards", type=Path,
                    help="JSON list of {shard, config_hash} whose final attempt reached its job limit")
    a = ap.parse_args()
    config = load_config(a.config)
    block = validate(config)
    records = collect({GROUP: config}, a.runs, declared=declared_shards, job_limit=load_job_limit(a.job_limit_shards))
    result = analyze(block, config, records, wave2_scores(a.wave2), a.output)
    print(json.dumps({k: result[k] for k in ("block", "requested", "coverage", "families", "selected_rho")}, indent=2))


if __name__ == "__main__":
    main()
