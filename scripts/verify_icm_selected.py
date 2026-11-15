"""Independent checks for the selected-rho Causal-ICM addendum (docs/V1_ICM_SELECTED.md).

  arithmetic  Check replicate_scores.csv against the run records, then recompute every
              declared contrast, Holm adjustment, verdict, count, headline row, readout
              state (stress), identity count and selected-rho count, without importing
              the analyzer. Family definitions and sizes are written out here.
  reproduce   Refit every recipe twice on the block's first seed (core: OBS 512 and RCT
              128; stress: RCT 128 at levels 0 and 3 of every dial) and compare with the
              records within 1e-12.

    python scripts/verify_icm_selected.py arithmetic --block core --analysis icmsel-core-analysis \
        --runs out-core/*/ --wave2 results/v1_wave2 --output arith-core.json
    python scripts/verify_icm_selected.py reproduce --config configs/v1_icm_selected_core.toml \
        --runs out-core/*/ --output reproduce-core.json
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from verify_v1_wave2 import as_float, check_cells, check_headline

GROUP = "gp"
METHODS = ("rct_selected", "rct_t_ridge", "causal_icm_fixed", "causal_icm_rct_only", "causal_icm_selected")
RCT_ONLY = {"rct_selected", "rct_t_ridge", "causal_icm_rct_only"}
WAVE2 = ("rct_selected", "rct_t_ridge", "causal_icm_fixed", "causal_icm_rct_only")
GRID = tuple(k / 10 for k in range(11))
# (family, recipe, reference, kind)
FAMILIES = [("gp/causal_icm_selected", "causal_icm_selected", "rct_selected", "primary"),
            ("gp/causal_icm_selected vs causal_icm_rct_only", "causal_icm_selected", "causal_icm_rct_only", "matched_obs"),
            ("gp/causal_icm_selected vs causal_icm_fixed", "causal_icm_selected", "causal_icm_fixed", "selection")]
BLOCK = {
    "core": {"seeds": list(range(70000, 70064)), "conditions": 124, "family_size": 120, "split": "transport",
             "split_size": 24},
    "stress": {"seeds": list(range(80000, 80064)), "conditions": 64, "family_size": 64, "split": "extrapolation",
               "split_size": 16},
}
HEAVY_TAIL = "structural:M11"
TRANSPORT = {"structural:M9", "addendum:C08", "paper:experimental_grounding_kps",
             "paper:integrative_r_learner_kps_normal_target_extension",
             "paper:integrative_r_learner_multivariate_paper", "paper:elastic_integrative_gen_data_overlap"}
DIALS = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection",
         "trial_shrink", "obs_size", "target_shift", "noise_shape")
EXTRAPOLATION = {"trial_shrink", "target_shift"}
OBS_ONLY = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection", "obs_size")
TOL = 1e-12
SAME_TOL = 1e-9


def close(a, b, rel=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-12))


def dial(scenario: str) -> str:
    return scenario.split("@")[0].split(":")[1]


def in_split(block: str, scenario: str) -> bool:
    return dial(scenario) in EXTRAPOLATION if block == "stress" else scenario.split("@")[0] in TRANSPORT


def load_records(runs: list[Path], job_limit: frozenset = frozenset()) -> tuple[dict, set]:
    """Records keyed (group, scenario, seed, method), and the requested cells left
    unrecorded by runs declared as reaching their job limit on the final attempt."""
    out, unrecorded, seen = {}, set(), set()
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        config = manifest["config"]
        if manifest["config_hash"] in seen:
            raise SystemExit(f"{run}: shard given twice")
        seen.add(manifest["config_hash"])
        if tuple(config["run"]["methods"]) != METHODS:
            raise SystemExit(f"{run}: not a shard of this addendum")
        path = run / "records.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        limited = manifest["config_hash"] in job_limit
        if limited and lines:
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                lines = lines[:-1]
        got = set()
        for line in lines:
            row = json.loads(line)
            key = (GROUP, row["scenario"], int(row["seed"]), row["method"])
            if key in out:
                raise SystemExit(f"{run}: duplicate record {key}")
            out[key] = row
            got.add(key)
        if limited:
            unrecorded |= {(GROUP, f"{sc['kind']}:{sc['id']}" + (f"@{sc['label']}" if sc.get("label") else ""), seed, m)
                           for sc in config["scenarios"] for seed in config["run"]["seeds"]
                           for m in config["run"]["methods"]} - got
    return out, unrecorded


def wave2_scores(packet: Path) -> dict:
    out = {}
    for path in sorted((packet / "replicate_scores").glob("*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["group"] == GROUP and r["method"] in WAVE2:
                out[(r["scenario"], int(r["seed"]), r["method"])] = as_float(r["cate_mse"]) if r["status"] == "ok" else None
    return out


def arithmetic(block: str, analysis: Path, runs: list[Path], wave2: Path, job_limit: frozenset = frozenset()) -> dict:
    from scipy.special import stdtr

    spec = BLOCK[block]
    split = spec["split"]
    mse, table, scenarios, seeds = {}, {}, set(), set()
    with (analysis / "replicate_scores.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["group"], r["scenario"], int(r["seed"]), r["method"])
            table[key] = r
            scenarios.add(r["scenario"])
            seeds.add(int(r["seed"]))
            if r["status"] == "ok":
                mse[key] = float(r["cate_mse"])
    seeds, scenarios = sorted(seeds), sorted(scenarios)
    mismatches = [] if seeds == spec["seeds"] and len(scenarios) == spec["conditions"] else [
        {"reason": f"expected the {block} block's seeds and {spec['conditions']} conditions"}]
    records, unrecorded = load_records(runs, job_limit)
    mismatches += check_cells(table, records, unrecorded)

    rho = {}
    for key, row in records.items():
        if key[3] != "causal_icm_selected" or row["status"] != "ok":
            continue
        info = row.get("fit_info") or {}
        value = info.get("selected_rho")
        if info.get("status") != "completed" or value not in GRID:
            mismatches.append({"cell": list(key), "reason": "successful selected fit without a completed selection"})
            continue
        rho[key] = value
        if as_float(table.get(key, {}).get("selected_rho")) != value:
            mismatches.append({"cell": list(key), "reason": "selected_rho differs from the record"})

    identity_file = json.loads((analysis / "identity_check.json").read_text())
    first = wave2_scores(wave2)
    reproduction = {}
    for m in WAVE2:
        keys = [(sc, s) for sc in scenarios for s in seeds]
        both = [(mse[(GROUP, sc, s, m)], first[(sc, s, m)]) for sc, s in keys
                if (GROUP, sc, s, m) in mse and first.get((sc, s, m)) is not None]
        differs = sum((sc, s, m) not in first or ((GROUP, sc, s, m) in mse) != (first[(sc, s, m)] is not None)
                      for sc, s in keys)
        worst = max((abs(a - b) for a, b in both), default=None)
        reproduction[m] = {"compared": len(both), "status_differs": differs, "exact": sum(a == b for a, b in both),
                           "max_abs_difference": worst,
                           "within_1e-9": differs == 0 and worst is not None and worst <= SAME_TOL}
        got = identity_file["wave2_reproduction"].get(m, {})
        if (got.get("within_1e-9") != reproduction[m]["within_1e-9"] or got.get("compared") != len(both)
                or got.get("exact") != reproduction[m]["exact"] or got.get("status_differs") != differs
                or not close(got.get("max_abs_difference"), worst)):
            mismatches.append({"identity": f"wave2:{m}", "recomputed": reproduction[m], "reported": got})
    half = [k for k, v in rho.items() if v == 0.5 and (*k[:3], "causal_icm_fixed") in mse]
    at_half = {"cells": len(half), "differ": sum(mse[k] != mse[(*k[:3], "causal_icm_fixed")] for k in half)}
    if identity_file.get("selected_at_0.5_vs_fixed") != at_half:
        mismatches.append({"identity": "selected_at_0.5_vs_fixed", "recomputed": at_half,
                           "reported": identity_file.get("selected_at_0.5_vs_fixed")})
    rct_only = {}
    for m in sorted(RCT_ONLY):
        if block == "core":
            rct_only[m] = sum(
                mse.get((GROUP, sc, s, m)) != mse.get((GROUP, sc.replace("@o512_", "@o2048_"), s, m))
                for sc in scenarios if "@o512_" in sc for s in seeds)
        else:
            rct_only[m] = sum(
                any(mse.get((GROUP, f"stress:{d}@l{i}_r{r}", s, m)) != mse.get((GROUP, f"stress:{d}@l0_r{r}", s, m))
                    for i in (1, 2, 3))
                for d in OBS_ONLY for r in (128, 512) for s in seeds)
    name = "rct_only_obs_size_mismatches" if block == "core" else "rct_only_level_mismatches"
    if identity_file.get(name) != rct_only:
        mismatches.append({"identity": name, "recomputed": rct_only, "reported": identity_file.get(name)})

    with (analysis / "rho_selection.csv").open(newline="") as fh:
        rho_rows = {r["scenario"]: r for r in csv.DictReader(fh)}
    for sc in scenarios:
        chosen = [rho[(GROUP, sc, s, "causal_icm_selected")] for s in seeds if (GROUP, sc, s, "causal_icm_selected") in rho]
        got = rho_rows.get(sc, {})
        if any(int(got.get(f"rho_{r:.1f}", -1)) != chosen.count(r) for r in GRID) or int(got.get("successful", -1)) != len(chosen):
            mismatches.append({"rho_selection": sc, "recomputed": [chosen.count(r) for r in GRID]})

    with (analysis / "contrasts.csv").open(newline="") as fh:
        rows_in = list(csv.DictReader(fh))
    reported = {(r["family"], r["scenario"]): r for r in rows_in}
    if len(reported) != len(rows_in):
        mismatches.append({"reason": "contrasts.csv has duplicate rows"})
    counts, verdicts = {}, {}
    for fam, rec, ref, _ in FAMILIES:
        pairs = [(sc, sc) for sc in scenarios] if block == "stress" else [
            (sc, sc.replace("@o2048_", "@o512_") if ref in RCT_ONLY else sc)
            for sc in scenarios if not sc.startswith(HEAVY_TAIL + "@")]
        if (len(pairs), sum(in_split(block, sc) for sc, _ in pairs)) != (spec["family_size"], spec["split_size"]):
            mismatches.append({"family": fam, "reason": "family size differs from the declared size"})
        rows = []
        for sc, rk in pairs:
            both = [s for s in seeds if (GROUP, sc, s, rec) in mse and (GROUP, rk, s, ref) in mse]
            d = np.array([mse[(GROUP, sc, s, rec)] - mse[(GROUP, rk, s, ref)] for s in both])
            p = 1.0
            if len(both) == len(seeds) and float(np.std(d, ddof=1)) > 0:
                t = float(np.mean(d)) / (float(np.std(d, ddof=1)) / math.sqrt(len(d)))
                p = float(2 * stdtr(len(d) - 1, -abs(t)))
            rows.append({"sc": sc, "complete": len(both) == len(seeds), "p": p,
                         "mean": float(np.mean(d)) if len(d) else None})
        order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p"]))
            rows[i]["holm"] = running
        tally = defaultdict(int)
        for r in rows:
            v = ("untested" if not r["complete"] else "better" if r["holm"] <= 0.05 and r["mean"] < 0
                 else "worse" if r["holm"] <= 0.05 and r["mean"] > 0 else "unresolved")
            verdicts[(fam, r["sc"])] = v
            tally[f"{split if in_split(block, r['sc']) else 'other'}_{v}"] += 1
            got = reported.get((fam, r["sc"]))
            if (got is None or got["verdict"] != v or not close(r["mean"], as_float(got["mean_difference"]))
                    or not close(r["p"], float(got["p_two_sided"])) or not close(r["holm"], float(got["holm_p"]))):
                mismatches.append({"family": fam, "scenario": r["sc"], "recomputed": r, "reported": got})
        counts[fam] = {"size": len(rows), **tally}
    if len(reported) != sum(c["size"] for c in counts.values()):
        mismatches.append({"reason": "reported contrast rows differ from the declared families"})
    result = json.loads((analysis / "analysis.json").read_text())
    if set(result["families"]) != set(counts):
        mismatches.append({"reason": "reported family names differ from the declared families"})
    for fam, c in counts.items():
        got = {k: v for k, v in result["families"].get(fam, {}).items() if k != "kind" and v}
        if got != {k: v for k, v in c.items() if v}:
            mismatches.append({"family": fam, "reason": "headline counts differ", "recomputed": c, "reported": got})
    totals = {f"{r:.1f}": list(rho.values()).count(r) for r in GRID}
    if result.get("selected_rho") != totals:
        mismatches.append({"reason": "selected-rho totals differ", "recomputed": totals})
    mismatches += check_headline(analysis / "headline.md", counts, split)

    states = 0
    if block == "stress":
        with (analysis / "readout.csv").open(newline="") as fh:
            readout_rows = list(csv.DictReader(fh))
        readout = {(r["family"], r["dial"], int(r["rct"])): (r["state"], r["untested_at"]) for r in readout_rows}
        if len(readout) != len(readout_rows):
            mismatches.append({"reason": "readout.csv has duplicate rows"})
        fam = FAMILIES[0][0]
        for d in DIALS:
            for rct in (128, 512):
                seq = [verdicts[(fam, f"stress:{d}@l{level}_r{rct}")] for level in range(4)]
                want = ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
                        else "kept" if seq[3] == "better" else "stopped")
                want = (want, ",".join(str(i) for i in (0, 3) if seq[i] == "untested"))
                states += 1
                if readout.get((fam, d, rct)) != want:
                    mismatches.append({"readout": [fam, d, rct], "recomputed": want, "reported": readout.get((fam, d, rct))})
        if len(readout) != states:
            mismatches.append({"reason": "readout rows differ from the primary family's 16 states"})
    return {"check": "arithmetic", "block": block, "families": counts, "wave2_reproduction": reproduction,
            "selected_at_0.5_vs_fixed": at_half, name: rct_only, "selected_rho": totals,
            "readout_states": states, "mismatches": mismatches}


def reproduce(config, runs: list[Path]) -> dict:
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records, _ = load_records(runs)
    seed = config.seeds[0]
    stress = config.scenarios[0].kind == "stress"
    scenarios = [s for s in config.scenarios if s.parameters["n_rct_train"] == 128
                 and (s.parameters["level"] in (0, 3) if stress else s.parameters["n_obs"] == 512)]
    checks, mismatches = [], []
    for s in scenarios:
        problems = [_generate(s, seed), _generate(s, seed)]
        for method in config.methods:
            row = records[(GROUP, s.key, seed, method)]
            outcomes = []
            for problem in problems:
                try:
                    p = np.asarray(make_method(method, **dict(config.method_options.get(method, {}))).fit_predict(
                        problem.training_view(), problem.eval_X, seed=method_seed(s.key, seed, method)), dtype=float)
                    cate_metrics(p, problem.eval_tau)
                    diff = p - problem.eval_tau
                    outcomes.append({"status": "ok", "mse": float(diff @ diff / diff.size)})
                except Exception as exc:
                    outcomes.append({"status": "failed", "stage": "method_or_metric", "error_type": type(exc).__name__})
            if row["status"] == "ok":
                ok = all(o["status"] == "ok" and math.isclose(o["mse"], row["metrics"]["cate_mse"], rel_tol=TOL, abs_tol=TOL)
                         for o in outcomes)
            else:
                ok = all(o["status"] == "failed" and o["error_type"] == row.get("error_type")
                         and o["stage"] == row.get("stage") for o in outcomes)
            checks.append({"scenario": s.key, "method": method, "recorded_status": row["status"],
                           "repeats": outcomes, "tolerance": TOL, "match": ok})
            if not ok:
                mismatches.append([s.key, method])
    return {"check": "reproduce", "seed": seed, "cases": len(checks), "refits": 2 * len(checks),
            "mismatches": mismatches, "details": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("arithmetic", "reproduce"))
    ap.add_argument("--block", choices=tuple(BLOCK))
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--wave2", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--job-limit-shards", type=Path,
                    help="JSON list of {shard, config_hash} whose final attempt reached its job limit")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    if a.check == "arithmetic":
        if not (a.block and a.analysis and a.wave2):
            ap.error("arithmetic needs --block, --analysis and --wave2")
        limited = frozenset(e["config_hash"] for e in json.loads(a.job_limit_shards.read_text())) \
            if a.job_limit_shards else frozenset()
        result = arithmetic(a.block, a.analysis, a.runs, a.wave2, limited)
    else:
        from obs_rct_benchmark.runner import load_config

        if a.config is None:
            ap.error("reproduce needs --config")
        result = reproduce(load_config(a.config), a.runs)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
