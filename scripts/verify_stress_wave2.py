"""Independent checks for the wave-2 recipes on stress study 1 (docs/V1_STRESS_WAVE2.md).

  arithmetic  Check replicate_scores.csv against the run records, then recompute every
              declared contrast, Holm adjustment, verdict, count, readout state and the
              stress-study-1 data identity, without importing the analyzer. Family
              definitions and sizes are written out here.
  reproduce   For one group, refit every recipe twice on the first seed at RCT 128,
              levels 0 and 3 of every dial, and compare with the records: 1e-12 for CPU
              recipes, a relative 1e-6 for TabPFN recipes run on a GPU.

    python scripts/verify_stress_wave2.py arithmetic --analysis stress-wave2-analysis \
        --runs outputs/*/ --stress1 results/stress_cornet --output arith.json
    python scripts/verify_stress_wave2.py reproduce --group gp --config configs/v1_stress_wave2_gp.toml \
        --runs outputs/gp/* --output reproduce-gp.json
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

DIALS = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection",
         "trial_shrink", "obs_size", "target_shift", "noise_shape")
EXTRAPOLATION = {"trial_shrink", "target_shift"}
GPU_RECIPES = {"tsr_tabpfn_v2", "tsr_tabpfn_v25", "tsr_tabpfn_v3", "tsr_tabpfn_lmcgp",
               "tabpfn_v2_obs_only", "tabpfn_v25_obs_only", "tabpfn_v3_obs_only", "tabpfn_obs_only"}
GROUP_RECIPES = {
    "native_r": ("integrative_r_linear", "integrative_r_quadratic", "rct_causal_forest"),
    "gp": ("causal_icm_fixed", "causal_icm_rct_only"),
    "tabpfn": ("lmcgp_rct_only", "tsr_tabpfn_v2", "tsr_tabpfn_v25", "tsr_tabpfn_v3",
               "tabpfn_v2_obs_only", "tabpfn_v25_obs_only", "tabpfn_v3_obs_only"),
    "tsr_legacy": ("tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only"),
}
# (family, group, recipe, reference, kind). RCT-only recipes against rct_selected are descriptive.
PRIMARY = [("native_r", "integrative_r_linear"), ("native_r", "integrative_r_quadratic"), ("gp", "causal_icm_fixed"),
           *[("tabpfn", f"tsr_tabpfn_{t}") for t in ("v2", "v25", "v3")],
           *[("tabpfn", f"tabpfn_{t}_obs_only") for t in ("v2", "v25", "v3")],
           ("tsr_legacy", "tsr_tabpfn_lmcgp"), ("tsr_legacy", "tabpfn_obs_only")]
FAMILIES = [(f"{g}/{m}", g, m, "rct_selected", "primary") for g, m in PRIMARY] + [
    ("gp/causal_icm_fixed vs causal_icm_rct_only", "gp", "causal_icm_fixed", "causal_icm_rct_only", "matched_obs"),
    *[(f"tabpfn/tsr_tabpfn_{t} vs lmcgp_rct_only", "tabpfn", f"tsr_tabpfn_{t}", "lmcgp_rct_only", "matched_obs")
      for t in ("v2", "v25", "v3")],
    ("tsr_legacy/tsr_tabpfn_lmcgp vs lmcgp_rct_only", "tsr_legacy", "tsr_tabpfn_lmcgp", "lmcgp_rct_only", "matched_obs"),
    *[(f"tabpfn/{k.format(t)} vs {k.format('v2')}", "tabpfn", k.format(t), k.format("v2"), "tabpfn_release")
      for k in ("tsr_tabpfn_{}", "tabpfn_{}_obs_only") for t in ("v25", "v3")],
]
TOL = 1e-12
GPU_RTOL = 1e-6
SAME_DATA_TOL = 1e-9


def as_float(text):
    return None if text in ("", None) else float(text)


def close(a, b, rel=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-12))


def split(scenario: str) -> tuple[str, int, int]:
    """'stress:<dial>@l<level>_r<rct>' -> (dial, level, rct)."""
    head, label = scenario.split("@")
    level, rct = label.split("_")
    return head.split(":")[1], int(level[1:]), int(rct[1:])


def load_records(runs: list[Path]) -> dict:
    out = {}
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        methods = set(manifest["config"]["run"]["methods"])
        group = next((g for g, ms in GROUP_RECIPES.items() if methods == {"rct_selected", "rct_t_ridge", *ms}), None)
        if group is None:
            raise SystemExit(f"{run}: not a stress wave-2 shard")
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            out[(group, row["scenario"], int(row["seed"]), row["method"])] = row
    return out


def arithmetic(analysis: Path, runs: list[Path], stress1: Path) -> dict:
    from scipy.special import stdtr

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
    mismatches = [] if seeds == list(range(80000, 80064)) and len(scenarios) == 64 else [
        {"reason": "expected seeds 80000-80063 and 64 conditions"}]
    records = load_records(runs)
    if set(records) != set(table):
        mismatches.append({"reason": "replicate_scores.csv cells differ from the run records"})
    for key, row in records.items():
        got = table.get(key)
        if got is not None and (got["status"] != row["status"] or as_float(got["cate_mse"]) !=
                                (row["metrics"]["cate_mse"] if row["status"] == "ok" else None)):
            mismatches.append({"cell": list(key), "reason": "replicate score differs from its record"})

    first = {}
    for path in sorted((stress1 / "replicate_scores").glob("*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["method"] in ("rct_selected", "rct_t_ridge") and r["status"] == "ok":
                first[(r["scenario"], int(r["seed"]), r["method"])] = float(r["cate_mse"])
    reported_identity = json.loads((analysis / "identity_check.json").read_text())["stress1_reproduction"]
    same_data = {}
    for g in GROUP_RECIPES:
        for m in ("rct_selected", "rct_t_ridge"):
            both = [(mse[(g, sc, s, m)], first[(sc, s, m)]) for sc in scenarios for s in seeds
                    if (g, sc, s, m) in mse and (sc, s, m) in first]
            worst = max((abs(a - b) for a, b in both), default=None)
            ok = len(both) == len(scenarios) * len(seeds) and worst is not None and worst <= SAME_DATA_TOL
            same_data[f"{g}:{m}"] = {"compared": len(both), "max_abs_difference": worst, "within_1e-9": ok}
            got = reported_identity.get(f"{g}:{m}", {})
            if got.get("within_1e-9") != ok or not close(got.get("max_abs_difference"), worst):
                mismatches.append({"identity": f"{g}:{m}", "recomputed": same_data[f"{g}:{m}"], "reported": got})

    with (analysis / "contrasts.csv").open(newline="") as fh:
        reported = {(r["family"], r["scenario"]): r for r in csv.DictReader(fh)}
    counts, verdicts = {}, {}
    for name, g, rec, ref, _ in FAMILIES:
        if (len(scenarios), sum(split(sc)[0] in EXTRAPOLATION for sc in scenarios)) != (64, 16):
            mismatches.append({"family": name, "reason": "family size differs from the declared 64 (16 extrapolation)"})
        rows = []
        for sc in scenarios:
            both = [s for s in seeds if (g, sc, s, rec) in mse and (g, sc, s, ref) in mse]
            d = np.array([mse[(g, sc, s, rec)] - mse[(g, sc, s, ref)] for s in both])
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
            verdicts[(name, r["sc"])] = v
            tally[f"{'extrapolation' if split(r['sc'])[0] in EXTRAPOLATION else 'other'}_{v}"] += 1
            got = reported.get((name, r["sc"]))
            if (got is None or got["verdict"] != v or not close(r["mean"], as_float(got["mean_difference"]))
                    or not close(r["p"], float(got["p_two_sided"])) or not close(r["holm"], float(got["holm_p"]))):
                mismatches.append({"family": name, "scenario": r["sc"], "recomputed": r, "reported": got})
        counts[name] = {"size": len(rows), **tally}
    if len(reported) != sum(c["size"] for c in counts.values()):
        mismatches.append({"reason": "reported contrast rows differ from the declared families"})
    reported_families = json.loads((analysis / "analysis.json").read_text())["families"]
    if set(reported_families) != set(counts):
        mismatches.append({"reason": "reported family names differ from the declared families"})
    for name, c in counts.items():
        got = {k: v for k, v in reported_families.get(name, {}).items() if k != "kind" and v}
        if got != {k: v for k, v in c.items() if v}:
            mismatches.append({"family": name, "reason": "headline counts differ", "recomputed": c, "reported": got})

    with (analysis / "readout.csv").open(newline="") as fh:
        readout = {(r["family"], r["dial"], int(r["rct"])): r["state"] for r in csv.DictReader(fh)}
    states = 0
    for name, *_, kind in FAMILIES:
        if kind != "primary":
            continue
        for dial in DIALS:
            for rct in (128, 512):
                seq = [verdicts[(name, f"stress:{dial}@l{level}_r{rct}")] for level in range(4)]
                want = ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
                        else "stopped" if seq[3] != "better" else "kept")
                states += 1
                if readout.get((name, dial, rct)) != want:
                    mismatches.append({"readout": [name, dial, rct], "recomputed": want,
                                       "reported": readout.get((name, dial, rct))})
    if len(readout) != states:
        mismatches.append({"reason": "reported readout rows differ from the declared primary families"})
    return {"check": "arithmetic", "families": counts, "same_data": same_data, "readout_states": states,
            "mismatches": mismatches}


def reproduce(group: str, config, runs: list[Path]) -> dict:
    from obs_rct_benchmark import tsr
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records = load_records(runs)
    seed = config.seeds[0]
    scenarios = [s for s in config.scenarios
                 if s.parameters["n_rct_train"] == 128 and s.parameters["level"] in (0, 3)]
    checks, mismatches = [], []
    for s in scenarios:
        problems = [_generate(s, seed), _generate(s, seed)]
        for method in config.methods:
            row = records[(group, s.key, seed, method)]
            outcomes = []
            for problem in problems:
                tsr._STAGE1_CACHE.clear()  # each refit fits its own Stage 1
                try:
                    p = np.asarray(make_method(method, **dict(config.method_options.get(method, {}))).fit_predict(
                        problem.training_view(), problem.eval_X, seed=method_seed(s.key, seed, method)), dtype=float)
                    cate_metrics(p, problem.eval_tau)
                    diff = p - problem.eval_tau
                    outcomes.append({"status": "ok", "mse": float(diff @ diff / diff.size)})
                except Exception as exc:
                    outcomes.append({"status": "failed", "stage": "method_or_metric", "error_type": type(exc).__name__})
            rel = GPU_RTOL if method in GPU_RECIPES else TOL
            if row["status"] == "ok":
                ok = all(o["status"] == "ok" and math.isclose(o["mse"], row["metrics"]["cate_mse"], rel_tol=rel, abs_tol=TOL)
                         for o in outcomes)
            else:
                ok = all(o["status"] == "failed" and o["error_type"] == row.get("error_type")
                         and o["stage"] == row.get("stage") for o in outcomes)
            checks.append({"scenario": s.key, "method": method, "recorded_status": row["status"],
                           "repeats": outcomes, "tolerance": rel, "match": ok})
            if not ok:
                mismatches.append([s.key, method])
    return {"check": "reproduce", "group": group, "seed": seed, "cases": len(checks),
            "refits": 2 * len(checks), "mismatches": mismatches, "details": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("arithmetic", "reproduce"))
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--stress1", type=Path)
    ap.add_argument("--group")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    if a.check == "arithmetic":
        result = arithmetic(a.analysis, a.runs, a.stress1)
    else:
        from obs_rct_benchmark.runner import load_config

        result = reproduce(a.group, load_config(a.config), a.runs)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
