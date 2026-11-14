"""Independent checks for the v1 core wave-2 addendum (docs/V1_CORE_WAVE2.md).

  arithmetic  Check replicate_scores.csv against the run records, then recompute every
              declared contrast, Holm adjustment, verdict and count without importing
              the analyzer. Family definitions and sizes are written out here.
  reproduce   For one group, refit every recipe twice on the first seed at OBS 512 /
              RCT 128 and compare with the records: 1e-12 for CPU recipes, a relative
              1e-6 for TabPFN recipes run on a GPU.

    python scripts/verify_v1_wave2.py arithmetic --analysis wave2-analysis --runs outputs/*/ --output arith.json
    python scripts/verify_v1_wave2.py reproduce --group native_r --config configs/v1_wave2_native_r.toml \
        --runs outputs/native_r/* --output reproduce-native_r.json
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

TRANSPORT = {
    "structural:M9", "addendum:C08", "paper:experimental_grounding_kps",
    "paper:integrative_r_learner_kps_normal_target_extension",
    "paper:integrative_r_learner_multivariate_paper",
    "paper:elastic_integrative_gen_data_overlap",
}
RCT_ONLY = {"rct_selected", "rct_t_ridge", "rct_causal_forest", "causal_icm_rct_only", "lmcgp_rct_only"}
GPU_RECIPES = {"tsr_tabpfn_v2", "tsr_tabpfn_v25", "tsr_tabpfn_v3", "tsr_tabpfn_lmcgp",
               "tabpfn_v2_obs_only", "tabpfn_v25_obs_only", "tabpfn_v3_obs_only", "tabpfn_obs_only"}
GROUP_RECIPES = {
    "native_r": ("integrative_r_linear", "integrative_r_quadratic", "rct_causal_forest"),
    "gp": ("causal_icm_fixed", "causal_icm_rct_only"),
    "tabpfn": ("lmcgp_rct_only", "tsr_tabpfn_v2", "tsr_tabpfn_v25", "tsr_tabpfn_v3",
               "tabpfn_v2_obs_only", "tabpfn_v25_obs_only", "tabpfn_v3_obs_only"),
    "tsr_legacy": ("tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only"),
}
# (family, group, recipe, reference); sizes follow from RCT_ONLY and the M11 exclusion.
FAMILIES = [(f"{g}/{m}", g, m, "rct_selected") for g, ms in GROUP_RECIPES.items() for m in ms] + [
    ("gp/causal_icm_fixed vs causal_icm_rct_only", "gp", "causal_icm_fixed", "causal_icm_rct_only"),
    *[(f"tabpfn/tsr_tabpfn_{t} vs lmcgp_rct_only", "tabpfn", f"tsr_tabpfn_{t}", "lmcgp_rct_only") for t in ("v2", "v25", "v3")],
    ("tsr_legacy/tsr_tabpfn_lmcgp vs lmcgp_rct_only", "tsr_legacy", "tsr_tabpfn_lmcgp", "lmcgp_rct_only"),
    *[(f"tabpfn/{k.format(t)} vs {k.format('v2')}", "tabpfn", k.format(t), k.format("v2"))
      for k in ("tsr_tabpfn_{}", "tabpfn_{}_obs_only") for t in ("v25", "v3")],
]
TOL = 1e-12
GPU_RTOL = 1e-6


def as_float(text):
    return None if text in ("", None) else float(text)


def close(a, b, rel=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-12))


def load_records(runs: list[Path]) -> dict:
    out = {}
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        methods = set(manifest["config"]["run"]["methods"])
        group = next((g for g, ms in GROUP_RECIPES.items() if methods == {"rct_selected", "rct_t_ridge", *ms}), None)
        if group is None:
            raise SystemExit(f"{run}: not a wave-2 shard")
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            out[(group, row["scenario"], int(row["seed"]), row["method"])] = row
    return out


def arithmetic(analysis: Path, runs: list[Path]) -> dict:
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
    seeds = sorted(seeds)
    mismatches = [] if len(seeds) == 64 and len(scenarios) == 124 else [{"reason": "expected 64 seeds and 124 conditions"}]
    records = load_records(runs)
    if set(records) != set(table):
        mismatches.append({"reason": "replicate_scores.csv cells differ from the run records"})
    for key, row in records.items():
        got = table.get(key)
        if got is not None and (got["status"] != row["status"] or as_float(got["cate_mse"]) !=
                                (row["metrics"]["cate_mse"] if row["status"] == "ok" else None)):
            mismatches.append({"cell": list(key), "reason": "replicate score differs from its record"})
    with (analysis / "contrasts.csv").open(newline="") as fh:
        reported = {(r["family"], r["scenario"]): r for r in csv.DictReader(fh)}
    counts = {}
    for name, g, rec, ref in FAMILIES:
        fam = sorted(sc for sc in scenarios if not sc.startswith("structural:M11@")
                     and (rec not in RCT_ONLY or "@o512_" in sc))
        declared = (60, 12) if rec in RCT_ONLY else (120, 24)
        if (len(fam), sum(sc.split("@")[0] in TRANSPORT for sc in fam)) != declared:
            mismatches.append({"family": name, "reason": "family size differs from the declared size"})
        rows = []
        for sc in fam:
            rk = sc.replace("@o2048_", "@o512_") if ref in RCT_ONLY else sc
            both = [s for s in seeds if (g, sc, s, rec) in mse and (g, rk, s, ref) in mse]
            d = np.array([mse[(g, sc, s, rec)] - mse[(g, rk, s, ref)] for s in both])
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
            tally[f"{'transport' if r['sc'].split('@')[0] in TRANSPORT else 'other'}_{v}"] += 1
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
    return {"check": "arithmetic", "families": counts, "mismatches": mismatches}


def reproduce(group: str, config, runs: list[Path]) -> dict:
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records = load_records(runs)
    seed = config.seeds[0]
    scenarios = [s for s in config.scenarios if s.parameters["n_obs"] == 512 and s.parameters["n_rct_train"] == 128]
    checks, mismatches = [], []
    for s in scenarios:
        problems = [_generate(s, seed), _generate(s, seed)]
        for method in config.methods:
            row = records[(group, s.key, seed, method)]
            outcomes = []
            for problem in problems:
                from obs_rct_benchmark import tsr

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
    ap.add_argument("--group")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    if a.check == "arithmetic":
        result = arithmetic(a.analysis, a.runs)
    else:
        from obs_rct_benchmark.runner import load_config

        result = reproduce(a.group, load_config(a.config), a.runs)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
