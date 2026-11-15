"""Independent checks for the selected-rho Causal-ICM addendum (docs/V1_ICM_SELECTED.md).

  arithmetic  Rebuild the block's requested cells from its frozen configuration,
              authenticate the run records (one record per cell, record and selection
              identities, valid metrics), then recompute every published output of the
              analyzer and compare: replicate_scores.csv, condition_scores.csv,
              rho_selection.csv, contrasts.csv, identity_check.json, analysis.json and
              headline.md (whole text); for the stress block also dose_response.csv and
              readout.csv. It imports no analysis script.
  reproduce   Refit every recipe twice on the block's first seed (core: OBS 512 and RCT
              128; stress: RCT 128 at levels 0 and 3 of every dial) and compare with the
              records within 1e-12. Cells without a record are listed as unavailable.

    python scripts/verify_icm_selected.py arithmetic --config configs/v1_icm_selected_core.toml \
        --analysis icmsel-core-analysis --runs out-core/*/ --wave2 results/v1_wave2 --output arith-core.json
    python scripts/verify_icm_selected.py reproduce --config configs/v1_icm_selected_core.toml \
        --runs out-core/*/ --output reproduce-core.json
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

GROUP = "gp"
METHODS = ("rct_selected", "rct_t_ridge", "causal_icm_fixed", "causal_icm_rct_only", "causal_icm_selected")
OPTIONS = {"rct_selected": {"ridge_alpha": 1.0, "n_estimators": 100, "min_samples_leaf": 5},
           "rct_t_ridge": {"ridge_alpha": 1.0}, "causal_icm_fixed": {"rho": 0.5}, "causal_icm_rct_only": {"rho": 0.5}}
RCT_ONLY = {"rct_selected", "rct_t_ridge", "causal_icm_rct_only"}
WAVE2 = ("rct_selected", "rct_t_ridge", "causal_icm_fixed", "causal_icm_rct_only")
GRID = tuple(k / 10 for k in range(11))
SELECTOR = "causal_icm_weighted_rct_cv_v1"
# (family, recipe, reference, kind)
FAMILIES = [("gp/causal_icm_selected", "causal_icm_selected", "rct_selected", "primary"),
            ("gp/causal_icm_selected vs causal_icm_rct_only", "causal_icm_selected", "causal_icm_rct_only", "matched_obs"),
            ("gp/causal_icm_selected vs causal_icm_fixed", "causal_icm_selected", "causal_icm_fixed", "selection")]
BLOCK = {
    "core": {"seeds": tuple(range(70000, 70064)), "conditions": 124, "family_size": 120, "split": "transport",
             "split_size": 24},
    "stress": {"seeds": tuple(range(80000, 80064)), "conditions": 64, "family_size": 64, "split": "extrapolation",
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
IDENTITY_RULE = ("contrasts pair recipes fitted in the same shard. The block ran on wave 2's data and fits only if "
                 "the data check passed and all four refitted recipes reproduce wave 2's gp scores within 1e-9 with "
                 "the same statuses. A selected fit that chose 0.5 must equal causal_icm_fixed exactly")


def cell_text(value) -> str:
    """How csv.DictWriter writes a value."""
    return "" if value is None else str(value)


def as_float(text):
    return None if text in ("", None) else float(text)


def close(a, b, rel=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-12))


def dial(scenario: str) -> str:
    return scenario.split("@")[0].split(":")[1]


def level_rct(scenario: str) -> tuple[int, int]:
    level, rct = scenario.split("@")[1].split("_")
    return int(level[1:]), int(rct[1:])


def in_split(block: str, scenario: str) -> bool:
    return dial(scenario) in EXTRAPOLATION if block == "stress" else scenario.split("@")[0] in TRANSPORT


def read_csv(path: Path, key: tuple[str, ...], mismatches: list) -> dict:
    """Rows keyed by `key`; a duplicate key is a mismatch."""
    rows = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            k = tuple(r[f] for f in key)
            if k in rows:
                mismatches.append({"file": path.name, "reason": "duplicate row", "key": list(k)})
            rows[k] = r
    return rows


def compare_rows(name: str, reported: dict, expected: dict, mismatches: list) -> None:
    """Every expected row and field, and no other row. Numbers within a relative 1e-9;
    everything else as written."""
    if set(reported) != set(expected):
        mismatches.append({"file": name, "reason": "row set differs", "missing": sorted(set(expected) - set(reported))[:5],
                           "extra": sorted(set(reported) - set(expected))[:5]})
    for k, want in expected.items():
        got = reported.get(k)
        if got is None:
            continue
        for field, value in want.items():
            text = got.get(field)
            if isinstance(value, float) and not isinstance(value, bool):
                ok = text not in (None, "") and close(float(text), value)
            else:
                ok = text == cell_text(value)
            if not ok:
                mismatches.append({"file": name, "key": list(k), "field": field, "recomputed": value, "reported": text})


def block_of(config) -> str:
    for name, spec in BLOCK.items():
        if config.seeds == spec["seeds"]:
            if (config.methods != METHODS or {m: dict(o) for m, o in config.method_options.items()} != OPTIONS
                    or len(config.scenarios) != spec["conditions"]):
                raise SystemExit("configuration differs from the declared addendum")
            return name
    raise SystemExit("seeds match neither declared block")


def on_grid(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value in GRID


def valid_metrics(row: dict) -> bool:
    m = row.get("metrics")
    if row["status"] == "failed":
        return m is None
    if not isinstance(m, dict):
        return False
    mse, root = m.get("cate_mse"), m.get("root_pehe")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (mse, root)):
        return False
    return math.isclose(root * root, mse, rel_tol=1e-10, abs_tol=1e-12)


def partition(config) -> dict:
    """{config hash: shard} for the block's declared shards, rebuilt here from the frozen
    configuration: core, one shard per seed and preset; stress, one per seed, dial and
    trial size; scenarios in configuration order."""
    from dataclasses import replace

    stress = block_of(config) == "stress"
    groups = defaultdict(list)
    for s in config.scenarios:
        groups[(s.id, s.parameters["n_rct_train"]) if stress else (s.kind, s.id)].append(s)
    out = {}
    for seed in config.seeds:
        for scenarios in groups.values():
            shard = replace(config, scenarios=tuple(scenarios), seeds=(seed,))
            out[shard.config_hash] = shard
    return out


def load_records(config, runs: list[Path], job_limit: frozenset = frozenset()) -> tuple[dict, set, list]:
    """Records keyed (group, scenario, seed, method) for exactly the block's declared shards;
    the cells a declared job-limit run left unrecorded; and problems found."""
    from obs_rct_benchmark.runner import method_seed, record_id

    shards = partition(config)
    out, unrecorded, seen, problems, provenance = {}, set(), set(), [], set()
    if not job_limit <= set(shards):
        problems.append({"reason": "a declared job-limit shard is not a declared shard"})
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        shard = shards.get(manifest.get("config_hash"))
        if shard is None or manifest.get("config") != shard.payload():
            problems.append({"run": run.name, "reason": "not a declared shard of the block"})
            continue
        if shard.config_hash in seen:
            problems.append({"run": run.name, "reason": "shard given twice"})
        seen.add(shard.config_hash)
        provenance.add(json.dumps({k: manifest.get(k) for k in ("source", "environment", "external_methods")},
                                  sort_keys=True))
        requested = {(GROUP, sc.key, seed, m) for sc in shard.scenarios for seed in shard.seeds for m in METHODS}
        path = run / "records.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        limited = shard.config_hash in job_limit
        if limited and lines:
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                lines = lines[:-1]
        got = set()
        for line in lines:
            row = json.loads(line)
            key = (GROUP, row["scenario"], int(row["seed"]), row["method"])
            if key in out or key not in requested:
                problems.append({"run": run.name, "cell": list(key), "reason": "duplicate or unrequested record"})
            if (row["config_hash"] != shard.config_hash or row["method_seed"] != method_seed(*key[1:])
                    or row["record_id"] != record_id(shard, *key[1:])
                    or row["status"] not in ("ok", "failed") or not valid_metrics(row)):
                problems.append({"run": run.name, "cell": list(key), "reason": "record identity, status or metrics"})
            out[key] = row
            got.add(key)
        if limited:
            unrecorded |= requested - got
        elif got != requested:
            problems.append({"run": run.name, "reason": f"{len(requested - got)} requested cells lack a record"})
    if seen != set(shards):
        problems.append({"reason": f"{len(set(shards) - seen)} declared shards are missing"})
    if not job_limit <= seen:
        problems.append({"reason": "a declared job-limit shard was not given"})
    if len(provenance) > 1:
        problems.append({"reason": "source, environment or external-method provenance differs between shards"})
    for key, row in out.items():
        if key[3] != "causal_icm_selected":
            continue
        twin = out.get((*key[:3], "causal_icm_fixed"))
        if twin is not None and twin["method_seed"] != row["method_seed"]:
            problems.append({"cell": list(key), "reason": "selected and fixed method seeds differ"})
        info = row.get("fit_info") or {}
        if row["status"] == "ok" and (info.get("selector") != SELECTOR or info.get("status") != "completed"
                                      or not on_grid(info.get("selected_rho")) or info.get("fold_seed") != row["method_seed"]):
            problems.append({"cell": list(key), "reason": "successful selected fit without a completed selection"})
    return out, unrecorded, problems


def wave2_scores(packet: Path) -> dict:
    out = {}
    for path in sorted((packet / "replicate_scores").glob("*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["group"] == GROUP and r["method"] in WAVE2:
                out[(r["scenario"], int(r["seed"]), r["method"])] = as_float(r["cate_mse"]) if r["status"] == "ok" else None
    return out


def paired(d: np.ndarray) -> dict:
    """Two-sided paired t; identical differences give p = 1 (declared rule)."""
    from scipy.special import stdtr, stdtrit

    n, mean = len(d), float(np.mean(d))
    se = float(np.std(d, ddof=1)) / math.sqrt(n)
    if np.all(d == d[0]) or se == 0.0:
        return {"mean": mean, "mcse": 0.0, "lo": mean, "hi": mean, "p": 1.0, "zero": True}
    q = float(stdtrit(n - 1, 0.975))
    return {"mean": mean, "mcse": se, "lo": mean - q * se, "hi": mean + q * se,
            "p": float(2 * stdtr(n - 1, -abs(mean / se))), "zero": False}


def arithmetic(config, analysis: Path, runs: list[Path], wave2: Path, job_limit: frozenset = frozenset()) -> dict:
    from scipy.special import stdtrit

    block = block_of(config)
    spec = BLOCK[block]
    split = spec["split"]
    seeds = list(config.seeds)
    scenarios = [s.key for s in config.scenarios]
    records, unrecorded, mismatches = load_records(config, runs, job_limit)
    for key in unrecorded:
        records[key] = {"status": "failed", "metrics": None, "stage": "job_limit", "error_type": "JobLimit"}
    if set(records) != {(GROUP, sc, seed, m) for sc in scenarios for seed in seeds for m in METHODS}:
        return {"check": "arithmetic", "block": block, "mismatches": mismatches + [
            {"reason": "the records do not cover the declared cells; nothing else was recomputed"}]}
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}
    rho = {k: float(r["fit_info"]["selected_rho"]) for k, r in records.items()
           if k[3] == "causal_icm_selected" and r["status"] == "ok" and on_grid((r.get("fit_info") or {}).get("selected_rho"))}

    compare_rows("replicate_scores.csv", read_csv(analysis / "replicate_scores.csv", ("group", "scenario", "seed", "method"),
                                                  mismatches),
                 {(k[0], k[1], str(k[2]), k[3]): {
                     "status": r["status"], "cate_mse": r["metrics"]["cate_mse"] if r["status"] == "ok" else None,
                     "root_pehe": r["metrics"]["root_pehe"] if r["status"] == "ok" else None,
                     "error_type": r.get("error_type") or None, "stage": r.get("stage") or None,
                     "selected": ((r.get("fit_info") or {}).get("selected") or None) if k[3] == "rct_selected" else None,
                     "selected_rho": rho.get(k)} for k, r in records.items()}, mismatches)

    # Identities.
    first = wave2_scores(wave2)
    reproduction = {}
    for m in WAVE2:
        keys = [(sc, s) for sc in scenarios for s in seeds]
        both = [(mse[(GROUP, sc, s, m)], first[(sc, s, m)]) for sc, s in keys
                if (GROUP, sc, s, m) in mse and first.get((sc, s, m)) is not None]
        differs = sum((sc, s, m) not in first or ((GROUP, sc, s, m) in mse) != (first[(sc, s, m)] is not None)
                      for sc, s in keys)
        worst = max((abs(a - b) for a, b in both), default=None)
        reproduction[m] = {"compared": len(both), "expected": len(keys), "status_differs": differs,
                           "exact": sum(a == b for a, b in both), "max_abs_difference": worst,
                           "within_1e-9": differs == 0 and worst is not None and worst <= SAME_TOL}
    half = [k for k, v in rho.items() if v == 0.5]
    at_half = {"cells": len(half),
               "fixed_not_successful": sum((*k[:3], "causal_icm_fixed") not in mse for k in half),
               "differ": sum(mse.get((*k[:3], "causal_icm_fixed")) != mse[k] for k in half)}
    rct_only = {}
    for m in sorted(RCT_ONLY):
        if block == "core":
            rct_only[m] = sum(mse.get((GROUP, sc, s, m)) != mse.get((GROUP, sc.replace("@o512_", "@o2048_"), s, m))
                              for sc in scenarios if "@o512_" in sc for s in seeds)
        else:
            rct_only[m] = sum(
                len({mse.get((GROUP, f"stress:{d}@l{i}_r{r}", s, m)) for i in range(4)}) > 1
                for d in OBS_ONLY for r in (128, 512) for s in seeds)
    identity_name = "rct_only_obs_size_mismatches" if block == "core" else "rct_only_level_mismatches"
    reported_identity = json.loads((analysis / "identity_check.json").read_text())
    for m, want in reproduction.items():
        got = reported_identity.get("wave2_reproduction", {}).get(m, {})
        if (any(got.get(f) != want[f] for f in ("compared", "expected", "status_differs", "exact", "within_1e-9"))
                or not close(got.get("max_abs_difference"), want["max_abs_difference"])):
            mismatches.append({"identity": f"wave2:{m}", "recomputed": want, "reported": got})
    for name, want in (("selected_at_0.5_vs_fixed", at_half), (identity_name, rct_only), ("rule", IDENTITY_RULE)):
        if reported_identity.get(name) != want:
            mismatches.append({"identity": name, "recomputed": want, "reported": reported_identity.get(name)})
    if set(reported_identity) != {"wave2_reproduction", "selected_at_0.5_vs_fixed", identity_name, "rule"}:
        mismatches.append({"identity": "keys", "reported": sorted(reported_identity)})

    # Condition scores, selection diagnostics and (stress) dose-response.
    cond, rho_rows, dose = {}, {}, {}
    for sc in scenarios:
        extra = {"dial": dial(sc), "level": level_rct(sc)[0], "rct": level_rct(sc)[1]} if block == "stress" else {}
        for m in METHODS:
            vals = [mse[(GROUP, sc, s, m)] for s in seeds if (GROUP, sc, s, m) in mse]
            cond[(sc, m)] = {**extra, split: in_split(block, sc), "successful": len(vals),
                             "failed": len(seeds) - len(vals), "mean_mse": float(np.mean(vals)) if vals else None}
            pairs = np.array([(mse[(GROUP, sc, s, m)], mse[(GROUP, sc, s, "rct_selected")]) for s in seeds
                              if (GROUP, sc, s, m) in mse and (GROUP, sc, s, "rct_selected") in mse]).reshape(-1, 2)
            if block == "stress" and m != "rct_selected" and len(pairs) >= 2:
                row = {"pairs": len(pairs),
                       "ratio_of_means": float(pairs[:, 0].sum() / pairs[:, 1].sum()) if (pairs[:, 1] > 0).any() else None,
                       "geometric_mean_ratio": None, "gmr_ci95_lower": None, "gmr_ci95_upper": None}
                if (pairs > 0).all():
                    logs = np.log(pairs[:, 0]) - np.log(pairs[:, 1])
                    se = float(np.std(logs, ddof=1)) / math.sqrt(len(logs))
                    q = float(stdtrit(len(logs) - 1, 0.975))
                    row.update(geometric_mean_ratio=math.exp(float(np.mean(logs))),
                               gmr_ci95_lower=math.exp(float(np.mean(logs)) - q * se),
                               gmr_ci95_upper=math.exp(float(np.mean(logs)) + q * se))
                dose[(str(extra["dial"]), str(extra["level"]), str(extra["rct"]), m)] = row
        attempts = [records[(GROUP, sc, s, "causal_icm_selected")] for s in seeds]
        chosen = [rho[(GROUP, sc, s, "causal_icm_selected")] for s in seeds if (GROUP, sc, s, "causal_icm_selected") in rho]
        infos = [(r["status"], r.get("fit_info") or {}) for r in attempts]
        weights = [i["participation_weights"] for _, i in infos if isinstance(i.get("participation_weights"), dict)]
        failed = [i.get("status") for st, i in infos if st != "ok"]
        rho_rows[(sc,)] = {**extra, "attempts": len(attempts), "successful": len(chosen),
                           **{f"rho_{r:.1f}": chosen.count(r) for r in GRID},
                           "mean_rho": float(np.mean(chosen)) if chosen else None,
                           "successful_fewer_than_5_folds": sum(i.get("folds_used") != 5 for st, i in infos if st == "ok"),
                           "failed_selecting": failed.count("selecting"), "failed_final_fit": failed.count("final_fit"),
                           "failed_other": sum(f not in ("selecting", "final_fit") for f in failed),
                           "weights_recorded": len(weights),
                           "weights_not_converged": sum(w.get("converged") is not True for w in weights)}
    compare_rows("condition_scores.csv", read_csv(analysis / "condition_scores.csv", ("scenario", "method"), mismatches),
                 cond, mismatches)
    compare_rows("rho_selection.csv", read_csv(analysis / "rho_selection.csv", ("scenario",), mismatches),
                 rho_rows, mismatches)
    if block == "stress":
        compare_rows("dose_response.csv", read_csv(analysis / "dose_response.csv", ("dial", "level", "rct", "method"),
                                                   mismatches), dose, mismatches)

    # Contrasts, Holm and verdicts.
    expected, counts, verdicts = {}, {}, {}
    for fam, rec, ref, kind in FAMILIES:
        pairs = [(sc, sc) for sc in scenarios] if block == "stress" else [
            (sc, sc.replace("@o2048_", "@o512_") if ref in RCT_ONLY else sc)
            for sc in scenarios if not sc.startswith(HEAVY_TAIL + "@")]
        if (len(pairs), sum(in_split(block, sc) for sc, _ in pairs)) != (spec["family_size"], spec["split_size"]):
            mismatches.append({"family": fam, "reason": "family size differs from the declared size"})
        rows = []
        for sc, rk in pairs:
            both = [s for s in seeds if (GROUP, sc, s, rec) in mse and (GROUP, rk, s, ref) in mse]
            d = np.array([mse[(GROUP, sc, s, rec)] - mse[(GROUP, rk, s, ref)] for s in both])
            complete = len(both) == len(seeds)
            stats = paired(d) if complete else {"mean": float(np.mean(d)) if len(d) else None, "mcse": None,
                                                "lo": None, "hi": None, "p": 1.0, "zero": None}
            rows.append({"sc": sc, "rk": rk, "complete": complete, "pairs": len(both),
                         "rf": sum((GROUP, sc, s, rec) not in mse for s in seeds),
                         "ff": sum((GROUP, rk, s, ref) not in mse for s in seeds), **stats})
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
            expected[(fam, r["sc"])] = {
                "kind": kind, "recipe": rec, "reference": ref, "reference_scenario": r["rk"],
                split: in_split(block, r["sc"]), "pairs": r["pairs"], "recipe_failures": r["rf"],
                "reference_failures": r["ff"], "complete": r["complete"], "mean_difference": r["mean"],
                "mcse": r["mcse"], "ci95_lower": r["lo"], "ci95_upper": r["hi"], "p_two_sided": r["p"],
                "zero_variance": r["zero"], "holm_p": r["holm"], "verdict": v}
        counts[fam] = {"kind": kind, "size": len(rows), **{
            f"{g}_{v}": tally[f"{g}_{v}"] for g in ("other", split) for v in ("better", "worse", "unresolved", "untested")}}
    compare_rows("contrasts.csv", read_csv(analysis / "contrasts.csv", ("family", "scenario"), mismatches),
                 expected, mismatches)

    states = {}
    if block == "stress":
        fam = FAMILIES[0][0]
        for d in DIALS:
            for rct in (128, 512):
                seq = [verdicts[(fam, f"stress:{d}@l{level}_r{rct}")] for level in range(4)]
                st = ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
                      else "kept" if seq[3] == "better" else "stopped")
                states[(fam, d, str(rct))] = {**{f"level_{i}": v for i, v in enumerate(seq)}, "state": st,
                                              "untested_at": ",".join(str(i) for i in range(4) if seq[i] == "untested")}
        compare_rows("readout.csv", read_csv(analysis / "readout.csv", ("family", "dial", "rct"), mismatches),
                     states, mismatches)

    totals = {f"{r:.1f}": list(rho.values()).count(r) for r in GRID}
    failures = Counter((k[3], r.get("stage"), r.get("error_type")) for k, r in records.items() if r["status"] != "ok")
    want_analysis = {"protocol": "docs/V1_ICM_SELECTED.md", "block": block, "requested": len(records),
                     "coverage": dict(Counter(r["status"] for r in records.values())),
                     "failures": sorted(([m, st, e, n] for (m, st, e), n in failures.items()), key=str),
                     "families": counts, "selected_rho": totals}
    got_analysis = json.loads((analysis / "analysis.json").read_text())
    got_analysis["failures"] = sorted(([f["method"], f["stage"], f["error_type"], f["count"]]
                                       for f in got_analysis.get("failures", [])), key=str)
    for field, want in want_analysis.items():
        if got_analysis.get(field) != want:
            mismatches.append({"file": "analysis.json", "field": field, "recomputed": want,
                               "reported": got_analysis.get(field)})

    label = split.capitalize()
    lines = [f"| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | {label}: better / worse |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fam, c in counts.items():
        tot = [c[f"other_{v}"] + c[f"{split}_{v}"] for v in ("better", "worse", "unresolved", "untested")]
        lines.append(f"| `{fam}` | {c['kind']} | {c['size']} | " + " | ".join(map(str, tot))
                     + f" | {c[split + '_better']} / {c[split + '_worse']} |")
    lines += ["", "| Recipe refitted from wave 2 | Equal to wave 2 | Largest difference | Within 1e-9, same statuses |",
              "| --- | ---: | ---: | --- |"]
    for m, r in reproduction.items():
        lines.append(f"| `{m}` | {r['exact']} of {r['expected']} | {r['max_abs_difference']} | "
                     f"{'yes' if r['within_1e-9'] else 'no'} |")
    lines += ["", f"Successful selected fits that chose 0.5: {at_half['cells']}; differing from `causal_icm_fixed` "
              f"or without a successful `causal_icm_fixed` twin: {at_half['differ']}.", "",
              "| Selected rho | " + " | ".join(f"{r:.1f}" for r in GRID) + " |",
              "| --- | " + " | ".join("---:" for _ in GRID) + " |",
              "| Successful fits | " + " | ".join(str(totals[f"{r:.1f}"]) for r in GRID) + " |"]
    if (analysis / "headline.md").read_text() != "\n".join(lines) + "\n":
        mismatches.append({"file": "headline.md", "reason": "text differs from the recomputed headline"})
    return {"check": "arithmetic", "block": block, "families": counts, "wave2_reproduction": reproduction,
            "selected_at_0.5_vs_fixed": at_half, identity_name: rct_only, "selected_rho": totals,
            "readout_states": len(states), "mismatches": mismatches}


def reproduce(config, runs: list[Path], job_limit: frozenset = frozenset()) -> dict:
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records, _, problems = load_records(config, runs, job_limit)
    seed = config.seeds[0]
    stress = config.scenarios[0].kind == "stress"
    scenarios = [s for s in config.scenarios if s.parameters["n_rct_train"] == 128
                 and (s.parameters["level"] in (0, 3) if stress else s.parameters["n_obs"] == 512)]
    checks, mismatches, unavailable = [], [], []
    for s in scenarios:
        problems_ = [_generate(s, seed), _generate(s, seed)]
        for method in config.methods:
            row = records.get((GROUP, s.key, seed, method))
            if row is None:
                unavailable.append([s.key, method])
                continue
            outcomes = []
            for problem in problems_:
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
    # A cell without a record cannot be replayed: the check is then incomplete, which fails it.
    return {"check": "reproduce", "seed": seed, "cases": len(checks), "refits": 2 * len(checks),
            "unavailable": unavailable, "record_problems": problems,
            "mismatches": mismatches + problems + [{"unavailable": u} for u in unavailable], "details": checks}


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("arithmetic", "reproduce"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--wave2", type=Path)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--job-limit-shards", type=Path,
                    help="JSON list of {shard, config_hash} whose final attempt reached its job limit")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    limited = frozenset(e["config_hash"] for e in json.loads(a.job_limit_shards.read_text())) \
        if a.job_limit_shards else frozenset()
    config = load_config(a.config)
    if a.check == "arithmetic":
        if not (a.analysis and a.wave2):
            ap.error("arithmetic needs --analysis and --wave2")
        result = arithmetic(config, a.analysis, a.runs, a.wave2, limited)
    else:
        result = reproduce(config, a.runs, limited)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
