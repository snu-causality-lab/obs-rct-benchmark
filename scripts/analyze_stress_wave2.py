"""Analyze the wave-2 recipes on stress study 1 exactly as declared in docs/V1_STRESS_WAVE2.md.

  python scripts/analyze_stress_wave2.py --configs configs/v1_stress_wave2_*.toml \
      --runs outputs/*/ --stress1 results/stress_cornet --output stress-wave2-analysis

Every declared shard of every group must be present once with a terminal record for
every requested cell; a group's shards share one source, environment and external-method
provenance (checked by the wave-2 collector). Each contrast pairs two recipes fitted in
the same shard. Outputs: replicate_scores.csv, condition_scores.csv, dose_response.csv,
contrasts.csv, readout.csv, identity_check.json, headline.md and analysis.json.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

from analyze_stress import parse
from analyze_v1_core import ALPHA, holm, paired_t
from analyze_v1_wave2 import IDENTITY_TOL, RCT_ONLY, collect, load_job_limit, write_csv
from make_stress_wave2_configs import GROUPS, shards as declared_shards, validate
from obs_rct_benchmark.stress_dgp import EXTRAPOLATION_DIALS

REFERENCE = "rct_selected"
# Dials that change only the OBS sample: trial rows, targets and method seeds are the same
# at every level, so an RCT-only recipe must score identically across levels.
OBS_ONLY_DIALS = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection", "obs_size")


def families() -> list[dict]:
    """Every declared Holm family, named group/recipe[ vs reference]: 11 primary, 5 matched, 4 release."""
    def fam(group, kind, recipe, reference):
        name = f"{group}/{recipe}" + ("" if reference == REFERENCE else f" vs {reference}")
        return {"family": name, "kind": kind, "recipe": recipe, "reference": reference, "group": group}
    out = [fam(g, "primary", m, REFERENCE) for g, recipes in GROUPS.items() for m in recipes if m not in RCT_ONLY]
    out.append(fam("gp", "matched_obs", "causal_icm_fixed", "causal_icm_rct_only"))
    for tag in ("v2", "v25", "v3"):
        out.append(fam("tabpfn", "matched_obs", f"tsr_tabpfn_{tag}", "lmcgp_rct_only"))
    out.append(fam("tsr_legacy", "matched_obs", "tsr_tabpfn_lmcgp", "lmcgp_rct_only"))
    for kind in ("tsr_tabpfn_{}", "tabpfn_{}_obs_only"):
        for tag in ("v25", "v3"):
            out.append(fam("tabpfn", "tabpfn_release", kind.format(tag), kind.format("v2")))
    return out


class AnalysisError(ValueError):
    """The inputs do not match the declared addendum; nothing is reported."""


def stress1_scores(packet: Path) -> dict:
    """rct_selected and rct_t_ridge scores published by stress study 1."""
    out = {}
    for path in sorted((packet / "replicate_scores").glob("*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["method"] in ("rct_selected", "rct_t_ridge") and r["status"] == "ok":
                out[(r["scenario"], int(r["seed"]), r["method"])] = float(r["cate_mse"])
    return out


def state(seq: list[str]) -> str:
    """The stress plan's readout; "untested" at level 3 counts as not better."""
    return ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
            else "stopped" if seq[3] != "better" else "kept")


def analyze(configs: dict, records: dict, stress1: dict, output: Path) -> dict:
    from scipy import stats

    output.mkdir(parents=True, exist_ok=False)
    any_config = next(iter(configs.values()))
    seeds = list(any_config.seeds)
    scenarios = [s.key for s in any_config.scenarios]
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}

    write_csv(output / "replicate_scores.csv", [
        {"group": k[0], "scenario": k[1], "seed": k[2], "method": k[3], "status": r["status"],
         "cate_mse": r["metrics"]["cate_mse"] if r["status"] == "ok" else "",
         "error_type": r.get("error_type") or "", "stage": r.get("stage") or "",
         "selected": ((r.get("fit_info") or {}).get("selected") or "") if k[3] == REFERENCE else ""}
        for k, r in sorted(records.items())])

    identity = {}
    for group in configs:
        for m in ("rct_selected", "rct_t_ridge"):
            pairs = [(mse.get((group, sc, s, m)), stress1.get((sc, s, m))) for sc in scenarios for s in seeds]
            both = [(a, b) for a, b in pairs if a is not None and b is not None]
            worst = max((abs(a - b) for a, b in both), default=None)
            identity[f"{group}:{m}"] = {
                "compared": len(both), "expected": len(scenarios) * len(seeds),
                "exact": sum(a == b for a, b in both), "max_abs_difference": worst,
                "within_1e-9": len(both) == len(scenarios) * len(seeds) and worst is not None and worst <= IDENTITY_TOL}
    level_mismatches = {}
    for group, config in configs.items():
        for m in (x for x in config.methods if x in RCT_ONLY):
            bad = 0
            for dial in OBS_ONLY_DIALS:
                for rct in (128, 512):
                    for s in seeds:
                        vals = [mse.get((group, f"stress:{dial}@l{level}_r{rct}", s, m)) for level in range(4)]
                        bad += any(v != vals[0] for v in vals[1:])
            level_mismatches[f"{group}:{m}"] = bad
    (output / "identity_check.json").write_text(json.dumps(
        {"stress1_reproduction": identity, "rct_only_level_mismatches": level_mismatches,
         "rule": ("contrasts pair recipes fitted in the same shard. A group ran on the stress-study-1 data only if "
                  "its rct_t_ridge scores reproduce stress study 1 within 1e-9; rct_selected drift is reported "
                  "separately")}, indent=2) + "\n")

    conditions, dose = [], []
    for group, config in configs.items():
        for sc in scenarios:
            dial, level, rct = parse(sc)
            for m in config.methods:
                vals = [mse[(group, sc, s, m)] for s in seeds if (group, sc, s, m) in mse]
                conditions.append({"group": group, "scenario": sc, "dial": dial, "level": level, "rct": rct,
                                   "method": m, "successful": len(vals), "failed": len(seeds) - len(vals),
                                   "mean_mse": math.fsum(vals) / len(vals) if vals else None})
                pairs = [(mse[(group, sc, s, m)], mse[(group, sc, s, REFERENCE)]) for s in seeds
                         if (group, sc, s, m) in mse and (group, sc, s, REFERENCE) in mse]
                if m != REFERENCE and len(pairs) >= 2:
                    logs = [math.log(a / b) for a, b in pairs]
                    mean = math.fsum(logs) / len(logs)
                    se = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (len(logs) - 1) / len(logs))
                    q = float(stats.t.ppf(0.975, len(logs) - 1))
                    dose.append({"group": group, "dial": dial, "level": level, "rct": rct, "method": m,
                                 "pairs": len(pairs),
                                 "ratio_of_means": math.fsum(a for a, _ in pairs) / math.fsum(b for _, b in pairs),
                                 "geometric_mean_ratio": math.exp(mean),
                                 "gmr_ci95_lower": math.exp(mean - q * se), "gmr_ci95_upper": math.exp(mean + q * se)})
    write_csv(output / "condition_scores.csv", conditions)
    write_csv(output / "dose_response.csv", dose)

    rows_all, summary = [], {}
    for fam in families():
        g, rec, ref = fam["group"], fam["recipe"], fam["reference"]
        rows = []
        for sc in scenarios:
            dial, level, rct = parse(sc)
            ok = [s for s in seeds if (g, sc, s, rec) in mse and (g, sc, s, ref) in mse]
            diffs = [mse[(g, sc, s, rec)] - mse[(g, sc, s, ref)] for s in ok]
            row = {"family": fam["family"], "kind": fam["kind"], "recipe": rec, "reference": ref, "scenario": sc,
                   "dial": dial, "level": level, "rct": rct, "extrapolation": dial in EXTRAPOLATION_DIALS,
                   "pairs": len(ok), "recipe_failures": sum((g, sc, s, rec) not in mse for s in seeds),
                   "reference_failures": sum((g, sc, s, ref) not in mse for s in seeds),
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
            f"{grp}_{v}": sum(1 for r in rows if r["verdict"] == v and r["extrapolation"] == (grp == "extrapolation"))
            for grp in ("other", "extrapolation") for v in ("better", "worse", "unresolved", "untested")}}
        rows_all += rows
    write_csv(output / "contrasts.csv", rows_all)

    verdict = {(r["family"], r["dial"], r["level"], r["rct"]): r["verdict"] for r in rows_all}
    readout = []
    for fam in families():
        if fam["kind"] != "primary":
            continue
        for dial in dict.fromkeys(parse(sc)[0] for sc in scenarios):
            for rct in (128, 512):
                seq = [verdict[(fam["family"], dial, level, rct)] for level in range(4)]
                readout.append({"family": fam["family"], "dial": dial, "rct": rct,
                                **{f"level_{i}": v for i, v in enumerate(seq)}, "state": state(seq),
                                # A state read from an untested level reflects failed fits, not a loss.
                                "untested_at": ",".join(str(i) for i in (0, 3) if seq[i] == "untested")})
    write_csv(output / "readout.csv", readout)

    lines = ["| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | Extrapolation: better / worse |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, f in summary.items():
        tot = {v: f[f"other_{v}"] + f[f"extrapolation_{v}"] for v in ("better", "worse", "unresolved", "untested")}
        lines.append(f"| `{name}` | {f['kind']} | {f['size']} | {tot['better']} | {tot['worse']} | {tot['unresolved']} "
                     f"| {tot['untested']} | {f['extrapolation_better']} / {f['extrapolation_worse']} |")
    lines += ["", "Each family pairs recipes fitted in one group's environment. Each group has its own `rct_selected`; "
              "ratios set side by side across groups, or against stress study 1, are descriptive unless that "
              "group's `rct_selected` reproduces stress study 1.", "",
              "| Group | `rct_selected` equal to stress study 1 | Largest difference | `rct_t_ridge` within 1e-9 of stress study 1 |",
              "| --- | ---: | ---: | --- |"]
    for group in configs:
        sel, ridge = identity[f"{group}:rct_selected"], identity[f"{group}:rct_t_ridge"]
        lines.append(f"| `{group}` | {sel['exact']} of {sel['expected']} | {sel['max_abs_difference']} | "
                     f"{'yes' if ridge['within_1e-9'] else 'no'} |")
    (output / "headline.md").write_text("\n".join(lines) + "\n")

    failures = Counter((k[0], k[3], r.get("stage"), r.get("error_type"))
                       for k, r in records.items() if r["status"] != "ok")
    analysis = {"protocol": "docs/V1_STRESS_WAVE2.md", "requested": len(records),
                "coverage": dict(Counter(r["status"] for r in records.values())),
                "failures": [{"group": g, "method": m, "stage": st, "error_type": e, "count": n}
                             for (g, m, st, e), n in sorted(failures.items(), key=str)],
                "families": summary}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=Path, nargs="+", required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--stress1", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--job-limit-shards", type=Path,
                    help="JSON list of {shard, config_hash} whose final attempt reached its job limit")
    a = ap.parse_args()
    configs = {}
    for path in a.configs:
        config = load_config(path)
        configs[validate(config)] = config
    if set(configs) != set(GROUPS):
        raise SystemExit("one configuration per declared group is required")
    records = collect(configs, a.runs, declared=declared_shards, job_limit=load_job_limit(a.job_limit_shards))
    result = analyze(configs, records, stress1_scores(a.stress1), a.output)
    print(json.dumps({k: result[k] for k in ("requested", "coverage", "families")}, indent=2)[:6000])


if __name__ == "__main__":
    main()
