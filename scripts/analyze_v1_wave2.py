"""Analyze the v1 core wave-2 addendum exactly as declared in docs/V1_CORE_WAVE2.md.

Usage:
  python scripts/analyze_v1_wave2.py --configs configs/v1_wave2_native_r.toml configs/v1_wave2_gp.toml \
      configs/v1_wave2_tabpfn.toml --runs outputs/*/ --wave1 results/v1_core --output wave2-analysis

Every declared shard of every group must be present once with a terminal record for
every requested cell. Shards of one group must share one source and environment;
external-method provenance must agree across a group's shards. Each contrast pairs two
recipes fitted in the same shard. Outputs: replicate_scores.csv, condition_scores.csv,
contrasts.csv, identity_check.json, headline.md and analysis.json.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

from analyze_v1_core import ALPHA, HEAVY_TAIL, TRANSPORT, holm, paired_t, preset
from make_v1_wave2_configs import GROUPS, shards as declared_shards, validate

REFERENCE = "rct_selected"
RCT_ONLY = frozenset({"rct_selected", "rct_t_ridge", "rct_causal_forest", "causal_icm_rct_only", "lmcgp_rct_only"})
# Stage 1 provenance each successful TabPFN fit must carry: (package, release or None).
TABPFN_STAGE = {**{m: ("10.0.0", r) for m, r in (
    ("tsr_tabpfn_v2", "v2"), ("tsr_tabpfn_v25", "v2.5"), ("tsr_tabpfn_v3", "v3"),
    ("tabpfn_v2_obs_only", "v2"), ("tabpfn_v25_obs_only", "v2.5"), ("tabpfn_v3_obs_only", "v3"))},
    "tsr_tabpfn_lmcgp": ("2.0.9", None), "tabpfn_obs_only": ("2.0.9", None)}
IDENTITY_TOL = 1e-9


def families() -> list[dict]:
    """Every declared Holm family, named group/recipe[ vs reference]."""
    def fam(group, kind, recipe, reference):
        name = f"{group}/{recipe}" + ("" if reference == REFERENCE else f" vs {reference}")
        return {"family": name, "kind": kind, "recipe": recipe, "reference": reference, "group": group}
    out = [fam(g, "primary", m, REFERENCE) for g, recipes in GROUPS.items() for m in recipes]
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


def collect(configs: dict, runs: list[Path], *, check_source: bool = True, declared=declared_shards) -> dict:
    """Authenticated terminal records keyed (group, scenario, seed, method). `declared`
    yields a group configuration's (name, shard) pairs; the stress addendum passes its own."""
    from obs_rct_benchmark.runner import method_seed, record_id, source_provenance

    expected = {}
    for group, config in configs.items():
        for name, shard in declared(config):
            expected[shard.config_hash] = (group, name, shard)
    seen, records, identity, external = set(), {}, {}, {}
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        group, name, shard = expected.get(manifest.get("config_hash"), (None, None, None))
        if shard is None or manifest.get("config") != shard.payload():
            raise AnalysisError(f"{run}: not a declared wave-2 shard")
        if name in seen:
            raise AnalysisError(f"{run}: shard {name} given twice")
        seen.add(name)
        ident = json.dumps({"source": manifest["source"], "environment": manifest["environment"]}, sort_keys=True)
        if identity.setdefault(group, ident) != ident:
            raise AnalysisError(f"{run}: source or environment differs within group {group}")
        ext = json.dumps(manifest.get("external_methods"), sort_keys=True)
        if external.setdefault(group, ext) != ext:
            raise AnalysisError(f"{run}: external-method provenance differs within group {group}")
        if check_source and manifest["source"] != source_provenance():
            raise AnalysisError(f"{run}: producer source differs from the analysis package")
        h = shard.config_hash
        cells = {(s.key, seed, m) for s in shard.scenarios for seed in shard.seeds for m in shard.methods}
        got = set()
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            cell = (row["scenario"], int(row["seed"]), row["method"])
            if cell not in cells or cell in got:
                raise AnalysisError(f"{run}: unrequested or duplicate cell {cell}")
            if row["config_hash"] != h or row["method_seed"] != method_seed(*cell) or row["record_id"] != record_id(shard, *cell):
                raise AnalysisError(f"{run}: record identity mismatch for {cell}")
            if row["status"] not in {"ok", "failed"}:
                raise AnalysisError(f"{cell}: status {row['status']!r} is not terminal")
            if row["status"] == "ok" and row["method"] in TABPFN_STAGE:
                info = row.get("fit_info") or {}
                package, release = TABPFN_STAGE[row["method"]]
                if (info.get("stage1_device") != "cuda"
                        or info.get("stage1_model_sha256") != dict(shard.method_options[row["method"]])["model_sha256"]
                        or info.get("stage1_tabpfn_package") != package
                        or info.get("stage1_tabpfn_release") != release):
                    raise AnalysisError(f"{cell}: TabPFN stage lacks CUDA, the pinned checkpoint, package or release")
            got.add(cell)
            records[(group, *cell)] = row
        if got != cells:
            raise AnalysisError(f"{run}: {len(cells - got)} requested cells lack a terminal record")
    missing = {n for _, n, _ in expected.values()} - seen
    if missing:
        raise AnalysisError(f"{len(missing)} declared shards are missing, e.g. {sorted(missing)[:3]}")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wave1_scores(packet: Path) -> dict:
    out = {}
    for path in sorted((packet / "replicate_scores").glob("*.csv")):
        for r in csv.DictReader(path.open(newline="")):
            if r["method"] in ("rct_selected", "rct_t_ridge") and r["status"] == "ok":
                out[(r["scenario"], int(r["seed"]), r["method"])] = float(r["cate_mse"])
    return out


def analyze(configs: dict, records: dict, wave1: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    any_config = next(iter(configs.values()))
    seeds = list(any_config.seeds)
    scenarios = [s.key for s in any_config.scenarios]
    mse = {k: float(r["metrics"]["cate_mse"]) for k, r in records.items() if r["status"] == "ok"}

    replicate = [{"group": k[0], "scenario": k[1], "seed": k[2], "method": k[3], "status": r["status"],
                  "cate_mse": r["metrics"]["cate_mse"] if r["status"] == "ok" else "",
                  "root_pehe": r["metrics"]["root_pehe"] if r["status"] == "ok" else "",
                  "error_type": r.get("error_type") or "", "stage": r.get("stage") or "",
                  "selected": ((r.get("fit_info") or {}).get("selected") or "") if k[3] == REFERENCE else ""}
                 for k, r in sorted(records.items())]
    write_csv(output / "replicate_scores.csv", replicate)

    identity = {}
    for group in configs:
        for m in ("rct_selected", "rct_t_ridge"):
            pairs = [(mse.get((group, sc, s, m)), wave1.get((sc, s, m))) for sc in scenarios for s in seeds]
            both = [(a, b) for a, b in pairs if a is not None and b is not None]
            worst = max((abs(a - b) for a, b in both), default=None)
            identity[f"{group}:{m}"] = {
                "compared": len(both), "expected": len(scenarios) * len(seeds),
                "exact": sum(a == b for a, b in both), "max_abs_difference": worst,
                "within_1e-9": len(both) == len(scenarios) * len(seeds) and worst is not None and worst <= IDENTITY_TOL}
    obs_size = {}
    for group in configs:
        for m in (x for x in (*GROUPS[group], REFERENCE) if x in RCT_ONLY):
            diffs = []
            for sc in scenarios:
                if "@o512_" not in sc:
                    continue
                twin = sc.replace("@o512_", "@o2048_")
                for s in seeds:
                    a, b = mse.get((group, sc, s, m)), mse.get((group, twin, s, m))
                    if (a is None) != (b is None) or (a is not None and a != b):
                        diffs.append((sc, s))
            obs_size[f"{group}:{m}"] = len(diffs)
    (output / "identity_check.json").write_text(json.dumps(
        {"wave1_reproduction": identity, "rct_only_obs_size_mismatches": obs_size,
         "rule": ("contrasts pair recipes fitted in the same shard. A group ran on the wave-1 data only if its "
                  "rct_t_ridge scores reproduce wave 1 within 1e-9; rct_selected drift is reported separately")},
        indent=2) + "\n")

    conditions = []
    for group, config in configs.items():
        for sc in scenarios:
            for m in config.methods:
                vals = [mse[(group, sc, s, m)] for s in seeds if (group, sc, s, m) in mse]
                conditions.append({"group": group, "scenario": sc, "method": m, "transport": preset(sc) in TRANSPORT,
                                   "successful": len(vals), "failed": len(seeds) - len(vals),
                                   "mean_mse": math.fsum(vals) / len(vals) if vals else None})
    write_csv(output / "condition_scores.csv", conditions)

    rows_all, summary = [], {}
    for fam in families():
        g, rec, ref = fam["group"], fam["recipe"], fam["reference"]
        rows = []
        for sc in scenarios:
            if preset(sc) == HEAVY_TAIL or (rec in RCT_ONLY and "@o512_" not in sc):
                continue
            rk = sc.replace("@o2048_", "@o512_") if ref in RCT_ONLY else sc
            ok = [s for s in seeds if (g, sc, s, rec) in mse and (g, rk, s, ref) in mse]
            diffs = [mse[(g, sc, s, rec)] - mse[(g, rk, s, ref)] for s in ok]
            row = {"family": fam["family"], "kind": fam["kind"], "recipe": rec, "reference": ref, "scenario": sc,
                   "transport": preset(sc) in TRANSPORT, "pairs": len(ok),
                   "recipe_failures": sum((g, sc, s, rec) not in mse for s in seeds),
                   "reference_failures": sum((g, rk, s, ref) not in mse for s in seeds),
                   "complete": len(ok) == len(seeds),
                   "mean_difference": math.fsum(diffs) / len(diffs) if diffs else None,
                   "wins_failure_as_loss": sum(1 for s in ok if mse[(g, sc, s, rec)] < mse[(g, rk, s, ref)]),
                   "wins_denominator": sum((g, rk, s, ref) in mse for s in seeds)}
            row.update(paired_t(diffs) if row["complete"] else
                       {"mcse": None, "ci95_lower": None, "ci95_upper": None, "p_two_sided": 1.0, "zero_variance": None})
            rows.append(row)
        for row, p in zip(rows, holm([r["p_two_sided"] for r in rows])):
            row["holm_p"] = p
            row["verdict"] = ("untested" if not row["complete"] else
                              "better" if p <= ALPHA and row["mean_difference"] < 0 else
                              "worse" if p <= ALPHA and row["mean_difference"] > 0 else "unresolved")
        summary[fam["family"]] = {"kind": fam["kind"], "size": len(rows), **{
            f"{grp}_{v}": sum(1 for r in rows if r["verdict"] == v and r["transport"] == (grp == "transport"))
            for grp in ("other", "transport") for v in ("better", "worse", "unresolved", "untested")}}
        rows_all += rows
    write_csv(output / "contrasts.csv", rows_all)

    lines = ["| Family | Kind | Conditions | Better | Worse | Unresolved | Untested | Transport: better / worse |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, f in summary.items():
        tot = {v: f[f"other_{v}"] + f[f"transport_{v}"] for v in ("better", "worse", "unresolved", "untested")}
        lines.append(f"| `{name}` | {f['kind']} | {f['size']} | {tot['better']} | {tot['worse']} | {tot['unresolved']} "
                     f"| {tot['untested']} | {f['transport_better']} / {f['transport_worse']} |")
    (output / "headline.md").write_text("\n".join(lines) + "\n")

    failures = Counter((k[0], k[3], r.get("stage"), r.get("error_type"))
                       for k, r in records.items() if r["status"] != "ok")
    analysis = {"protocol": "docs/V1_CORE_WAVE2.md", "requested": len(records),
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
    ap.add_argument("--wave1", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    configs = {}
    for path in a.configs:
        config = load_config(path)
        configs[validate(config)] = config
    if set(configs) != set(GROUPS):
        raise SystemExit("one configuration per declared group is required")
    records = collect(configs, a.runs)
    result = analyze(configs, records, wave1_scores(a.wave1), a.output)
    print(json.dumps({k: result[k] for k in ("requested", "coverage", "families")}, indent=2)[:6000])


if __name__ == "__main__":
    main()
