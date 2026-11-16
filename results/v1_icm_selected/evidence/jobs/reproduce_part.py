"""Run the frozen checker's reproducibility check in parts, one condition per job, and merge them.

The check (scripts/verify_icm_selected.py `reproduce`) refits every recipe twice, one condition
after another, in one process. At the stress block's 8,192-row level the two selection refits
alone take most of the 12-hour partition limit, so one job cannot hold the whole check. Each part
calls the same frozen function on one condition; records are authenticated against the whole
block, exactly as in one run. `merge` rebuilds the single-run output from the parts.

  reproduce_part.py run --code C --config CFG --scenario KEY --runs R... --output part.json [--job-limit-shards J]
  reproduce_part.py merge --code C --config CFG --parts P... --output checks.json
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys


def checker(code: Path):
    sys.path.insert(0, str(code / "scripts"))
    spec = importlib.util.spec_from_file_location("verify_icm_selected", code / "scripts" / "verify_icm_selected.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected(config) -> list[str]:
    """The conditions the frozen check replays, in its order (its own filter, restated)."""
    stress = config.scenarios[0].kind == "stress"
    return [s.key for s in config.scenarios if s.parameters["n_rct_train"] == 128
            and (s.parameters["level"] in (0, 3) if stress else s.parameters["n_obs"] == 512)]


def run(a, V, full):
    if a.scenario not in selected(full):
        raise SystemExit(f"{a.scenario} is not replayed by the check")
    limited = frozenset(e["config_hash"] for e in json.loads(a.job_limit_shards.read_text())) \
        if a.job_limit_shards else frozenset()
    whole = V.load_records
    V.load_records = lambda config, runs, job_limit=frozenset(): whole(full, runs, job_limit)
    part = replace(full, scenarios=tuple(s for s in full.scenarios if s.key == a.scenario))
    result = V.reproduce(part, a.runs, limited)
    a.output.write_text(json.dumps({"scenario": a.scenario, "result": result}, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, default=str)[:2000])
    raise SystemExit(int(bool(result["mismatches"])))


def merge(a, full):
    parts = {}
    for path in a.parts:
        p = json.loads(path.read_text())
        if p["scenario"] in parts:
            raise SystemExit(f"{p['scenario']} appears in two parts")
        parts[p["scenario"]] = p["result"]
    want = selected(full)
    if sorted(parts) != sorted(want):
        raise SystemExit(f"parts cover {sorted(parts)}, the check replays {want}")
    first = parts[want[0]]
    if any(parts[k]["record_problems"] != first["record_problems"] or parts[k]["seed"] != first["seed"] for k in want):
        raise SystemExit("parts disagree on the authenticated records")
    checks = [c for k in want for c in parts[k]["details"]]
    unavailable = [u for k in want for u in parts[k]["unavailable"]]
    mismatches = [[c["scenario"], c["method"]] for c in checks if not c["match"]]
    problems = first["record_problems"]
    result = {"check": "reproduce", "seed": first["seed"], "cases": len(checks), "refits": 2 * len(checks),
              "unavailable": unavailable, "record_problems": problems,
              "mismatches": mismatches + problems + [{"unavailable": u} for u in unavailable], "details": checks,
              "parts": len(want)}
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("run", "merge"))
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--scenario")
    ap.add_argument("--runs", type=Path, nargs="+")
    ap.add_argument("--parts", type=Path, nargs="+")
    ap.add_argument("--job-limit-shards", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    V = checker(a.code)
    from obs_rct_benchmark.runner import load_config
    full = load_config(a.config)
    if a.step == "run":
        if not (a.scenario and a.runs):
            ap.error("run needs --scenario and --runs")
        run(a, V, full)
    else:
        if not a.parts:
            ap.error("merge needs --parts")
        merge(a, full)


if __name__ == "__main__":
    main()
