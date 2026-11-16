"""Run one preflight shard (seed 69999, outside the grid) with records kept, pass them
through the addendum's collector, rho check and the checker's record loader, print
counts only, then delete the records. Usage: python pipeline_check.py <code dir> <scratch dir>"""
import json, shutil, sys
from collections import Counter
from pathlib import Path

from obs_rct_benchmark.runner import load_config, run_benchmark
import analyze_icm_selected as A
import make_icm_selected_configs as M
import verify_icm_selected as V

code, scratch = Path(sys.argv[1]), Path(sys.argv[2])
config = load_config(code / "configs/v1_icm_selected_core.toml")
name, shard = next((n, s) for n, s in M.shards(config, preflight=True) if n.endswith("structural__M0.toml"))
run = scratch / Path(name).stem
run_benchmark(shard, run)
manifest = json.loads((run / "run_manifest.json").read_text())
A.declared_shards = lambda c, preflight=False: iter([(name, shard)])  # one shard stands for the block
records = A.collect(config, [run])
rho = {k: r["fit_info"]["selected_rho"] for k, r in records.items() if k[3] == A.SELECTED and r["status"] == "ok"}
V.partition = lambda c: {shard.config_hash: shard}  # one shard stands for the block
loaded, _, problems = V.load_records(config, [run])
fixed_seed = {k[:3]: r["method_seed"] for k, r in records.items() if k[3] == "causal_icm_fixed"}
print(json.dumps({
    "cells": len(records), "statuses": Counter(f"{k[3]}:{r['status']}" for k, r in records.items()),
    "selected_with_grid_rho": len(rho), "checker_loads_same_cells": set(loaded) == set(records), "checker_record_problems": problems,
    "selected_shares_fixed_seed": all(r["method_seed"] == fixed_seed[k[:3]] for k, r in records.items() if k[3] == A.SELECTED),
    "fit_info_keys": sorted(next(iter(records[k] for k in rho))["fit_info"]),
    "external_methods": sorted((manifest.get("external_methods") or {})),
}, indent=1, default=str))
shutil.rmtree(run)
