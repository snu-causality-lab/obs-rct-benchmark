"""Arm B: benchmark CorNet DGP translation and adapter on the Figure 3 grid for one mechanism seed.

For each sample seed and cell, one problem is generated and fitted twice, with and without the
adapter's OBS-based standardization, so the two fits are paired. Failures are recorded, not retried.

Usage: python benchmark_grid.py --code SRC_DIR --cornet DIR --mechanism-seed MS --out DIR
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

N_OBS = (25, 77, 130, 183, 236, 288, 341, 394, 447, 500)
N_RCT = (25, 50, 100)
BRIDGE = ((512, 128), (512, 512))
SAMPLE_SEEDS = tuple(range(10))
N_EVAL = 10_000

parser = argparse.ArgumentParser()
parser.add_argument("--code", required=True)
parser.add_argument("--cornet", required=True)
parser.add_argument("--mechanism-seed", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.environ["CORNET_SOURCE"] = str(Path(args.cornet).resolve())
sys.path.insert(0, str(Path(args.code).resolve()))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from obs_rct_benchmark.cornet import OfficialCorNet, cornet_provenance  # noqa: E402
from obs_rct_benchmark.paper_dgp import generate_paper_problem, make_paper_config  # noqa: E402

provenance = cornet_provenance()
if not provenance["verified"]:
    raise SystemExit(f"CorNet source not verified: {provenance}")

cells = [(n, m, "figure3") for m in N_RCT for n in N_OBS] + [(n, m, "bridge") for n, m in BRIDGE]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows, mechanism_hashes = [], set()
start = time.time()
for seed in SAMPLE_SEEDS:
    for n_obs, n_rct, block in cells:
        cfg = make_paper_config(
            name=f"cornet_fig3_o{n_obs}_r{n_rct}", paper_id="cornet_official_n_conf",
            mechanism_seed=args.mechanism_seed, n_obs=n_obs, n_rct_train=n_rct,
            n_rct_tune=4, n_rct_test=4, n_eval=N_EVAL,  # minimum role size; unused by CorNet
        )
        problem = generate_paper_problem(cfg, seed=seed)
        mechanism_hashes.add(problem.metadata["mechanism_sha256"])
        view = problem.training_view()
        for standardize in (False, True):
            row = dict(mechanism_seed=args.mechanism_seed, sample_seed=seed, n_obs=n_obs, n_rct=n_rct,
                       block=block, standardize=standardize, status="success", cate_mse="", root_pehe="",
                       seconds="", error_type="")
            t0 = time.time()
            try:
                prediction = OfficialCorNet(standardize=standardize).fit_predict(view, problem.eval_X, seed=seed)
                mse = float(np.mean((np.asarray(prediction) - problem.eval_tau) ** 2))
                if not math.isfinite(mse):
                    raise FloatingPointError("nonfinite MSE")
                row.update(cate_mse=repr(mse), root_pehe=repr(math.sqrt(mse)))
            except Exception as exc:  # recorded, never retried
                row.update(status="failure", error_type=type(exc).__name__)
                traceback.print_exc()
            row["seconds"] = f"{time.time() - t0:.3f}"
            rows.append(row)
    print(f"sample seed {seed} done after {time.time() - start:.0f}s", flush=True)

with open(out / "fits.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
record = {
    "arm": "B_benchmark",
    "mechanism_seed": args.mechanism_seed,
    "mechanism_sha256": sorted(mechanism_hashes),
    "n_fits": len(rows),
    "n_failures": sum(r["status"] != "success" for r in rows),
    "elapsed_seconds": time.time() - start,
    "cornet_provenance": provenance,
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")},
}
(out / "fits.json").write_text(json.dumps(record, indent=2))
print(json.dumps({k: record[k] for k in ("mechanism_seed", "n_fits", "n_failures", "elapsed_seconds")}))
