"""Arm B: benchmark DGP translation and Integrative R adapter at the paper's settings.

One problem per seed is generated and fitted once with the quadratic recipe, using the
native panel's options. Failures are recorded, never retried. INTEGRATIVE_R_SOURCE and
INTEGRATIVE_R_RSCRIPT must be set by the caller.

Usage: python benchmark_grid.py --code SRC_DIR --cell CELL --seed-from S --seed-to S --out DIR
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

KPS = "integrative_r_learner_kps_code"
MULTI = "integrative_r_learner_multivariate_paper"
CELLS = {
    "t1_n200": dict(paper_id=KPS, n_obs=2000, n_rct_train=200),
    "t1_n500": dict(paper_id=KPS, n_obs=2000, n_rct_train=500),
    "t1_n1000": dict(paper_id=KPS, n_obs=2000, n_rct_train=1000),
    "t2_J1_b0": dict(paper_id=MULTI, n_obs=500, n_rct_train=244,
                     int_r_zero_cate_coefficients=8, int_r_confounding_strength=0.0),
    "t2_J1_b25": dict(paper_id=MULTI, n_obs=500, n_rct_train=244,
                      int_r_zero_cate_coefficients=8, int_r_confounding_strength=2.5),
    "t2_J9_b0": dict(paper_id=MULTI, n_obs=500, n_rct_train=244,
                     int_r_zero_cate_coefficients=0, int_r_confounding_strength=0.0),
    "t2_J9_b25": dict(paper_id=MULTI, n_obs=500, n_rct_train=244,
                      int_r_zero_cate_coefficients=0, int_r_confounding_strength=2.5),
    "bridge_r128": dict(paper_id=KPS, n_obs=512, n_rct_train=128),
    "bridge_r512": dict(paper_id=KPS, n_obs=512, n_rct_train=512),
}
N_EVAL = 100_000
METHOD_OPTIONS = dict(k_folds=5, timeout_seconds=1200)  # native-panel options

parser = argparse.ArgumentParser()
parser.add_argument("--code", required=True)
parser.add_argument("--cell", required=True, choices=sorted(CELLS))
parser.add_argument("--seed-from", type=int, required=True)
parser.add_argument("--seed-to", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

code = Path(args.code).resolve()
sys.path.insert(0, str(code))
import numpy as np  # noqa: E402
import obs_rct_benchmark  # noqa: E402
from obs_rct_benchmark.integrative_r import integrative_r_provenance  # noqa: E402
from obs_rct_benchmark.methods import make_method  # noqa: E402
from obs_rct_benchmark.paper_dgp import generate_paper_problem, make_paper_config  # noqa: E402

if not Path(obs_rct_benchmark.__file__).resolve().is_relative_to(code):
    raise SystemExit(f"benchmark imported from {obs_rct_benchmark.__file__}, not the snapshot")
provenance = integrative_r_provenance()
if not provenance["verified"] or provenance["runtime"].get("availability") != "available":
    raise SystemExit(f"Integrative R source or runtime not verified: {provenance}")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fits_csv = out / "fits.csv"
if fits_csv.exists():
    raise SystemExit("fits.csv already exists; use a new directory")
columns = ["cell", "seed", "status", "n_obs", "n_rct", "cate_mse", "root_pehe", "seconds",
           "native_seed", "error_type", "message"]
with fits_csv.open("w", newline="") as fh:
    csv.writer(fh).writerow(columns)

spec = CELLS[args.cell]
mechanism_hashes, counts, start = set(), {"success": 0, "failure": 0}, time.time()
for seed in range(args.seed_from, args.seed_to + 1):
    row = dict(cell=args.cell, seed=seed, status="success", n_obs=spec["n_obs"], n_rct=spec["n_rct_train"],
               cate_mse="", root_pehe="", seconds="", native_seed="", error_type="", message="")
    t0 = time.time()
    method = None
    try:
        cfg = make_paper_config(name=f"intr_replay_{args.cell}", n_rct_tune=4, n_rct_test=4,
                                n_eval=N_EVAL, **spec)
        problem = generate_paper_problem(cfg, seed=seed)
        mechanism_hashes.add(problem.metadata["mechanism_sha256"])
        method = make_method("integrative_r_quadratic", **METHOD_OPTIONS)
        prediction = method.fit_predict(problem.training_view(), problem.eval_X, seed=seed)
        mse = float(np.mean((np.asarray(prediction) - problem.eval_tau) ** 2))
        if not math.isfinite(mse):
            raise FloatingPointError("nonfinite MSE")
        row.update(cate_mse=repr(mse), root_pehe=repr(math.sqrt(mse)),
                   native_seed=method.provenance_["native_seed"])
    except Exception as exc:  # recorded, never retried
        row.update(status="failure", error_type=type(exc).__name__, message=str(exc)[:300].replace("\n", " "))
        traceback.print_exc()
        diagnostics = getattr(method, "last_run_diagnostics_", None) or {}
        for name, text in diagnostics.items():
            (out / f"seed-{seed:03d}-{name}.log").write_text(text)
    row["seconds"] = f"{time.time() - t0:.3f}"
    counts[row["status"]] += 1
    with fits_csv.open("a", newline="") as fh:
        csv.writer(fh).writerow([row[c] for c in columns])
    print(f"cell={args.cell} seed={seed} status={row['status']} seconds={row['seconds']}", flush=True)

record = {
    "arm": "B_benchmark",
    "cell": args.cell,
    "spec": spec,
    "n_eval": N_EVAL,
    "method": "integrative_r_quadratic",
    "method_options": METHOD_OPTIONS,
    "seeds": [args.seed_from, args.seed_to],
    "mechanism_sha256": sorted(mechanism_hashes),
    "counts": counts,
    "elapsed_seconds": time.time() - start,
    "benchmark_module": str(Path(obs_rct_benchmark.__file__).resolve()),
    "integrative_r_provenance": provenance,
    "versions": {"python": platform.python_version(), "numpy": np.__version__},
    "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")},
}
(out / "fits.json").write_text(json.dumps(record, indent=2, default=str))
print(json.dumps({k: record[k] for k in ("cell", "seeds", "counts", "elapsed_seconds")}))
