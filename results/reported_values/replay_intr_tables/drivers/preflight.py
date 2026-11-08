"""Preflight: sources, runtimes, and one replicate of each simulation in each arm.

It also runs the README example at seed 0 for readout R0. Seeds other than 0 lie outside
the replay seeds. Outputs are execution checks only, not replay results.

Usage: python preflight.py --code SRC_DIR --driver AUTHOR_REPLAY_R --out DIR
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--code", required=True)
parser.add_argument("--driver", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
code = Path(args.code).resolve()
rscript = os.environ["INTEGRATIVE_R_RSCRIPT"]
source = os.environ["INTEGRATIVE_R_SOURCE"]
checks = {"node": os.environ.get("SLURMD_NODENAME"), "python": platform.python_version()}

sys.path.insert(0, str(code))
import numpy as np  # noqa: E402
import obs_rct_benchmark  # noqa: E402
from obs_rct_benchmark.integrative_r import integrative_r_provenance  # noqa: E402
from obs_rct_benchmark.methods import make_method  # noqa: E402
from obs_rct_benchmark.paper_dgp import generate_paper_problem, make_paper_config  # noqa: E402

checks["benchmark_module"] = str(Path(obs_rct_benchmark.__file__).resolve())
checks["benchmark_from_snapshot"] = Path(obs_rct_benchmark.__file__).resolve().is_relative_to(code)
provenance = integrative_r_provenance()
checks["intr_source_verified"] = bool(provenance["verified"])
checks["intr_runtime"] = provenance["runtime"]
probe = subprocess.run([rscript, "--vanilla", "-e", 'cat(as.character(packageVersion("mvtnorm")))'],
                       capture_output=True, text=True, timeout=300)
checks["mvtnorm"] = probe.stdout.strip() if probe.returncode == 0 else None

arm_a = {}
for cell, seed in (("readme_J8_b25", 0), ("t1_n1000", 100000)):
    target = out / f"author-{cell}"
    t0 = time.time()
    proc = subprocess.run([rscript, "--vanilla", args.driver, source, str(target), cell, str(seed), str(seed)],
                          capture_output=True, text=True, timeout=1500)
    rows = list(csv.DictReader((target / "replicates.csv").open())) if (target / "replicates.csv").exists() else []
    arm_a[cell] = {"returncode": proc.returncode, "seconds": round(time.time() - t0, 1), "rows": rows,
                   "stderr_tail": proc.stderr[-1500:]}
checks["arm_a"] = arm_a

arm_b = {}
specs = {
    "t1_n1000": dict(paper_id="integrative_r_learner_kps_code", n_obs=2000, n_rct_train=1000),
    "t2_J9_b25": dict(paper_id="integrative_r_learner_multivariate_paper", n_obs=500, n_rct_train=244,
                      int_r_zero_cate_coefficients=0, int_r_confounding_strength=2.5),
}
for cell, spec in specs.items():
    cfg = make_paper_config(name=f"preflight_{cell}", n_rct_tune=4, n_rct_test=4, n_eval=1000, **spec)
    problem = generate_paper_problem(cfg, seed=100000)
    method = make_method("integrative_r_quadratic", k_folds=5, timeout_seconds=1200)
    t0 = time.time()
    try:
        prediction = method.fit_predict(problem.training_view(), problem.eval_X, seed=100000)
        value = float(np.sqrt(np.mean((np.asarray(prediction) - problem.eval_tau) ** 2)))
        arm_b[cell] = {"ok": bool(np.isfinite(value)), "root_pehe": value}
    except Exception as exc:  # reported, not retried
        diagnostics = getattr(method, "last_run_diagnostics_", None) or {}
        arm_b[cell] = {"ok": False, "error": repr(exc), "stderr_tail": diagnostics.get("stderr", "")[-1500:]}
    arm_b[cell]["seconds"] = round(time.time() - t0, 1)
checks["arm_b"] = arm_b

ok = (checks["benchmark_from_snapshot"] and checks["intr_source_verified"]
      and checks["intr_runtime"].get("availability") == "available" and checks["mvtnorm"]
      and all(v["returncode"] == 0 and len(v["rows"]) == 1 and v["rows"][0]["status"] == "success"
              for v in arm_a.values())
      and all(v["ok"] for v in arm_b.values()))
checks["ok"] = bool(ok)
(out / "preflight.json").write_text(json.dumps(checks, indent=2, default=str))
print(json.dumps(checks, indent=2, default=str))
sys.exit(0 if ok else 1)
