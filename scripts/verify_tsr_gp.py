"""Check the TSR study's GP against GPy, an independent implementation.

For the first seed of every preset at OBS 512 / RCT 128 (docs/TSR_STUDY.md, check 5),
fit `lmcgp_rct_only`'s GP to the raw RCT training outcomes, rebuild the same
coregionalized kernel in GPy at the fitted hyperparameters, and compare posterior
effect means on the first 200 evaluation rows. Stage 2 of TSR runs the same GP code
on residuals, so this covers both recipes' Stage 2.

Requires GPy (the `causal-icm` extra). Usage:
  python scripts/verify_tsr_gp.py --config configs/tsr_study.toml --output tsr_gp_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from obs_rct_benchmark import tsr
from obs_rct_benchmark.runner import _generate, load_config

LABEL, SEED, N_QUERY, REL_TOL = "o512_r128", 20000, 200, 1e-6


def gpy_tau(gp: tsr.LMCGP, query: np.ndarray) -> np.ndarray:
    import GPy

    d = gp.x.shape[1]
    parts, log_noise = tsr._unpack(gp.theta, d)
    kernel = None
    for log_ls, w, log_kap in parts:
        rbf = GPy.kern.RBF(d, variance=1.0, lengthscale=np.exp(log_ls), ARD=True, active_dims=list(range(d)))
        coreg = GPy.kern.Coregionalize(1, output_dim=2, rank=1, active_dims=[d])
        coreg.W[:] = np.asarray(w).reshape(2, 1)
        coreg.kappa[:] = np.exp(log_kap)
        term = rbf * coreg
        kernel = term if kernel is None else kernel + term
    X = np.column_stack([gp.x, gp.t])
    model = GPy.models.GPRegression(X, gp.ys[:, None], kernel=kernel,
                                    noise_var=float(np.exp(log_noise)) + tsr._JITTER)
    qs = (query - gp.x_mean) / gp.x_scale
    n = len(qs)
    mu1, _ = model.predict_noiseless(np.column_stack([qs, np.ones(n)]))
    mu0, _ = model.predict_noiseless(np.column_stack([qs, np.zeros(n)]))
    return gp.y_scale * (mu1[:, 0] - mu0[:, 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    config = load_config(a.config)
    rows, worst = [], 0.0
    for scenario in config.scenarios:
        if scenario.label != LABEL:
            continue
        problem = _generate(scenario, SEED)
        rct = problem.training_view().rct_train
        gp = tsr.LMCGP().fit(rct.X, rct.T, rct.Y)
        query = np.asarray(problem.eval_X, float)[:N_QUERY]
        ours, theirs = gp.tau_mean(query), gpy_tau(gp, query)
        err = float(np.max(np.abs(ours - theirs)) / gp.y_scale)
        worst = max(worst, err)
        rows.append({"scenario": scenario.key, "d": int(rct.X.shape[1]), "converged": gp.converged,
                     "max_abs_diff_over_outcome_scale": err})
    report = {"check": "TSR GP vs GPy coregionalized GP at fitted hyperparameters", "seed": SEED, "label": LABEL,
              "query_rows": N_QUERY, "tolerance": REL_TOL, "worst": worst, "passed": worst <= REL_TOL and len(rows) == 31,
              "presets": rows}
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("worst", "passed")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
