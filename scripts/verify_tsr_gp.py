"""Check the TSR study's GP against GPy, an independent implementation.

For the first seed of every preset at OBS 512 / RCT 128 (docs/TSR_STUDY.md, check 5),
fit `lmcgp_rct_only`'s GP to the raw RCT training outcomes. Then rebuild the same
coregionalized kernel in GPy and compare:

* the negative log marginal likelihood at the default and at the fitted
  hyperparameters, which checks the objective the optimizer used; and
* the posterior effect mean on the first 200 evaluation rows at the fitted
  hyperparameters.

Stage 2 of TSR runs the same GP code on residuals, so this covers both recipes'
Stage 2. GPy pins an older SciPy, so this check runs in its own environment; the
report records versions, the TSR source hash and every fitted hyperparameter vector.

Requires GPy (the `causal-icm` extra). Usage:
  python scripts/verify_tsr_gp.py --config configs/tsr_study.toml --output tsr_gp_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from obs_rct_benchmark import tsr
from obs_rct_benchmark.runner import _generate, load_config

LABEL, SEED, N_QUERY = "o512_r128", 20000, 200
MEAN_TOL, NLML_TOL = 1e-6, 1e-6  # mean: relative to outcome scale; NLML: relative to |NLML| + 1


def gpy_model(gp: tsr.LMCGP, theta: np.ndarray):
    import GPy

    d = gp.x.shape[1]
    parts, log_noise = tsr._unpack(theta, d)
    kernel = None
    for log_ls, w, log_kap in parts:
        rbf = GPy.kern.RBF(d, variance=1.0, lengthscale=np.exp(log_ls), ARD=True, active_dims=list(range(d)))
        coreg = GPy.kern.Coregionalize(1, output_dim=2, rank=1, active_dims=[d])
        coreg.W[:] = np.asarray(w).reshape(2, 1)
        coreg.kappa[:] = np.exp(log_kap)
        term = rbf * coreg
        kernel = term if kernel is None else kernel + term
    return GPy.models.GPRegression(np.column_stack([gp.x, gp.t]), gp.ys[:, None], kernel=kernel,
                                   noise_var=float(np.exp(log_noise)) + tsr._JITTER)


def gpy_tau(gp: tsr.LMCGP, query: np.ndarray) -> np.ndarray:
    model = gpy_model(gp, gp.theta)
    qs = (query - gp.x_mean) / gp.x_scale
    n = len(qs)
    mu1, _ = model.predict_noiseless(np.column_stack([qs, np.ones(n)]))
    mu0, _ = model.predict_noiseless(np.column_stack([qs, np.zeros(n)]))
    return gp.y_scale * (mu1[:, 0] - mu0[:, 0])


def ours_nlml(gp: tsr.LMCGP, theta: np.ndarray) -> float:
    sqd = [(gp.x[:, j][:, None] - gp.x[:, j][None, :]) ** 2 for j in range(gp.x.shape[1])]
    return tsr.negative_log_marginal_likelihood(theta, gp.x, gp.t, gp.ys, sqd)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    config = load_config(a.config)
    rows, worst_mean, worst_nlml = [], 0.0, 0.0
    for scenario in config.scenarios:
        if scenario.label != LABEL:
            continue
        problem = _generate(scenario, SEED)
        rct = problem.training_view().rct_train
        gp = tsr.LMCGP().fit(rct.X, rct.T, rct.Y)
        query = np.asarray(problem.eval_X, float)[:N_QUERY]
        mean_err = float(np.max(np.abs(gp.tau_mean(query) - gpy_tau(gp, query))) / gp.y_scale)
        nlml = {}
        for name, theta in (("default", tsr.default_theta(gp.x.shape[1])), ("fitted", gp.theta)):
            ours, theirs = ours_nlml(gp, theta), float(-gpy_model(gp, theta).log_likelihood())
            nlml[name] = {"ours": ours, "gpy": theirs, "relative_difference": abs(ours - theirs) / (abs(theirs) + 1.0)}
        worst_mean = max(worst_mean, mean_err)
        worst_nlml = max(worst_nlml, *(v["relative_difference"] for v in nlml.values()))
        rows.append({"scenario": scenario.key, "d": int(rct.X.shape[1]), "converged": gp.converged,
                     "iterations": gp.iterations, "max_abs_mean_diff_over_outcome_scale": mean_err,
                     "nlml": nlml, "fitted_theta": gp.theta.tolist()})
    versions = {}
    for package in ("numpy", "scipy", "GPy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"check": "TSR GP vs GPy coregionalized GP", "seed": SEED, "label": LABEL, "query_rows": N_QUERY,
              "mean_tolerance": MEAN_TOL, "nlml_tolerance": NLML_TOL,
              "worst_mean": worst_mean, "worst_nlml": worst_nlml,
              "tsr_source_sha256": hashlib.sha256(Path(tsr.__file__).read_bytes()).hexdigest(),
              "versions": versions,
              "passed": worst_mean <= MEAN_TOL and worst_nlml <= NLML_TOL and len(rows) == 31,
              "presets": rows}
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("worst_mean", "worst_nlml", "passed")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
