"""Re-run integrative_r_quadratic on the CorNet paper design at one wave-2 seed to capture
why it fails at three of four budgets (wave 2: 192 failures, 64/64 seeds at o2048_r128,
o512_r512, o2048_r512; none at o512_r128). Reports error text, exception chain, the R
process's stderr tail and wall time, never metrics. Usage: python intr_trace.py RUN_DIR SEED
"""
import sys
import time
import traceback
from pathlib import Path

from obs_rct_benchmark.runner import _generate, load_config, method_seed
from obs_rct_benchmark.methods import make_method

run, seed = Path(sys.argv[1]), int(sys.argv[2])
config = load_config(run / "main-native_r" / f"native_r-seed-{seed}-paper__cornet_official_n_conf.toml")
for method in ("integrative_r_quadratic", "integrative_r_linear"):
    if method not in config.methods:
        print(f"{method}: not in shard"); continue
    for scenario in config.scenarios:
        problem = _generate(scenario, seed)
        training = problem.training_view()
        estimator = make_method(method, **dict(config.method_options.get(method, {})))
        rct, obs = training.rct_train, training.obs_train
        info = f"n_rct={rct.n} n_obs={obs.n} d={rct.X.shape[1]} n_eval={len(problem.eval_X)}"
        start = time.monotonic()
        try:
            estimator.fit_predict(training, problem.eval_X, seed=method_seed(scenario.key, seed, method))
            print(f"{method} {scenario.key}: ok in {time.monotonic() - start:.0f}s ({info})")
        except Exception as exc:
            chain, e = [], exc
            while e is not None:
                chain.append(f"{type(e).__name__}: {e}"); e = e.__cause__
            print(f"{method} {scenario.key}: FAILED after {time.monotonic() - start:.0f}s ({info})")
            print("  chain: " + " <- ".join(chain))
            prov = getattr(estimator, "last_run_provenance_", {})
            print(f"  status={prov.get('status')} returncode={prov.get('returncode')}")
            diag = getattr(estimator, "last_run_diagnostics_", {})
            for name in ("stderr", "stdout"):
                tail = diag.get(name, "").strip().splitlines()[-15:]
                if tail: print(f"  {name} tail:\n    " + "\n    ".join(tail))
            print("  " + traceback.format_exc().strip().splitlines()[-1])
        sys.stdout.flush()
