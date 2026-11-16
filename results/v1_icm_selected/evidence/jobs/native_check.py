"""Real-GPy check of causal_icm_selected: timing, the fit record, and that the
final fit equals the fixed recipe at the selected rho with the same seed."""
import json
import sys
import time

import numpy as np

from obs_rct_benchmark.causal_icm import OfficialCausalICM, SelectedRhoCausalICM
from obs_rct_benchmark.runner import _generate, load_config, method_seed

config = load_config(sys.argv[1])
for scenario in config.scenarios:
    for seed in config.seeds:
        problem = _generate(scenario, seed)
        training = problem.training_view()
        mseed = method_seed(scenario.key, seed, "causal_icm_selected")
        assert mseed == method_seed(scenario.key, seed, "causal_icm_fixed")
        selected = SelectedRhoCausalICM()
        t0 = time.time()
        pred = selected.fit_predict(training, problem.eval_X, seed=mseed)
        t_sel = time.time() - t0
        rho = selected.fit_info_["selected_rho"]
        t0 = time.time()
        again = OfficialCausalICM(rho=rho).fit_predict(training, problem.eval_X, seed=mseed)
        t_fix = time.time() - t0
        fixed = OfficialCausalICM(rho=0.5).fit_predict(training, problem.eval_X, seed=mseed)
        mse = lambda p: float(np.mean((p - problem.eval_tau) ** 2))
        print(json.dumps({
            "scenario": scenario.key, "seed": seed, "selected_rho": rho,
            "identical_to_fixed_at_selected_rho": bool(np.array_equal(pred, again)),
            "mse_selected": mse(pred), "mse_fixed_0.5": mse(fixed),
            "seconds_selected": round(t_sel, 1), "seconds_one_fixed_fit": round(t_fix, 1),
            "weights": selected.fit_info_["participation_weights"],
            "scores": selected.fit_info_["scores"], "folds_used": selected.fit_info_["folds_used"],
        }), flush=True)
