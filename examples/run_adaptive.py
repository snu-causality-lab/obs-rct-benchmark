"""Run a tiny outcome-budgeted example; use the reference config for a study.

Run from the repository root with the package and baselines extra installed:
python examples/run_adaptive.py --output outputs/adaptive-example
"""

import argparse

from obs_rct_benchmark.adaptive_runner import AdaptiveConfig, run_adaptive_benchmark
from obs_rct_benchmark.runner import ScenarioConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = AdaptiveConfig(
        scenarios=(ScenarioConfig("structural", "M3", {"n_obs": 120, "n_eval": 128}),),
        seeds=(7, 11), budgets=(48, 80), pool_size=160,
        pilot_size=24, tune_size=8,
        estimators=("rct_t_ridge", "naive_pooled_t_ridge"),
        scope="Tiny sequential recruitment example; not ranking evidence",
    )
    result = run_adaptive_benchmark(config, args.output, resume=args.resume)
    print(result["status"], result["coverage"])
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
