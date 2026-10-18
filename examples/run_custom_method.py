"""Register a fixed-default custom estimator without receiving simulation truth.

Run this example on a compute server: python examples/run_custom_method.py
The plug-in boundary is an information contract, not a sandbox for untrusted
code. A custom method must not inspect evaluator files or reach outside it.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from obs_rct_benchmark.data import TrainingView
from obs_rct_benchmark.methods import register_method
from obs_rct_benchmark.runner import load_config, run_benchmark


class RCTConstantEffect:
    """Difference in observed RCT arm means, constant across covariates."""

    def fit_predict(self, training: TrainingView, eval_X: np.ndarray, *,
                    seed: int) -> np.ndarray:
        sample = training.rct_train
        if any(np.count_nonzero(sample.T == arm) < 2 for arm in (0, 1)):
            raise ValueError("each RCT arm needs at least two observations")
        effect = sample.Y[sample.T == 1].mean() - sample.Y[sample.T == 0].mean()
        return np.full(eval_X.shape[0], effect, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/custom-method-example")
    args = parser.parse_args()
    register_method("custom_rct_constant", RCTConstantEffect,
                    provenance="public-example:RCT-arm-mean-difference:v1")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "quickstart.toml")
    config = replace(config, methods=("rct_t_ridge", "custom_rct_constant"))
    summary = run_benchmark(config, args.output)
    raise SystemExit(0 if summary["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
