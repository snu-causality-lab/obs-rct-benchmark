"""Exercise both file adapters on invented records, without external datasets.

These are schema/integration fixtures, not IHDP or ACIC performance results.
Run from an installed checkout: python examples/run_synthetic_file_smoke.py --output outputs/file-fixtures
"""

import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, run_benchmark


def run(output: Path) -> None:
    controls = dict(n_obs=120, n_rct_train=60, n_rct_tune=30,
                    n_rct_test=30, n_eval=128, noise_std=0.2, gamma=3.0)
    config = BenchmarkConfig(
        scenarios=(
            ScenarioConfig("empirical", "ihdp", controls | {
                "hidden_columns": tuple(f"x{i}" for i in range(2, 26)),
                "visible_binning": (("x1", 2),),
            }),
            ScenarioConfig("empirical", "acic2016", controls | {
                "hidden_columns": ("x2", "category"),
            }),
        ),
        methods=("rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge", "rct_dr_ridge"),
        scope="synthetic file-format fixtures only; no IHDP/ACIC data or empirical performance claim",
    )
    variables = ("OBSRCT_IHDP_TRAIN", "OBSRCT_IHDP_TEST", "OBSRCT_ACIC_X", "OBSRCT_ACIC_RESPONSE")
    previous = {name: os.environ.get(name) for name in variables}
    try:
        with tempfile.TemporaryDirectory(prefix="obsrct-file-fixtures-") as directory:
            fixture = Path(directory)
            visible = np.repeat([0.0, 1.0], 32)
            hidden = np.tile(np.repeat([0.0, 1.0], 16), 2)
            X = np.zeros((64, 25))
            X[:, 0], X[:, 1] = visible, hidden
            mu0 = 2 * visible + hidden
            mu1 = mu0 + 1 + visible + 2 * hidden
            np.savez(fixture / "fixture.npz", x=X, t=np.zeros(64),
                     yf=np.full(64, 9999.0), ycf=np.full(64, -9999.0), mu0=mu0, mu1=mu1)
            with (fixture / "x.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x1", "x2", "category"])
                writer.writerows(zip(visible, hidden, np.where(hidden == 0, "A", "B")))
            with (fixture / "response.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["z", "y0", "y1", "mu0", "mu1"])
                writer.writerows(zip(np.zeros(64), np.full(64, 9999.0),
                                     np.full(64, -9999.0), mu0, mu1))
            os.environ["OBSRCT_IHDP_TRAIN"] = str(fixture / "fixture.npz")
            os.environ.pop("OBSRCT_IHDP_TEST", None)
            os.environ["OBSRCT_ACIC_X"] = str(fixture / "x.csv")
            os.environ["OBSRCT_ACIC_RESPONSE"] = str(fixture / "response.csv")
            result = run_benchmark(config, output)
            if result["status"] != "complete":
                raise RuntimeError("synthetic file-format smoke did not complete")
            before = (output / "records.jsonl").read_bytes()
            run_benchmark(config, output, resume=True)
            if before != (output / "records.jsonl").read_bytes():
                raise RuntimeError("completed records changed on resume")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
