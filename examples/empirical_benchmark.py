"""Use locally supplied IHDP/ACIC files without exporting rows or file paths.

Run on a compute server with the benchmark and ``baselines`` extra installed:

    python examples/empirical_benchmark.py --dataset ihdp-npz \
        --train "$OBSRCT_IHDP_TRAIN" --test "$OBSRCT_IHDP_TEST" \
        --keep x1 --bin x1=2 --gamma 2

This illustrative design hides every source column except x1 and displays its
two empirical quantile bins. Its CATE is the exact within-bin mean treatment
effect under the uniform loaded-row distribution. Choose retained columns and
coarsening for the scientific question; masking a few columns while retaining
unique continuous rows alone creates no residual ambiguity in this distribution.

All OBS/RCT roles draw rows independently with replacement. Outcomes are newly
generated from the selected arm means; historical factual outcomes are ignored.
The output contains aggregate metrics and sanitized mechanism metadata only.
"""

import argparse
import json

from obs_rct_benchmark.datasets import (
    EmpiricalDGPConfig,
    generate_empirical_problem,
    load_acic2016,
    load_ihdp_csv,
    load_ihdp_npz,
)
from obs_rct_benchmark.evaluation import cate_metrics
from obs_rct_benchmark.methods import make_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ihdp-npz", "ihdp-csv", "acic2016"), required=True)
    parser.add_argument("--train", help="IHDP NPZ training file, or CEVAE CSV file")
    parser.add_argument("--test", help="optional IHDP NPZ test file to pool")
    parser.add_argument("--x", help="ACIC covariate CSV")
    parser.add_argument("--response", help="ACIC response CSV")
    parser.add_argument("--replication", type=int, default=0, help="zero-based IHDP NPZ replication")
    parser.add_argument("--keep", nargs="+", required=True, help="source column names visible to the learner")
    parser.add_argument("--bin", action="append", default=[], metavar="COLUMN=BINS",
                        help="explicit fixed-population quantile coarsening; may repeat")
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--outcome-model", choices=("loaded_means", "synthetic"), default="loaded_means")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.dataset == "ihdp-npz":
        if not args.train:
            parser.error("ihdp-npz requires --train")
        population = load_ihdp_npz(args.train, args.test, replication=args.replication)
    elif args.dataset == "ihdp-csv":
        if not args.train:
            parser.error("ihdp-csv requires --train")
        population = load_ihdp_csv(args.train)
    else:
        if not args.x or not args.response:
            parser.error("acic2016 requires --x and --response")
        population = load_acic2016(args.x, args.response)
    source_columns = tuple(dict.fromkeys(population.feature_groups))
    if set(args.keep) - set(source_columns):
        parser.error("--keep contains source column names absent from the loaded file")
    try:
        binning = tuple((name, int(bins)) for name, bins in (value.rsplit("=", 1) for value in args.bin))
    except ValueError:
        parser.error("--bin must have the form COLUMN=POSITIVE_INTEGER")
    cfg = EmpiricalDGPConfig(
        name="local_empirical_example",
        dataset_id="acic2016" if args.dataset == "acic2016" else "ihdp",
        replication=args.replication,
        hidden_columns=tuple(name for name in source_columns if name not in args.keep),
        visible_binning=binning,
        gamma=args.gamma,
        outcome_model=args.outcome_model,
    )
    problem = generate_empirical_problem(population, cfg, args.seed)
    method = make_method("rct_t_ridge")
    prediction = method.fit_predict(problem.training_view(), problem.eval_X, seed=args.seed)
    print(json.dumps({"metrics": cate_metrics(prediction, problem.eval_tau),
                      "metadata": dict(problem.metadata)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
