#!/usr/bin/env python3
"""Fixed descriptive risks and a complete, one-sided strict-win family.

The binomial endpoint concerns finite-query comparisons across seed blocks.
It never supplies an uncertainty statement for mean fitted-method risk.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

FUSION = ("cornet_official", "datafusionppi", "integrative_r_linear",
          "integrative_r_quadratic", "causal_icm_fixed")
ALPHA = .05
WIN_HALF_WIDTH = .25
N_SEEDS = 64
FAMILY_SIZE = 72


def contrasts(config):
    """Enumerate the entire prospective family, without reading outcomes."""
    return [dict(scenario_key=condition["scenario_key"], n_rct=budget,
                 method=method, reference=reference)
            for condition in config["conditions"]
            for budget in config["rct_budgets"]
            for method, reference in (
                *((m, "rct_causal_forest") for m in FUSION),
                ("causal_icm_fixed", "causal_icm_rct_only"))]


def analysis_contract(config):
    return dict(comparisons=contrasts(config), family_size=FAMILY_SIZE, alpha=ALPHA,
                seed_blocks=N_SEEDS, primary_metrics=["cate_mse", "root_pehe"],
                mean_risk_inference_eligible=False,
                secondary_endpoint="strict finite-query MSE win; ties are nonwins",
                null="strict-win probability <= 0.5", alternative="greater",
                multiplicity="Holm over all 72 comparisons; incomplete family withheld",
                pointwise_interval="two-sided 95% Clopper-Pearson",
                target_absolute_tolerance=1e-12, target_relative_tolerance=0.,
                score_absolute_tolerance=1e-12, score_relative_tolerance=1e-12,
                simultaneous_win_probability_half_width=WIN_HALF_WIDTH,
                adaptive_seed_extension=False)


def holm(pvalues):
    """Adjusted p-values, preserving input order and the whole family size."""
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    adjusted = [None] * len(pvalues)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def summarize(records, config):
    """Require one terminal record per requested fit; never shrink families.

    Authentication and independent rescoring precede this pure aggregation.
    Caller supplies records with case, method, status and metrics dictionaries.
    """
    from scipy.stats import binomtest

    family = contrasts(config)
    seeds = config["seeds"]
    if len(seeds) != N_SEEDS or len(family) != FAMILY_SIZE:
        raise ValueError("this analysis requires the frozen 64-seed/72-contrast design")
    expected = {(c["scenario_key"], b, s, m)
                for c in config["conditions"] for b in config["rct_budgets"]
                for s in seeds for m in config["methods"]}
    indexed = {}
    for record in records:
        case = record["case"]
        key = (case["scenario_key"], case["n_rct"], case["seed"], record["method"])
        if key not in expected or key in indexed:
            raise ValueError("extra or duplicate terminal record")
        if record["status"] not in {"success", "failure", "timeout"}:
            raise ValueError("nonterminal or unsupported record status")
        if record["status"] == "success":
            for metric in ("cate_mse", "root_pehe"):
                value = record["metrics"][metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid successful score")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("missing terminal records: collection is not complete")

    means = []
    for condition in config["conditions"]:
        for budget in config["rct_budgets"]:
            for method in config["methods"]:
                rows = [indexed[(condition["scenario_key"], budget, seed, method)] for seed in seeds]
                successful = [r for r in rows if r["status"] == "success"]
                means.append(dict(
                    scenario_key=condition["scenario_key"], n_rct=budget, method=method,
                    requested=len(rows), successful=len(successful),
                    status_counts=dict(Counter(r["status"] for r in rows)),
                    conditional_on_success=len(successful) != len(rows),
                    means={metric: _mean([r["metrics"][metric] for r in successful])
                           for metric in ("cate_mse", "root_pehe")},
                    mean_risk_inference_eligible=False,
                    mean_risk_inference_reason="fitted-risk moment and mean precision conditions not established",
                ))

    paired = []
    for comparison in family:
        differences = {metric: [] for metric in ("cate_mse", "root_pehe")}
        wins = losses = ties = 0
        for seed in seeds:
            prefix = (comparison["scenario_key"], comparison["n_rct"], seed)
            left = indexed[(*prefix, comparison["method"])]
            right = indexed[(*prefix, comparison["reference"])]
            if left["status"] != "success" or right["status"] != "success":
                continue
            for metric in differences:
                differences[metric].append(left["metrics"][metric] - right["metrics"][metric])
            delta = differences["cate_mse"][-1]
            wins += delta < 0
            losses += delta > 0
            ties += delta == 0
        complete = len(differences["cate_mse"]) == N_SEEDS
        row = dict(comparison, complete_pairs=len(differences["cate_mse"]),
                   required_pairs=N_SEEDS, complete=complete, wins=wins, losses=losses, ties=ties,
                   paired_mean_differences={m: _mean(v) if complete else None for m, v in differences.items()},
                   mean_risk_inference_eligible=False, strict_win_probability=None,
                   exact_pointwise_95_interval=None, simultaneous_95_interval=None,
                   one_sided_pvalue=None, holm_pvalue=None, reject_majority_nonwin=None)
        if complete:
            result = binomtest(wins, N_SEEDS, .5, alternative="greater")
            # Use a separate two-sided call: alternative='greater' would make
            # proportion_ci a one-sided interval, contrary to this protocol.
            interval = binomtest(wins, N_SEEDS).proportion_ci(.95, method="exact")
            probability = wins / N_SEEDS
            row.update(strict_win_probability=probability,
                       exact_pointwise_95_interval=[float(interval.low), float(interval.high)],
                       simultaneous_95_interval=[max(0., probability - WIN_HALF_WIDTH),
                                                 min(1., probability + WIN_HALF_WIDTH)],
                       one_sided_pvalue=float(result.pvalue))
        paired.append(row)
    complete_family = all(row["complete"] for row in paired)
    if complete_family:
        for row, adjusted in zip(paired, holm([r["one_sided_pvalue"] for r in paired])):
            row.update(holm_pvalue=adjusted, reject_majority_nonwin=adjusted <= ALPHA)
    else:
        # Complete pointwise calculations are descriptive when any family
        # member is incomplete. No completed subset becomes a smaller family.
        for row in paired:
            row["simultaneous_95_interval"] = None
    return dict(
        schema_version=1,
        scope="inspected-mechanism development comparison; finite-query strict wins",
        expected_fits=len(expected), terminal_fits=len(records),
        status_counts=dict(Counter(r["status"] for r in records)),
        family_size=FAMILY_SIZE, family_complete=complete_family,
        family_withheld_reason=None if complete_family else "one or more required paired fits failed",
        alpha=ALPHA, test="exact binomial, one-sided greater than 0.5; Holm over all 72",
        ties="nonwins; ties do not establish harm",
        precision=dict(endpoint="finite-query strict-win probability across independent seed blocks",
                       simultaneous_half_width=WIN_HALF_WIDTH,
                       union_failure_bound=2 * FAMILY_SIZE * math.exp(-2 * N_SEEDS * WIN_HALF_WIDTH**2),
                       eligible=complete_family),
        method_summaries=means, paired_comparisons=paired,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="frozen campaign directory")
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    # The execution module verifies immutable bindings and retained predictions
    # before this module is permitted to aggregate terminal records.
    from run_native_panel import verified_records
    config, records = verified_records(args.output)
    summary = summarize(records, config)
    args.analysis.parent.mkdir(parents=True, exist_ok=True)
    if args.analysis.exists():
        raise FileExistsError(args.analysis)
    args.analysis.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
