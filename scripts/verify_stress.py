"""Independent checks for stress study 1 (docs/V1_STRESS_CORNET.md).

  targets     Recompute tau, the OBS arm shifts and propensities from the plan's formulas
              (written out here, not imported) on the first seed, at 1e-12; check sizes
              and covariate scales.
  arithmetic  Recompute every contrast, Holm adjustment and verdict from replicate_scores.csv
              after checking it against the run records, without importing the analyzer.
  reproduce   Refit every recipe twice on the first seed and compare with the records (1e-12).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

TOL = 1e-12
OBS_RECIPES = ("cornet_raw", "cornet_official", "datafusionppi", "obs_t_ridge", "obs_t_extra_trees",
               "naive_pooled_t_ridge", "naive_pooled_t_extra_trees")


V = 0.1225
OBS_SIZES = (8192, 2048, 512, 128)


def shape(X, level):
    s = X[:, 0] + X[:, 1]
    c = [0, 1, 2, 4][level]
    return np.sqrt(V / 2) * s if c == 0 else np.sqrt(V / ((1 - np.exp(-4 * c * c)) / 2)) * np.sin(c * s)


def plan_formulas(X, dial, level):
    """The plan's equations, written independently of stress_dgp.py: (tau, b0, b1, e_obs, sigma)."""
    x1, x2, x4 = X[:, 0], X[:, 1], X[:, 3]
    tau = 1 + 0.5 * np.tanh(x1) + 0.5 * x4
    zero = np.zeros(len(X))
    if dial == "hidden_modifier":
        r = level / 3
        b0, b1 = zero, -0.5 * (np.sqrt(1 - r) * x1 + np.sqrt(r) * x4)
    elif dial == "bias_complexity":
        b1 = shape(X, level)
        b0 = -b1
    elif dial == "common_shift":
        b0 = b1 = shape(X, level)
    elif dial == "obs_selection":
        b0, b1 = zero, zero
    elif dial == "obs_size":
        b0, b1 = zero, 0.5 * x1 + 0.25 * x2 ** 2
    else:
        b0, b1 = zero, 0.5 * x1
    k = [0, 1, 2, 4][level] if dial == "obs_selection" else 0
    gamma = [0, 0.5, 1, 1.5][level] if dial == "noise_shape" else 0
    return tau, b0, b1, 1 / (1 + np.exp(-k * x1)), np.exp(gamma * x2 / 2 - gamma ** 2 / 4)


def regenerate(dial, level, params, seed):
    """Redraw every role from the plan's sampling order: X, then T, then noise."""
    streams = [np.random.default_rng(c) for c in np.random.SeedSequence([seed, 7319]).spawn(6)]
    scale = [1, 0.75, 0.5, 0.35][level] if dial == "trial_shrink" else 1.0
    out = {}
    for i, (role, key) in enumerate((("obs_train", "n_obs"), ("rct_train", "n_rct_train"),
                                      ("rct_tune", "n_rct_tune"), ("rct_test", "n_rct_test"))):
        rows = int(params[key])
        drawn = OBS_SIZES[0] if (role == "obs_train" and dial == "obs_size") else rows
        X = streams[i].normal(size=(drawn, 6)) * (1.0 if role == "obs_train" else scale)
        tau, b0, b1, e, sigma = plan_formulas(X, dial, level)
        if role != "obs_train":
            e, b0, b1 = np.full(drawn, 0.5), np.zeros(drawn), np.zeros(drawn)
        T = (streams[i].uniform(size=drawn) < e).astype(float)
        mu0 = 0.5 * X[:, 0] + 0.5 * np.sin(X[:, 1]) + 0.25 * X[:, 2] ** 2
        Y = mu0 + T * tau + np.where(T == 1, b1, b0) + sigma * streams[i].normal(size=drawn)
        out[role] = (X[:rows], T[:rows], Y[:rows])
    shift = [0, 0.5, 1, 1.5][level] if dial == "target_shift" else 0.0
    escale = 1.0 if dial in ("trial_shrink", "target_shift") else scale
    ex = streams[4].normal(size=(int(params["n_eval"]), 6)) * escale
    ex[:, 0] += shift
    out["eval"] = (ex, plan_formulas(ex, dial, level)[0])
    return out


def targets(config) -> dict:
    from obs_rct_benchmark.runner import _generate

    seed, checks, bad = config.seeds[0], [], []
    for s in config.scenarios:
        problem = _generate(s, seed)
        dial, level = s.id, int(s.parameters["level"])
        ref = regenerate(dial, level, s.parameters, seed)
        ok = True
        for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
            sample = getattr(problem, role)
            for got, want in zip((sample.X, sample.T, sample.Y), ref[role]):
                ok &= got.shape == want.shape and bool(np.allclose(got, want, rtol=0, atol=TOL))
        ok &= bool(np.allclose(problem.rct_train.known_propensity, 0.5, rtol=0, atol=0))
        ok &= bool(np.allclose(problem.eval_X, ref["eval"][0], rtol=0, atol=TOL))
        ok &= bool(np.allclose(problem.eval_tau, ref["eval"][1], rtol=0, atol=TOL))
        checks.append({"scenario": s.key, "match": bool(ok)})
        if not ok:
            bad.append(s.key)
    return {"check": "targets", "seed": seed, "cases": len(checks), "mismatches": bad, "details": checks}


def load_records(runs):
    out = {}
    for run in runs:
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            out[(row["scenario"], int(row["seed"]), row["method"])] = row
    return out


def arithmetic(analysis: Path, runs) -> dict:
    from scipy.special import stdtr

    table, mse, seeds, scen = {}, {}, set(), []
    with (analysis / "replicate_scores.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["scenario"], int(r["seed"]), r["method"])
            table[key] = r
            seeds.add(key[1])
            if r["scenario"] not in scen:
                scen.append(r["scenario"])
            if r["status"] == "ok":
                mse[key] = float(r["cate_mse"])
    seeds, mismatches = sorted(seeds), []
    records = load_records(runs)
    if set(records) != set(table) or any(
            table[k]["status"] != r["status"] or (r["status"] == "ok" and float(table[k]["cate_mse"]) != r["metrics"]["cate_mse"])
            for k, r in records.items() if k in table):
        mismatches.append({"reason": "replicate_scores.csv differs from the run records"})
    if len(scen) != 64 or len(seeds) != 64:
        mismatches.append({"reason": "expected 64 conditions and 64 seeds"})
    with (analysis / "contrasts.csv").open(newline="") as fh:
        reported = {(r["family"], r["scenario"]): r for r in csv.DictReader(fh)}
    counts, verdicts = {}, {}
    for recipe in OBS_RECIPES:
        rows = []
        for sc in scen:
            both = [s for s in seeds if (sc, s, recipe) in mse and (sc, s, "rct_selected") in mse]
            d = np.array([mse[(sc, s, recipe)] - mse[(sc, s, "rct_selected")] for s in both])
            p = 1.0
            if len(both) == len(seeds) and float(np.std(d, ddof=1)) > 0:
                t = float(np.mean(d)) / (float(np.std(d, ddof=1)) / math.sqrt(len(d)))
                p = float(2 * stdtr(len(d) - 1, -abs(t)))
            rows.append({"sc": sc, "complete": len(both) == len(seeds), "p": p, "mean": float(np.mean(d)) if len(d) else None})
        order, running = sorted(range(len(rows)), key=lambda i: rows[i]["p"]), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p"]))
            rows[i]["holm"] = running
        tally = defaultdict(int)
        for r in rows:
            v = ("untested" if not r["complete"] else "better" if r["holm"] <= 0.05 and r["mean"] < 0
                 else "worse" if r["holm"] <= 0.05 and r["mean"] > 0 else "unresolved")
            r["verdict"] = v
            extrap = r["sc"].split("@")[0].split(":")[1] in ("trial_shrink", "target_shift")
            tally[f"{'extrapolation' if extrap else 'other'}_{v}"] += 1
            verdicts[(recipe, r["sc"])] = v
            got = reported.get((recipe, r["sc"]))
            if got is None or got["verdict"] != v or not math.isclose(float(got["holm_p"]), r["holm"], rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append({"family": recipe, "scenario": r["sc"], "recomputed": r})
        counts[recipe] = dict(tally)
    reported_families = json.loads((analysis / "analysis.json").read_text())["families"]
    for recipe, tally in counts.items():
        got = {k: v for k, v in reported_families.get(recipe, {}).items() if k != "size" and v}
        if got != tally:
            mismatches.append({"family": recipe, "reason": "verdict counts differ", "recomputed": tally, "reported": got})
    with (analysis / "readout.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            seq = [verdicts.get((r["recipe"], f"stress:{r['dial']}@l{i}_r{r['rct']}")) for i in range(4)]
            state = ("no_level0_win" if seq[0] != "better" else "broke" if seq[3] == "worse"
                     else "stopped" if seq[3] != "better" else "kept")
            if seq != [r[f"level_{i}"] for i in range(4)] or state != r["state"]:
                mismatches.append({"readout": [r["recipe"], r["dial"], r["rct"]], "recomputed": [seq, state]})
    return {"check": "arithmetic", "families": counts, "mismatches": mismatches}


def reproduce(config, runs) -> dict:
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records, seed, checks, bad = load_records(runs), config.seeds[0], [], []
    for s in config.scenarios:
        problems = [_generate(s, seed), _generate(s, seed)]
        for m in config.methods:
            row, outs = records[(s.key, seed, m)], []
            for problem in problems:
                try:
                    p = np.asarray(make_method(m, **dict(config.method_options.get(m, {}))).fit_predict(
                        problem.training_view(), problem.eval_X, seed=method_seed(s.key, seed, m)), dtype=float)
                    cate_metrics(p, problem.eval_tau)
                    d = p - problem.eval_tau
                    outs.append({"status": "ok", "mse": float(d @ d / d.size)})
                except Exception as exc:
                    outs.append({"status": "failed", "error_type": type(exc).__name__})
            ok = (all(o["status"] == "ok" and math.isclose(o["mse"], row["metrics"]["cate_mse"], rel_tol=TOL, abs_tol=TOL) for o in outs)
                  if row["status"] == "ok" else
                  all(o["status"] == "failed" and o["error_type"] == row.get("error_type") for o in outs))
            checks.append({"scenario": s.key, "method": m, "match": ok})
            if not ok:
                bad.append([s.key, m])
    return {"check": "reproduce", "seed": seed, "cases": len(checks), "mismatches": bad}


def main() -> None:
    from obs_rct_benchmark.runner import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("targets", "arithmetic", "reproduce"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    config = load_config(a.config)
    result = (targets(config) if a.check == "targets" else
              arithmetic(a.analysis, a.runs) if a.check == "arithmetic" else reproduce(config, a.runs))
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:3000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
