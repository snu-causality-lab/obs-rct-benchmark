"""Independent checks for the v1 core study (docs/V1_CORE_STUDY.md).

  arithmetic  Check replicate_scores.csv against the run records, then recompute
              every primary contrast, Holm adjustment, verdict and headline count
              from it, without importing the analyzer. Family sizes are fixed here.
  selector    On the first seed at OBS 512 / RCT 128, regenerate every preset's
              data and recompute rct_selected's fold scores and choice with code
              written separately from the recipe (public candidate recipes,
              own folds, own nuisance and loss).
  reproduce   On the same subset, refit every recorded recipe twice and compare
              each score with its record; failures must reproduce stage and type.

    python scripts/verify_v1_core.py arithmetic --analysis outputs/v1core-analysis --config configs/v1_core_study.toml --runs outputs/v1core/group-* --output check-arith.json
    python scripts/verify_v1_core.py selector --config configs/v1_core_study.toml --runs outputs/v1core/group-* --output check-selector.json
    python scripts/verify_v1_core.py reproduce --config configs/v1_core_study.toml --runs outputs/v1core/group-* --output check-reproduce.json

Each subcommand writes a JSON report and exits nonzero on any mismatch.
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
REFERENCE = "rct_selected"
RCT_ONLY = {"rct_selected", "rct_dr_extra_trees", "rct_dr_ridge", "rct_t_extra_trees", "rct_t_ridge"}
# Declared family sizes (all, transport) outside M11: docs/V1_CORE_STUDY.md.
FAMILY_SIZES = {"datafusionppi": (116, 20), **{m: (60, 12) for m in RCT_ONLY}}
DEFAULT_FAMILY_SIZE = (120, 24)
TRANSPORT = {
    "structural:M9", "addendum:C08", "paper:experimental_grounding_kps",
    "paper:integrative_r_learner_kps_normal_target_extension",
    "paper:integrative_r_learner_multivariate_paper",
    "paper:elastic_integrative_gen_data_overlap",
}


def close(a, b, tol=TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def as_float(text):
    return None if text in ("", None) else float(text)


# ---------------------------------------------------------------- arithmetic

def arithmetic(analysis: Path, config, runs: list[Path]) -> dict:
    from scipy.special import stdtr

    mse, seeds, presence, table = {}, set(), defaultdict(set), {}
    with (analysis / "replicate_scores.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            seeds.add(int(r["seed"]))
            presence[r["method"]].add(r["scenario"])
            table[(r["scenario"], int(r["seed"]), r["method"])] = r
            if r["status"] == "ok":
                mse[(r["scenario"], int(r["seed"]), r["method"])] = float(r["cate_mse"])
    seeds = sorted(seeds)
    mismatches = []
    records = load_records(config, runs)
    if set(records) != set(table):
        mismatches.append({"reason": "replicate_scores.csv cells differ from the run records",
                           "only_records": len(set(records) - set(table)),
                           "only_csv": len(set(table) - set(records))})
    for key, row in records.items():
        got = table.get(key)
        if got is None:
            continue
        expected = row["metrics"]["cate_mse"] if row["status"] == "ok" else None
        if got["status"] != row["status"] or as_float(got["cate_mse"]) != expected:
            mismatches.append({"cell": list(key), "reason": "replicate score differs from its record"})
    with (analysis / "primary_contrasts.csv").open(newline="") as fh:
        reported = {(r["family"], r["scenario"]): r for r in csv.DictReader(fh)}
    counts = {}
    for recipe in sorted(presence):
        if recipe == REFERENCE:
            continue
        family = sorted(sc for sc in presence[recipe]
                        if not sc.startswith("structural:M11@")
                        and (recipe not in RCT_ONLY or "@o512_" in sc))
        size, transport = FAMILY_SIZES.get(recipe, DEFAULT_FAMILY_SIZE)
        if len(family) != size or sum(sc.split("@")[0] in TRANSPORT for sc in family) != transport:
            mismatches.append({"family": recipe, "reason": "family size differs from the declared size",
                               "found": len(family), "declared": size})
        rows = []
        for sc in family:
            rk = sc.replace("@o2048_", "@o512_")
            both = [s for s in seeds if (sc, s, recipe) in mse and (rk, s, REFERENCE) in mse]
            d = np.array([mse[(sc, s, recipe)] - mse[(rk, s, REFERENCE)] for s in both])
            complete = len(both) == len(seeds)
            p = 1.0
            if complete:
                sd = float(np.std(d, ddof=1))
                if sd > 0:
                    t = float(np.mean(d)) / (sd / math.sqrt(len(d)))
                    p = float(2 * stdtr(len(d) - 1, -abs(t)))
            rows.append({"scenario": sc, "complete": complete, "p": p,
                         "mean": float(np.mean(d)) if len(d) else None,
                         "wins": sum(1 for s in both if mse[(sc, s, recipe)] < mse[(rk, s, REFERENCE)])})
        m = len(rows)
        order = sorted(range(m), key=lambda i: rows[i]["p"])
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
            rows[i]["holm"] = running
        tally = defaultdict(int)
        for r in rows:
            verdict = ("untested" if not r["complete"] else
                       "better" if r["holm"] <= 0.05 and r["mean"] < 0 else
                       "worse" if r["holm"] <= 0.05 and r["mean"] > 0 else "unresolved")
            group = "transport" if r["scenario"].split("@")[0] in TRANSPORT else "other"
            tally[f"{group}_{verdict}"] += 1
            got = reported.get((recipe, r["scenario"]))
            if (got is None or got["verdict"] != verdict or not close(r["mean"], as_float(got["mean_difference"]))
                    or not close(r["p"], float(got["p_two_sided"]), 1e-9)
                    or not close(r["holm"], float(got["holm_p"]), 1e-9)
                    or int(got["wins_failure_as_loss"]) != r["wins"]):
                mismatches.append({"family": recipe, "scenario": r["scenario"], "recomputed": r,
                                   "reported": got})
        counts[recipe] = {"size": m, **tally}
    if len(reported) != sum(c["size"] for c in counts.values()):
        mismatches.append({"reason": "reported contrast rows differ from the recomputed families"})
    reported_families = json.loads((analysis / "analysis.json").read_text())["families"]
    for recipe, c in counts.items():
        got = reported_families.get(recipe, {})
        if any(got.get(k, 0) != v for k, v in c.items()) or got.get("size") != c["size"]:
            mismatches.append({"family": recipe, "reason": "headline counts differ",
                               "recomputed": c, "reported": got})
    return {"check": "arithmetic", "families": counts, "mismatches": mismatches}


# ---------------------------------------------------------------- run records

def load_records(config, runs: list[Path]) -> dict:
    from make_v1_core_configs import shards

    declared = {shard.config_hash for _, shard in shards(config)}
    records = {}
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        if manifest.get("config_hash") not in declared:
            raise SystemExit(f"{run}: not a declared shard")
        for line in (run / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            records[(row["scenario"], int(row["seed"]), row["method"])] = row
    return records


def subset(config):
    seed = config.seeds[0]
    scenarios = [s for s in config.scenarios
                 if s.parameters["n_obs"] == 512 and s.parameters["n_rct_train"] == 128]
    if len(scenarios) != 31:
        raise SystemExit("expected one OBS 512 / RCT 128 condition per preset")
    return seed, scenarios


# ---------------------------------------------------------------- selector

def recompute_selection(problem, seed: int, options: dict) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor

    from obs_rct_benchmark.data import SourceSample, TrainingView
    from obs_rct_benchmark.methods import make_method

    train, tune, obs = problem.rct_train, problem.rct_tune, problem.obs_train
    X, T, Y, e = train.X, train.T, train.Y, train.known_propensity
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(T), dtype=int)
    for arm in (0.0, 1.0):
        idx = np.where(T == arm)[0]
        rng.shuffle(idx)
        for position, row in enumerate(idx):
            fold[row] = position % 5
    names = ("rct_dr_extra_trees", "rct_dr_ridge", "rct_t_extra_trees", "rct_t_ridge")
    held = {n: np.full(len(T), np.nan) for n in names}
    failed: set[str] = set()
    residual = np.full(len(T), np.nan)
    for k in range(5):
        fit, out = fold != k, fold == k
        m_hat = ExtraTreesRegressor(n_estimators=options["n_estimators"],
                                    min_samples_leaf=options["min_samples_leaf"],
                                    max_features=1.0, random_state=(seed + 11 + k) % 2**32, n_jobs=1)
        m_hat.fit(X[fit], Y[fit])
        residual[out] = Y[out] - m_hat.predict(X[out])
        view = TrainingView(
            obs_train=SourceSample(obs.X, obs.T, np.zeros(len(obs.T)), source="obs", role="obs_train"),
            rct_train=SourceSample(X[fit], T[fit], Y[fit], source="rct", role="rct_train",
                                   known_propensity=e[fit]),
            rct_tune=SourceSample(tune.X, tune.T, np.zeros(len(tune.T)), source="rct", role="rct_tune",
                                  known_propensity=tune.known_propensity))
        for n in names:
            if n in failed:
                continue
            kwargs = ({"ridge_alpha": options["ridge_alpha"]} if n.endswith("ridge") else
                      {"n_estimators": options["n_estimators"], "min_samples_leaf": options["min_samples_leaf"]})
            try:
                values = np.asarray(make_method(n, **kwargs).fit_predict(view, X[out], seed=seed), dtype=float)
                if values.shape != (int(out.sum()),) or not np.all(np.isfinite(values)):
                    raise FloatingPointError
                held[n][out] = values
            except Exception:
                failed.add(n)
    scores = {}
    for n in names:
        loss = (residual - (T - e) * held[n]) ** 2 / (e * (1 - e))
        scores[n] = None if n in failed or not np.all(np.isfinite(loss)) else float(loss.sum() / len(loss))
    finite = [n for n in names if scores[n] is not None]
    chosen = min(finite, key=lambda n: (scores[n], names.index(n))) if finite else None
    return {"scores": scores, "selected": chosen, "unavailable": sorted(n for n in names if scores[n] is None),
            "max_weight": float((1 / (e * (1 - e))).max())}


def selector(config, runs: list[Path]) -> dict:
    from obs_rct_benchmark.runner import _generate, method_seed

    records = load_records(config, runs)
    seed, scenarios = subset(config)
    options = dict(config.method_options[REFERENCE])
    checks, mismatches = [], []
    for s in scenarios:
        row = records[(s.key, seed, REFERENCE)]
        info = row.get("fit_info") or {}
        got = recompute_selection(_generate(s, seed), method_seed(s.key, seed, REFERENCE), options)
        ok = (got["selected"] == info.get("selected") and close(got["max_weight"], info.get("max_weight"))
              and got["unavailable"] == sorted(info.get("unavailable") or {})
              and all(close(v, (info.get("scores") or {}).get(n)) for n, v in got["scores"].items()))
        checks.append({"scenario": s.key, "recorded": info, "recomputed": got, "match": ok})
        if not ok:
            mismatches.append(s.key)
    return {"check": "selector", "seed": seed, "cases": len(checks), "mismatches": mismatches,
            "tolerance": TOL, "details": checks}


# ---------------------------------------------------------------- reproduce

def reproduce(config, runs: list[Path]) -> dict:
    from obs_rct_benchmark.evaluation import cate_metrics
    from obs_rct_benchmark.methods import make_method
    from obs_rct_benchmark.runner import _generate, method_seed

    records = load_records(config, runs)
    seed, scenarios = subset(config)
    checks, mismatches = [], []
    for s in scenarios:
        problems = []
        for _ in range(2):
            try:
                problems.append((_generate(s, seed), None))
            except Exception as exc:
                problems.append((None, type(exc).__name__))
        for method in config.methods:
            row = records.get((s.key, seed, method))
            if row is None:
                continue  # declared not applicable
            outcomes = []
            for problem, generation_error in problems:
                if generation_error is not None:
                    outcomes.append({"status": "failed", "stage": "simulation", "error_type": generation_error})
                    continue
                try:
                    p = np.asarray(make_method(method, **dict(config.method_options.get(method, {}))).fit_predict(
                        problem.training_view(), problem.eval_X, seed=method_seed(s.key, seed, method)), dtype=float)
                    cate_metrics(p, problem.eval_tau)  # reproduces the runner's failure types only
                    d = p - problem.eval_tau
                    outcomes.append({"status": "ok", "mse": float(d @ d / d.size)})
                except Exception as exc:
                    outcomes.append({"status": "failed", "stage": "method_or_metric",
                                     "error_type": type(exc).__name__})
            if row["status"] == "ok":
                ok = all(o["status"] == "ok" and close(o["mse"], row["metrics"]["cate_mse"]) for o in outcomes)
            else:
                ok = all(o["status"] == "failed" and o["error_type"] == row.get("error_type")
                         and o["stage"] == row.get("stage") for o in outcomes)
            checks.append({"scenario": s.key, "method": method, "recorded_status": row["status"],
                           "repeats": outcomes, "match": ok})
            if not ok:
                mismatches.append([s.key, method])
    return {"check": "reproduce", "seed": seed, "cases": len(checks), "refits": 2 * len(checks),
            "mismatches": mismatches, "tolerance": TOL, "details": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("arithmetic", "selector", "reproduce"))
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--runs", type=Path, nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    from obs_rct_benchmark.runner import load_config

    config = load_config(a.config)
    if a.check == "arithmetic":
        result = arithmetic(a.analysis, config, a.runs)
    else:
        result = (selector if a.check == "selector" else reproduce)(config, a.runs)
    a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, default=str)[:4000])
    raise SystemExit(int(bool(result["mismatches"])))


if __name__ == "__main__":
    main()
