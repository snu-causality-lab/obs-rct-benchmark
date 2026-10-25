"""Independently check compact catalog arithmetic, without importing the producer.

Run on GSDS, after the catalog analyzer has written its complete directory:
  python independent_reaggregate.py --config code-v2/configs/catalog_study.toml \
      --analysis catalog-analysis-v1 --output independent-reaggregation-v1.json

Requires Python 3.11+ and SciPy. Uses csv, math.fsum and explicit quantile/Holm
arithmetic rather than the producer's NumPy summary helpers. SciPy supplies only
Student-t quantiles and survival probabilities. No model is fitted, dataset
generated, or original prediction verified. Input/output hashes bind the exact
compact CSVs checked; they do not authenticate their producer by themselves.
Inputs must be complete and immutable during checking. A pass establishes
arithmetic consistency, including disclosed failures; read coverage separately.
Incomplete contrasts have no descriptive paired summary, and any incomplete
eligible contrast withholds the entire Holm decision set. Condition means use
successful replicates. M11 retains canonical t(3) noise, so squared-loss variance
need not exist. The excluded official adapter requires constant propensity,
whereas that one multivariate Integrative-R design has varying propensity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

import scipy
from scipy.stats import t


FUSION = ("cornet_official", "datafusionppi")
CONTROLS = ("rct_t_extra_trees", "rct_dr_extra_trees")
METHODS = {"rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
           "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
           "rct_dr_ridge", "rct_dr_extra_trees", *FUSION}
UNSUPPORTED = "paper:integrative_r_learner_multivariate_paper"
HEAVY = "structural:M11"
TOLERANCE = 1e-12
CONFIG_SHA256 = "17b5f87d466ea10eefe6c3c4dd4699182c709b9c8e8a984a477bc28468344645"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None
                and len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"missing or duplicate columns: {path}")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            f"malformed CSV: {path}")
    return rows


def number(raw):
    value = float(raw)
    require(math.isfinite(value), "nonfinite numerical input")
    return value


def check_row(actual, expected, identity):
    for key, value in expected.items():
        require(key in actual, f"missing column {key}: {identity}")
        raw = actual[key]
        if value is None:
            equal = raw == ""
        elif isinstance(value, bool):
            equal = raw == str(value)
        elif isinstance(value, int):
            equal = raw == str(value)
        elif isinstance(value, float):
            equal = math.isclose(number(raw), value,
                                 rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        elif isinstance(value, list):
            parsed = json.loads(raw)
            equal = (isinstance(parsed, list) and len(parsed) == len(value)
                     and all(isinstance(a, (int, float)) and not isinstance(a, bool)
                             and math.isfinite(a) and math.isclose(a, b,
                                 rel_tol=TOLERANCE, abs_tol=TOLERANCE)
                             for a, b in zip(parsed, value)))
        else:
            equal = raw == value
        require(equal, f"mismatch {identity} {key}: observed={raw!r}, expected={value!r}")


def summary(values, inferential=True):
    count = len(values)
    result = dict(n=count, mean=None, mcse=None, ci95=None,
                  p_two_sided_zero=None, sample_median=None, sample_q10=None,
                  sample_q25=None, sample_q75=None, sample_q90=None)
    if not count:
        return result
    ordered = sorted(values)

    def quantile(probability):
        position = (count - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    average = math.fsum(values) / count
    result.update(mean=average, sample_median=quantile(.5), sample_q10=quantile(.1),
                  sample_q25=quantile(.25), sample_q75=quantile(.75),
                  sample_q90=quantile(.9))
    if inferential and count >= 2:
        se = math.sqrt(math.fsum((value - average) ** 2 for value in values)
                       / (count - 1) / count)
        half = float(t.ppf(.975, count - 1)) * se
        pvalue = ((1.0 if average == 0 else 0.0) if se == 0
                  else 2.0 * float(t.sf(abs(average / se), count - 1)))
        result.update(mcse=se, ci95=[average - half, average + half],
                      p_two_sided_zero=pvalue)
    return result


def run(config_path, analysis):
    require(digest(config_path) == CONFIG_SHA256, "configuration differs from frozen catalog protocol")
    config = tomllib.loads(config_path.read_text())
    seeds, methods = config["run"]["seeds"], config["run"]["methods"]
    require(seeds == list(range(20000, 20064)), "unexpected seed grid")
    require(len(methods) == 10 and set(methods) == METHODS, "unexpected recipe grid")
    scenarios = {f"{row['kind']}:{row['id']}@{row['label']}": row
                 for row in config["scenarios"]}
    require(len(scenarios) == len(config["scenarios"]) == 124,
            "expected 124 unique declared conditions")
    presets = {key.split("@", 1)[0] for key in scenarios}
    require(len(presets) == 31, "expected 31 presets")
    budgets = {"o512_r128", "o512_r512", "o2048_r128", "o2048_r512"}
    require(all({key.split("@", 1)[1] for key in scenarios
                 if key.split("@", 1)[0] == preset} == budgets for preset in presets),
            "a preset is missing a budget")

    def applicable(scenario, method):
        return not (scenario.split("@", 1)[0] == UNSUPPORTED
                    and method == "datafusionppi")

    expected = {(scenario, seed, method) for scenario in scenarios for seed in seeds
                for method in methods if applicable(scenario, method)}
    require(len(expected) == 79104, "unexpected applicability count")
    files = sorted(analysis.glob("designs/*/replicate_scores.csv"))
    require(len(files) == 31, "expected one replicate CSV per preset")
    records, inputs = {}, {}
    for path in files:
        inputs[path.relative_to(analysis).as_posix()] = digest(path)
        for row in read_csv(path):
            key = (row["scenario"], int(row["seed"]), row["method"])
            require(key in expected and key not in records,
                    f"undeclared or duplicate replicate cell: {key}")
            require(path.parent.name == key[0].split("@", 1)[0].replace(":", "__"),
                    f"replicate stored under wrong preset: {key}")
            status = row["status"]
            require(status in {"ok", "failed", "missing"}, f"invalid status: {key}")
            if status == "ok":
                mse, root = number(row["cate_mse"]), number(row["root_pehe"])
                require(mse >= 0 and root >= 0 and math.isclose(root * root, mse,
                        rel_tol=1e-10, abs_tol=1e-12), f"invalid metric identity: {key}")
            else:
                require(row["cate_mse"] == row["root_pehe"] == "",
                        f"unsuccessful cell has a score: {key}")
                if status == "failed":
                    require(bool(row["stage"] and row["error_type"]),
                            f"failed cell lacks stage/type: {key}")
                mse = root = None
            records[key] = (status, mse, root)
    require(set(records) == expected, "compact CSVs omit requested applicable cells")

    scores, contrasts = {}, {}
    for scenario in scenarios:
        eligible = scenario.split("@", 1)[0] != HEAVY
        for method in methods:
            applies = applicable(scenario, method)
            cells = [records[(scenario, seed, method)] for seed in seeds] if applies else []
            successes = [cell for cell in cells if cell[0] == "ok"]
            scores[(scenario, method)] = dict(
                applicable=applies, expected=len(cells), successful=len(successes),
                failed=sum(cell[0] == "failed" for cell in cells),
                missing=sum(cell[0] == "missing" for cell in cells),
                **summary([cell[1] for cell in successes], eligible),
                mean_root_pehe=(math.fsum(cell[2] for cell in successes) / len(successes)
                                if successes else None),
                inference_eligible=applies and eligible)
        for fusion in FUSION:
            if not applicable(scenario, fusion):
                continue
            for control in CONTROLS:
                pairs = [(records[(scenario, seed, fusion)],
                          records[(scenario, seed, control)]) for seed in seeds]
                good = [(left, right) for left, right in pairs
                        if left[0] == right[0] == "ok"]
                complete = len(good) == len(seeds)
                values = [left[1] - right[1] for left, right in good] if complete else []
                contrasts[(scenario, fusion, control)] = dict(
                    complete=complete, expected_pairs=len(seeds), successful_pairs=len(good),
                    inference_eligible=eligible, **summary(values, eligible))
    eligible_keys = [key for key, row in contrasts.items() if row["inference_eligible"]]
    require(len(contrasts) == 488 and len(eligible_keys) == 472,
            "unexpected supported or eligible contrast grid")
    family_complete = all(contrasts[key]["complete"]
                          and contrasts[key]["p_two_sided_zero"] is not None
                          for key in eligible_keys)
    adjusted = {}
    if family_complete:
        ranked = sorted(eligible_keys, key=lambda key: contrasts[key]["p_two_sided_zero"])
        running = 0.0
        for rank, key in enumerate(ranked):
            running = max(running, min(1.0, (len(ranked) - rank)
                                      * contrasts[key]["p_two_sided_zero"]))
            adjusted[key] = running
    for key, row in contrasts.items():
        pvalue = adjusted.get(key)
        decision = ("not_assessed_heavy_tail" if not row["inference_eligible"]
                    else "withheld_incomplete_family")
        if pvalue is not None:
            decision = (("favorable" if row["mean"] < 0 else "adverse")
                        if pvalue <= .05 else "unresolved")
        row.update(holm_p=pvalue, holm_reject_05=None if pvalue is None else pvalue <= .05,
                   decision=decision)
    checked = {}
    for filename, table, keys in [
            ("condition_scores.csv", scores, ("scenario", "method")),
            ("primary_contrasts.csv", contrasts, ("scenario", "method", "reference"))]:
        path = analysis / filename
        seen = set()
        for row in read_csv(path):
            key = tuple(row[field] for field in keys)
            require(key in table and key not in seen, f"undeclared or duplicate summary: {key}")
            check_row(row, table[key], key)
            seen.add(key)
        require(seen == set(table), f"missing summary rows: {filename}")
        checked[filename] = len(seen)
        inputs[filename] = digest(path)
    na_path = analysis / "not_applicable.csv"
    na_expected = {(scenario, "datafusionppi") for scenario in scenarios
                   if scenario.split("@", 1)[0] == UNSUPPORTED}
    na_seen = set()
    for row in read_csv(na_path):
        key = (row["scenario"], row["method"])
        require(key in na_expected and key not in na_seen, "invalid or duplicate N/A row")
        check_row(row, dict(seed_count=len(seeds)), key)
        require(bool(row["reason"]), "N/A row lacks declared reason")
        na_seen.add(key)
    require(na_seen == na_expected and len(na_seen) == 4, "missing N/A rows")
    inputs["not_applicable.csv"] = digest(na_path)
    coverage = dict(nominal_cells=79360, not_applicable_cells=256, expected=len(expected),
                    recorded=sum(row[0] != "missing" for row in records.values()),
                    successful=sum(row[0] == "ok" for row in records.values()),
                    failed=sum(row[0] == "failed" for row in records.values()),
                    missing=sum(row[0] == "missing" for row in records.values()),
                    supported_primary_contrasts=488, nominal_primary_contrasts=496,
                    mean_t_family_size=472, descriptive_heavy_tail_contrasts=16,
                    mean_t_family_complete=family_complete, not_applicable_primary_contrasts=8)
    report = json.loads((analysis / "summary.json").read_text())
    require(report["coverage"] == coverage, "coverage differs from independent arithmetic")
    manifest = json.loads((analysis / "manifest.json").read_text())
    require((set(inputs) | {"summary.json"}) <= set(manifest["output_sha256"]),
            "manifest omits checked scientific inputs")
    for relative, expected_hash in manifest["output_sha256"].items():
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts,
                "unsafe manifest output path")
        require(digest(analysis / relative) == expected_hash, f"output hash mismatch: {relative}")
    for filename in ("summary.json", "manifest.json"):
        inputs[filename] = digest(analysis / filename)
    require(all(digest(analysis / relative) == expected_hash
                for relative, expected_hash in inputs.items()),
            "an input changed while the arithmetic was checked")
    require(digest(config_path) == CONFIG_SHA256, "configuration changed while checking")
    return dict(schema_version=1, scope="independent compact-CSV arithmetic check; no prediction replay",
                config_file_sha256=digest(config_path), checker_source_sha256=digest(Path(__file__)),
                environment=dict(python=sys.version, scipy=scipy.__version__),
                arithmetic_tolerance=dict(absolute=TOLERANCE, relative=TOLERANCE),
                input_sha256=inputs, checked_rows=checked, coverage=coverage,
                incomplete_eligible_contrasts=sum(not contrasts[key]["complete"] for key in eligible_keys),
                status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    result = run(args.config, args.analysis)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "coverage", "checked_rows", "incomplete_eligible_contrasts")}, sort_keys=True))


if __name__ == "__main__":
    main()
