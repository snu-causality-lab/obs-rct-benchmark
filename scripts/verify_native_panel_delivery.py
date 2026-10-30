"""Check published native-panel scalar arithmetic without importing its producer.

This checks the exact incomplete collection and its public aggregate reductions.
It fits no estimator and does not replay predictions or authenticate omitted raw
arrays. Row-to-original-record binding relies on the server export and its pinned
delivery manifest, not on recomputing the supplied record-digest strings. The
separately published verification receipt records the earlier raw-array check.
Requires Python 3.11+ and SciPy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tomllib


CONFIG_SHA256 = "33d434a1e7fcfb2819ea3d6ed0723fbd99dd599351f50e27cb43ec1a844ecde6"
SUMMARY_SHA256 = "e3377f07655a0163fa7c26fa159c05f5a5512cee8ddc2c7f02b1a9ec814b2b19"
CAMPAIGN_SHA256 = "56003f39f113bbc84f673d0603f1fa2396030bdb64a809edde4d2473dd496c91"
EXPECTED_MISSING = {
    ("addendum:C08@trial_target", 512, 50040, "rct_causal_forest"),
    ("addendum:C08@trial_target", 512, 50042, "rct_causal_forest"),
    ("structural:M0", 512, 50047, "rct_causal_forest"),
}
FUSION = ("cornet_official", "datafusionppi", "integrative_r_linear",
          "integrative_r_quadratic", "causal_icm_fixed")
METRICS = ("cate_mse", "root_pehe")
SCALAR_FILES = (
    "scalar_records/structural_M0.csv", "scalar_records/structural_M2.csv",
    "scalar_records/structural_M10.csv", "scalar_records/addendum_C08_trial_target.csv",
    "scalar_records/paper_integrative_r_learner_kps_code.csv",
    "scalar_records/paper_cornet_official_n_conf.csv",
)
DATA_FILES = ("summary.json", "scalar_records.csv", "method_summaries.csv",
              "paired_comparisons.csv", "non_successful_fits.csv",
              "evidence/verification_receipt.json", "evidence/producer_identity.json", *SCALAR_FILES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=invalid)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "missing or duplicate CSV columns")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), "malformed CSV")
    return rows


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " fields")
        for key, value in expected.items():
            same(actual[key], value, label + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " list")
        for i, value in enumerate(expected):
            same(actual[i], value, label + f"/{i}")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label + " number")
    else:
        require(type(actual) is type(expected) and actual == expected, label + " value")


def key(row):
    return row["scenario_key"], int(row["n_rct"]), int(row["seed"]), row["method"]


def read_scalar_records(analysis):
    index = read_csv(analysis/"scalar_records.csv")
    require({r["path"] for r in index} == set(SCALAR_FILES) and len(index) == 6,
            "scalar shard index differs")
    rows = []
    for entry in index:
        path = analysis/entry["path"]
        require(sha256(path) == entry["sha256"], "scalar shard identity differs")
        shard = read_csv(path)
        require(len(shard) == int(entry["rows"]) == 1408
                and all(r["scenario_key"] == entry["scenario_key"] for r in shard),
                "scalar shard coverage differs")
        rows.extend(shard)
    return sorted(rows, key=key)


def parse_records(rows, config, expected_missing):
    expected = {(c["scenario_key"], b, s, m) for c in config["conditions"]
                for b in config["rct_budgets"] for s in config["seeds"] for m in config["methods"]}
    indexed = {}
    paths = set()
    for row in rows:
        identity = key(row)
        require(identity in expected and identity not in indexed, "extra or duplicate requested fit")
        require(row["status"] in {"success", "failure", "timeout", "missing"}, "invalid fit status")
        parsed = dict(row)
        for metric in METRICS:
            value = row[metric]
            require((value != "") == (row["status"] == "success"), "score/status mismatch")
            parsed[metric] = float(value) if value else None
            require(parsed[metric] is None or (math.isfinite(parsed[metric]) and parsed[metric] >= 0),
                    "invalid scalar score")
        if row["status"] == "success":
            same(parsed["root_pehe"], math.sqrt(parsed["cate_mse"]), "root-PEHE")
        if row["status"] == "missing":
            require(row["record_path"] == row["record_sha256"] == "", "invented missing record")
        else:
            path = Path(row["record_path"])
            require(not path.is_absolute() and ".." not in path.parts
                    and len(path.parts) == 3 and path.name == "record.json"
                    and path.parts[1] == row["method"], "invalid relative terminal path")
            require(row["record_path"] not in paths, "duplicate terminal path")
            paths.add(row["record_path"])
            require(len(row["record_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in row["record_sha256"]), "invalid record digest")
        indexed[identity] = parsed
    require(set(indexed) == expected, "requested fit union differs")
    require({k for k, r in indexed.items() if r["status"] == "missing"} == expected_missing,
            "missing identities differ")
    return indexed


def check_arithmetic(indexed, config, summary):
    from scipy.stats import beta
    method_rows = summary["method_summaries"]
    require(len(method_rows) == len(config["conditions"]) * len(config["rct_budgets"]) * len(config["methods"]),
            "method-summary row count")
    observed_methods = {(r["scenario_key"], r["n_rct"], r["method"]): r for r in method_rows}
    require(len(observed_methods) == len(method_rows), "duplicate method summary")
    seeds = config["seeds"]
    for c in config["conditions"]:
        for b in config["rct_budgets"]:
            for method in config["methods"]:
                rows = [indexed[(c["scenario_key"], b, s, method)] for s in seeds]
                successful = [r for r in rows if r["status"] == "success"]
                terminal = [r for r in rows if r["status"] != "missing"]
                expected = dict(scenario_key=c["scenario_key"], n_rct=b, method=method,
                    requested=len(seeds), successful=len(successful), terminal=len(terminal),
                    missing=len(rows)-len(terminal), status_counts=dict(Counter(r["status"] for r in terminal)),
                    conditional_on_success=len(successful) != len(seeds),
                    means={m: math.fsum(r[m] for r in successful)/len(successful) if successful else None for m in METRICS},
                    mean_risk_inference_eligible=False,
                    mean_risk_inference_reason="fitted-risk moment and mean precision conditions not established")
                same(observed_methods[(c["scenario_key"], b, method)], expected, "method summary")
    pairs = [(c["scenario_key"], b, m, ref) for c in config["conditions"]
             for b in config["rct_budgets"] for m, ref in
             [*((m, "rct_causal_forest") for m in FUSION), ("causal_icm_fixed", "causal_icm_rct_only")]]
    actual_pairs = {(r["scenario_key"], r["n_rct"], r["method"], r["reference"]): r
                    for r in summary["paired_comparisons"]}
    require(len(actual_pairs) == len(summary["paired_comparisons"]) == len(pairs)
            and set(actual_pairs) == set(pairs), "comparison family differs")
    for scenario, budget, method, reference in pairs:
        diffs = {m: [] for m in METRICS}
        wins = losses = ties = absent = failed = 0
        for seed in seeds:
            a, b = (indexed[(scenario, budget, seed, m)] for m in (method, reference))
            if "missing" in {a["status"], b["status"]}:
                absent += 1
            elif a["status"] != "success" or b["status"] != "success":
                failed += 1
            else:
                wins += a["cate_mse"] < b["cate_mse"]
                losses += a["cate_mse"] > b["cate_mse"]
                ties += a["cate_mse"] == b["cate_mse"]
                for metric in METRICS:
                    diffs[metric].append(a[metric]-b[metric])
        n = len(seeds)
        complete = absent + failed == 0
        interval = ([0. if wins == 0 else float(beta.ppf(.025, wins, n-wins+1)),
                     1. if wins == n else float(beta.ppf(.975, wins+1, n-wins))] if complete else None)
        expected = dict(scenario_key=scenario, n_rct=budget, method=method, reference=reference,
            complete_pairs=wins+losses+ties, required_pairs=n, missing_pairs=absent, failed_pairs=failed,
            complete=complete, wins=wins, losses=losses, ties=ties,
            paired_mean_differences={m: math.fsum(v)/n if complete else None for m, v in diffs.items()},
            mean_risk_inference_eligible=False, strict_win_probability=wins/n if complete else None,
            exact_pointwise_95_interval=interval, simultaneous_95_interval=None,
            one_sided_pvalue=math.fsum(math.comb(n, k) for k in range(wins, n+1))/2**n if complete else None,
            holm_pvalue=None, reject_majority_nonwin=None)
        same(actual_pairs[(scenario, budget, method, reference)], expected, "paired comparison")


def flattened(row):
    result = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for name, value in v.items():
                result[k + "__" + name] = value
        else:
            result[k] = v
    return result


def check_flat_csv(path, source_rows):
    rows = read_csv(path)
    require(len(rows) == len(source_rows), "CSV summary row count")
    for actual, original in zip(rows, source_rows):
        expected = flattened(original)
        for k, v in expected.items():
            require(k in actual, "missing summary CSV field")
            if v is None:
                require(actual[k] == "", "CSV null differs")
            elif isinstance(v, list):
                same(json.loads(actual[k]), v, "CSV interval")
            elif isinstance(v, bool):
                require(actual[k] == str(v), "CSV boolean differs")
            elif isinstance(v, (int, float)):
                same(float(actual[k]), float(v), "CSV number")
            else:
                require(actual[k] == v, "CSV string differs")
        # Unioned status columns are zero when not present in the original row.
        for k, v in actual.items():
            if k not in expected:
                require(k.startswith("status_counts__") and v == "0", "unexpected CSV field")


def run(config_path, analysis, expected_manifest_sha256=None):
    require(sha256(config_path) == CONFIG_SHA256, "frozen configuration differs")
    require(sha256(analysis/"summary.json") == SUMMARY_SHA256, "original verified summary differs")
    config = tomllib.loads(config_path.read_text())
    require(config["seeds"] == list(range(50000, 50064)), "seed grid differs")
    if expected_manifest_sha256 is not None:
        require(sha256(analysis/"DELIVERY_MANIFEST.json") == expected_manifest_sha256,
                "pinned delivery manifest differs")
    manifest = read_json(analysis/"DELIVERY_MANIFEST.json")
    require(set(manifest["files"]) == set(DATA_FILES), "delivery file set differs")
    for name, digest in manifest["files"].items():
        require(sha256(analysis/name) == digest, "delivery digest differs: " + name)
    summary = read_json(analysis/"summary.json")
    receipt = read_json(analysis/"evidence/verification_receipt.json")
    require(receipt["verified_summary_sha256"] == SUMMARY_SHA256
            and receipt["manifest_sha256"] == CAMPAIGN_SHA256
            and receipt["verification_job_id"] == "731708"
            and receipt["bootstrap_sha256"] == summary["verification"]["bootstrap_sha256"],
            "verification receipt differs")
    producer = read_json(analysis/"evidence/producer_identity.json")
    require(producer["scientific_fitter_revision"] == "83c45c3acca5ffd316a48153647b078280d648cb"
            and producer["campaign_manifest_sha256"] == CAMPAIGN_SHA256
            and producer["config_sha256"] == CONFIG_SHA256
            and producer["original_summary_sha256"] == SUMMARY_SHA256
            and producer["verification"] == summary["verification"], "producer identity differs")
    for name, field in (("scripts/run_native_panel.py","source_runner_sha256"),
                        ("scripts/analyze_native_panel.py","source_analysis_sha256"),
                        ("scripts/verify_native_panel.py","source_truth_checker_sha256")):
        require(producer["source_files_sha256"][name] == summary["verification"][field],
                "producer verifier source differs")
    scalar_rows = read_scalar_records(analysis)
    indexed = parse_records(scalar_rows, config, EXPECTED_MISSING)
    counts = Counter(r["status"] for r in indexed.values())
    require(counts == {"success": 8350, "failure": 64, "timeout": 31, "missing": 3}, "coverage differs")
    require(summary["expected_fits"] == len(indexed) == 8448 and summary["terminal_fits"] == 8445
            and summary["missing_fits"] == 3 and summary["status_counts"] == {k:v for k,v in counts.items() if k != "missing"},
            "summary coverage differs")
    require({(r["scenario_key"], r["n_rct"], r["seed"], r["method"]) for r in summary["missing_identities"]}
            == EXPECTED_MISSING, "summary missing union differs")
    require(summary["family_size"] == 72 and summary["family_complete"] is False
            and summary["precision"]["eligible"] is False
            and summary["pointwise_results"] == "descriptive_only"
            and summary["analysis_kind"] == "separate_incomplete_collection_analysis", "inference permission differs")
    h = hashlib.sha256()
    for row in sorted((r for r in indexed.values() if r["status"] != "missing"), key=lambda r:r["record_path"]):
        h.update(row["record_path"].encode()+b"\0"+row["record_sha256"].encode()+b"\n")
    require(h.hexdigest() == receipt["record_inventory_sha256"]
            == summary["verification"]["terminal_record_path_and_sha256_digest"], "terminal inventory differs")
    check_arithmetic(indexed, config, summary)
    check_flat_csv(analysis/"method_summaries.csv", summary["method_summaries"])
    check_flat_csv(analysis/"paired_comparisons.csv", summary["paired_comparisons"])
    non_success = read_csv(analysis/"non_successful_fits.csv")
    require(non_success == [r for r in scalar_rows if r["status"] != "success"],
            "non-success disclosure differs")
    return dict(schema_version=1, passed=True, requested_fits=8448, terminal_fits=8445,
                status_counts=dict(counts), method_rows=132, paired_rows=72,
                all_joint_decisions_withheld=True, simultaneous_precision_withheld=True,
                original_summary_sha256=SUMMARY_SHA256, terminal_inventory_sha256=h.hexdigest(),
                scope="public scalar arithmetic and identities; no estimator fit or raw prediction replay",
                delivery_manifest_sha256=sha256(analysis/"DELIVERY_MANIFEST.json"),
                input_sha256={"config":sha256(config_path), **{name:sha256(analysis/name) for name in DATA_FILES}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True,
                        help="published delivery-manifest digest; obtain independently of edited inputs")
    args = parser.parse_args()
    result = run(args.config, args.analysis, args.expected_manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "input_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
