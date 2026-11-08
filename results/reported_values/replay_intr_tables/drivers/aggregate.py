"""Aggregate both arms and compare them with the paper. Descriptive only.

Usage: python aggregate.py --results DIR --preflight DIR --reported CSV --native CSV --out DIR
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

A_CELLS = ("t1_n200", "t1_n500", "t1_n1000", "t2_J1_b0", "t2_J1_b25", "t2_J9_b0", "t2_J9_b25")
B_CELLS = A_CELLS + ("bridge_r128", "bridge_r512")
SEEDS = set(range(100))
A_METHODS = {"int rlearner": "rmse_int", "rct rlearner": "rmse_rct", "os rlearner": "rmse_os",
             "naive rlearner": "rmse_naive"}
CONDITION_CELL = {
    ("wu2022_table1", "n=200"): "t1_n200", ("wu2022_table1", "n=500"): "t1_n500",
    ("wu2022_table1", "n=1000"): "t1_n1000", ("wu2022_table1", "all n"): "t1_n500",
    ("wu2022_table2", "J=1, b=0"): "t2_J1_b0", ("wu2022_table2", "J=1, b=2.5"): "t2_J1_b25",
    ("wu2022_table2", "J=9, b=0"): "t2_J9_b0", ("wu2022_table2", "J=9, b=2.5"): "t2_J9_b25",
    ("wu2022_table2", "J=1, all b"): "t2_J1_b0", ("wu2022_table2", "J=9, all b"): "t2_J9_b0",
}
PINNED_SOURCES = {
    "utils.R": "2b85e2e56db83e33427f6f43aa1b59d995bafb6d46bfc1aec8e5f2e6a215525b",
    "intRlearner.R": "98563e2423f41153fcbb2548f3b54ac779c72bcb8d154608bf51f778086fdce8",
}
README_EXAMPLE = {"rmse_int": 0.8664778, "rmse_rct": 1.5589467, "rmse_os": 1.4570438, "rmse_naive": 1.3032480}
NATIVE_CELLS = {"bridge_r128": "128", "bridge_r512": "512"}

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True)
parser.add_argument("--preflight", required=True)
parser.add_argument("--reported", required=True)
parser.add_argument("--native", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
results, out = Path(args.results), Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def summary(values):
    n = len(values)
    sd = stdev(values) if n > 1 else float("nan")
    return {"n": n, "mean": mean(values) if n else float("nan"), "sd": sd,
            "se": sd / math.sqrt(n) if n > 1 else float("nan"),
            "min": min(values) if n else float("nan"), "max": max(values) if n else float("nan")}


def read_rows(arm, filename):
    rows, problems = {}, []
    for path in sorted((results / arm).glob(f"*/seeds-*/{filename}")):
        for row in csv.DictReader(path.open(newline="")):
            key = (row["cell"], int(row["seed"]))
            if key in rows:
                problems.append(f"duplicate {key}")
            rows[key] = row
    return rows, problems


a_rows, a_problems = read_rows("author", "replicates.csv")
b_rows, b_problems = read_rows("benchmark", "fits.csv")
source_checks = []
for path in sorted((results / "author").glob("*/seeds-*/sources.sha256")):
    for line in path.read_text().splitlines():
        digest, name = line.split()
        source_checks.append(PINNED_SOURCES.get(Path(name).name) == digest)

coverage, cell_rows, timing = [], [], []
for arm, cells, rows, metric_columns in (
        ("A", A_CELLS, a_rows, list(A_METHODS.values())), ("B", B_CELLS, b_rows, ["root_pehe"])):
    for cell in cells:
        present = {s for (c, s) in rows if c == cell}
        statuses = [rows[(cell, s)]["status"] for s in sorted(present)]
        coverage.append({"arm": arm, "cell": cell, "expected": len(SEEDS), "present": len(present & SEEDS),
                         "missing": len(SEEDS - present), "success": statuses.count("success"),
                         "not_success": len(statuses) - statuses.count("success")})
        ok = [rows[(cell, s)] for s in sorted(present) if rows[(cell, s)]["status"] == "success"]
        seconds = [float(r["seconds"]) for r in rows.values() if r["cell"] == cell and r["seconds"]]
        timing.append({"arm": arm, "cell": cell, "median_seconds": median(seconds) if seconds else "",
                       "max_seconds": max(seconds) if seconds else ""})
        for column in metric_columns:
            s = summary([float(r[column]) for r in ok])
            method = {"rmse_int": "int rlearner", "rmse_rct": "rct rlearner", "rmse_os": "os rlearner",
                      "rmse_naive": "naive rlearner", "root_pehe": "integrative_r_quadratic"}[column]
            cell_rows.append({"arm": arm, "cell": cell, "method": method, **s})


def cell_stat(arm, cell, method):
    for row in cell_rows:
        if row["arm"] == arm and row["cell"] == cell and row["method"] == method:
            return row
    raise KeyError((arm, cell, method))


reported = [r for r in csv.DictReader(open(args.reported, newline=""))
            if r["experiment_id"] in ("wu2022_table1", "wu2022_table2") and r["method_paper"] in A_METHODS]
r1 = []
for r in reported:
    cell = CONDITION_CELL[(r["experiment_id"], r["condition"])]
    s = cell_stat("A", cell, r["method_paper"])
    paper, paper_se = float(r["value"]), float(r["uncertainty"])
    tolerance = 2 * math.sqrt(paper_se ** 2 + s["se"] ** 2) + 0.005
    r1.append({"experiment_id": r["experiment_id"], "condition": r["condition"], "method": r["method_paper"],
               "paper": paper, "paper_se": paper_se, "cell": cell, "rerun_mean": s["mean"],
               "rerun_se": s["se"], "rerun_n": s["n"], "difference": s["mean"] - paper,
               "tolerance": tolerance, "agrees": abs(s["mean"] - paper) <= tolerance})

r2 = []
for cell in A_CELLS:
    a, b = cell_stat("A", cell, "int rlearner"), cell_stat("B", cell, "integrative_r_quadratic")
    paper = next(x["paper"] for x in r1 if x["cell"] == cell and x["method"] == "int rlearner")
    tolerance = 2 * math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    r2.append({"cell": cell, "paper": paper, "a_mean": a["mean"], "a_se": a["se"], "a_n": a["n"],
               "b_mean": b["mean"], "b_se": b["se"], "b_n": b["n"], "b_minus_a": b["mean"] - a["mean"],
               "tolerance": tolerance, "agrees": abs(b["mean"] - a["mean"]) <= tolerance,
               "b_minus_paper": b["mean"] - paper})

native = [r for r in csv.DictReader(open(args.native, newline=""))
          if r["method"] == "integrative_r_quadratic" and r["status"] == "success"]
bridge = []
for cell, n_rct in NATIVE_CELLS.items():
    n = summary([float(r["root_pehe"]) for r in native if r["n_rct"] == n_rct])
    b = cell_stat("B", cell, "integrative_r_quadratic")
    bridge.append({"cell": cell, "replay_mean": b["mean"], "replay_se": b["se"], "replay_n": b["n"],
                   "native_mean": n["mean"], "native_se": n["se"], "native_n": n["n"],
                   "difference": b["mean"] - n["mean"]})

rct_sizes = []
for cell in ("t2_J1_b0", "t2_J1_b25", "t2_J9_b0", "t2_J9_b25"):
    sizes = [float(r["n_rct"]) for (c, _), r in a_rows.items() if c == cell and r["n_rct"]]
    rct_sizes.append({"cell": cell, **summary(sizes)})

# Same seed and J give the same RCT sample whatever b is, so the RCT-only fit must not change with b.
identity = {}
for j in ("J1", "J9"):
    pairs = [(a_rows.get((f"t2_{j}_b0", s)), a_rows.get((f"t2_{j}_b25", s))) for s in sorted(SEEDS)]
    pairs = [(x, y) for x, y in pairs if x and y and x["status"] == y["status"] == "success"]
    identity[j] = {"pairs": len(pairs),
                   "identical": sum(abs(float(x["rmse_rct"]) - float(y["rmse_rct"])) < 1e-12 for x, y in pairs)}

r0 = {}
readme_csv = Path(args.preflight) / "author-readme_J8_b25" / "replicates.csv"
if readme_csv.exists():
    row = next(csv.DictReader(readme_csv.open(newline="")))
    r0 = {k: {"readme": v, "rerun": float(row[k]) if row[k] else None,
              "difference": float(row[k]) - v if row[k] else None} for k, v in README_EXAMPLE.items()}


def write(name, rows):
    with (out / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


for name, rows in (("coverage.csv", coverage), ("cell_summary.csv", cell_rows), ("timing.csv", timing),
                   ("r1_reproduction.csv", r1), ("r2_benchmark.csv", r2), ("bridge.csv", bridge),
                   ("rct_sizes.csv", rct_sizes)):
    write(name, rows)
readouts = {
    "problems": a_problems + b_problems,
    "author_sources_match_pin": f"{sum(source_checks)} of {len(source_checks)}",
    "r0_readme_example": r0,
    "r1_agree": f"{sum(x['agrees'] for x in r1)} of {len(r1)}",
    "r1_int_rlearner": [{k: x[k] for k in ("experiment_id", "condition", "paper", "rerun_mean", "rerun_se", "agrees")}
                        for x in r1 if x["method"] == "int rlearner"],
    "r2_agree": f"{sum(x['agrees'] for x in r2)} of {len(r2)}",
    "r2": r2,
    "bridge": bridge,
    "rct_sizes": rct_sizes,
    "rct_only_identical_across_b": identity,
    "coverage_incomplete": [c for c in coverage if c["missing"] or c["not_success"]],
    "reported_sha256": hashlib.sha256(Path(args.reported).read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(Path(args.native).read_bytes()).hexdigest(),
}
(out / "readouts.json").write_text(json.dumps(readouts, indent=2))
print(json.dumps({k: readouts[k] for k in ("problems", "author_sources_match_pin", "r1_agree", "r2_agree")}, indent=2))
