"""Aggregate Arm A and Arm B outputs into the readouts fixed in PROTOCOL.md.

Usage: python aggregate.py --results DIR --reported CSV --out DIR
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

N_OBS = (25, 77, 130, 183, 236, 288, 341, 394, 447, 500)
N_RCT = (25, 50, 100)
NATIVE_PANEL = {(512, 128): 1.2463, (512, 512): 1.2158}

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True)
parser.add_argument("--reported", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
res, out = Path(args.results), Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def describe(values):
    v = np.asarray(values, dtype=float)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                min=float(v.min()), max=float(v.max()))


def write(name, rows):
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ paper values (Figure 3)
paper = {}
for r in csv.DictReader(open(args.reported)):
    if r["experiment_id"] != "hatt2022_fig3":
        continue
    m = int(r["panel"].split("=")[1])
    n = int(float(r["x_plotted"]))
    key = {"tau_CorNet": "cornet", "tau_Conf": "conf", "tau_Unc": "unc"}[r["method_paper"]]
    paper[(key, m, n)] = float(r["value"])
assert len(paper) == 90, len(paper)

# ------------------------------------------------------------------ Arm A: network-level curves
arm_a = defaultdict(dict)  # (method, m, n) -> {seed: mean root PEHE over 10 reps}
a_rows, a_meta = [], []
for seed_dir in sorted((res / "author").glob("seed-*")):
    meta = json.loads((seed_dir / "replay.json").read_text())
    data = np.load(seed_dir / "replay.npz")
    sizes = [int(v) for v in data["n_conf_sizes"]]
    assert sizes == list(N_OBS), sizes
    a_meta.append(dict(seed=meta["seed"], elapsed_seconds=round(meta["elapsed_seconds"], 1),
                       gap_mean_square=meta["network_gap_mean_square_on_last_test_draw"]))
    for method in ("cornet", "conf", "unc"):
        for m in N_RCT:
            roots = np.sqrt(data[f"mse_{method}_rct{m}"])  # (10 reps, 10 sizes)
            for j, n in enumerate(N_OBS):
                arm_a[(method, m, n)][meta["seed"]] = float(roots[:, j].mean())
                a_rows.append(dict(seed=meta["seed"], method=method, n_rct=m, n_obs=n,
                                   mean_root_pehe=float(roots[:, j].mean()), sd_root_pehe_ddof0=float(roots[:, j].std())))
write("arm_a_network_curves.csv", a_rows)
write("arm_a_networks.csv", a_meta)

# ------------------------------------------------------------------ Arm B: network-level curves and paired differences
fits = []
for mech_dir in sorted((res / "benchmark").glob("mech-*")):
    fits.extend(csv.DictReader(open(mech_dir / "fits.csv")))
n_fail = sum(r["status"] != "success" for r in fits)
by = defaultdict(list)
pair = defaultdict(dict)
for r in fits:
    if r["status"] != "success":
        continue
    key = (int(r["mechanism_seed"]), int(r["n_obs"]), int(r["n_rct"]), r["standardize"] == "True")
    by[key].append(float(r["root_pehe"]))
    pair[(int(r["mechanism_seed"]), int(r["sample_seed"]), int(r["n_obs"]), int(r["n_rct"]))][r["standardize"] == "True"] = float(r["root_pehe"])
b_rows = [dict(mechanism_seed=k[0], n_obs=k[1], n_rct=k[2], standardize=k[3], n_success=len(v),
               mean_root_pehe=float(np.mean(v))) for k, v in sorted(by.items())]
write("arm_b_network_curves.csv", b_rows)
arm_b = defaultdict(dict)  # (std, m, n) -> {ms: mean}
for k, v in by.items():
    arm_b[(k[3], k[2], k[1])][k[0]] = float(np.mean(v))

# ------------------------------------------------------------------ cell summaries
cells = []
for m in N_RCT:
    for n in N_OBS:
        row = dict(n_rct=m, n_obs=n)
        for method in ("cornet", "conf", "unc"):
            d = describe(list(arm_a[(method, m, n)].values()))
            row.update({f"paper_{method}": paper[(method, m, n)],
                        f"A_{method}_mean": d["mean"], f"A_{method}_sd": d["sd"],
                        f"A_{method}_min": d["min"], f"A_{method}_max": d["max"]})
        for std in (False, True):
            d = describe(list(arm_b[(std, m, n)].values()))
            tag = "B_std" if std else "B_raw"
            row.update({f"{tag}_mean": d["mean"], f"{tag}_sd": d["sd"], f"{tag}_min": d["min"], f"{tag}_max": d["max"],
                        f"{tag}_networks": d["n"]})
        cells.append(row)
write("cell_summary.csv", cells)

# ------------------------------------------------------------------ readouts
readouts = {"arm_b_failures": n_fail, "arm_b_fits": len(fits), "arm_a_networks": len(a_meta)}
r1 = {}
for method in ("cornet", "conf", "unc"):
    inside = [c for c in cells if c[f"A_{method}_min"] <= c[f"paper_{method}"] <= c[f"A_{method}_max"]]
    key = next(c for c in cells if c["n_rct"] == 100 and c["n_obs"] == 500)
    r1[method] = dict(points_within_network_range=len(inside), points=len(cells),
                      key_point=dict(paper=key[f"paper_{method}"], A_mean=key[f"A_{method}_mean"],
                                     A_min=key[f"A_{method}_min"], A_max=key[f"A_{method}_max"]))
readouts["R1_reproduction"] = r1
within = [c for c in cells if abs(c["B_raw_mean"] - c["A_cornet_mean"]) < max(c["B_raw_sd"], c["A_cornet_sd"])]
readouts["R2_translation"] = dict(
    points_within_larger_sd=len(within), points=len(cells),
    mean_difference_B_raw_minus_A=float(np.mean([c["B_raw_mean"] - c["A_cornet_mean"] for c in cells])),
    key_point={k: next(c for c in cells if c["n_rct"] == 100 and c["n_obs"] == 500)[k]
               for k in ("A_cornet_mean", "A_cornet_sd", "B_raw_mean", "B_raw_sd", "B_std_mean", "B_std_sd")},
)
r3 = {}
for n, m in ((512, 128), (512, 512), (500, 25), (500, 50), (500, 100)):
    per_mech = defaultdict(list)
    for (ms, s, nn, mm), v in pair.items():
        if (nn, mm) == (n, m) and len(v) == 2:
            per_mech[ms].append(v[True] - v[False])
    means = [float(np.mean(x)) for x in per_mech.values()]
    r3[f"obs{n}_rct{m}"] = dict(mechanisms=len(means), mean_difference_std_minus_raw=float(np.mean(means)),
                                sd_across_mechanisms=float(np.std(means, ddof=1)),
                                mechanisms_positive=int(sum(x > 0 for x in means)))
readouts["R3_standardization"] = r3
bridge = {}
for (n, m), ref in NATIVE_PANEL.items():
    entry = {"native_panel_reference": ref}
    for std in (False, True):
        d = describe(list(arm_b[(std, m, n)].values()))
        entry["B_std" if std else "B_raw"] = d
        entry[("B_std" if std else "B_raw") + "_mechanism_2025"] = arm_b[(std, m, n)].get(2025)
    bridge[f"obs{n}_rct{m}"] = entry
readouts["bridge"] = bridge
readouts["arm_a_gap_mean_square"] = describe([r["gap_mean_square"] for r in a_meta])
(out / "readouts.json").write_text(json.dumps(readouts, indent=2))
print(json.dumps(readouts, indent=2))
