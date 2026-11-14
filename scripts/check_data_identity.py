"""Check that every environment generates the same data, before launch
(docs/V1_CORE_WAVE2.md, docs/V1_STRESS_WAVE2.md).

  dump     Generate every scenario of a configuration at one seed in this environment and
           save every sample role's arrays: OBS, trial training, tuning and test rows,
           evaluation covariates and effects, and target-reference covariates.
  compare  Compare dumps made in other environments with a reference dump.

    python scripts/check_data_identity.py dump --config configs/v1_wave2_gp.toml --seed 70000 --output data-gp.npz
    python scripts/check_data_identity.py compare --reference data-native_r.npz --others data-gp.npz --output data.json

A recipe that uses only trial rows cannot show that OBS rows agree; this check covers every role.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

TOL = 1e-12
ROLES = ("obs_train", "rct_train", "rct_tune", "rct_test")


def dump(config, seed: int, output: Path) -> dict:
    from obs_rct_benchmark.runner import _generate

    arrays = {}
    for s in config.scenarios:
        p = _generate(s, seed)
        for role in ROLES:
            sample = getattr(p, role)
            for name in ("X", "T", "Y"):
                arrays[f"{s.key}|{role}|{name}"] = np.asarray(getattr(sample, name), dtype=float)
            if sample.known_propensity is not None:
                arrays[f"{s.key}|{role}|known_propensity"] = np.asarray(sample.known_propensity, dtype=float)
        arrays[f"{s.key}|eval|X"] = np.asarray(p.eval_X, dtype=float)
        arrays[f"{s.key}|eval|tau"] = np.asarray(p.eval_tau, dtype=float)
        if p.target_reference is not None:
            arrays[f"{s.key}|target_reference|X"] = np.asarray(p.target_reference.X, dtype=float)
    np.savez_compressed(output, **arrays)
    meta = {"seed": seed, "config_hash": config.config_hash, "scenarios": len(config.scenarios),
            "arrays": len(arrays), "numpy": np.__version__, "python": sys.version.split()[0],
            "executable": sys.executable}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def compare(reference: Path, others: list[Path]) -> dict:
    ref = np.load(reference)
    out = {"reference": reference.name, "tolerance": TOL, "environments": {}}
    for path in others:
        got = np.load(path)
        roles, problems = {}, []
        if set(got.files) != set(ref.files):
            problems.append("array names differ from the reference")
        for key in sorted(set(got.files) & set(ref.files)):
            role = key.split("|")[1]
            a, b = ref[key], got[key]
            r = roles.setdefault(role, {"arrays": 0, "exact": 0, "max_abs_difference": 0.0})
            r["arrays"] += 1
            if a.shape != b.shape:
                problems.append(f"{key}: shape {b.shape} differs from {a.shape}")
                continue
            r["exact"] += bool(np.array_equal(a, b))
            if a.size:
                r["max_abs_difference"] = max(r["max_abs_difference"], float(np.max(np.abs(a - b))))
        same = not problems and all(r["max_abs_difference"] <= TOL for r in roles.values())
        out["environments"][path.name] = {"same_data": same, "roles": roles, "problems": problems}
    out["all_same"] = all(e["same_data"] for e in out["environments"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("dump", "compare"))
    ap.add_argument("--config", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--others", type=Path, nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be new")
    if a.check == "dump":
        from obs_rct_benchmark.runner import load_config

        print(json.dumps(dump(load_config(a.config), a.seed, a.output), indent=2))
        return
    result = compare(a.reference, a.others)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:4000])
    raise SystemExit(int(not result["all_same"]))


if __name__ == "__main__":
    main()
