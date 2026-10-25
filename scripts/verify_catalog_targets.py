"""Check all 31 scored CATE targets with a separate equation implementation.

Run this bounded check on a compute node before fitting the full catalog.
It uses the first declared seed and the low-budget condition, not selected
successful fits. No producer truth helper or target-surface helper is called
by the equation checker. CorNet's parameter initialization is shared, while
its forward calculation is implemented separately and that limitation is
recorded explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from obs_rct_benchmark.paper_dgp import PaperDGPConfig, _mechanism
from obs_rct_benchmark.runner import (
    BenchmarkConfig, _generate, build_dgp_config, environment_provenance,
    load_config, source_provenance,
)
from obs_rct_benchmark.structural_dgp import StructuralDGPConfig


PAPER_IDS = (
    "cornet_official_n_conf", "experimental_grounding_kps",
    "integrative_r_learner_kps_normal_target_extension",
    "integrative_r_learner_kps_code", "integrative_r_learner_multivariate_paper",
    "elastic_integrative_gen_data_overlap", "power_likelihood_paper_text",
    "power_likelihood_supplement_code", "cvci_official_linear_bias",
)
PRESETS = (tuple(("structural", f"M{i}") for i in range(12))
           + tuple(("addendum", f"C{i:02d}") for i in range(1, 11))
           + tuple(("paper", name) for name in PAPER_IDS))
ROLE_FIELDS = {"n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval",
               "n_target_reference"}
ABSOLUTE_TOLERANCE = 1e-12


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(str(array.shape).encode("ascii") + array.tobytes()).hexdigest()


def independent_structural_target(X: np.ndarray, config: StructuralDGPConfig) -> np.ndarray:
    """Canonical target equations; source bias and noise do not define CATE."""
    if (config.signal_basis != "axis_aligned" or config.covariate_rho != 0
            or config.covariate_marginals != "gaussian" or config.u_predictability != 0):
        raise ValueError("this checker covers canonical catalog targets only")
    cell = config.mechanism
    if cell == "M1":
        return 1.0 + .5 * X[:, 1]
    if cell == "M8":
        return .5 + 2.25 * (X[:, 0] > NormalDist().inv_cdf(.8))
    if cell == "M9":
        return .5 + 1.5 * (X[:, 0] > NormalDist().inv_cdf(.7))
    if cell in {"M11", "C03"}:
        return np.where(X[:, 1] > 0.0, 2.0, .5)
    if cell == "C05":
        return np.zeros(X.shape[0])
    if cell == "C06":
        return X[:, 1].copy()
    smooth = {"M0", "M2", "M3", "M4", "M5", "M6", "M7", "M10",
              "C01", "C02", "C04", "C07", "C08", "C09", "C10"}
    if cell in smooth:
        return 1.0 + .5 * np.tanh(X[:, 1]) + .25 * np.sin(X[:, 2])
    raise ValueError("unknown canonical structural target")


def independent_cornet_target(X: np.ndarray, parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    """Three explicit layers and arm heads, without _cornet_components."""
    representation = np.asarray(X, dtype=np.float64)
    for layer in range(3):
        weight = np.asarray(parameters[f"rep_weight_{layer}"])
        bias = np.asarray(parameters[f"rep_bias_{layer}"])
        if weight.shape != (X.shape[1], X.shape[1]) or bias.shape != (X.shape[1],):
            raise ValueError("CorNet mechanism has unexpected layer dimensions")
        # The explicit contraction follows the stated linear equation through
        # an independent arithmetic path from the generator's BLAS calls.
        representation = np.einsum("ij,kj->ik", representation, weight) + bias
        if layer < 2:
            representation = np.tanh(representation)
    means = []
    for arm in (0, 1):
        base = np.einsum("ij,j->i", representation, parameters[f"base_weight_{arm}"])
        correction = np.einsum("ij,j->i", representation,
                               parameters[f"correction_weight_{arm}"])
        means.append(base + parameters[f"base_bias_{arm}"][0]
                     + correction + parameters[f"correction_bias_{arm}"][0])
    return means[1] - means[0]


def _check_cornet_parameters(config: PaperDGPConfig, parameters: Mapping[str, np.ndarray]) -> None:
    expected_zeros = int(config.cornet_sparse_ratio * config.d)
    for arm in (0, 1):
        if not math.isclose(float(np.linalg.norm(parameters[f"base_weight_{arm}"])),
                            config.cornet_outcome_norm, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("CorNet base-head norm differs from the declared mechanism")
        weight = parameters[f"correction_weight_{arm}"]
        if np.count_nonzero(weight == 0) != expected_zeros or weight[0] == 0:
            raise ValueError("CorNet correction sparsity differs from the declared mechanism")


def independent_paper_target(X: np.ndarray, config: PaperDGPConfig,
                             mechanism: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    cell = config.paper_id
    if cell == "cornet_official_n_conf":
        parameters = _mechanism(config) if mechanism is None else mechanism
        _check_cornet_parameters(config, parameters)
        return independent_cornet_target(X, parameters)
    if cell == "experimental_grounding_kps":
        return 1.0 + X[:, 0] + .25 * X[:, 0] ** 2
    if cell in {"integrative_r_learner_kps_normal_target_extension",
                "integrative_r_learner_kps_code"}:
        return 1.0 + 2.0 * X[:, 0] + config.int_r_cf * X[:, 0] ** 2
    if cell == "integrative_r_learner_multivariate_paper":
        nonzero = config.d - config.int_r_zero_cate_coefficients
        return 1.0 + np.sum(X[:, :nonzero], axis=1)
    if cell == "elastic_integrative_gen_data_overlap":
        return X[:, 0] + X[:, 1]
    if cell.startswith("power_likelihood_"):
        rho0 = math.tanh(config.power_copula_intercept / 2.0)
        rho1 = math.tanh((config.power_copula_intercept
                          + config.power_copula_treatment_coef) / 2.0)
        C, Z = X[:, 0], X[:, 1]
        return .1 + .1 * C + config.noise_std * (rho1 - rho0) * (Z - 1.0 - C)
    if cell == "cvci_official_linear_bias":
        return np.full(X.shape[0], config.cvci_true_te)
    raise ValueError("unknown paper target")


def _select_cases(config: BenchmarkConfig):
    if config.seeds != tuple(range(20000, 20064)) or len(config.scenarios) != 124:
        raise ValueError("expected the frozen 64-seed, 124-condition catalog study")
    expected = {(kind, name, f"o{obs}_r{trial}") for kind, name in PRESETS
                for obs in (512, 2048) for trial in (128, 512)}
    if {(s.kind, s.id, s.label) for s in config.scenarios} != expected:
        raise ValueError("catalog study does not contain the exact 31 native presets")
    for scenario in config.scenarios:
        role_fields = ROLE_FIELDS - {"n_target_reference"} if scenario.kind == "paper" else ROLE_FIELDS
        if (set(scenario.parameters) != role_fields
                or scenario.parameters["n_rct_tune"] != 32
                or scenario.parameters["n_rct_test"] != 64
                or scenario.parameters["n_eval"] != 2048
                or (scenario.kind != "paper" and scenario.parameters["n_target_reference"] != 0)
                or scenario.label != f"o{scenario.parameters['n_obs']}_r{scenario.parameters['n_rct_train']}"):
            raise ValueError("catalog target check permits only declared role budgets")
    cases = tuple(s for s in config.scenarios if s.label == "o512_r128")
    if {(s.kind, s.id) for s in cases} != set(PRESETS) or len(cases) != 31:
        raise ValueError("the low-budget target check does not cover all 31 presets")
    return cases


def verify_targets(config: BenchmarkConfig) -> list[dict[str, Any]]:
    cases = _select_cases(config)
    rows = []
    for scenario in cases:
        controls = build_dgp_config(scenario)
        problem = _generate(scenario, config.seeds[0])
        expected = (independent_paper_target(problem.eval_X, controls)
                    if scenario.kind == "paper" else
                    independent_structural_target(problem.eval_X, controls))
        actual = np.asarray(problem.eval_tau)
        if (expected.shape != actual.shape or expected.ndim != 1
                or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
            raise ValueError(f"malformed independent target: {scenario.key}")
        maximum = float(np.max(np.abs(expected - actual)))
        if maximum > ABSOLUTE_TOLERANCE:
            raise ValueError(f"independent target differs for {scenario.key}: {maximum:.8g}")
        rows.append({"scenario": scenario.key, "kind": scenario.kind, "preset": scenario.id,
                     "sample_seed": config.seeds[0], "dimension": controls.d,
                     "n_eval": len(actual), "n_obs": controls.n_obs,
                     "n_rct_train": controls.n_rct_train,
                     "resolved_config": asdict(controls),
                     "eval_X_sha256": _array_hash(problem.eval_X),
                     "eval_tau_sha256": _array_hash(actual),
                     "mechanism_sha256": problem.metadata.get("mechanism_sha256"),
                     "max_absolute_target_difference": maximum, "passed": True,
                     "shared_initialization": scenario.id == "cornet_official_n_conf"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/catalog_study.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("target-check output must be new")
    config = load_config(args.config)
    rows = verify_targets(config)
    protocol = Path(__file__).parents[1] / "docs/CATALOG_STUDY.md"
    result = {"schema_version": 1, "scope": "first retained sample seed, low-budget condition of all 31 native presets",
              "source": source_provenance(), "environment": environment_provenance(),
              "study_config_hash": config.config_hash,
              "study_config_file_sha256": _file_hash(args.config),
              "protocol_sha256": _file_hash(protocol),
              "checker_source_sha256": _file_hash(Path(__file__)),
              "target_cases": len(rows), "absolute_tolerance": ABSOLUTE_TOLERANCE,
              "max_absolute_target_difference": max(row["max_absolute_target_difference"] for row in rows),
              "remaining_shared_dependency": "CorNet parameters use the producer's seeded mechanism initialization; its forward pass is independently implemented. Other targets use separate analytic equations. This check does not independently verify every sampler or OBS-discrepancy diagnostic.",
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("target_cases", "absolute_tolerance",
                                                   "max_absolute_target_difference")}))


if __name__ == "__main__":
    main()
