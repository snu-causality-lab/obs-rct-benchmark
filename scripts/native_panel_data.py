"""Frozen native/default panel inputs, trial targets and label accounting.

Understood as: preserve six native mechanisms while producing only the observed
fitting arrays and independently scored trial-population query targets. Native
factories generate four tuning and four test outcomes; those eight outcomes are
discarded. The method view instead aliases two already charged fitting rows in
its ignored compatibility tuning field. This is a study-specific role contract.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import tomllib

import numpy as np


OBSERVED_ARRAYS = ("obs_X", "obs_T", "obs_Y", "rct_X", "rct_T", "rct_Y",
                   "rct_known_propensity", "eval_X")
METHODS = (
    "rct_causal_forest", "rct_t_extra_trees", "rct_dr_extra_trees",
    "obs_t_extra_trees", "naive_pooled_t_extra_trees", "cornet_official",
    "datafusionppi", "integrative_r_linear", "integrative_r_quadratic",
    "causal_icm_fixed", "causal_icm_rct_only",
)
TREE_OPTIONS = {"ridge_alpha": 1.0, "n_estimators": 100, "min_samples_leaf": 5}
METHOD_OPTIONS = {
    **{name: dict(TREE_OPTIONS) for name in METHODS[1:5]},
    "rct_causal_forest": {"num_trees": 2000, "num_threads": 1, "timeout_seconds": 600},
    "cornet_official": {"standardize": True, "source_epochs": 500,
        "correction_epochs": 200, "width": 25, "learning_rate": 0.1,
        "outcome_penalty": 0.1, "correction_penalty": 0.0},
    "datafusionppi": {"learner": "DR", "covariate_shift": True},
    "integrative_r_linear": {"k_folds": 5, "timeout_seconds": 1200},
    "integrative_r_quadratic": {"k_folds": 5, "timeout_seconds": 1200},
    "causal_icm_fixed": {"rho": 0.5},
    "causal_icm_rct_only": {"rho": 0.5},
}
CONDITIONS = (
    {"scenario_key": "structural:M0", "target_law": "N(0,I_5)",
     "target_role": "trial_full_X_CATE", "native_dimension": 5,
     "discrepancy": "none",
     "assumption_tags": ["clean_OBS", "shared_source_CATE", "nonpolynomial_effect"]},
    {"scenario_key": "structural:M2", "target_law": "N(0,I_5)",
     "target_role": "trial_full_X_CATE", "native_dimension": 5,
     "discrepancy": "hidden_treatment_selection_smooth",
     "assumption_tags": ["confounded_OBS", "shared_source_CATE", "nonpolynomial_effect"]},
    {"scenario_key": "structural:M10", "target_law": "N(0,I_5)",
     "target_role": "trial_full_X_CATE", "native_dimension": 5,
     "discrepancy": "hidden_treatment_selection_and_OBS_effect_shift",
     "assumption_tags": ["confounded_OBS", "OBS_effect_equality_violated",
                         "trial_equals_canonical_target", "nonpolynomial_effect"]},
    {"scenario_key": "addendum:C08@trial_target", "target_law": "N(0,I_5)",
     "target_role": "trial_full_X_CATE", "native_dimension": 5,
     "discrepancy": "hidden_participation_trial_effect_shift",
     "assumption_tags": ["unconfounded_OBS", "OBS_effect_equality_violated",
                         "trial_differs_from_canonical_target", "new_trial_target"]},
    {"scenario_key": "paper:integrative_r_learner_kps_code", "target_law": "Uniform[-1,1]",
     "target_role": "trial_full_X_CATE", "native_dimension": 1,
     "discrepancy": "hidden_treatment_selection_linear",
     "assumption_tags": ["confounded_OBS", "shared_source_CATE", "trial_support_covered",
                         "quadratic_IR_effect_basis_correct", "linear_IR_effect_basis_misspecified"]},
    {"scenario_key": "paper:cornet_official_n_conf", "target_law": "N(0,I_25)",
     "target_role": "trial_full_X_CATE", "native_dimension": 25,
     "discrepancy": "OBS_effect_shift_without_treatment_confounding",
     "assumption_tags": ["unconfounded_OBS", "OBS_effect_equality_violated",
                         "native_CorNet_construction_favorable", "nonlinear_effect"]},
)
ANALYSIS = "descriptive_mse_root_pehe_and_exact_win_family_v1"
TRUTH_PRECISION = {"c08_quadrature_order": 256, "target_atol": 1e-12,
                   "target_rtol": 0.0, "native_observed_factory_unchanged": True}
_FIELDS = {"schema_version", "scope", "description", "generator", "n_obs", "rct_budgets",
           "n_eval", "seeds", "preflight_seed", "tune_role", "factory_rct_tune",
           "factory_rct_test", "conditions", "methods", "method_options", "resources", "analysis",
           "truth_precision"}
_RESOURCE_FIELDS = {"status", "default_fit_timeout_seconds", "method_fit_timeout_seconds",
                    "memory_gib", "num_cpus", "num_threads", "preflight_manifest_sha256"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_integer(value):
    return type(value) is int and value > 0


def validate_config(config):
    """Reject drift in the scientific grid; resource caps await measured review."""
    if not isinstance(config, dict) or set(config) != _FIELDS:
        raise ValueError("native panel configuration has missing or unknown fields")
    fixed = {"schema_version": 1, "generator": "native_trial_panel_v1", "n_obs": 512,
             "rct_budgets": [128, 512], "n_eval": 2048,
             "seeds": list(range(50000, 50064)), "preflight_seed": 49999,
             "tune_role": "first_two_training_rows_alias_ignored",
             "factory_rct_tune": 4, "factory_rct_test": 4,
             "conditions": list(CONDITIONS), "methods": list(METHODS),
             "method_options": METHOD_OPTIONS, "analysis": ANALYSIS,
             "truth_precision": TRUTH_PRECISION}
    for name, expected in fixed.items():
        if canonical(config[name]) != canonical(expected):
            raise ValueError(f"native panel has a changed {name}")
    for name in ("scope", "description"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must explicitly describe the development comparison")
    resources = config["resources"]
    if not isinstance(resources, dict) or set(resources) != _RESOURCE_FIELDS:
        raise ValueError("resource configuration has missing or unknown fields")
    if resources["status"] not in {"provisional_pending_preflight", "preflight_reviewed"}:
        raise ValueError("unsupported resource readiness status")
    for name in ("default_fit_timeout_seconds", "num_cpus", "num_threads"):
        if not _positive_integer(resources[name]):
            raise ValueError(f"resource {name} must be a positive integer")
    if resources["num_cpus"] != 1 or resources["num_threads"] != 1:
        raise ValueError("this panel uses one CPU and one numerical thread per fit")
    memory = resources["memory_gib"]
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError("memory_gib must be finite and positive")
    timeouts = resources["method_fit_timeout_seconds"]
    if (not isinstance(timeouts, dict) or set(timeouts) != set(METHODS)
            or any(not _positive_integer(value) for value in timeouts.values())):
        raise ValueError("declare a positive outer fit timeout for every recipe")
    proof = resources["preflight_manifest_sha256"]
    if not isinstance(proof, str) or (proof and not re.fullmatch(r"[0-9a-f]{64}", proof)):
        raise ValueError("invalid preflight manifest SHA256")
    if resources["status"] == "preflight_reviewed" and not proof:
        raise ValueError("reviewed resource status requires a preflight manifest hash")
    canonical(config)
    return config


def load_config(path):
    path = Path(path)
    config = (tomllib.loads(path.read_text()) if path.suffix == ".toml"
              else json.loads(path.read_text()))
    return validate_config(config)


load_panel = load_config


def _case_id(scenario_key, n_rct, seed, *, profile=False):
    stem = scenario_key.replace(":", "__").replace("@", "__")
    return ("preflight_" if profile else "") + f"{stem}_r{n_rct}_s{seed}"


def _make_cases(config, seeds, *, profile=False):
    return [{"id": _case_id(condition["scenario_key"], budget, seed, profile=profile),
             "scenario_key": condition["scenario_key"], "n_rct": budget, "seed": seed}
            for condition in config["conditions"]
            for budget in config["rct_budgets"] for seed in seeds]


def cases(config):
    validate_config(config)
    return _make_cases(config, config["seeds"])


def profile_cases(config):
    validate_config(config)
    return _make_cases(config, [config["preflight_seed"]], profile=True)


def _validate_case(config, case, *, profile=False):
    if not isinstance(case, dict) or set(case) != {"id", "scenario_key", "n_rct", "seed"}:
        raise ValueError("case must declare exactly id, scenario_key, n_rct and seed")
    keys = {row["scenario_key"] for row in config["conditions"]}
    allowed_seeds = [config["preflight_seed"]] if profile else config["seeds"]
    if (case["scenario_key"] not in keys or type(case["n_rct"]) is not int
            or case["n_rct"] not in config["rct_budgets"] or type(case["seed"]) is not int
            or case["seed"] not in allowed_seeds
            or case["id"] != _case_id(case["scenario_key"], case["n_rct"], case["seed"], profile=profile)):
        raise ValueError("case is not in the frozen panel or declared preflight")


def factory_config(config, case, *, profile=False):
    """Native factory parameters; no scientific or dimensional overrides."""
    validate_config(config)
    _validate_case(config, case, profile=profile)
    from obs_rct_benchmark.paper_dgp import make_paper_config
    from obs_rct_benchmark.structural_dgp import make_structural_addendum_config, make_structural_config
    kwargs = {"name": case["scenario_key"], "n_obs": config["n_obs"],
              "n_rct_train": case["n_rct"], "n_rct_tune": config["factory_rct_tune"],
              "n_rct_test": config["factory_rct_test"], "n_eval": config["n_eval"]}
    kind, native_id = case["scenario_key"].split("@", 1)[0].split(":", 1)
    if kind == "paper":
        return make_paper_config(paper_id=native_id, **kwargs)
    if kind == "addendum":
        return make_structural_addendum_config(cell=native_id, **kwargs)
    return make_structural_config(mechanism=native_id, **kwargs)


@lru_cache(maxsize=1)
def c08_selected_mean():
    """Panel-only selected-normal moment, fixed at 256 Hermite nodes.

    Accuracy policy: retain absolute target tolerance 1e-12 against independent
    adaptive integration. Increasing evaluator quadrature changes neither the
    native observed-data sampling nor any retained canonical study's truth.
    The native participation sigmoid clips its input to [-30, 30].
    """
    nodes, weights = np.polynomial.hermite.hermgauss(TRUTH_PRECISION["c08_quadrature_order"])
    z = np.sqrt(2.0) * nodes
    acceptance = 1.0 / (1.0 + np.exp(-np.clip(2.0 * z, -30.0, 30.0)))
    # The common normalizing factor 1/sqrt(pi) cancels in this ratio.
    return float(np.dot(weights, z * acceptance) / np.dot(weights, acceptance))


def generate_case(config, case, *, profile=False):
    """Return flat observed arrays plus evaluator-only ``eval_tau``.

    No method receives this dictionary: the driver must pass OBSERVED_ARRAYS
    alone into its fit subprocess. C08's canonical target is intentionally not
    scored; its newly declared target is the trial conditional effect.
    """
    cfg = factory_config(config, case, profile=profile)
    if case["scenario_key"].startswith("paper:"):
        from obs_rct_benchmark.paper_dgp import generate_paper_problem
        problem = generate_paper_problem(cfg, case["seed"])
    else:
        from obs_rct_benchmark.structural_dgp import generate_structural_problem
        problem = generate_structural_problem(cfg, case["seed"])
    trial, obs = problem.rct_train, problem.obs_train
    tau = problem.eval_tau
    if case["scenario_key"] == "addendum:C08@trial_target":
        tau = problem.eval_tau + 2.0 * c08_selected_mean()
    return {"obs_X": obs.X, "obs_T": obs.T, "obs_Y": obs.Y,
            "rct_X": trial.X, "rct_T": trial.T, "rct_Y": trial.Y,
            "rct_known_propensity": trial.known_propensity,
            "eval_X": problem.eval_X, "eval_tau": np.asarray(tau, dtype=np.float64)}


def make_source_sample(arrays, source):
    from obs_rct_benchmark.data import SourceSample
    if source not in {"obs", "rct"}:
        raise ValueError("source must be obs or rct")
    return SourceSample(arrays[source + "_X"], arrays[source + "_T"], arrays[source + "_Y"],
                        source, source + "_train", known_propensity=(
                            arrays["rct_known_propensity"] if source == "rct" else None))


def make_training_view(arrays):
    from obs_rct_benchmark.data import SourceSample, TrainingView
    return TrainingView(make_source_sample(arrays, "obs"), make_source_sample(arrays, "rct"),
        SourceSample(arrays["rct_X"][:2], arrays["rct_T"][:2], arrays["rct_Y"][:2],
                     "rct", "rct_tune", known_propensity=arrays["rct_known_propensity"][:2]))


def case_method_seed(case, method):
    from obs_rct_benchmark.runner import method_seed
    if method not in METHODS:
        raise ValueError("method is not in the frozen panel")
    label = "causal_icm_fixed" if method == "causal_icm_rct_only" else method
    return method_seed(case["scenario_key"], case["seed"], label)


def case_metadata(config, case, *, profile=False):
    validate_config(config)
    _validate_case(config, case, profile=profile)
    condition = next(row for row in config["conditions"] if row["scenario_key"] == case["scenario_key"])
    return {"condition": json.loads(canonical(condition)),
        "generated_unique_outcomes": {"obs": config["n_obs"], "rct": case["n_rct"] + 8},
        "method_view_unique_outcomes": {"obs": config["n_obs"], "rct": case["n_rct"]},
        "discarded_factory_outcomes": {"rct_tune": 4, "rct_test": 4},
        "compatibility_tuning_rows": 2, "compatibility_tuning_unique_outcomes": 0,
        "target_reference_rows": 0, "query_rows": config["n_eval"],
        "query_outcomes_generated": 0, "profile": profile,
        "rct_budget_pairing": "same_native_seed_streams_not_claimed_nested",
        "target_mapping": ("tau_rct_new_trial_target" if case["scenario_key"] == "addendum:C08@trial_target"
                           else "native_eval_tau_equals_trial_CATE"),
        "truth_precision": {"target_atol": TRUTH_PRECISION["target_atol"],
            "target_rtol": TRUTH_PRECISION["target_rtol"],
            "selected_mean_quadrature_order": (TRUTH_PRECISION["c08_quadrature_order"]
                if case["scenario_key"] == "addendum:C08@trial_target" else None),
            "native_observed_factory_unchanged": True}}


def array_identity(arrays):
    return {name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
                   "sha256": hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()}
            for name, value in sorted(arrays.items())}
