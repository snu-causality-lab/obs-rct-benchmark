"""Synthetic-file adapter checks and independent finite-population truth checks."""

from dataclasses import fields, replace
from fractions import Fraction
import json

import numpy as np
import pytest

from obs_rct_benchmark.datasets import (
    EmpiricalDGPConfig,
    EmpiricalPopulation,
    generate_empirical_problem,
    load_acic2016,
    load_ihdp_csv,
    load_ihdp_npz,
)


SMALL = dict(n_obs=24, n_rct_train=16, n_rct_tune=12, n_rct_test=12, n_eval=32)


def _population():
    return EmpiricalPopulation(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        ("visible", "hidden"),
        mu0=np.array([0, 2, 10, 12.0]),
        mu1=np.array([1, 5, 14, 18.0]),
        provenance={"source_id": "fixture"},
    )


def _config(**changes):
    return EmpiricalDGPConfig(**({"name": "fixture", "hidden_columns": ("hidden",),
                                  "noise_std": 0} | SMALL | changes))


def test_empirical_truth_matches_independent_four_row_enumeration():
    problem = generate_empirical_problem(_population(), _config(), seed=17)
    truth = problem.truth
    group = problem.eval_X[:, 0].astype(int)
    # Each X has two equiprobable hidden values. At e=1/2 and Gamma=2,
    # P(T=1|X,U=low)=1/3 and P(T=1|X,U=high)=2/3. Enumerating their
    # potential means gives the following exact conditional expectations.
    np.testing.assert_allclose(truth.eval["mu0_target"], np.array([1, 11])[group])
    np.testing.assert_allclose(truth.eval["mu1_target"], np.array([3, 16])[group])
    np.testing.assert_allclose(problem.eval_tau, np.array([2, 5])[group])
    np.testing.assert_allclose(truth.eval["mu0_obs_regression"], np.array([2 / 3, 32 / 3])[group])
    np.testing.assert_allclose(truth.eval["mu1_obs_regression"], np.array([11 / 3, 50 / 3])[group])
    np.testing.assert_allclose(truth.eval["confounding_bias_0"], -1 / 3)
    np.testing.assert_allclose(truth.eval["confounding_bias_1"], 2 / 3)
    np.testing.assert_allclose(truth.eval["gamma_required"], 2)
    np.testing.assert_allclose(truth.eval["obs_propensity"], 0.5)
    for arm in (0, 1):
        np.testing.assert_array_equal(truth.eval[f"source_shift_{arm}"], 0)
    p = truth.obs_train["individual_treatment_probability"]
    assert np.isin(np.round(p, 12), np.round([1 / 3, 2 / 3], 12)).all()
    view = problem.training_view()
    assert {field.name for field in fields(view)} == {
        "obs_train", "rct_train", "rct_tune", "target_reference"
    }
    assert view.obs_train.known_propensity is None
    assert not hasattr(view, "truth")
    assert not hasattr(view, "population")
    assert not view.obs_train.X.flags.writeable


def test_unequal_hidden_support_preserves_marginal_and_attains_gamma():
    population = EmpiricalPopulation(
        np.array([[0, 0], [0, 0], [0, 1]], dtype=float), ("visible", "hidden"),
        mu0=np.array([0, 0, 3.0]), mu1=np.array([1, 1, 7.0]),
    )
    cfg = _config(gamma=3, marginal_propensity=0.25)
    problem = generate_empirical_problem(population, cfg, seed=19)
    # Bounds at e=1/4 are [1/10,1/2]. The centered 2:1 support hits
    # the upper bound first: p(low)=1/8, p(high)=1/2; their mean is 1/4.
    np.testing.assert_allclose(problem.truth.eval["gamma_required"], 3)
    np.testing.assert_allclose(problem.truth.eval["mu1_obs_regression"], 5)
    np.testing.assert_allclose(problem.truth.eval["mu0_obs_regression"], 2 / 3)
    p = problem.truth.obs_train["individual_treatment_probability"]
    assert np.isin(np.round(p, 12), [0.125, 0.5]).all()
    assert problem.metadata["diagnostics"]["max_group_marginal_error"] < 1e-14


def test_unique_visible_rows_require_explicit_estimand_choice():
    population = EmpiricalPopulation(np.array([[0, 0], [1, 1], [2, 0], [3, 1.0]]),
                                     ("visible", "hidden"), np.zeros(4), np.arange(4.0))
    with pytest.raises(ValueError, match="no within-visible-group variation"):
        generate_empirical_problem(population, _config(), 1)
    negative = generate_empirical_problem(population, _config(allow_no_hidden_ambiguity=True), 1)
    assert not negative.metadata["diagnostics"]["has_hidden_ambiguity"]
    np.testing.assert_allclose(negative.truth.eval["gamma_required"], 1)
    coarsened = generate_empirical_problem(population, _config(visible_binning=(("visible", 2),)), 1)
    assert coarsened.metadata["diagnostics"]["has_hidden_ambiguity"]
    # Binning creates {0,1} and {2,3}; target effects are their averages.
    np.testing.assert_allclose(coarsened.eval_tau, np.array([0.5, 2.5])[coarsened.eval_X[:, 0].astype(int)])
    assert set(coarsened.metadata["diagnostics"]["visible_binning"]["visible"]) == {
        "requested_bins", "realized_bins", "edges_sha256"
    }


def test_gamma_one_is_negative_control_and_synthetic_cate_is_visible():
    cfg = _config(gamma=1, outcome_model="synthetic", synthetic_cate_intercept=2,
                  synthetic_cate_scale=3, hidden_outcome_loading_0=4,
                  hidden_outcome_loading_1=-7)
    problem = generate_empirical_problem(_population(), cfg, 9)
    # The single visible coordinate standardizes from {0,1} to {-1,+1};
    # the hidden rank is centered conditionally, so it contributes no CATE.
    np.testing.assert_allclose(problem.eval_tau, 2 + 3 * (2 * problem.eval_X[:, 0] - 1))
    np.testing.assert_allclose(problem.truth.eval["confounding_bias_0"], 0, atol=1e-14)
    np.testing.assert_allclose(problem.truth.eval["confounding_bias_1"], 0, atol=1e-14)


def test_gamma_one_subnormal_propensity_preserves_constant_arm_means_exactly():
    population = replace(_population(), mu0=np.full(4, 0.25), mu1=np.full(4, 0.25))
    cfg = _config(gamma=1, marginal_propensity=float(np.nextafter(0.0, 1.0)))
    problem = generate_empirical_problem(population, cfg, 23)
    # At Gamma=1, selection is independent of the hidden row. Multiplying
    # .25 by the smallest subnormal propensity first would underflow to zero;
    # normalized selection weights must preserve the constant exactly.
    for truth in (problem.truth.eval, problem.truth.obs_train):
        for arm in (0, 1):
            np.testing.assert_array_equal(truth[f"mu{arm}_obs_regression"], 0.25)
            np.testing.assert_array_equal(truth[f"confounding_bias_{arm}"], 0.0)


def test_near_one_unequal_support_fails_when_control_mass_is_not_calibrated():
    # Exact arithmetic over the represented float64 probabilities establishes
    # why a check only on the much larger treated mass is insufficient.
    epsilon = Fraction(1, 2**53)
    nominal_control_mass = 6 * epsilon
    low_probability = 1 - 7 * epsilon
    high_probability = 1 - 4 * epsilon
    represented_control_mass = (3 * (1 - low_probability) + (1 - high_probability)) / 4
    assert represented_control_mass == Fraction(25, 4) * epsilon
    assert represented_control_mass / nominal_control_mass == Fraction(25, 24)
    assert Fraction.from_float(float(1 - 3 * 2**-52)) == 1 - nominal_control_mass
    assert Fraction.from_float(float(low_probability)) == low_probability
    assert Fraction.from_float(float(high_probability)) == high_probability

    population = EmpiricalPopulation(
        np.array([[0, 0], [0, 0], [0, 0], [0, 1]], dtype=float),
        ("visible", "hidden"), mu0=np.ones(4), mu1=np.ones(4),
    )
    cfg = _config(gamma=1.5, marginal_propensity=1 - 3 * 2**-52)
    # A constant potential outcome has conditional mean one. Dividing the
    # represented control numerator by nominal mass would instead invent
    # 25/24; the unrepresentable calibrated mechanism must fail closed.
    with pytest.raises(RuntimeError, match="calibration"):
        generate_empirical_problem(population, cfg, 29)


def test_hidden_ambiguity_counts_all_rows_of_a_varying_group():
    population = EmpiricalPopulation(np.array([[0, 0], [0, 1], [0, 2.0]]),
                                     ("visible", "hidden"), np.zeros(3), np.arange(3.0))
    problem = generate_empirical_problem(population, _config(), 4)
    # The middle hidden score equals its group mean, yet its conditional hidden
    # distribution still has three possible values.
    assert problem.metadata["diagnostics"]["hidden_ambiguity_row_fraction"] == 1


def test_role_streams_are_independent_reproducible_and_reference_is_stable():
    cfg = _config()
    first = generate_empirical_problem(_population(), cfg, 3)
    second = generate_empirical_problem(_population(), cfg, 3)
    larger_obs = generate_empirical_problem(_population(), replace(cfg, n_obs=cfg.n_obs + 1), 3)
    with_reference = generate_empirical_problem(_population(), replace(cfg, n_target_reference=7), 3)
    larger_reference = generate_empirical_problem(_population(), replace(cfg, n_target_reference=9), 3)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        np.testing.assert_array_equal(getattr(first, role).Y, getattr(second, role).Y)
    np.testing.assert_array_equal(first.rct_train.Y, larger_obs.rct_train.Y)
    np.testing.assert_array_equal(first.eval_X, with_reference.eval_X)
    np.testing.assert_array_equal(with_reference.target_reference.X, larger_reference.target_reference.X[:7])
    assert {field.name for field in fields(with_reference.target_reference)} == {"X", "role"}
    np.testing.assert_allclose(first.rct_train.known_propensity, 0.5)


def _write_ihdp(path, *, rows=2, replications=2, bad_key=None):
    X = np.zeros((rows, 25, replications))
    X[:, 0, :] = np.arange(rows)[:, None]
    payload = dict(x=X, t=np.zeros((rows, replications)),
                   yf=np.full((rows, replications), 99_999.0),
                   ycf=np.full((rows, replications), -99_999.0),
                   mu0=np.tile(np.arange(replications), (rows, 1)).astype(float),
                   mu1=1 + np.tile(np.arange(replications), (rows, 1)).astype(float))
    if bad_key:
        payload[bad_key] = np.zeros((rows + 1, replications))
    np.savez(path, **payload)


def test_ihdp_npz_selects_one_replication_and_pools_splits(tmp_path):
    train, test = tmp_path / "train.npz", tmp_path / "test.npz"
    _write_ihdp(train)
    _write_ihdp(test, rows=3)
    population = load_ihdp_npz(train, test, replication=1)
    assert population.X_full.shape == (5, 25)
    np.testing.assert_array_equal(population.mu0, 1)
    np.testing.assert_array_equal(population.mu1, 2)
    assert population.provenance["replication"] == 1
    assert population.provenance["pooling"] == "train_test"
    assert [item["role"] for item in population.provenance["files"]] == ["ihdp_train", "ihdp_test"]
    assert str(tmp_path) not in json.dumps(dict(population.provenance))
    assert not hasattr(population, "yf") and not hasattr(population, "t")
    with pytest.raises(ValueError, match="replication index"):
        load_ihdp_npz(train, replication=2)
    _write_ihdp(test, bad_key="ycf")
    with pytest.raises(ValueError, match="ycf must have shape"):
        load_ihdp_npz(train, test)


def test_ihdp_csv_layout_and_reassigned_rct_outcomes(tmp_path):
    values = np.zeros((4, 30))
    values[:, 0] = [0, 1, 0, 1]
    values[:, 1:3] = 99_999
    values[:, 4] = 1  # New mu0=0, mu1=1, regardless of old factual Y.
    values[:, 5] = [0, 0, 1, 1]
    values[:, 6] = [0, 1, 0, 1]
    values[:, 18] = 2  # Preserve CEVAE's historical x14 coding.
    path = tmp_path / "ihdp.csv"
    np.savetxt(path, values, delimiter=",")
    population = load_ihdp_csv(path)
    np.testing.assert_array_equal(population.X_full[:, 13], 2)
    cfg = _config(hidden_columns=tuple(f"x{i}" for i in range(2, 26)))
    problem = generate_empirical_problem(population, cfg, 13)
    for role in ("obs_train", "rct_train", "rct_tune", "rct_test"):
        sample = getattr(problem, role)
        np.testing.assert_array_equal(sample.Y, sample.T)
    np.savetxt(path, values[:, :-1], delimiter=",")
    with pytest.raises(ValueError, match="exactly 30"):
        load_ihdp_csv(path)


def test_acic_encoding_mapping_and_row_alignment(tmp_path):
    x = tmp_path / "x.csv"
    y = tmp_path / "zymu_1.csv"
    x.write_text("id,x1,cat\na,0,A\nb,0,B\nc,1,A\nd,1,B\n")
    y.write_text("id,z,y.0,y.1,mu.0,mu.1\na,0,999,999,0,1\nb,1,999,999,2,5\n"
                 "c,0,999,999,10,14\nd,1,999,999,12,18\n")
    mapping = {"y0": "y.0", "y1": "y.1", "mu0": "mu.0", "mu1": "mu.1"}
    population = load_acic2016(x, y, column_map=mapping, row_id_columns=("id", "id"))
    assert population.feature_names == ("x1", "cat[level_0]", "cat[level_1]")
    assert population.feature_groups == ("x1", "cat", "cat")
    np.testing.assert_array_equal(population.X_full[:, 1:], [[1, 0], [0, 1], [1, 0], [0, 1]])
    problem = generate_empirical_problem(population, _config(hidden_columns=("cat",)), 7)
    assert problem.obs_train.d == 1
    assert problem.metadata["dataset_provenance"]["preprocessing"]["row_alignment"] == "verified_ids"
    assert str(tmp_path) not in json.dumps(dict(problem.metadata))
    with pytest.raises(ValueError, match="response columns"):
        load_acic2016(x, y, row_id_columns=("id", "id"))
    y.write_text(y.read_text().replace("a,0", "b,0", 1))
    with pytest.raises(ValueError, match="row IDs"):
        load_acic2016(x, y, column_map=mapping, row_id_columns=("id", "id"))


def test_adapter_rejects_missing_malformed_and_mixed_columns(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    y.write_text("z,y0,y1,mu0,mu1\n0,0,1,0,1\n1,0,1,0,1\n")
    x.write_text("numeric,mixed\n0,A\n1,1\n")
    with pytest.raises(ValueError, match="mixed numeric/text"):
        load_acic2016(x, y)
    population = load_acic2016(x, y, categorical_columns=("mixed",))
    assert population.X_full.shape == (2, 3)
    x.write_text("numeric,mixed\n0,NA\n1,A\n")
    with pytest.raises(ValueError, match="missing CSV"):
        load_acic2016(x, y)
    x.write_text("numeric,mixed\n0,A\n")
    with pytest.raises(ValueError, match="matching row counts"):
        load_acic2016(x, y)


@pytest.mark.parametrize("kwargs", [
    {"gamma": 0.5}, {"gamma": float("inf")}, {"noise_std": -1},
    {"marginal_propensity": 0}, {"rct_propensity": 1}, {"n_obs": True},
    {"hidden_columns": ()}, {"visible_binning": (("visible", 0),)},
    {"replication": -1},
])
def test_empirical_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


def test_population_provenance_rejects_paths_and_raw_records():
    population = _population()
    with pytest.raises(ValueError, match="provenance accepts"):
        replace(population, provenance={"rows": [[1, 2]]})
    with pytest.raises(ValueError, match="path-free"):
        replace(population, provenance={"source_id": "/private/file"})
    with pytest.raises(ValueError, match="path-free"):
        replace(population, provenance={"files": [{"role": "../x.csv", "sha256": "a" * 64}]})
