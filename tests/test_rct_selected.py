"""The v1 core reference: out-of-fold known-propensity selection (docs/V1_CORE_STUDY.md)."""

import json

import numpy as np
import pytest

from obs_rct_benchmark import methods
from obs_rct_benchmark.data import SourceSample, TrainingView
from obs_rct_benchmark.methods import (
    SELECTOR_CANDIDATES, RCTSelected, TLearner, known_propensity_loss,
    treatment_stratified_folds,
)
from obs_rct_benchmark.runner import method_seed


def tau(X):
    return 1.0 + X[:, 0]


def view(n=120, seed=0, propensity=None, tune_shift=0.0, obs_shift=0.0):
    rng = np.random.default_rng(seed)
    def sample(source, role, rows, shift):
        X = rng.normal(size=(rows, 2))
        e = np.full(rows, 0.5) if propensity is None or source == "obs" else propensity(X)
        T = (rng.uniform(size=rows) < e).astype(float)
        Y = X[:, 1] + T * tau(X) + 0.1 * rng.normal(size=rows) + shift
        return SourceSample(X, T, Y, source=source, role=role,
                            known_propensity=None if source == "obs" else e)
    return TrainingView(obs_train=sample("obs", "obs_train", 60, obs_shift),
                        rct_train=sample("rct", "rct_train", n, 0.0),
                        rct_tune=sample("rct", "rct_tune", 32, tune_shift))


class Fixed:
    """A candidate whose prediction is a fixed function of X."""

    def __init__(self, function, fail=False):
        self.function, self.fail = function, fail

    def fit_predict(self, training, X, *, seed):
        if self.fail:
            raise ArithmeticError("planned failure")
        return self.function(np.asarray(X))


def with_candidates(monkeypatch, table):
    monkeypatch.setattr(RCTSelected, "candidate", lambda self, name: table[name])


def test_loss_matches_a_hand_calculation():
    residual = np.array([1.0, -0.5, 2.0])
    T = np.array([1.0, 0.0, 1.0])
    e = np.array([0.5, 0.25, 0.8])
    f = np.array([0.3, 1.0, -1.0])
    hand = [((1.0 - 0.5 * 0.3) ** 2) / 0.25,
            ((-0.5 + 0.25 * 1.0) ** 2) / 0.1875,
            ((2.0 - 0.2 * -1.0) ** 2) / 0.16]
    assert known_propensity_loss(residual, T, e, f) == pytest.approx(hand, abs=1e-12, rel=0)


def test_folds_are_stratified_balanced_and_seeded():
    T = np.array([0.0] * 23 + [1.0] * 17)
    fold = treatment_stratified_folds(T, 5, 9)
    for arm in (0, 1):
        counts = np.bincount(fold[T == arm], minlength=5)
        assert counts.max() - counts.min() <= 1
    assert np.array_equal(fold, treatment_stratified_folds(T, 5, 9))
    assert not np.array_equal(fold, treatment_stratified_folds(T, 5, 10))


def test_choice_and_prediction_ignore_tuning_and_obs_outcomes():
    eval_X = np.random.default_rng(5).normal(size=(40, 2))
    a, b = RCTSelected(), RCTSelected()
    first = a.fit_predict(view(), eval_X, seed=17)
    second = b.fit_predict(view(tune_shift=50.0, obs_shift=-30.0), eval_X, seed=17)
    assert np.array_equal(first, second)
    assert a.fit_info_ == b.fit_info_
    assert a.fit_info_["selected"] in SELECTOR_CANDIDATES
    assert set(a.fit_info_["scores"]) == set(SELECTOR_CANDIDATES)


def test_lower_out_of_fold_loss_wins(monkeypatch):
    with_candidates(monkeypatch, {
        "rct_dr_extra_trees": Fixed(lambda X: tau(X) + 3.0),
        "rct_dr_ridge": Fixed(lambda X: tau(X) - 2.0),
        "rct_t_extra_trees": Fixed(lambda X: tau(X)),
        "rct_t_ridge": Fixed(lambda X: np.zeros(len(X))),
    })
    estimator = RCTSelected()
    eval_X = np.array([[0.0, 0.0], [1.0, 2.0]])
    prediction = estimator.fit_predict(view(n=400), eval_X, seed=3)
    assert estimator.fit_info_["selected"] == "rct_t_extra_trees"
    assert prediction == pytest.approx(tau(eval_X), abs=0)
    assert estimator.fit_info_["max_weight"] == pytest.approx(4.0)


def test_exact_tie_goes_to_the_earlier_candidate(monkeypatch):
    with_candidates(monkeypatch, {name: Fixed(tau) for name in SELECTOR_CANDIDATES})
    estimator = RCTSelected()
    estimator.fit_predict(view(), np.zeros((2, 2)), seed=1)
    assert estimator.fit_info_["selected"] == SELECTOR_CANDIDATES[0]


def test_failed_candidates_are_unavailable_and_all_failing_fails(monkeypatch):
    table = {name: Fixed(tau, fail=True) for name in SELECTOR_CANDIDATES}
    table["rct_t_ridge"] = Fixed(lambda X: tau(X) + 10.0)
    table["rct_dr_ridge"] = Fixed(lambda X: np.full(len(X), np.nan))
    with_candidates(monkeypatch, table)
    estimator = RCTSelected()
    estimator.fit_predict(view(), np.zeros((2, 2)), seed=1)
    info = estimator.fit_info_
    assert info["selected"] == "rct_t_ridge"
    assert info["scores"]["rct_dr_extra_trees"] is None
    assert info["unavailable"]["rct_dr_extra_trees"] == "fold 0: ArithmeticError"
    assert info["unavailable"]["rct_dr_ridge"] == "fold 0: FloatingPointError"
    table["rct_t_ridge"] = Fixed(tau, fail=True)
    with pytest.raises(RuntimeError, match="every rct_selected candidate"):
        estimator.fit_predict(view(), np.zeros((2, 2)), seed=1)
    assert estimator.fit_info_["selected"] is None


def test_chosen_ridge_refit_equals_the_standalone_recipe(monkeypatch):
    real = RCTSelected.candidate
    def only_ridge(self, name):
        return real(self, name) if name == "rct_t_ridge" else Fixed(tau, fail=True)
    monkeypatch.setattr(RCTSelected, "candidate", only_ridge)
    training, eval_X = view(), np.random.default_rng(2).normal(size=(10, 2))
    prediction = RCTSelected().fit_predict(training, eval_X, seed=123)
    standalone = TLearner(source="rct", learner="ridge").fit_predict(training, eval_X, seed=999)
    assert np.array_equal(prediction, standalone)


def test_unclipped_weight_is_reported_for_variable_propensity():
    def e(X):
        return 1.0 / (1.0 + np.exp(-(-1.0 - X[:, 0] + X[:, 1])))
    training = view(propensity=e)
    estimator = RCTSelected()
    estimator.fit_predict(training, np.zeros((3, 2)), seed=4)
    p = training.rct_train.known_propensity
    assert estimator.fit_info_["max_weight"] == pytest.approx(float(np.max(1 / (p * (1 - p)))))


def test_cornet_raw_fixes_standardization_and_shares_the_cornet_seed():
    estimator = methods.make_method("cornet_raw", source_epochs=5)
    assert estimator.standardize is False and estimator.source_epochs == 5
    with pytest.raises(ValueError, match="standardize=False"):
        methods.make_method("cornet_raw", standardize=True)
    for seed in (70000, 70063):
        assert (method_seed("structural:M0@o512_r128", seed, "cornet_raw")
                == method_seed("structural:M0@o512_r128", seed, "cornet_official"))
    assert (method_seed("structural:M0", 70000, "rct_selected")
            != method_seed("structural:M0", 70000, "rct_t_ridge"))


def test_runner_keeps_fit_info_of_a_failed_fit(tmp_path):
    from obs_rct_benchmark.runner import load_config, run_benchmark

    class FailsAfterRecording:
        def fit_predict(self, training, eval_X, *, seed):
            self.fit_info_ = {"selected": None, "unavailable": {"x": "fold 0: ArithmeticError"}}
            raise RuntimeError("planned")

    if "failing_info_probe" not in methods.method_names():
        methods.register_method("failing_info_probe", FailsAfterRecording, provenance="test-only v1")
    (tmp_path / "c.toml").write_text(
        'schema_version = 1\n[run]\npurpose = "smoke"\nseeds = [1]\n'
        'methods = ["rct_t_ridge", "failing_info_probe"]\nreference_method = "rct_t_ridge"\nscope = "test"\n'
        '[[scenarios]]\nkind = "structural"\nid = "M0"\nlabel = "t"\n[scenarios.parameters]\n'
        'n_obs = 40\nn_rct_train = 30\nn_rct_tune = 4\nn_rct_test = 4\nn_eval = 10\n')
    run_benchmark(load_config(tmp_path / "c.toml"), tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out" / "records.jsonl").read_text().splitlines()]
    failed = [r for r in rows if r["method"] == "failing_info_probe"]
    assert failed[0]["status"] == "failed" and failed[0]["fit_info"]["unavailable"] == {"x": "fold 0: ArithmeticError"}
