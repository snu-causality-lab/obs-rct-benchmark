"""Independent query/cost calculations and observed-information boundaries."""

from dataclasses import replace
import json

import numpy as np
import pytest

from obs_rct_benchmark import adaptive
from obs_rct_benchmark.adaptive import (
    GeometryPolicy, OutcomeResidualPolicy, RecruitmentError, RecruitmentSession,
    RecruitmentView, acquired_training, recruit, replay_adaptive_records,
    run_adaptive_problem,
)
from obs_rct_benchmark.adaptive_runner import (
    AdaptiveConfig, load_adaptive_config, run_adaptive_benchmark, verify_adaptive_run,
)
from obs_rct_benchmark.data import SourceSample
from obs_rct_benchmark.runner import ScenarioConfig
from obs_rct_benchmark.structural_dgp import generate_structural_problem, make_structural_config


def problem():
    return generate_structural_problem(make_structural_config(
        name="M0", mechanism="M0", n_obs=24, n_rct_train=48,
        n_rct_tune=4, n_rct_test=4, n_eval=8), 5)


def config():
    return AdaptiveConfig((ScenarioConfig("structural", "M0", {"n_obs": 24, "n_eval": 8}),),
                          seeds=(5,), policies=("uniform", "geometry"),
                          estimators=("rct_t_ridge",), budgets=(16, 24),
                          pilot_size=8, tune_size=4, pool_size=48)


def test_query_boundary_no_replacement_and_exact_charges():
    p = problem()
    session = RecruitmentSession(p.obs_train, p.rct_train, budget=8)
    initial = session.view()
    assert initial.revealed is None and len(initial.revealed_indices) == 0
    assert not hasattr(initial, "T") and not hasattr(initial, "Y")
    assert not hasattr(initial, "eval_X") and not hasattr(initial, "eval_tau")
    assert not hasattr(initial, "truth") and not hasattr(initial, "rct_test")
    assert not initial.pool_X.flags.writeable
    got = session.reveal([3, 1])
    assert session.cost == 2
    np.testing.assert_array_equal(got.Y, p.rct_train.Y[[3, 1]])
    assert session.view().remaining_budget == 6
    assert len(initial.revealed_indices) == 0  # Earlier views do not mutate.
    for bad in ([1], [2, 2], [True], [0.5], [-1], [48]):
        with pytest.raises(ValueError):
            session.reveal(bad)
    assert session.cost == 2
    with pytest.raises(ValueError):
        session.reveal([0, 2, 4, 5, 6, 7, 8])
    assert session.cost == 2


def test_policies_share_pilot_and_fixed_budget_prefixes():
    p = problem()
    orders = [recruit(p, adaptive.make_policy(name), budget=24, pilot_size=8, seed=seed)
              for name, seed in zip(adaptive.POLICY_NAMES, (10, 20, 30))]
    for order in orders:
        assert len(order) == len(set(order.tolist())) == 24
        np.testing.assert_array_equal(order[:8], orders[0][:8])
    for name in adaptive.POLICY_NAMES:
        short = recruit(p, adaptive.make_policy(name), budget=16, pilot_size=8, seed=19)
        long = recruit(p, adaptive.make_policy(name), budget=24, pilot_size=8, seed=19)
        np.testing.assert_array_equal(short, long[:16])


def test_outcome_priority_changes_from_purchased_outcomes_only():
    # Independent symmetric four-candidate example. Zero OBS predictions make
    # a pilot Y=10 yield squared pseudo-outcome residual 400 at p=1/2.
    X = np.array([[-1.], [-.9], [.9], [1.]])
    obs = SourceSample(np.zeros((4, 1)), [0, 0, 1, 1], [0, 0, 0, 0], "obs", "obs_train")
    def view(outcomes):
        trial = SourceSample(X[[0, 3]], [0, 1], outcomes, "rct", "rct_train")
        return RecruitmentView(obs, X, np.array([0, 3]), trial, 2)
    left, right = view([10, 0]), view([0, 10])
    assert OutcomeResidualPolicy().select(left, 1, seed=3).tolist() == [1]
    assert OutcomeResidualPolicy().select(right, 1, seed=3).tolist() == [2]
    np.testing.assert_array_equal(GeometryPolicy().select(left, 2, seed=3),
                                  GeometryPolicy().select(right, 2, seed=3))


def test_policy_never_receives_unqueried_assignments_or_outcomes():
    p = problem()
    class Spy:
        def select(self, view, n, *, seed):
            assert isinstance(view, RecruitmentView)
            assert view.revealed.n == 8
            assert set(view.__dict__) == {"obs_train", "pool_X", "revealed_indices", "revealed", "remaining_budget"}
            assert not hasattr(view, "eval_X") and not hasattr(view, "rct_test")
            np.testing.assert_array_equal(view.revealed.Y, p.rct_train.Y[view.revealed_indices])
            return view.available_indices[:n]
    order = recruit(p, Spy(), budget=16, pilot_size=8, seed=4)
    assert len(order) == 16
    training = acquired_training(p, order, tune_size=4)
    assert training.rct_train.n == 12 and training.rct_tune.n == 4
    assert len(set(order[:4]) & set(order[4:])) == 0


def test_metric_and_cost_replay_match_independent_constant_prediction(monkeypatch):
    class Constant:
        def fit_predict(self, training, eval_X, *, seed):
            assert training.rct_train.n + training.rct_tune.n == 16
            assert training.rct_tune.n == 4
            return np.full(len(eval_X), 2.)
    monkeypatch.setattr(adaptive, "make_method", lambda *a, **k: Constant())
    p = problem()
    rows = run_adaptive_problem(p, budgets=(16,), pilot_size=8, tune_size=4,
                               policies=("uniform",), estimators=("rct_t_ridge",), seed=5)
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    expected = sum((2.-float(t))**2 for t in p.eval_tau)/len(p.eval_tau)
    assert rows[0]["metrics"]["cate_mse"] == pytest.approx(expected)
    assert rows[0]["query_cost"] == rows[0]["train_cost"]+rows[0]["tune_cost"] == 16
    replay_adaptive_records(p, rows, pilot_size=8, tune_size=4)
    altered = dict(rows[0], query_cost=15)
    with pytest.raises(ValueError, match="cost"):
        replay_adaptive_records(p, [altered], pilot_size=8, tune_size=4)
    false_success = dict(rows[0], stage="recruitment", query_indices=rows[0]["query_indices"][:8],
                         query_cost=8, pilot_cost=8, tune_cost=4, train_cost=4)
    with pytest.raises(ValueError, match="recruitment"):
        replay_adaptive_records(p, [false_success], pilot_size=8, tune_size=4)


def test_recruitment_failure_preserves_purchased_pilot_and_failed_cost_checks(monkeypatch):
    class Broken:
        def select(self, view, n, *, seed):
            assert view.revealed.n == 8
            raise RuntimeError("do not export this /private/path")
    monkeypatch.setattr(adaptive, "make_policy", lambda name: Broken())
    p = problem()
    with pytest.raises(RecruitmentError) as info:
        recruit(p, Broken(), budget=16, pilot_size=8, seed=4)
    assert len(info.value.query_indices) == 8
    rows = run_adaptive_problem(p, budgets=(16,), pilot_size=8, tune_size=4,
                               policies=("uniform",), estimators=("rct_t_ridge",), seed=5)
    row = rows[0]
    assert row["status"] == "failed" and row["stage"] == "recruitment"
    assert row["query_cost"] == row["pilot_cost"] == 8
    assert row["train_cost"] == row["tune_cost"] == 4
    assert row["metrics"] is None and "/private" not in json.dumps(row)
    replay_adaptive_records(p, rows, pilot_size=8, tune_size=4)
    with pytest.raises(ValueError, match="cost"):
        replay_adaptive_records(p, [dict(row, train_cost=12)], pilot_size=8, tune_size=4)


def test_terminal_final_method_failure_retains_budget_and_replay_checks_cost(monkeypatch):
    class Broken:
        def fit_predict(self, *a, **k):
            raise RuntimeError("method failed")
    monkeypatch.setattr(adaptive, "make_method", lambda *a, **k: Broken())
    p = problem()
    rows = run_adaptive_problem(p, budgets=(16,), pilot_size=8, tune_size=4,
                               policies=("uniform",), estimators=("rct_t_ridge",), seed=5)
    row = rows[0]
    assert row["stage"] == "method_or_metric" and row["query_cost"] == 16
    replay_adaptive_records(p, rows, pilot_size=8, tune_size=4)
    with pytest.raises(ValueError, match="cost"):
        replay_adaptive_records(p, [dict(row, query_cost=15)], pilot_size=8, tune_size=4)


def test_adaptive_runner_resume_and_independent_replay(tmp_path):
    output = tmp_path/"run"
    c = config()
    result = run_adaptive_benchmark(c, output)
    assert result["status"] == "complete"
    assert result["coverage"]["successful"] == 4
    original = (output/"records.jsonl").read_bytes()
    run_adaptive_benchmark(c, output, resume=True)
    assert (output/"records.jsonl").read_bytes() == original
    verified = verify_adaptive_run(c, output)
    assert verified == {"records": 4, "successful_replayed": 4, "terminal_failures": 0}
    manifest = json.loads((output/"run_manifest.json").read_text())
    assert manifest["config"]["track"] == "adaptive_fixed_pool_v1"
    assert "within_estimator_policy_comparisons" in result
    assert str(tmp_path) not in "".join(x.read_text() for x in output.iterdir())
    with pytest.raises(ValueError, match="provenance"):
        run_adaptive_benchmark(replace(c, budgets=(16,)), output, resume=True)
    rows = (output/"records.jsonl").read_text()
    parsed = [json.loads(line) for line in rows.splitlines()]
    parsed[0].update(stage="recruitment", query_indices=parsed[0]["query_indices"][:8],
                     query_cost=8, pilot_cost=8, tune_cost=4, train_cost=4)
    (output/"records.jsonl").write_text("".join(json.dumps(row)+"\n" for row in parsed))
    with pytest.raises(ValueError, match="recruitment"):
        run_adaptive_benchmark(c, output, resume=True)
    (output/"records.jsonl").write_text(rows+rows.splitlines()[0]+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        run_adaptive_benchmark(c, output, resume=True)


def test_failed_cells_are_not_retried_or_used_for_comparison(tmp_path, monkeypatch):
    class Broken:
        calls = 0
        def fit_predict(self, *a, **k):
            Broken.calls += 1
            raise ValueError("fail")
    monkeypatch.setattr(adaptive, "make_method", lambda *a, **k: Broken())
    output = tmp_path/"failed"
    result = run_adaptive_benchmark(config(), output)
    assert result["coverage"]["failed"] == 4 and Broken.calls == 4
    assert not result["paired_comparisons"]["geometry__rct_t_ridge"]["overall"]["eligible_for_comparison"]
    run_adaptive_benchmark(config(), output, resume=True)
    assert Broken.calls == 4


def test_reference_configuration_and_invalid_contracts():
    from pathlib import Path
    c = load_adaptive_config(Path(__file__).parents[1]/"configs/adaptive_reference.toml")
    assert len(c.seeds) == 32 and len(c.scenarios) == 3
    assert len(c.scenario_keys)*len(c.seeds)*len(c.methods) == 1152
    assert c.purpose == "reference"
    for kwargs in ({"budgets": (True,)}, {"budgets": (16, 16)},
                   {"pilot_size": 16}, {"tune_size": 8}, {"pool_size": 12},
                   {"policies": ("geometry",)}, {"purpose": "scientific_winner"}):
        with pytest.raises(ValueError):
            replace(config(), **kwargs)
