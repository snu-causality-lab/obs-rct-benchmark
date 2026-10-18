"""Information-contract checks with the real optional public implementation."""

from dataclasses import replace
import os

import numpy as np
import pytest

from obs_rct_benchmark.cornet import OfficialCorNet, _official_class
from obs_rct_benchmark.structural_dgp import (
    generate_structural_problem, make_structural_config,
)


def test_checkout_is_required(monkeypatch):
    monkeypatch.delenv("CORNET_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="CORNET_SOURCE"):
        _official_class()


def test_full_problem_is_rejected_before_importing_torch():
    with pytest.raises(TypeError, match="TrainingView"):
        OfficialCorNet().fit_predict(object(), np.zeros((2, 5)), seed=9)


@pytest.mark.skipif(not os.environ.get("CORNET_SOURCE"), reason="external CorNet not configured")
def test_official_recipe_ignores_unused_tuning_labels():
    pytest.importorskip("torch")
    cfg = make_structural_config(
        name="correction-contract", mechanism="M0", n_obs=80,
        n_rct_train=60, n_rct_tune=20, n_rct_test=10, n_eval=16,
    )
    problem = generate_structural_problem(cfg, 77)
    view = problem.training_view()
    altered = replace(view, rct_tune=replace(view.rct_tune, Y=view.rct_tune.Y + 1000.0))
    method = OfficialCorNet(source_epochs=3, correction_epochs=3, width=8)
    first = method.fit_predict(view, problem.eval_X, seed=11)
    second = method.fit_predict(altered, problem.eval_X, seed=11)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (cfg.n_eval,)
    assert np.isfinite(first).all()
