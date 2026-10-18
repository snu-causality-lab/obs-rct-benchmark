"""Method boundaries, independent constant-effect data, and optional provenance."""

from dataclasses import fields
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark.data import SourceSample, TrainingView
from obs_rct_benchmark.methods import (
    DATAFUSIONPPI_COMMIT, DataFusionPPI, MethodUnavailableError, RCTDRLearner,
    TLearner, _verify_dfppi_install,
)
from obs_rct_benchmark import methods


def constant_effect_view():
    X = np.tile(np.arange(8, dtype=float)[:, None], (8, 2))
    T = np.tile([0.0, 1.0], 32)
    def sample(source, role, effect):
        return SourceSample(X=X, T=T, Y=np.where(T == 1, effect, 0.0),
                            source=source, role=role)
    return TrainingView(obs_train=sample("obs", "obs_train", 10),
                        rct_train=sample("rct", "rct_train", 2),
                        rct_tune=sample("rct", "rct_tune", -100))


def test_training_contract_excludes_truth_test_and_outcomes_of_target():
    assert {f.name for f in fields(TrainingView)} == {
        "obs_train", "rct_train", "rct_tune", "target_reference"}
    training = constant_effect_view()
    assert not any(hasattr(training, name) for name in
                   ("eval_tau", "rct_test", "truth", "eval_region_score"))
    assert training.obs_train.known_propensity is None
    assert not training.rct_train.X.flags.writeable


@pytest.mark.parametrize("learner", ["ridge", "extra_trees"])
def test_t_learners_respect_source_and_do_not_tune_on_eval_truth(learner):
    training = constant_effect_view()
    eval_X = np.array([[0, 0], [4, 4]], dtype=float)
    for source, effect in (("rct", 2), ("obs", 10), ("pooled", 6)):
        prediction = TLearner(source=source, learner=learner).fit_predict(
            training, eval_X, seed=31)
        assert prediction == pytest.approx([effect, effect], abs=1e-10)


def test_cross_fitted_dr_recovers_noiseless_constant_effect():
    prediction = RCTDRLearner().fit_predict(constant_effect_view(),
                                           np.zeros((4, 2)), seed=3)
    assert prediction == pytest.approx(np.full(4, 2.0), abs=1e-10)


def test_official_adapter_rejects_modified_sources_and_shadowed_imports(tmp_path, monkeypatch):
    for name in tuple(sys.modules):
        if name == "dfppi" or name.startswith("dfppi."):
            monkeypatch.delitem(sys.modules, name)
    package = tmp_path / "dfppi"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    (package / "api.py").write_text("# fixture algorithm\n")
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in package.glob("*.py")}
    monkeypatch.setattr(methods, "DATAFUSIONPPI_FILE_SHA256", expected)
    class Distribution:
        version = "0.1.0"
        def read_text(self, name):
            return json.dumps({"vcs_info": {"vcs": "git", "commit_id": DATAFUSIONPPI_COMMIT}})
        def locate_file(self, name):
            return package
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    assert _verify_dfppi_install() == package
    # Correct installation metadata cannot bless an edited algorithm file.
    (package / "api.py").write_text("# altered algorithm\n")
    with pytest.raises(MethodUnavailableError, match="unverified"):
        _verify_dfppi_install()
    snapshot = methods.datafusionppi_provenance()
    assert not snapshot["matches_pinned_source"]
    assert snapshot["package_file_sha256"] != expected
    assert str(tmp_path) not in json.dumps(snapshot)
    (package / "api.py").write_text("# fixture algorithm\n")
    # Even verified distribution files cannot bless a shadow package import.
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(tmp_path / "shadow" / "__init__.py")))
    with pytest.raises(MethodUnavailableError, match="unverified"):
        _verify_dfppi_install()
    with pytest.raises(ValueError):
        DataFusionPPI(learner="not-an-official-learner")


def test_methods_reject_a_simulation_object_in_place_of_training_view():
    for method in (TLearner(), RCTDRLearner(), DataFusionPPI()):
        with pytest.raises(TypeError, match="TrainingView"):
            method.fit_predict(object(), np.zeros((2, 2)), seed=0)
