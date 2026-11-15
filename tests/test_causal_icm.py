"""Mocked adapter checks; real numerical fidelity is a separate server check."""

from dataclasses import replace
import hashlib
import json
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark import causal_icm as adapter
from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView


def _training():
    x = np.arange(18, dtype=float).reshape(6, 3) / 10
    arm = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    obs = SourceSample(x + 3, arm, np.arange(6) + 20, "obs", "obs_train")
    rct = SourceSample(x, arm, np.arange(6) - 2, "rct", "rct_train")
    tune = SourceSample(x + 6, arm, np.arange(6) + 100, "rct", "rct_tune")
    return TrainingView(obs, rct, tune, TargetCovariateSample(x + 9))


def _locked_checkout(tmp_path, monkeypatch, *, code=None):
    """A synthetic cell, never copied author implementation."""
    code = code or (
        "import GPy\nimport numpy as np\n"
        "def ICM(*args, **kwargs):\n    return ('synthetic-control', 'synthetic-treated')\n"
    )
    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["raise AssertionError('setup executed')"]},
            {"cell_type": "code", "source": [code]},
            {"cell_type": "code", "source": ["raise AssertionError('simulation executed')"]},
        ]
    }
    root = tmp_path / "official"
    root.mkdir()
    path = root / "example.ipynb"
    contents = json.dumps(notebook).encode()
    path.write_bytes(contents)
    monkeypatch.setenv("CAUSAL_ICM_SOURCE", str(root))
    monkeypatch.setattr(adapter, "CAUSAL_ICM_NOTEBOOK", "example.ipynb")
    monkeypatch.setattr(adapter, "CAUSAL_ICM_CELL_INDEX", 1)
    monkeypatch.setattr(adapter, "CAUSAL_ICM_NOTEBOOK_SHA256", hashlib.sha256(contents).hexdigest())
    monkeypatch.setattr(adapter, "CAUSAL_ICM_CELL_SHA256", hashlib.sha256(code.encode()).hexdigest())

    def git(command, **kwargs):
        action = command[3:]
        if action == ["rev-parse", "--show-toplevel"]:
            return str(root) + "\n"
        if action == ["rev-parse", "HEAD"]:
            return adapter.CAUSAL_ICM_REVISION + "\n"
        if action == ["status", "--porcelain", "--untracked-files=no"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(adapter.subprocess, "check_output", git)
    return root, path, git


def test_missing_checkout_and_wrong_information_object_rejected(monkeypatch):
    monkeypatch.delenv("CAUSAL_ICM_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="CAUSAL_ICM_SOURCE"):
        adapter._official_icm()
    with pytest.raises(TypeError, match="TrainingView"):
        adapter.OfficialCausalICM().fit_predict(object(), np.zeros((2, 3)), seed=9)


@pytest.mark.parametrize("options", [
    {"rho": -0.1}, {"rho": 1.1}, {"rho": np.inf}, {"rho": np.nan},
    {"rho": True}, {"rho": "0.5"}, {"rho": None}, {"rho": 1j},
    {"effect_modifier_columns": []}, {"effect_modifier_columns": [0, 0]},
    {"effect_modifier_columns": [-1]}, {"effect_modifier_columns": [True]},
    {"effect_modifier_columns": [1.0]},
])
def test_fixed_recipe_options_rejected(options):
    with pytest.raises(ValueError):
        adapter.OfficialCausalICM(**options)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_seed_rejected_before_optional_loading(monkeypatch, seed):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    with pytest.raises(ValueError, match="seed"):
        adapter.OfficialCausalICM().fit_predict(_training(), np.zeros((2, 3)), seed=seed)


@pytest.mark.parametrize("query", [np.zeros(3), np.zeros((2, 2)), np.zeros((0, 3)),
                                   np.full((2, 3), np.nan)])
def test_query_rejected_before_optional_loading(monkeypatch, query):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    with pytest.raises(ValueError, match="eval_X"):
        adapter.OfficialCausalICM().fit_predict(_training(), query, seed=2)


def test_arm_and_column_boundaries_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    training = _training()
    one_arm = replace(training.obs_train, T=np.zeros(6))
    with pytest.raises(ValueError, match="both treatment arms"):
        adapter.OfficialCausalICM().fit_predict(
            replace(training, obs_train=one_arm), np.zeros((2, 3)), seed=3,
        )
    with pytest.raises(ValueError, match="dimension"):
        adapter.OfficialCausalICM(effect_modifier_columns=(3,)).fit_predict(
            training, np.zeros((2, 3)), seed=3,
        )


def test_native_call_and_trial_target_mapping(monkeypatch):
    calls, queries = [], []

    class Model:
        def __init__(self, arm):
            self.arm = arm

        def predict_noiseless(self, query):
            queries.append(query.copy())
            # A nonzero source coordinate would contaminate the trial target.
            mean = query[:, :1] + self.arm * (2 + query[:, 1:2]) + 100 * query[:, -1:]
            return mean, np.ones_like(mean)

    def native(**kwargs):
        calls.append(kwargs)
        return Model(0), Model(1)

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    training = _training()
    query = np.array([[3., 7., 11.], [-2., 4., 5.]])
    method = adapter.OfficialCausalICM(rho=0.7, effect_modifier_columns=[2, 0])
    result = method.fit_predict(training, query, seed=11)
    np.testing.assert_array_equal(result, 2 + query[:, 0])
    assert result.shape == (2,)
    assert result.dtype == np.float64
    assert method.effect_modifier_columns == (2, 0)
    assert method.provenance_["effect_modifier_columns"] == [2, 0]
    assert method.provenance_["rho_zero_is_rct_only"] is False
    assert len(calls) == 1
    passed = calls[0]
    assert set(passed) == {"X_E", "X_O", "T_E", "T_O", "Y_E", "Y_O", "r", "ID", "AD", "rho"}
    assert (passed["r"], passed["ID"], passed["AD"], passed["rho"]) == (2, 2, [0, 1], .7)
    np.testing.assert_array_equal(passed["X_E"], training.rct_train.X[:, (2, 0)])
    np.testing.assert_array_equal(passed["X_O"], training.obs_train.X[:, (2, 0)])
    for actual, expected in [(passed["Y_E"], training.rct_train.Y),
                             (passed["Y_O"], training.obs_train.Y),
                             (passed["T_E"], training.rct_train.T),
                             (passed["T_O"], training.obs_train.T)]:
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    assert len(queries) == 2
    for actual in queries:
        np.testing.assert_array_equal(actual[:, :-1], query[:, (2, 0)])
        np.testing.assert_array_equal(actual[:, -1], 0)


def test_all_x_is_explicit_default_and_queries_do_not_enter_fit(monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return tuple(SimpleNamespace(predict_noiseless=lambda x: (np.zeros((len(x), 1)), None))
                     for _ in range(2))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    method, training = adapter.OfficialCausalICM(), _training()
    method.fit_predict(training, np.ones((2, 3)), seed=11)
    method.fit_predict(training, np.full((4, 3), 2000), seed=11)
    assert calls[0]["ID"] == 3 and calls[0]["AD"] == [0, 1, 2]
    for key in calls[0]:
        np.testing.assert_array_equal(calls[0][key], calls[1][key])


def test_subset_with_covariate_dependent_randomization_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    training = _training()
    training = replace(training, rct_train=replace(
        training.rct_train, known_propensity=np.linspace(.2, .7, 6)))
    with pytest.raises(ValueError, match="constant RCT assignment"):
        adapter.OfficialCausalICM(effect_modifier_columns=(0,)).fit_predict(
            training, np.zeros((2, 3)), seed=4)


def test_tuning_and_target_reference_cannot_change_native_fit(monkeypatch):
    def native(**kwargs):
        value = sum(np.sum(kwargs[k]) for k in ("X_E", "X_O", "Y_E", "Y_O"))
        return tuple(SimpleNamespace(predict_noiseless=lambda x, a=a: (
            np.full((len(x), 1), a * value), None)) for a in (0, 1))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    training = _training()
    changed = replace(training,
        rct_tune=replace(training.rct_tune, Y=training.rct_tune.Y + 100000),
        target_reference=TargetCovariateSample(np.ones((30, 3)) * 10000))
    query = np.ones((2, 3))
    np.testing.assert_array_equal(
        adapter.OfficialCausalICM().fit_predict(training, query, seed=7),
        adapter.OfficialCausalICM().fit_predict(changed, query, seed=7),
    )


@pytest.mark.parametrize("failure", [False, True])
def test_scoped_native_seed_and_rng_restoration(monkeypatch, failure):
    def native(**kwargs):
        value = np.random.random() + random.random()
        if failure:
            raise RuntimeError("synthetic native failure")
        return tuple(SimpleNamespace(predict_noiseless=lambda x, a=a: (
            np.full((len(x), 1), a * value), None)) for a in (0, 1))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    np_state, py_state = np.random.get_state(), random.getstate()
    method, training, query = adapter.OfficialCausalICM(), _training(), np.zeros((2, 3))
    if failure:
        with pytest.raises(RuntimeError, match="synthetic native failure"):
            method.fit_predict(training, query, seed=17)
    else:
        one = method.fit_predict(training, query, seed=17)
        two = method.fit_predict(training, query, seed=17)
        np.testing.assert_array_equal(one, two)
        assert not np.array_equal(one, method.fit_predict(training, query, seed=18))
    restored = np.random.get_state()
    assert restored[0] == np_state[0] and restored[2:] == np_state[2:]
    np.testing.assert_array_equal(restored[1], np_state[1])
    assert random.getstate() == py_state


@pytest.mark.parametrize("bad_mean,error", [
    (np.zeros(2), ValueError), (np.zeros((1, 1)), ValueError),
    (np.full((2, 1), np.nan), FloatingPointError),
])
def test_native_shape_and_finite_predictions_fail_closed(monkeypatch, bad_mean, error):
    model = SimpleNamespace(predict_noiseless=lambda x: (bad_mean, None))
    monkeypatch.setattr(adapter, "_official_icm", lambda: lambda **kwargs: (model, model))
    with pytest.raises(error):
        adapter.OfficialCausalICM().fit_predict(_training(), np.zeros((2, 3)), seed=1)


def test_source_cell_provenance_and_no_paths_in_manifest(tmp_path, monkeypatch):
    root, _, _ = _locked_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_numerical_provenance", lambda: {"GPy": {"verified": True}})
    provenance = adapter.causal_icm_provenance()
    assert provenance["source_verified"] and provenance["verified"]
    assert provenance["nominal_helper_status"] == "unresolved_merge_markers_not_imported"
    assert str(root) not in json.dumps(provenance)


@pytest.mark.parametrize("changed", ["revision", "tracked", "contents", "top_level", "cell_hash"])
def test_modified_or_misidentified_source_rejected(tmp_path, monkeypatch, changed):
    root, path, git = _locked_checkout(tmp_path, monkeypatch)
    if changed == "contents":
        path.write_text(path.read_text() + " ")
    elif changed == "cell_hash":
        monkeypatch.setattr(adapter, "CAUSAL_ICM_CELL_SHA256", "0" * 64)
    else:
        def altered(command, **kwargs):
            if command[3:] == ["rev-parse", "HEAD"] and changed == "revision":
                return "0" * 40
            if command[3:4] == ["status"] and changed == "tracked":
                return " M other-file.py\n"
            if command[3:] == ["rev-parse", "--show-toplevel"] and changed == "top_level":
                return str(root.parent)
            return git(command, **kwargs)
        monkeypatch.setattr(adapter.subprocess, "check_output", altered)
    assert not adapter._source_snapshot()[0]["source_verified"]
    with pytest.raises(ValueError, match="locked source"):
        adapter._official_icm()


def test_only_locked_cell_executes_and_checkout_initializers_are_ignored(tmp_path, monkeypatch):
    root, _, _ = _locked_checkout(tmp_path, monkeypatch)
    (root / "__init__.py").write_text("raise AssertionError('initializer executed')")
    monkeypatch.setattr(adapter, "_numerical_provenance", lambda: {"GPy": {"verified": True}})
    monkeypatch.setitem(sys.modules, "GPy", ModuleType("GPy"))
    initial_path, initial_bytecode = list(sys.path), sys.dont_write_bytecode
    assert adapter._official_icm()() == ("synthetic-control", "synthetic-treated")
    assert sys.path == initial_path and sys.dont_write_bytecode == initial_bytecode


def test_native_cell_extra_statement_is_rejected_even_with_test_hashes(tmp_path, monkeypatch):
    _locked_checkout(tmp_path, monkeypatch, code=(
        "import GPy\nimport numpy as np\n"
        "raise AssertionError('extra statement executed')\n"
        "def ICM():\n    return None\n"))
    monkeypatch.setattr(adapter, "_numerical_provenance", lambda: {"GPy": {"verified": True}})
    with pytest.raises(ValueError, match="Unexpected executable statements"):
        adapter._official_icm()


def test_numerical_environment_rejected_before_native_import(tmp_path, monkeypatch):
    _locked_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_numerical_provenance", lambda: {"GPy": {"verified": False}})
    with pytest.raises(RuntimeError, match="numerical environment"):
        adapter._official_icm()


@pytest.mark.parametrize("case", ["valid", "wrong_version", "shadowed_spec", "shadowed_cached"])
def test_numerical_distribution_and_import_identity(tmp_path, monkeypatch, case):
    package = tmp_path / "GPy"
    package.mkdir()
    initializer = package / "__init__.py"
    initializer.write_text("# synthetic numerical package\n")
    expected = adapter.CAUSAL_ICM_NUMERICAL_VERSIONS["GPy"]
    distribution = SimpleNamespace(
        version="0.0" if case == "wrong_version" else expected,
        locate_file=lambda path: tmp_path / path,
    )
    monkeypatch.setattr(adapter, "CAUSAL_ICM_NUMERICAL_VERSIONS", {"GPy": expected})
    monkeypatch.setattr(adapter.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda name: SimpleNamespace(
        origin=str(tmp_path / "shadow.py") if case == "shadowed_spec" else str(initializer)))
    monkeypatch.delitem(sys.modules, "GPy", raising=False)
    if case == "shadowed_cached":
        monkeypatch.setitem(sys.modules, "GPy", SimpleNamespace(__file__=str(tmp_path / "other.py")))
    evidence = adapter._numerical_provenance()["GPy"]
    assert evidence["verified"] == (case == "valid")
    assert evidence["init_sha256"] == hashlib.sha256(initializer.read_bytes()).hexdigest()


@pytest.mark.parametrize("component", ["lists_and_dicts", "observable_array", "index_operations"])
@pytest.mark.parametrize("case", ["valid", "different_identity", "wrong_file", "wrong_spec", "wrong_owned_file",
                                  "unlisted_alias", "wrong_name", "wrong_paramz_version"])
def test_only_exact_gpy_paramz_aliases_with_canonical_identity_are_accepted(
        tmp_path, monkeypatch, component, case):
    for name in tuple(sys.modules):
        if name == "GPy" or name.startswith("GPy.") or name == "paramz" or name.startswith("paramz."):
            monkeypatch.delitem(sys.modules, name)
    versions = {name: adapter.CAUSAL_ICM_NUMERICAL_VERSIONS[name] for name in ("GPy", "paramz")}
    for name in versions:
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("# synthetic package\n")
    core = tmp_path / "paramz" / "core"
    core.mkdir()
    canonical_file = core / f"{component}.py"
    canonical_file.write_text("# synthetic compatibility target\n")
    canonical_name = f"paramz.core.{component}"
    canonical = ModuleType(canonical_name)
    canonical.__file__ = str(canonical_file)
    canonical.__spec__ = SimpleNamespace(name=canonical_name, origin=str(canonical_file))
    monkeypatch.setitem(sys.modules, canonical_name, canonical)
    alias_name = f"GPy.core.parameterization.{component}"
    aliased = canonical
    if case == "different_identity":
        aliased = ModuleType(canonical_name)
        aliased.__file__, aliased.__spec__ = canonical.__file__, canonical.__spec__
    elif case == "wrong_file":
        canonical.__file__ = str(tmp_path / "outside.py")
    elif case == "wrong_owned_file":
        canonical.__file__ = str(tmp_path / "GPy" / "__init__.py")
        canonical.__spec__ = SimpleNamespace(name=canonical_name, origin=canonical.__file__)
    elif case == "wrong_spec":
        canonical.__spec__ = SimpleNamespace(name=canonical_name, origin=str(tmp_path / "outside.py"))
    elif case == "unlisted_alias":
        alias_name = "GPy.core.parameterization.unlisted"
    elif case == "wrong_name":
        canonical.__name__ = "paramz.core.counterfeit"
    monkeypatch.setitem(sys.modules, alias_name, aliased)
    monkeypatch.setattr(adapter, "CAUSAL_ICM_NUMERICAL_VERSIONS", versions)

    def distribution(name):
        version = "0.0" if name == "paramz" and case == "wrong_paramz_version" else versions[name]
        return SimpleNamespace(version=version, locate_file=lambda relative: tmp_path / relative)

    monkeypatch.setattr(adapter.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda name: SimpleNamespace(
        origin=str(tmp_path / name / "__init__.py")))
    evidence = adapter._numerical_provenance()
    assert evidence["GPy"]["verified"] == (case == "valid")
    assert evidence["GPy"]["compatibility_aliases"] == adapter.GPY_PARAMZ_ALIASES
    if case == "valid":
        # Merely importing a legitimate alias must not change resume identity.
        monkeypatch.delitem(sys.modules, alias_name)
        monkeypatch.delitem(sys.modules, canonical_name)
        assert adapter._numerical_provenance() == evidence


@pytest.mark.parametrize("source", [None, "full", "both", "trial", 1, True, ["obs"]])
def test_source_control_rejects_ambiguous_source(source):
    with pytest.raises(ValueError, match="source must be"):
        adapter.CausalICMSourceControl(source=source)


@pytest.mark.parametrize("source", ["rct", "obs"])
def test_source_control_excludes_rows_and_preserves_trial_prediction(monkeypatch, source):
    calls, queries = [], []

    class Model:
        def __init__(self, arm):
            self.arm = arm

        def predict_noiseless(self, x):
            queries.append(x.copy())
            return self.arm * (2 + x[:, :1]) + 1000 * x[:, -1:], None

    def native(**kwargs):
        calls.append(kwargs)
        return Model(0), Model(1)

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    training, query = _training(), np.array([[3., 7., 11.], [-2., 4., 5.]])
    included = training.rct_train if source == "rct" else training.obs_train
    method = adapter.CausalICMSourceControl(source=source, rho=.5)
    result = method.fit_predict(training, query, seed=11)
    np.testing.assert_array_equal(result, 2 + query[:, 0])
    assert result.shape == (2,) and result.dtype == np.float64
    assert len(calls) == 1
    passed = calls[0]
    selected, excluded = ("E", "O") if source == "rct" else ("O", "E")
    for field in ("X", "T", "Y"):
        actual = passed[f"{field}_{selected}"]
        expected = getattr(included, field)
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
        assert passed[f"{field}_{excluded}"].shape == ((0, included.d) if field == "X" else (0,))
    assert (passed["r"], passed["ID"], passed["AD"], passed["rho"]) == (2, 3, [0, 1, 2], .5)
    for x in queries:
        np.testing.assert_array_equal(x[:, :-1], query)
        np.testing.assert_array_equal(x[:, -1], 0)
    access = method.provenance_["fitting_sources"]
    assert access[source] == {"availability": "included", "fit_rows": 6,
                              "arm_rows": {"0": 3, "1": 3}}
    assert access["obs" if source == "rct" else "rct"] == {
        "availability": "not_accessed", "fit_rows": 0, "arm_rows": {"0": 0, "1": 0}}
    assert method.provenance_["status"] == "completed"
    assert method.provenance_["prediction_source"] == 0


@pytest.mark.parametrize("source", ["rct", "obs"])
def test_source_control_does_not_read_excluded_source_or_other_roles(monkeypatch, source):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        value = sum(np.sum(kwargs[k]) for k in ("X_E", "X_O", "Y_E", "Y_O"))
        return tuple(SimpleNamespace(predict_noiseless=lambda x, a=a: (
            np.full((len(x), 1), a * value), None)) for a in (0, 1))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    training = _training()
    excluded_field = "obs_train" if source == "rct" else "rct_train"
    excluded = getattr(training, excluded_field)
    changed_sample = replace(excluded, X=excluded.X + 10000,
                             T=np.zeros(excluded.n), Y=excluded.Y - 100000)
    changed = replace(training, **{excluded_field: changed_sample})
    query = np.zeros((2, 3))
    first = adapter.CausalICMSourceControl(source).fit_predict(training, query, seed=7)
    second = adapter.CausalICMSourceControl(source).fit_predict(changed, query, seed=7)
    np.testing.assert_array_equal(first, second)

    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError(f"excluded role was accessed: {name}")

    # Deliberately poison an otherwise validated view to test adapter access,
    # including dimensions and propensities, independently of its constructor.
    for field in (excluded_field, "rct_tune", "target_reference"):
        object.__setattr__(changed, field, Unreadable())
    third = adapter.CausalICMSourceControl(source).fit_predict(changed, query, seed=7)
    np.testing.assert_array_equal(first, third)
    assert len(calls) == 3
    for name in calls[0]:
        np.testing.assert_array_equal(calls[0][name], calls[1][name])
        np.testing.assert_array_equal(calls[0][name], calls[2][name])


@pytest.mark.parametrize("source", ["rct", "obs"])
@pytest.mark.parametrize("rho", [0., .5, 1.])
def test_source_control_uses_row_exclusion_at_every_rho(monkeypatch, source, rho):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return tuple(SimpleNamespace(predict_noiseless=lambda x: (
            np.zeros((len(x), 1)), None)) for _ in (0, 1))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    method = adapter.CausalICMSourceControl(source=source, rho=rho)
    method.fit_predict(_training(), np.zeros((2, 3)), seed=1)
    assert calls[0]["rho"] == rho
    assert len(calls[0]["Y_O" if source == "rct" else "Y_E"]) == 0
    assert method.provenance_["rct_only_by_row_exclusion"] == (source == "rct")


@pytest.mark.parametrize("source", ["rct", "obs"])
def test_source_control_validates_only_included_arm_and_query_contract(monkeypatch, source):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    method = adapter.CausalICMSourceControl(source)
    with pytest.raises(TypeError, match="TrainingView"):
        method.fit_predict(object(), np.zeros((2, 3)), seed=1)
    training = _training()
    included_field = "rct_train" if source == "rct" else "obs_train"
    selected = getattr(training, included_field)
    changed = replace(training, **{included_field: replace(selected, T=np.zeros(selected.n))})
    with pytest.raises(ValueError, match="both treatment arms"):
        method.fit_predict(changed, np.zeros((2, 3)), seed=1)
    with pytest.raises(ValueError, match="included covariates"):
        method.fit_predict(training, np.zeros((2, 4)), seed=1)
    for seed in (-1, 2**32, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            method.fit_predict(training, np.zeros((2, 3)), seed=seed)


@pytest.mark.parametrize("source", ["rct", "obs"])
@pytest.mark.parametrize("failure", [False, True])
def test_source_control_restores_rng_even_if_loading_or_fitting_mutates_it(monkeypatch, source, failure):
    def native(**kwargs):
        value = np.random.random() + random.random()
        if failure:
            raise RuntimeError("synthetic control failure")
        return tuple(SimpleNamespace(predict_noiseless=lambda x, a=a: (
            np.full((len(x), 1), a * value), None)) for a in (0, 1))

    def load():
        np.random.random()
        random.random()
        return native

    monkeypatch.setattr(adapter, "_official_icm", load)
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    np_state, py_state = np.random.get_state(), random.getstate()
    method, training, query = adapter.CausalICMSourceControl(source), _training(), np.zeros((2, 3))
    if failure:
        with pytest.raises(RuntimeError, match="synthetic control failure"):
            method.fit_predict(training, query, seed=17)
        assert method.provenance_["status"] == "failed"
        assert method.provenance_["error_type"] == "RuntimeError"
    else:
        first = method.fit_predict(training, query, seed=17)
        np.testing.assert_array_equal(first, method.fit_predict(training, query, seed=17))
    restored = np.random.get_state()
    assert restored[0] == np_state[0] and restored[2:] == np_state[2:]
    np.testing.assert_array_equal(restored[1], np_state[1])
    assert random.getstate() == py_state


def test_source_control_load_failure_restores_rng_and_retains_attempt(monkeypatch):
    def load():
        np.random.random()
        random.random()
        raise ImportError("synthetic optional dependency failure")

    monkeypatch.setattr(adapter, "_official_icm", load)
    np_state, py_state = np.random.get_state(), random.getstate()
    method = adapter.CausalICMSourceControl("rct")
    with pytest.raises(ImportError, match="synthetic optional"):
        method.fit_predict(_training(), np.zeros((2, 3)), seed=3)
    restored = np.random.get_state()
    assert restored[0] == np_state[0] and restored[2:] == np_state[2:]
    np.testing.assert_array_equal(restored[1], np_state[1])
    assert random.getstate() == py_state
    assert method.provenance_["status"] == "failed"
    assert method.provenance_["error_type"] == "ImportError"
    assert method.provenance_["fitting_sources"]["obs"]["availability"] == "not_accessed"


@pytest.mark.parametrize("source", ["rct", "obs"])
@pytest.mark.parametrize("bad_mean,error", [
    (np.zeros(2), ValueError), (np.zeros((1, 1)), ValueError),
    (np.full((2, 1), np.nan), FloatingPointError),
])
def test_source_control_predictions_fail_closed(monkeypatch, source, bad_mean, error):
    model = SimpleNamespace(predict_noiseless=lambda x: (bad_mean, None))
    monkeypatch.setattr(adapter, "_official_icm", lambda: lambda **kwargs: (model, model))
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    method = adapter.CausalICMSourceControl(source)
    with pytest.raises(error):
        method.fit_predict(_training(), np.zeros((2, 3)), seed=1)
    assert method.provenance_["status"] == "failed"


@pytest.mark.parametrize("source", ["rct", "obs"])
def test_source_control_single_sample_matches_training_view_delegation(monkeypatch, source):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        value = sum(np.sum(kwargs[name]) for name in ("X_E", "X_O", "Y_E", "Y_O"))
        return tuple(SimpleNamespace(predict_noiseless=lambda x, a=a: (
            np.full((len(x), 1), a * value), None)) for a in (0, 1))

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    monkeypatch.setattr(adapter, "causal_icm_provenance", lambda: {"verified": True})
    training, query = _training(), np.zeros((2, 3))
    included = training.rct_train if source == "rct" else training.obs_train
    direct, wrapped = adapter.CausalICMSourceControl(source), adapter.CausalICMSourceControl(source)
    np.testing.assert_array_equal(
        direct.fit_predict_sample(included, query, seed=11),
        wrapped.fit_predict(training, query, seed=11),
    )
    assert direct.provenance_ == wrapped.provenance_
    for name in calls[0]:
        np.testing.assert_array_equal(calls[0][name], calls[1][name])


@pytest.mark.parametrize("source", ["rct", "obs"])
def test_source_control_single_sample_rejects_wrong_type_source_and_role(monkeypatch, source):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    training, query = _training(), np.zeros((2, 3))
    included = training.rct_train if source == "rct" else training.obs_train
    other = training.obs_train if source == "rct" else training.rct_train
    method = adapter.CausalICMSourceControl(source)
    for value in (training, object(), np.zeros((2, 3))):
        with pytest.raises(TypeError, match="SourceSample"):
            method.fit_predict_sample(value, query, seed=1)
    with pytest.raises(ValueError, match="declared source"):
        method.fit_predict_sample(other, query, seed=1)
    for role in ("rct_tune", "rct_test", "eval", "target_reference", "unknown"):
        with pytest.raises(ValueError, match="fitting role"):
            method.fit_predict_sample(replace(included, role=role), query, seed=1)


def _selection_training(n_rct=20, n_obs=30, d=2, rct_y=None):
    rng = np.random.default_rng(5)
    arm_r = np.tile([0., 1.], n_rct // 2)
    arm_o = np.tile([0., 1.], n_obs // 2)
    y_r = rng.normal(size=n_rct) if rct_y is None else np.asarray(rct_y, dtype=float)
    obs = SourceSample(rng.normal(size=(n_obs, d)) + 1, arm_o, rng.normal(size=n_obs) + 5, "obs", "obs_train")
    rct = SourceSample(rng.normal(size=(n_rct, d)), arm_r, y_r, "rct", "rct_train")
    tune = SourceSample(rng.normal(size=(4, d)), np.array([0., 1., 0., 1.]), np.full(4, 99.), "rct", "rct_tune")
    return TrainingView(obs, rct, tune, TargetCovariateSample(rng.normal(size=(3, d))))


class _RhoModel:
    """Predicts a constant set by rho and arm; the source column must be zero."""

    def __init__(self, value, queries):
        self.value, self.queries = value, queries

    def predict_noiseless(self, query):
        self.queries.append(query.copy())
        mean = np.full((query.shape[0], 1), self.value) + 100 * query[:, -1:]
        return mean, np.ones_like(mean)


def _rho_native(calls, queries, control=lambda rho: rho, treated=lambda rho: rho):
    def native(**kwargs):
        calls.append({**kwargs, "rng": np.random.random()})
        rho = kwargs["rho"]
        return _RhoModel(control(rho), queries), _RhoModel(treated(rho), queries)
    return native


def test_selection_picks_the_rho_minimizing_weighted_held_out_error(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(calls, queries))
    training = _selection_training(rct_y=np.full(20, .3))
    method = adapter.SelectedRhoCausalICM()
    result = method.fit_predict(training, np.zeros((3, 2)), seed=7)
    info = method.fit_info_
    assert info["selected_rho"] == .3 and info["status"] == "completed"
    assert info["grid"] == [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    assert info["folds_used"] == 5
    # 11 rho values x 5 folds, then one final fit on all rows at the chosen rho.
    assert len(calls) == 56
    final = calls[-1]
    assert final["rho"] == .3
    np.testing.assert_array_equal(final["X_E"], training.rct_train.X)
    np.testing.assert_array_equal(final["Y_O"], training.obs_train.Y)
    np.testing.assert_array_equal(result, np.zeros(3))
    assert all(np.all(q[:, -1] == 0) for q in queries)


def test_selection_collects_model_cycles_after_each_native_fit(monkeypatch):
    calls, queries, collected = [], [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(calls, queries))
    monkeypatch.setattr(adapter.gc, "collect", lambda: collected.append(len(calls)) or 0)
    adapter.SelectedRhoCausalICM().fit_predict(_selection_training(rct_y=np.full(20, .3)), np.zeros((3, 2)), seed=7)
    # One collection after each of the 55 selection fits, before the next one starts.
    assert collected == list(range(1, 56))


def _label_native(**kwargs):
    """Arm means that depend on every training array the native call receives."""
    rho = kwargs["rho"]
    control = rho * np.mean(kwargs["Y_O"][kwargs["T_O"] == 0])
    treated = np.mean(kwargs["Y_E"][kwargs["T_E"] == 1]) + rho
    return _RhoModel(control, []), _RhoModel(treated, [])


def test_selection_scores_match_a_direct_computation(monkeypatch):
    from sklearn.model_selection import KFold

    monkeypatch.setattr(adapter, "_official_icm", lambda: _label_native)
    weights = np.linspace(1, 3, 20)
    monkeypatch.setattr(adapter, "_participation_weights", lambda r, o: (weights, {"stub": True}))
    training = _selection_training()
    rct, obs = training.rct_train, training.obs_train
    method = adapter.SelectedRhoCausalICM()
    method.fit_predict(training, np.zeros((3, 2)), seed=11)
    splits = list(KFold(5, shuffle=True, random_state=11).split(np.zeros((50, 1))))
    for rho in adapter.CAUSAL_ICM_RHO_GRID:
        per_fold = []
        for train, held in splits:
            rows = held[held < 20]
            if rows.size == 0:
                continue
            tr, to = train[train < 20], train[train >= 20] - 20
            control = rho * np.mean(obs.Y[to][obs.T[to] == 0])
            treated = np.mean(rct.Y[tr][rct.T[tr] == 1]) + rho
            fitted = np.where(rct.T[rows] == 1, treated, control)
            per_fold.append(np.mean(weights[rows] * (rct.Y[rows] - fitted) ** 2))
        assert method.fit_info_["scores"][str(rho)] == pytest.approx(np.mean(per_fold), rel=1e-12)
    best = min(adapter.CAUSAL_ICM_RHO_GRID, key=lambda r: method.fit_info_["scores"][str(r)])
    assert method.fit_info_["selected_rho"] == best


def test_folds_train_on_remaining_pooled_rows_and_score_only_held_out_rct(monkeypatch):
    from sklearn.model_selection import KFold

    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(calls, queries))
    training = _selection_training()
    rct, obs = training.rct_train, training.obs_train
    adapter.SelectedRhoCausalICM().fit_predict(training, np.zeros((3, 2)), seed=3)
    splits = list(KFold(5, shuffle=True, random_state=3).split(np.zeros((50, 1))))
    selection = calls[:-1]
    for k, (train, held) in enumerate(splits):
        for j in range(11):
            call = selection[11 * k + j]
            tr, to = train[train < 20], train[train >= 20] - 20
            for key, expected in [("X_E", rct.X[tr]), ("T_E", rct.T[tr]), ("Y_E", rct.Y[tr]),
                                  ("X_O", obs.X[to]), ("T_O", obs.T[to]), ("Y_O", obs.Y[to])]:
                np.testing.assert_array_equal(call[key], expected)
        held_rct = held[held < 20]
        # Two arm models per fit, each queried at the held-out RCT rows only.
        np.testing.assert_array_equal(queries[22 * k][:, :-1], rct.X[held_rct])


def _same_numpy_state(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


@pytest.mark.parametrize("fail_at", [None, "selection", "final"])
def test_every_native_fit_starts_from_the_same_seeded_state_and_rng_is_restored(monkeypatch, fail_at):
    calls, queries = [], []
    inner = _rho_native(calls, queries)

    def native(**kwargs):
        models = inner(**kwargs)
        if (fail_at == "selection" and len(calls) == 7) or (fail_at == "final" and len(calls) == 56):
            raise RuntimeError("native failure")
        return models

    monkeypatch.setattr(adapter, "_official_icm", lambda: native)
    np.random.seed(123)
    np.random.standard_normal()  # leaves a cached Gaussian in the state
    random.seed(5)
    before, python_before = np.random.get_state(), random.getstate()
    method = adapter.SelectedRhoCausalICM()
    if fail_at is None:
        method.fit_predict(_selection_training(), np.zeros((3, 2)), seed=9)
    else:
        with pytest.raises(RuntimeError, match="native failure"):
            method.fit_predict(_selection_training(), np.zeros((3, 2)), seed=9)
    assert _same_numpy_state(np.random.get_state(), before)
    assert random.getstate() == python_before
    assert {call["rng"] for call in calls} == {np.random.RandomState(9).random_sample()}
    if fail_at == "final":
        assert method.fit_info_["status"] == "final_fit"


def test_ties_choose_the_smaller_rho(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(
        calls, queries, control=lambda rho: 0.0, treated=lambda rho: 0.0))
    method = adapter.SelectedRhoCausalICM()
    method.fit_predict(_selection_training(), np.zeros((3, 2)), seed=2)
    assert method.fit_info_["selected_rho"] == 0.0


def test_selected_rho_final_fit_is_the_fixed_recipe(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(
        calls, queries, control=lambda rho: 0.0, treated=lambda rho: 10 * rho))
    training = _selection_training(rct_y=np.tile([0., 5.], 10))
    selected = adapter.SelectedRhoCausalICM()
    result = selected.fit_predict(training, np.ones((3, 2)), seed=4)
    assert selected.fit_info_["selected_rho"] == .5
    fixed_calls = []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(
        fixed_calls, [], control=lambda rho: 0.0, treated=lambda rho: 10 * rho))
    fixed = adapter.OfficialCausalICM(rho=.5).fit_predict(training, np.ones((3, 2)), seed=4)
    np.testing.assert_array_equal(result, fixed)
    assert calls[-1]["rng"] == fixed_calls[0]["rng"]


def test_selection_failures_keep_the_partial_record(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(
        calls, queries, control=lambda rho: np.nan if rho == .4 else rho))
    method = adapter.SelectedRhoCausalICM()
    with pytest.raises(FloatingPointError):
        method.fit_predict(_selection_training(), np.zeros((3, 2)), seed=1)
    assert method.fit_info_["status"] == "selecting"
    assert method.fit_info_["fold_scores"]["0.3"]


def _fold_seed(n_rct, n_obs, lacks):
    """A seed whose folds put every row of `lacks` (pooled indices) in one held-out fold."""
    from sklearn.model_selection import KFold

    for seed in range(10_000):
        splits = KFold(5, shuffle=True, random_state=seed).split(np.zeros((n_rct + n_obs, 1)))
        if any(set(lacks) <= set(held) for _, held in splits):
            return seed
    raise AssertionError("no such seed")


def test_a_source_may_lack_an_arm_in_a_fold_but_the_pooled_arm_may_not(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(calls, queries))
    training = _selection_training(n_rct=6, n_obs=30)
    rct = replace(training.rct_train, T=np.array([0., 0., 0., 0., 0., 1.]))
    seed = _fold_seed(6, 30, [5])
    method = adapter.SelectedRhoCausalICM()
    method.fit_predict(replace(training, rct_train=rct), np.zeros((3, 2)), seed=seed)
    assert any(not np.any(call["T_E"] == 1) for call in calls[:-1])
    obs = replace(training.obs_train, X=training.obs_train.X[:4], Y=training.obs_train.Y[:4],
                  T=np.array([0., 0., 0., 1.]))
    seed = _fold_seed(6, 4, [5, 9])
    with pytest.raises(ValueError, match="lacks a treatment arm"):
        adapter.SelectedRhoCausalICM().fit_predict(
            replace(training, rct_train=rct, obs_train=obs), np.zeros((3, 2)), seed=seed)


def test_folds_without_held_out_rct_rows_are_skipped(monkeypatch):
    from sklearn.model_selection import KFold

    calls, queries = [], []
    monkeypatch.setattr(adapter, "_official_icm", lambda: _rho_native(calls, queries))
    training = _selection_training(n_rct=4, n_obs=30)
    seed = next(s for s in range(10_000) if sum(
        np.any(held < 4) for _, held in KFold(5, shuffle=True, random_state=s).split(np.zeros((34, 1)))) < 5)
    used = sum(np.any(held < 4) for _, held in KFold(5, shuffle=True, random_state=seed).split(np.zeros((34, 1))))
    method = adapter.SelectedRhoCausalICM()
    method.fit_predict(training, np.zeros((3, 2)), seed=seed)
    assert method.fit_info_["folds_used"] == used < 5
    assert len(calls) == 11 * used + 1
    assert all(len(v) == used for v in method.fit_info_["fold_scores"].values())


def test_selected_recipe_rejects_bad_inputs_before_native_loading(monkeypatch):
    monkeypatch.setattr(adapter, "_official_icm", lambda: pytest.fail("native code loaded"))
    training = _selection_training()
    with pytest.raises(ValueError, match="seed"):
        adapter.SelectedRhoCausalICM().fit_predict(training, np.zeros((3, 2)), seed=-1)
    with pytest.raises(ValueError, match="eval_X"):
        adapter.SelectedRhoCausalICM().fit_predict(training, np.zeros((3, 3)), seed=0)
    one_arm = replace(training.obs_train, T=np.zeros(30))
    with pytest.raises(ValueError, match="both treatment arms"):
        adapter.SelectedRhoCausalICM().fit_predict(
            replace(training, obs_train=one_arm), np.zeros((3, 2)), seed=0)


def test_selected_recipe_shares_the_fixed_recipe_seed():
    from obs_rct_benchmark.runner import method_seed

    for seed in (70000, 70063):
        assert (method_seed("structural:M0@o512_r128", seed, "causal_icm_selected")
                == method_seed("structural:M0@o512_r128", seed, "causal_icm_fixed"))
    assert (method_seed("structural:M0", 70000, "causal_icm_selected")
            != method_seed("structural:M0", 70000, "causal_icm_rct_only"))


def test_participation_weights_are_inverse_default_logistic_probabilities():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    rct_X, obs_X = rng.normal(size=(15, 2)), rng.normal(size=(40, 2)) + .5
    weights, info = adapter._participation_weights(rct_X, obs_X)
    model = LogisticRegression().fit(np.vstack((rct_X, obs_X)), np.r_[np.ones(15), np.zeros(40)])
    np.testing.assert_allclose(weights, 1 / model.predict_proba(rct_X)[:, 1], rtol=1e-12)
    assert info["converged"] is True and info["covariates"] == "all_observed"
    assert weights.min() > 1
