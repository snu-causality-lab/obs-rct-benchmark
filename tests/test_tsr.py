"""TSR recipe and matched controls.

Most tests replace TabPFN with a small deterministic regressor, so they run without
the optional extra. The last test uses the real TabPFN v2 when it is installed.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from obs_rct_benchmark import methods, tsr
from obs_rct_benchmark.data import SourceSample, TargetCovariateSample, TrainingView


def _view(seed=0, n_obs=80, n_rct=40, d=3, tune_shift=0.0, reference=False):
    rng = np.random.default_rng(seed)

    def sample(n, source, role, shift=0.0):
        x = rng.normal(size=(n, d))
        t = rng.integers(0, 2, size=n).astype(float)
        y = np.sin(x[:, 0]) + t * (1 + x[:, 1]) + (0.5 if source == "obs" else 0.0) * t + 0.1 * rng.normal(size=n) + shift
        return SourceSample(x, t, y, source, role)

    obs = sample(n_obs, "obs", "obs_train")
    rct = sample(n_rct, "rct", "rct_train")
    tune = sample(10, "rct", "rct_tune", shift=tune_shift)
    target = TargetCovariateSample(rng.normal(size=(15, d))) if reference else None
    return TrainingView(obs, rct, tune, target)


class _FakeRegressor:
    """Least squares on [x, t, x*t]; counts fits so cache sharing can be checked."""

    fits = 0

    def fit(self, Z, y):
        type(self).fits += 1
        self.coef = np.linalg.lstsq(self._design(Z), y, rcond=None)[0]
        return self

    def predict(self, Z):
        return self._design(Z) @ self.coef

    @staticmethod
    def _design(Z):
        x, t = Z[:, :-1], Z[:, -1:]
        return np.column_stack([np.ones(len(Z)), x, t, x * t])


@pytest.fixture
def fake_stage1(monkeypatch):
    tsr._STAGE1_CACHE.clear()
    _FakeRegressor.fits = 0
    monkeypatch.setattr(tsr, "_tabpfn_regressor", lambda device: _FakeRegressor())
    monkeypatch.setattr(tsr, "_resolve_device", lambda device: "cpu")
    yield _FakeRegressor
    tsr._STAGE1_CACHE.clear()


def test_gp_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    x, t, y = rng.normal(size=(25, 2)), rng.integers(0, 2, 25), rng.normal(size=25)
    sqd = [(x[:, j][:, None] - x[:, j][None, :]) ** 2 for j in range(2)]
    theta = tsr.default_theta(2) + 0.1 * rng.normal(size=len(tsr.default_theta(2)))
    _, grad = tsr.negative_log_marginal_likelihood(theta, x, t, y, sqd)
    eps = 1e-6
    for i in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[i] += eps
        down[i] -= eps
        fd = (tsr.negative_log_marginal_likelihood(up, x, t, y, sqd)[0]
              - tsr.negative_log_marginal_likelihood(down, x, t, y, sqd)[0]) / (2 * eps)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-5)


def test_gp_effect_mean_matches_an_explicit_loop_kernel():
    rng = np.random.default_rng(2)
    X, T = rng.normal(size=(30, 2)), rng.integers(0, 2, 30).astype(float)
    Y = X[:, 0] + T * X[:, 1] + 0.1 * rng.normal(size=30)
    gp = tsr.LMCGP().fit(X, T, Y)
    parts, log_noise = tsr._unpack(gp.theta, 2)

    def k(a, ta, b, tb):  # written point by point, independently of the vectorized kernel
        total = 0.0
        for log_ls, w, log_kap in parts:
            B = np.outer(w, w) + np.diag(np.exp(log_kap))
            total += B[ta, tb] * np.exp(-0.5 * np.sum(((a - b) / np.exp(log_ls)) ** 2))
        return total

    xs, ts = gp.x, gp.t
    n = len(xs)
    K = np.array([[k(xs[i], ts[i], xs[j], ts[j]) for j in range(n)] for i in range(n)])
    K += (np.exp(log_noise) + tsr._JITTER) * np.eye(n)
    alpha = np.linalg.solve(K, (Y - Y.mean()) / Y.std())
    Q = rng.normal(size=(7, 2))
    qs = (Q - gp.x_mean) / gp.x_scale
    expected = [Y.std() * sum((k(q, 1, xs[j], ts[j]) - k(q, 0, xs[j], ts[j])) * alpha[j] for j in range(n)) for q in qs]
    np.testing.assert_allclose(gp.tau_mean(Q), expected, rtol=1e-8, atol=1e-10)


def test_tsr_is_obs_offset_plus_residual_gp_and_controls_differ_in_one_part(fake_stage1):
    view, ev = _view(), np.random.default_rng(9).normal(size=(20, 3))
    tsr_pred = methods.make_method("tsr_tabpfn_lmcgp").fit_predict(view, ev, seed=1)
    obs_pred = methods.make_method("tabpfn_obs_only").fit_predict(view, ev, seed=2)
    rct_pred = methods.make_method("lmcgp_rct_only").fit_predict(view, ev, seed=3)
    stage1 = tsr.fit_stage1(view.obs_train, device="cpu")
    rct = view.rct_train
    residual_gp = tsr.LMCGP().fit(rct.X, rct.T, rct.Y - stage1.mu(rct.X, rct.T))
    np.testing.assert_allclose(obs_pred, stage1.tau(ev), rtol=0, atol=1e-12)
    np.testing.assert_allclose(tsr_pred, obs_pred + residual_gp.tau_mean(ev), rtol=0, atol=1e-10)
    np.testing.assert_allclose(rct_pred, tsr.LMCGP().fit(rct.X, rct.T, rct.Y).tau_mean(ev), rtol=0, atol=1e-10)
    assert fake_stage1.fits == 1  # one Stage 1 fit shared by TSR, OBS-only and the check above


def test_predictions_ignore_seed_tuning_rows_and_target_reference(fake_stage1):
    ev = np.random.default_rng(4).normal(size=(12, 3))
    for name in ("tsr_tabpfn_lmcgp", "lmcgp_rct_only", "tabpfn_obs_only"):
        base = methods.make_method(name).fit_predict(_view(), ev, seed=0)
        other = methods.make_method(name).fit_predict(_view(tune_shift=50.0, reference=True), ev, seed=123)
        np.testing.assert_array_equal(base, other)
        assert base.shape == (12,) and np.isfinite(base).all()


def test_component_is_fixed_by_method_id_and_options_are_checked():
    with pytest.raises(ValueError, match="fixed by the method ID"):
        methods.make_method("tsr_tabpfn_lmcgp", component="obs")
    with pytest.raises(ValueError, match="device"):
        methods.make_method("lmcgp_rct_only", device="tpu")
    with pytest.raises(TypeError, match="TrainingView"):
        methods.make_method("lmcgp_rct_only").fit_predict(object(), np.zeros((2, 3)), seed=0)


def test_missing_or_wrong_tabpfn_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    with pytest.raises(methods.MethodUnavailableError, match="tsr"):
        tsr._tabpfn_regressor("cpu")
    fake = types.ModuleType("tabpfn")
    fake.__version__ = "9.9.9"
    fake.TabPFNRegressor = object
    monkeypatch.setitem(sys.modules, "tabpfn", fake)
    with pytest.raises(methods.MethodUnavailableError, match="2.0.9"):
        tsr._tabpfn_regressor("cpu")


def test_real_tabpfn_small_fit_is_finite_and_shared():
    pytest.importorskip("tabpfn")
    pytest.importorskip("torch")
    tsr._STAGE1_CACHE.clear()
    view, ev = _view(n_obs=120, n_rct=40), np.random.default_rng(5).normal(size=(10, 3))
    a = methods.make_method("tsr_tabpfn_lmcgp", device="cpu").fit_predict(view, ev, seed=0)
    b = methods.make_method("tabpfn_obs_only", device="cpu").fit_predict(view, ev, seed=7)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert len(tsr._STAGE1_CACHE) == 1
