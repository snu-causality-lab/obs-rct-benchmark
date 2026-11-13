"""Two-stage residual (TSR) estimator and its matched controls.

This is a reimplementation, not the authors' code. Gao et al. (2026),
"Observationally Informed Adaptive Causal Experimental Design" (arXiv:2603.03785),
pair an acquisition rule with a two-stage estimator. Only the estimator is
implemented here, as a static recipe; no acquisition is performed.

Stage 1 fits a TabPFN v2 regressor as an S-learner on OBS ``(X, T) -> Y`` and freezes
it. Stage 2 fits a two-output Gaussian process on the RCT training residuals
``Y - mu_obs(X, T)``. The prediction is ``tau_obs(x) + E[tau_residual(x)]``.

Three recipes share the same two components so that each control differs from TSR
in exactly one part:

* ``tsr_tabpfn_lmcgp``: Stage 1 offset plus the residual GP.
* ``lmcgp_rct_only``: the same GP fitted to raw RCT outcomes (no OBS).
* ``tabpfn_obs_only``: Stage 1 alone (no RCT).

Deviations from the paper are fixed here, before any benchmark result: TabPFN v2
(``tabpfn==2.0.9``) instead of TabPFN-2.5, and a linear-model-of-coregionalization GP
of our own instead of the original CMGP code. Stage 1 uses ``random_state=0`` for
every recipe and seed, and the GP fit has no random restarts, so predictions do not
depend on the runner's method seed. RCT tuning rows and target-reference covariates
are not used.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.metadata
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import SourceSample, TrainingView

FloatArray = NDArray[np.float64]
TABPFN_VERSION = "2.0.9"
STAGE1_RANDOM_STATE = 0
GP_MAXITER = 200
_Q = 2
_JITTER = 1e-8


# --------------------------------------------------------------------------------------
# Stage 2: linear model of coregionalization over (x, arm)
#   K((x,a),(x',a')) = sum_q B_q[a,a'] exp(-0.5 sum_d (x_d - x'_d)^2 / l_qd^2)
#   B_q = w_q w_q^T + diag(kappa_q); shared Gaussian noise; ML-II with analytic gradient.
# --------------------------------------------------------------------------------------

def _unpack(theta: FloatArray, d: int):
    parts, k = [], 0
    for _ in range(_Q):
        log_ls = theta[k:k + d]; k += d
        w = theta[k:k + 2]; k += 2
        log_kap = theta[k:k + 2]; k += 2
        parts.append((log_ls, w, log_kap))
    return parts, theta[k]


def default_theta(d: int) -> FloatArray:
    theta: list[float] = []
    for ls, w in ((1.0, (0.8, 0.8)), (2.0, (0.5, -0.5))):
        theta += [float(np.log(ls))] * d + list(w) + [float(np.log(0.05))] * 2
    theta.append(float(np.log(0.3)))
    return np.asarray(theta, dtype=float)


def theta_bounds(d: int) -> list[tuple[float, float]]:
    b: list[tuple[float, float]] = []
    for _ in range(_Q):
        b += [(np.log(0.05), np.log(50.0))] * d + [(-10.0, 10.0)] * 2 + [(np.log(1e-6), np.log(10.0))] * 2
    b.append((np.log(1e-4), np.log(10.0)))
    return b


def _coreg(w: FloatArray, log_kap: FloatArray) -> FloatArray:
    return np.outer(w, w) + np.diag(np.exp(log_kap))


def _rbf(xa: FloatArray, xb: FloatArray, log_ls: FloatArray) -> FloatArray:
    ls = np.exp(log_ls)
    a, b = xa / ls, xb / ls
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.exp(-0.5 * np.maximum(sq, 0.0))


def lmc_kernel(theta: FloatArray, xa: FloatArray, ta: NDArray[np.int64],
               xb: FloatArray, tb: NDArray[np.int64]) -> FloatArray:
    parts, _ = _unpack(theta, xa.shape[1])
    K = np.zeros((len(xa), len(xb)))
    for log_ls, w, log_kap in parts:
        K += _coreg(w, log_kap)[np.ix_(ta, tb)] * _rbf(xa, xb, log_ls)
    return K


def negative_log_marginal_likelihood(theta: FloatArray, x: FloatArray, t: NDArray[np.int64],
                                     y: FloatArray, sqd: list[FloatArray]) -> tuple[float, FloatArray]:
    from scipy.linalg import cho_factor, cho_solve

    n, d = x.shape
    parts, log_noise = _unpack(theta, d)
    noise = np.exp(log_noise)
    Rs, Bs = [], []
    K = np.zeros((n, n))
    for log_ls, w, log_kap in parts:
        R = _rbf(x, x, log_ls)
        B = _coreg(w, log_kap)[np.ix_(t, t)]
        Rs.append(R)
        Bs.append(B)
        K += B * R
    K[np.diag_indices(n)] += noise + _JITTER
    try:
        cf = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta)
    alpha = cho_solve(cf, y, check_finite=False)
    nll = 0.5 * y @ alpha + np.log(np.diag(cf[0])).sum() + 0.5 * n * np.log(2 * np.pi)
    W = np.outer(alpha, alpha) - cho_solve(cf, np.eye(n), check_finite=False)
    g = np.zeros_like(theta)
    k = 0
    for qi, (log_ls, w, log_kap) in enumerate(parts):
        R, B = Rs[qi], Bs[qi]
        M = W * B * R
        ls2 = np.exp(2 * log_ls)
        for dd in range(d):
            g[k + dd] = -0.5 * (M * sqd[dd]).sum() / ls2[dd]
        k += d
        WR = W * R
        for a in range(2):
            ia = (t == a).astype(float)
            E = np.outer(ia, w[t]) + np.outer(w[t], ia)
            g[k + a] = -0.5 * (WR * E).sum()
        k += 2
        kap = np.exp(log_kap)
        for a in range(2):
            ia = t == a
            g[k + a] = -0.5 * kap[a] * WR[np.ix_(ia, ia)].sum()
        k += 2
    g[k] = -0.5 * noise * np.trace(W)
    return float(nll), g


class LMCGP:
    """Two-output GP over (x, arm). Inputs are standardized with the fitted rows' own
    covariate means/scales; outcomes are standardized inside the fit."""

    def __init__(self, maxiter: int = GP_MAXITER):
        self.maxiter = maxiter

    def fit(self, X: FloatArray, T: FloatArray, Y: FloatArray) -> "LMCGP":
        from scipy.linalg import cho_solve
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        self.x_mean = X.mean(0)
        sd = X.std(0)
        self.x_scale = np.where(sd > 1e-12, sd, 1.0)
        self.x = (X - self.x_mean) / self.x_scale
        self.t = np.asarray(T, dtype=int)
        y = np.asarray(Y, dtype=float)
        self.y_mean, self.y_scale = float(y.mean()), float(max(y.std(), 1e-8))
        ys = (y - self.y_mean) / self.y_scale
        self.ys = ys
        d = self.x.shape[1]
        sqd = [(self.x[:, j][:, None] - self.x[:, j][None, :]) ** 2 for j in range(d)]
        result = minimize(negative_log_marginal_likelihood, default_theta(d),
                          args=(self.x, self.t, ys, sqd), jac=True, method="L-BFGS-B",
                          bounds=theta_bounds(d), options={"maxiter": self.maxiter})
        self.theta = np.asarray(result.x, dtype=float)
        self.nlml = float(result.fun)
        self.converged = bool(result.success)
        _, log_noise = _unpack(self.theta, d)
        K = lmc_kernel(self.theta, self.x, self.t, self.x, self.t)
        K[np.diag_indices(len(K))] += float(np.exp(log_noise)) + _JITTER
        L = np.linalg.cholesky(K)
        self.alpha = cho_solve((L, True), ys, check_finite=False)
        return self

    def tau_mean(self, X: FloatArray) -> FloatArray:
        xs = (np.asarray(X, dtype=float) - self.x_mean) / self.x_scale
        n = len(xs)
        k1 = lmc_kernel(self.theta, xs, np.ones(n, dtype=int), self.x, self.t)
        k0 = lmc_kernel(self.theta, xs, np.zeros(n, dtype=int), self.x, self.t)
        return self.y_scale * ((k1 - k0) @ self.alpha)


# --------------------------------------------------------------------------------------
# Stage 1: TabPFN v2 S-learner on OBS, shared across recipes within one process
# --------------------------------------------------------------------------------------

_STAGE1_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_STAGE1_CACHE_SIZE = 2


def _sample_digest(sample: SourceSample, device: str) -> str:
    h = hashlib.sha256()
    for arr in (sample.X, sample.T, sample.Y):
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(f"{device}|{STAGE1_RANDOM_STATE}|{TABPFN_VERSION}".encode())
    return h.hexdigest()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _tabpfn_regressor(device: str) -> Any:
    try:
        import tabpfn
        from tabpfn import TabPFNRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        from .methods import MethodUnavailableError

        raise MethodUnavailableError("TSR recipes need the optional 'tsr' extra (tabpfn==2.0.9)") from exc
    version = getattr(tabpfn, "__version__", None) or importlib.metadata.version("tabpfn")
    if version != TABPFN_VERSION:
        from .methods import MethodUnavailableError

        raise MethodUnavailableError(f"TSR recipes require tabpfn {TABPFN_VERSION}, found {version}")
    return TabPFNRegressor(device=device, random_state=STAGE1_RANDOM_STATE)


class Stage1:
    """Frozen OBS S-learner. ``mu(X, t)`` and ``tau(X)`` use one forward pass each."""

    def __init__(self, model: Any, device: str):
        self.model, self.device = model, device

    def mu(self, X: FloatArray, T: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(np.column_stack([X, T])), dtype=float)

    def tau(self, X: FloatArray) -> FloatArray:
        n = len(X)
        z = np.vstack([np.column_stack([X, np.ones(n)]), np.column_stack([X, np.zeros(n)])])
        p = np.asarray(self.model.predict(z), dtype=float)
        return p[:n] - p[n:]


def fit_stage1(obs: SourceSample, *, device: str = "auto") -> Stage1:
    device = _resolve_device(device)
    key = _sample_digest(obs, device)
    if key in _STAGE1_CACHE:
        _STAGE1_CACHE.move_to_end(key)
        return _STAGE1_CACHE[key]
    model = _tabpfn_regressor(device)
    model.fit(np.column_stack([obs.X, obs.T]), np.asarray(obs.Y, dtype=float))
    fitted = Stage1(model, device)
    _STAGE1_CACHE[key] = fitted
    while len(_STAGE1_CACHE) > _STAGE1_CACHE_SIZE:
        _STAGE1_CACHE.popitem(last=False)
    return fitted


# --------------------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------------------

_COMPONENTS = {"tsr": (True, True), "rct_gp": (False, True), "obs": (True, False)}


class TwoStageResidual:
    """Static recipe: ``component`` selects TSR or one of its matched controls."""

    def __init__(self, component: str, device: str = "auto", gp_maxiter: int = GP_MAXITER):
        if component not in _COMPONENTS:
            raise ValueError(f"unknown TSR component: {component}")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be 'auto', 'cuda' or 'cpu'")
        if int(gp_maxiter) < 1:
            raise ValueError("gp_maxiter must be positive")
        self.component, self.device, self.gp_maxiter = component, device, int(gp_maxiter)
        self.fit_info_: dict[str, Any] = {}

    def fit_predict(self, training: TrainingView, eval_X: FloatArray, *, seed: int) -> FloatArray:
        if not isinstance(training, TrainingView):
            raise TypeError("TSR recipes require a TrainingView")
        del seed  # predictions are deterministic by construction; see module docstring
        use_obs, use_rct = _COMPONENTS[self.component]
        eval_X = np.asarray(eval_X, dtype=float)
        prediction = np.zeros(len(eval_X))
        info: dict[str, Any] = {"component": self.component}
        rct = training.rct_train
        target = np.asarray(rct.Y, dtype=float)
        if use_obs:
            stage1 = fit_stage1(training.obs_train, device=self.device)
            info["stage1_device"] = stage1.device
            prediction = prediction + stage1.tau(eval_X)
            if use_rct:
                target = target - stage1.mu(rct.X, rct.T)
        if use_rct:
            gp = LMCGP(maxiter=self.gp_maxiter).fit(rct.X, rct.T, target)
            info.update(gp_converged=gp.converged, gp_nlml=gp.nlml)
            prediction = prediction + gp.tau_mean(eval_X)
        self.fit_info_ = info
        return np.asarray(prediction, dtype=float)


def tsr_provenance() -> dict[str, Any]:
    """Installed versions for manifest/resume identity, without importing torch/tabpfn."""
    out: dict[str, Any] = {"implementation": "obs_rct_benchmark.tsr (reimplementation; no public author code)",
                           "required_tabpfn": TABPFN_VERSION, "stage1_random_state": STAGE1_RANDOM_STATE,
                           "gp_maxiter_default": GP_MAXITER}
    for package in ("tabpfn", "torch", "scipy"):
        try:
            out[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[f"{package}_version"] = None
    return out
