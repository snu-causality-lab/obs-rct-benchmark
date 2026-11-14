"""Assumption-stress generators: one dial per stated assumption of a fusion method.

Each dial moves one assumption from "holds" (level 0) toward "clearly violated"
(level 3). The trial-population CATE stays identified from the randomized trial
alone, and where a dial changes the form of the OBS bias, the bias variance is held
fixed across levels so that the dial changes the assumption, not the amount of bias.
The first set follows the assumptions stated by CorNet (Hatt et al., 2022,
arXiv:2202.12891v1); docs/V1_STRESS_CORNET.md maps each dial to its source.

Common structure, d = 6, RCT assignment probability 0.5:

    mu0(x) = 0.5 x1 + 0.5 sin(x2) + 0.25 x3^2
    tau(x) = 1 + 0.5 tanh(x1) + 0.5 x4            (same in OBS, trial and target)
    OBS:   Y = mu_T(x) + b_T(x) + sigma(x) z,   T ~ Bernoulli(e_obs(x))
    trial: Y = mu_T(x) + sigma(x) z,            T ~ Bernoulli(0.5)

z ~ N(0, 1), sigma = 1 and e_obs = 0.5 unless a dial sets them. Covariates are
N(0, I) unless a dial changes a law. The target is tau at fresh draws from the trial's
covariate law, except in `trial_shrink` (OBS law) and `target_shift` (a shifted law),
the two extrapolation dials.

g_c below is a bias shape with E[g_c^2] = V = 0.1225 at every level:
level 0 is linear, g = sqrt(V/2) (x1 + x2); levels 1-3 are A_c sin(c (x1 + x2)) with
c = 1, 2, 4 and A_c = sqrt(V / ((1 - exp(-4 c^2)) / 2)).

Dials (`dial`, level 0..3):
  hidden_modifier  b0 = 0, b1 = -0.5 (sqrt(1 - r) x1 + sqrt(r) x4), r = level/3. The bias
                   variance is fixed; at r = 1 the OBS outcomes do not depend on the
                   effect modifier x4.
  bias_complexity  b1 = +g_c, b0 = -g_c: CATE bias 2 g_c of rising frequency.
  common_shift     b1 = b0 = g_c: no CATE bias, but per-arm shifts of rising frequency.
  obs_selection    no outcome shift; e_obs = sigmoid(k x1), k in {0, 1, 2, 4}.
  trial_shrink     b1 = 0.5 x1, b0 = 0; trial covariates N(0, s^2 I), s in
                   {1, 0.75, 0.5, 0.35}; target = OBS law N(0, I).
  obs_size         b1 = 0.5 x1 + 0.25 x2^2, b0 = 0; OBS rows n_obs in {8192, 2048, 512, 128},
                   nested: smaller OBS samples are prefixes of the 8,192-row draw.
  target_shift     b1 = 0.5 x1, b0 = 0; target covariates N(m e1, I), m in {0, 0.5, 1, 1.5}.
  noise_shape      b1 = 0.5 x1, b0 = 0; sigma(x) = exp(gamma x2 / 2 - gamma^2 / 4) in OBS and
                   trial (mean variance 1), gamma in {0, 0.5, 1, 1.5}.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import SimulationProblem, SimulationTruth, SourceSample, TargetCovariateSample

FloatArray = NDArray[np.float64]
STRESS_DGP_VERSION = "2.0.0"
STRESS_DIALS = ("hidden_modifier", "bias_complexity", "common_shift", "obs_selection",
                "trial_shrink", "obs_size", "target_shift", "noise_shape")
EXTRAPOLATION_DIALS = ("trial_shrink", "target_shift")
D = 6
BIAS_VARIANCE = 0.1225
FREQUENCIES = (0.0, 1.0, 2.0, 4.0)
SELECTION = (0.0, 1.0, 2.0, 4.0)
TRIAL_SCALE = (1.0, 0.75, 0.5, 0.35)
OBS_SIZES = (8192, 2048, 512, 128)
TARGET_SHIFT = (0.0, 0.5, 1.0, 1.5)
NOISE_GAMMA = (0.0, 0.5, 1.0, 1.5)


@dataclass(frozen=True)
class StressDGPConfig:
    name: str
    dial: str
    level: int
    n_obs: int = 2048
    n_rct_train: int = 128
    n_rct_tune: int = 32
    n_rct_test: int = 64
    n_eval: int = 2048
    n_target_reference: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if self.dial not in STRESS_DIALS:
            raise ValueError(f"unknown stress dial: {self.dial}")
        if isinstance(self.level, bool) or self.level not in (0, 1, 2, 3):
            raise ValueError("level must be 0, 1, 2 or 3")
        for key in ("n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval", "n_target_reference"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer")
            if value < (0 if key == "n_target_reference" else 4):
                raise ValueError(f"{key} is too small")
        if self.dial == "obs_size" and self.n_obs != OBS_SIZES[self.level]:
            raise ValueError("the obs_size dial sets n_obs to 8192, 2048, 512 or 128 by level")


def mu0(X: FloatArray) -> FloatArray:
    return 0.5 * X[:, 0] + 0.5 * np.sin(X[:, 1]) + 0.25 * X[:, 2] ** 2


def tau(X: FloatArray) -> FloatArray:
    return 1.0 + 0.5 * np.tanh(X[:, 0]) + 0.5 * X[:, 3]


def bias_shape(X: FloatArray, level: int) -> FloatArray:
    """g_c with E[g_c^2] = BIAS_VARIANCE under X ~ N(0, I)."""
    s = X[:, 0] + X[:, 1]
    c = FREQUENCIES[level]
    if c == 0.0:
        return math.sqrt(BIAS_VARIANCE / 2.0) * s
    return math.sqrt(BIAS_VARIANCE / ((1.0 - math.exp(-4.0 * c * c)) / 2.0)) * np.sin(c * s)


def obs_bias(X: FloatArray, cfg: StressDGPConfig) -> tuple[FloatArray, FloatArray]:
    """(b0, b1): the OBS-only shifts of the two arms' outcome means."""
    zero = np.zeros(len(X))
    if cfg.dial == "hidden_modifier":
        r = cfg.level / 3.0
        return zero, -0.5 * (math.sqrt(1.0 - r) * X[:, 0] + math.sqrt(r) * X[:, 3])
    if cfg.dial == "bias_complexity":
        g = bias_shape(X, cfg.level)
        return -g, g
    if cfg.dial == "common_shift":
        g = bias_shape(X, cfg.level)
        return g, g
    if cfg.dial == "obs_selection":
        return zero, zero
    if cfg.dial == "obs_size":
        return zero, 0.5 * X[:, 0] + 0.25 * X[:, 1] ** 2
    return zero, 0.5 * X[:, 0]  # trial_shrink, target_shift, noise_shape


def obs_propensity(X: FloatArray, cfg: StressDGPConfig) -> FloatArray:
    k = SELECTION[cfg.level] if cfg.dial == "obs_selection" else 0.0
    return 1.0 / (1.0 + np.exp(-k * X[:, 0]))


def noise_scale(X: FloatArray, cfg: StressDGPConfig) -> FloatArray:
    gamma = NOISE_GAMMA[cfg.level] if cfg.dial == "noise_shape" else 0.0
    return np.exp(gamma * X[:, 1] / 2.0 - gamma * gamma / 4.0)


def trial_scale(cfg: StressDGPConfig) -> float:
    return TRIAL_SCALE[cfg.level] if cfg.dial == "trial_shrink" else 1.0


def stress_truth_at(X: ArrayLike, cfg: StressDGPConfig) -> dict[str, FloatArray]:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError(f"X must have {D} columns")
    b0, b1 = obs_bias(x, cfg)
    t = tau(x)
    return {"tau_trial": t, "tau_target": t, "mu0": mu0(x), "mu1": mu0(x) + t,
            "confounding_bias_0": b0, "confounding_bias_1": b1, "tau_obs": t + b1 - b0}


def _source(rng: np.random.Generator, rows: int, cfg: StressDGPConfig, *, source: str, role: str) -> SourceSample:
    drawn = OBS_SIZES[0] if (source == "obs" and cfg.dial == "obs_size") else rows
    scale = trial_scale(cfg) if source == "rct" else 1.0
    X = rng.normal(size=(drawn, D)) * scale
    e = obs_propensity(X, cfg) if source == "obs" else np.full(drawn, 0.5)
    T = (rng.uniform(size=drawn) < e).astype(np.float64)
    if source == "obs":
        b0, b1 = obs_bias(X, cfg)
        shift = np.where(T == 1.0, b1, b0)
    else:
        shift = np.zeros(drawn)
    Y = mu0(X) + T * tau(X) + shift + noise_scale(X, cfg) * rng.normal(size=drawn)
    X, T, Y, e = X[:rows], T[:rows], Y[:rows], e[:rows]
    return SourceSample(X, T, Y, source=source, role=role,
                        known_propensity=None if source == "obs" else e)


def evaluation_law(cfg: StressDGPConfig) -> tuple[float, float]:
    """(mean shift of x1, common scale) of the target covariate law."""
    if cfg.dial == "trial_shrink":
        return 0.0, 1.0
    if cfg.dial == "target_shift":
        return TARGET_SHIFT[cfg.level], 1.0
    return 0.0, trial_scale(cfg)


def generate_stress_problem(cfg: StressDGPConfig, seed: int) -> SimulationProblem:
    """Independent streams for OBS, trial training/tuning/test, evaluation and reference rows."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence([seed, 7319]).spawn(6)]
    samples = {
        "obs_train": _source(streams[0], cfg.n_obs, cfg, source="obs", role="obs_train"),
        "rct_train": _source(streams[1], cfg.n_rct_train, cfg, source="rct", role="rct_train"),
        "rct_tune": _source(streams[2], cfg.n_rct_tune, cfg, source="rct", role="rct_tune"),
        "rct_test": _source(streams[3], cfg.n_rct_test, cfg, source="rct", role="rct_test"),
    }
    shift, scale = evaluation_law(cfg)

    def law(rng, rows):
        x = rng.normal(size=(rows, D)) * scale
        x[:, 0] += shift
        return x

    eval_x = law(streams[4], cfg.n_eval)
    truth = stress_truth_at(eval_x, cfg)
    target = TargetCovariateSample(law(streams[5], cfg.n_target_reference)) if cfg.n_target_reference else None
    return SimulationProblem(
        name=cfg.name, seed=seed, **samples, eval_X=eval_x, eval_tau=truth["tau_target"],
        eval_region_score=np.abs(truth["confounding_bias_1"] - truth["confounding_bias_0"]),
        target_reference=target,
        truth=SimulationTruth(eval=truth, obs_train=stress_truth_at(samples["obs_train"].X, cfg),
                              n_eval=cfg.n_eval, n_obs=cfg.n_obs,
                              definition="Closed-form mu0, tau, OBS arm shifts and noise scale; see stress_dgp.py."),
        metadata={"dgp_kind": "stress", "dgp_version": STRESS_DGP_VERSION, "config": asdict(cfg),
                  "target": {"x1_shift": shift, "scale": scale},
                  "implementation": "original assumption-stress construction; not a published DGP"},
    )
