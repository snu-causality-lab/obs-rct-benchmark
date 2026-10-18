"""Role-separated data objects.

Estimators receive :class:`TrainingView`, never the simulation truth stored in
:class:`SimulationProblem`. Arrays are copied and made read-only so an adapter
cannot accidentally mutate a shared replicate in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ESTIMATOR_CONSUMED_ROLES = frozenset(
    {
        "obs_train",
        "rct_train",
        "rct_tune",
        "target_reference",
    }
)


def _readonly_float(values: ArrayLike, *, ndim: int) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("arrays must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SourceSample:
    """One observed sample with an explicit source and role."""

    X: FloatArray
    T: FloatArray
    Y: FloatArray
    source: str
    role: str
    known_propensity: FloatArray | None = None

    def __post_init__(self) -> None:
        X = _readonly_float(self.X, ndim=2)
        T = _readonly_float(self.T, ndim=1)
        Y = _readonly_float(self.Y, ndim=1)
        if X.shape[0] != T.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError("X, T, and Y must have the same number of rows")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("samples must not be empty")
        if not np.isin(T, (0.0, 1.0)).all():
            raise ValueError("T must be binary")
        if self.source not in {"obs", "rct"}:
            raise ValueError(f"unknown source: {self.source}")
        propensity = self.known_propensity
        if self.source == "obs" and propensity is not None:
            raise ValueError(
                "observational treatment propensities are not exposed to estimators"
            )
        if self.source == "rct":
            if propensity is None:
                propensity = np.full(X.shape[0], 0.5, dtype=np.float64)
            propensity = _readonly_float(propensity, ndim=1)
            if propensity.shape != T.shape:
                raise ValueError("known RCT propensities must match treatment rows")
            if ((propensity <= 0.0) | (propensity >= 1.0)).any():
                raise ValueError("known RCT propensities must lie strictly in (0,1)")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "known_propensity", propensity)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class TargetCovariateSample:
    """Outcome-free covariates representing the declared target population."""

    X: FloatArray
    role: str = "target_reference"

    def __post_init__(self) -> None:
        X = _readonly_float(self.X, ndim=2)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("target-reference covariates must not be empty")
        if self.role != "target_reference":
            raise ValueError("target covariates must have role='target_reference'")
        object.__setattr__(self, "X", X)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class TrainingView:
    """The complete information contract available to an estimator."""

    obs_train: SourceSample
    rct_train: SourceSample
    rct_tune: SourceSample
    target_reference: TargetCovariateSample | None = None

    def __post_init__(self) -> None:
        samples = (
            (self.obs_train, "obs", "obs_train"),
            (self.rct_train, "rct", "rct_train"),
            (self.rct_tune, "rct", "rct_tune"),
        )
        dimensions = {sample.d for sample, _, _ in samples}
        if self.target_reference is not None:
            if not isinstance(self.target_reference, TargetCovariateSample):
                raise ValueError(
                    "target_reference must be an outcome-free "
                    "TargetCovariateSample"
                )
            dimensions.add(self.target_reference.d)
        if len(dimensions) != 1:
            raise ValueError("all training sources must have the same covariates")
        for sample, expected_source, expected_role in samples:
            if sample.source != expected_source or sample.role != expected_role:
                raise ValueError(
                    f"{expected_role} must have source={expected_source!r} "
                    f"and role={expected_role!r}"
                )


def _readonly_array_mapping(
    values: Mapping[str, ArrayLike], *, rows: int, label: str
) -> Mapping[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for key, raw in values.items():
        if not key or not isinstance(key, str):
            raise ValueError(f"{label} truth keys must be nonempty strings")
        array = _readonly_float(raw, ndim=1)
        if array.shape[0] != rows:
            raise ValueError(
                f"{label} truth {key!r} has {array.shape[0]} rows; expected {rows}"
            )
        result[key] = array
    return MappingProxyType(result)


@dataclass(frozen=True)
class SimulationTruth:
    """Evaluation-only structural truth.

    The flexible mappings let DGP assets expose mechanism-specific oracle
    quantities without widening :class:`TrainingView`. Normal estimators never
    receive this object; explicitly labeled oracle diagnostics may consume it.
    """

    eval: Mapping[str, FloatArray]
    obs_train: Mapping[str, FloatArray]
    n_eval: int
    n_obs: int
    definition: str

    def __post_init__(self) -> None:
        if self.n_eval < 1 or self.n_obs < 1:
            raise ValueError("truth row counts must be positive")
        object.__setattr__(
            self,
            "eval",
            _readonly_array_mapping(self.eval, rows=self.n_eval, label="eval"),
        )
        object.__setattr__(
            self,
            "obs_train",
            _readonly_array_mapping(
                self.obs_train, rows=self.n_obs, label="obs_train"
            ),
        )
        if not self.definition:
            raise ValueError("truth definition must be documented")


@dataclass(frozen=True)
class SimulationProblem:
    """One benchmark replicate, including evaluation-only simulation truth."""

    name: str
    seed: int
    obs_train: SourceSample
    rct_train: SourceSample
    rct_tune: SourceSample
    rct_test: SourceSample
    eval_X: FloatArray
    eval_tau: FloatArray
    eval_region_score: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    truth: SimulationTruth | None = None
    target_reference: TargetCovariateSample | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("problem name must not be empty")
        if isinstance(self.seed, bool) or not isinstance(
            self.seed, (int, np.integer)
        ) or self.seed < 0:
            raise ValueError("problem seed must be a nonnegative integer")
        samples = (
            (self.obs_train, "obs", "obs_train"),
            (self.rct_train, "rct", "rct_train"),
            (self.rct_tune, "rct", "rct_tune"),
            (self.rct_test, "rct", "rct_test"),
        )
        for sample, expected_source, expected_role in samples:
            if sample.source != expected_source or sample.role != expected_role:
                raise ValueError(
                    f"{expected_role} must have source={expected_source!r} "
                    f"and role={expected_role!r}"
                )
        eval_X = _readonly_float(self.eval_X, ndim=2)
        eval_tau = _readonly_float(self.eval_tau, ndim=1)
        region = _readonly_float(self.eval_region_score, ndim=1)
        if eval_X.shape[0] != eval_tau.shape[0] or eval_X.shape[0] != region.shape[0]:
            raise ValueError("evaluation arrays must have matching rows")
        if eval_X.shape[0] == 0:
            raise ValueError("evaluation sample must not be empty")
        dimensions = {
            self.obs_train.d,
            self.rct_train.d,
            self.rct_tune.d,
            self.rct_test.d,
            eval_X.shape[1],
        }
        if self.target_reference is not None:
            if not isinstance(self.target_reference, TargetCovariateSample):
                raise ValueError(
                    "target_reference must be an outcome-free "
                    "TargetCovariateSample"
                )
            dimensions.add(self.target_reference.d)
        if len(dimensions) != 1:
            raise ValueError("all problem arrays must share a covariate dimension")
        object.__setattr__(self, "eval_X", eval_X)
        object.__setattr__(self, "eval_tau", eval_tau)
        object.__setattr__(self, "eval_region_score", region)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.truth is not None:
            if self.truth.n_eval != eval_X.shape[0]:
                raise ValueError("truth and eval_X row counts differ")
            if self.truth.n_obs != self.obs_train.n:
                raise ValueError("truth and obs_train row counts differ")

    def training_view(self) -> TrainingView:
        return TrainingView(
            self.obs_train,
            self.rct_train,
            self.rct_tune,
            self.target_reference,
        )
