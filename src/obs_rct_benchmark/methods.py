"""Generic CATE baselines and optional adapters to pinned public implementations.

The method boundary contains observed training data and evaluation covariates.
Neither simulation truth nor test outcomes are passed across this boundary.
All baseline hyperparameters are fixed defaults, exposed below; no truth-based
model or hyperparameter selection is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .data import TrainingView

FloatArray = NDArray[np.float64]
DATAFUSIONPPI_COMMIT = "04c7f3447b02fa09f875e68e8c47100a7207d546"
# SHA256 of every Python source file in the official package at this commit.
# These are provenance data, not copied upstream implementation code.
DATAFUSIONPPI_FILE_SHA256 = {
    "__init__.py": "bd8c261da5b96a34d49509049428f8907a98f52901010c9c66548b3592ac9aec",
    "__main__.py": "13a1a5b340cdcfc1902b62be90e508c7c71886000d5bf087e7854aadf09fb35e",
    "aipwf.py": "0f86537af38af6c3c3f62b43cee3b7f7ea4fec160423cdd221e93e6dc1924c55",
    "api.py": "87d7add83dde597be18fc4cd2968e08e67032d21248e8c3969a3dcf61aa48965",
    "cli.py": "be1e3246ab9a4f34223a08c2e0e6dc16b7133b111c2bf3a21709ecf7f83404d3",
    "crossfit.py": "42fd6e1b863f2bbaab25b52ec1f82a513829d2e44e9d1d08d3d7291ce334415f",
    "drf.py": "af6655a43c5b825d928d64344ae8d081871b21851701aefb94eb0128e7336e05",
    "method.py": "9606ccb2e0581beb5c573d2b774635da04481a801b9430fb4a2abb96683c7e4e",
    "rf.py": "9588f094cf2475e71496f4a2aa485d2ffeb0c98919d35d807b25c73b2db0b0ac",
}


class CATEMethod(Protocol):
    """A method may use only the supplied TrainingView and eval_X.

    Implementations return one finite CATE prediction per evaluation row.
    Built-in recipes disclose fixed settings and any native internal selection.
    They do not implement the future v0.3 controlled-tuning lane, which requires
    a frozen normalized R-loss selector on independent RCT tuning rows and its
    declared fitting/nuisance budgets. Never select using evaluation truth or
    rct_test; eval_X is prediction-only.
    """

    def fit_predict(
        self, training: TrainingView, eval_X: FloatArray, *, seed: int
    ) -> FloatArray: ...


class MethodUnavailableError(RuntimeError):
    """An optional dependency is unavailable or has unverified provenance."""


def _regressor(learner: str, *, seed: int, ridge_alpha: float, n_estimators: int,
               min_samples_leaf: int) -> Any:
    try:
        if learner == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        if learner == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                max_features=1.0, random_state=seed, n_jobs=1,
            )
    except ImportError as exc:
        raise MethodUnavailableError("install the baselines extra") from exc
    raise ValueError("learner must be 'ridge' or 'extra_trees'")


@dataclass(frozen=True)
class TLearner:
    """Separate outcome regressions fitted to RCT, OBS, or naively pooled rows.

    Fixed defaults: standardized ridge alpha=1, or 100 extra trees with minimum
    leaf size 5 and all covariates considered at each split. The pooled method
    ignores source and is deliberately labelled naive.
    """

    source: str = "rct"
    learner: str = "ridge"
    ridge_alpha: float = 1.0
    n_estimators: int = 100
    min_samples_leaf: int = 5

    def __post_init__(self) -> None:
        if self.source not in {"rct", "obs", "pooled"}:
            raise ValueError("source must be rct, obs, or pooled")
        _check_options(self.learner, self.ridge_alpha, self.n_estimators,
                       self.min_samples_leaf)

    def fit_predict(self, training: TrainingView, eval_X: FloatArray, *,
                    seed: int) -> FloatArray:
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        samples = {"rct": (training.rct_train,), "obs": (training.obs_train,),
                   "pooled": (training.rct_train, training.obs_train)}[self.source]
        X = np.concatenate([sample.X for sample in samples])
        T = np.concatenate([sample.T for sample in samples])
        Y = np.concatenate([sample.Y for sample in samples])
        predictions = []
        for arm in (0, 1):
            mask = T == arm
            if np.count_nonzero(mask) < 2:
                raise ValueError("each treatment arm needs at least two rows")
            model = _regressor(self.learner, seed=(seed + arm) % (2**32),
                               ridge_alpha=self.ridge_alpha,
                               n_estimators=self.n_estimators,
                               min_samples_leaf=self.min_samples_leaf)
            model.fit(X[mask], Y[mask])
            predictions.append(model.predict(eval_X))
        return np.asarray(predictions[1] - predictions[0], dtype=np.float64)


def _check_options(learner: str, ridge_alpha: float, n_estimators: int,
                   min_samples_leaf: int) -> None:
    if learner not in {"ridge", "extra_trees"}:
        raise ValueError("unknown learner")
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive and finite")
    for value in (n_estimators, min_samples_leaf):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("tree counts and leaf sizes must be positive integers")


@dataclass(frozen=True)
class RCTDRLearner:
    """RCT-only DR learner with two-fold out-of-fold outcome regressions.

    Known randomization propensities are used without propensity estimation.
    Folds are stratified by treatment. A final fixed-default regressor learns
    the standard augmented inverse-propensity CATE pseudo-outcome.
    """

    learner: str = "ridge"
    ridge_alpha: float = 1.0
    n_estimators: int = 100
    min_samples_leaf: int = 5

    def __post_init__(self) -> None:
        _check_options(self.learner, self.ridge_alpha, self.n_estimators,
                       self.min_samples_leaf)

    def fit_predict(self, training: TrainingView, eval_X: FloatArray, *,
                    seed: int) -> FloatArray:
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        sample = training.rct_train
        X, T, Y = sample.X, sample.T, sample.Y
        propensity = sample.known_propensity
        if propensity is None:
            raise ValueError("RCT randomization probabilities are required")
        rng = np.random.default_rng(seed)
        fold = np.empty(sample.n, dtype=int)
        for arm in (0, 1):
            rows = np.flatnonzero(T == arm)
            if rows.size < 4:
                raise ValueError("cross-fitting needs four rows per treatment arm")
            rng.shuffle(rows)
            fold[rows] = np.arange(rows.size) % 2
        mu = np.empty((sample.n, 2), dtype=np.float64)
        for held_out in (0, 1):
            for arm in (0, 1):
                fit_rows = (fold != held_out) & (T == arm)
                model = _regressor(
                    self.learner, seed=(seed + 1 + 2 * held_out + arm) % (2**32),
                    ridge_alpha=self.ridge_alpha, n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                )
                model.fit(X[fit_rows], Y[fit_rows])
                mu[fold == held_out, arm] = model.predict(X[fold == held_out])
        pseudo = (mu[:, 1] - mu[:, 0] + T * (Y - mu[:, 1]) / propensity
                  - (1 - T) * (Y - mu[:, 0]) / (1 - propensity))
        final = _regressor(self.learner, seed=(seed + 7) % (2**32),
                           ridge_alpha=self.ridge_alpha,
                           n_estimators=self.n_estimators,
                           min_samples_leaf=self.min_samples_leaf)
        final.fit(X, pseudo)
        return np.asarray(final.predict(eval_X), dtype=np.float64)


@dataclass(frozen=True)
class DataFusionPPI:
    """Thin adapter to the pinned official dfppi.fuse_cate implementation.

    Uses only rct_train and obs_train. The upstream algorithm performs its own
    internal split; rct_tune is not passed. This adapter checks the installed
    distribution's PEP 610 Git commit, complete source hashes, and import
    resolution before importing its implementation.
    """

    learner: str = "DR"
    covariate_shift: bool = True

    def __post_init__(self) -> None:
        if self.learner not in {"DR", "R"}:
            raise ValueError("unsupported official DataFusionPPI learner")
        if not isinstance(self.covariate_shift, bool):
            raise ValueError("covariate_shift must be boolean")

    def fit_predict(self, training: TrainingView, eval_X: FloatArray, *,
                    seed: int) -> FloatArray:
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        package = _verify_dfppi_install()
        trial, obs = training.rct_train, training.obs_train
        propensity = trial.known_propensity
        if propensity is None or not np.all(propensity == propensity[0]):
            raise ValueError("official fuse_cate requires constant trial propensity")
        if trial.n < 30 or obs.n < 30:
            raise ValueError("official fuse_cate needs at least 30 rows per source")
        module = importlib.import_module("dfppi")
        if Path(module.__file__).resolve() != package / "__init__.py":
            raise MethodUnavailableError("DataFusionPPI import was shadowed")
        fit = module.fuse_cate(
            trial={"x": trial.X, "a": trial.T, "y": trial.Y},
            obs={"x": obs.X, "a": obs.T, "y": obs.Y},
            trial_propensity=float(propensity[0]), learner=self.learner,
            covariate_shift=self.covariate_shift, seed=seed, shuffle=True,
        )
        return np.asarray(fit.predict(eval_X), dtype=np.float64).reshape(-1)


def _dfppi_snapshot() -> tuple[dict[str, Any], Path | None]:
    snapshot: dict[str, Any] = {
        "availability": "unavailable", "expected_revision": DATAFUSIONPPI_COMMIT,
        "installed_version": None, "declared_git_revision": None,
        "package_file_sha256": {}, "import_resolution": "unavailable",
        "matches_pinned_source": False,
    }
    try:
        distribution = importlib.metadata.distribution("dfppi")
    except importlib.metadata.PackageNotFoundError:
        return snapshot, None
    try:
        package = Path(distribution.locate_file("dfppi")).resolve()
        snapshot["installed_version"] = distribution.version
    except (OSError, ValueError, TypeError):
        snapshot["availability"] = "unreadable"
        return snapshot, None
    try:
        raw = distribution.read_text("direct_url.json")
        direct_url = json.loads(raw or "{}")
        vcs = direct_url.get("vcs_info", {})
        if vcs.get("vcs") == "git":
            revision = vcs.get("commit_id")
            if (isinstance(revision, str) and len(revision) == 40
                    and all(c in "0123456789abcdef" for c in revision)):
                snapshot["declared_git_revision"] = revision
    except (ValueError, TypeError, AttributeError, OSError):
        pass
    try:
        files = {
            path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
        snapshot["package_file_sha256"] = files
        snapshot["availability"] = "installed"
        snapshot["matches_pinned_source"] = files == DATAFUSIONPPI_FILE_SHA256
        spec = importlib.util.find_spec("dfppi")
        expected_init = package / "__init__.py"
        correct = spec is not None and spec.origin is not None and Path(spec.origin).resolve() == expected_init
        for name, cached in tuple(sys.modules.items()):
            if name == "dfppi" or name.startswith("dfppi."):
                origin = getattr(cached, "__file__", None)
                if origin is None or not Path(origin).resolve().is_relative_to(package):
                    correct = False
        snapshot["import_resolution"] = "verified_distribution" if correct else "shadowed"
    except (OSError, ValueError, ImportError, AttributeError):
        snapshot["availability"] = "unreadable"
    return snapshot, package


def datafusionppi_provenance() -> dict[str, Any]:
    """Snapshot actual external files without importing or executing dfppi.

    Missing or changed dependencies remain reportable in a run manifest; the
    adapter rejects them when its cell executes. Returned values contain no
    installation paths.
    """
    return _dfppi_snapshot()[0]


def _verify_dfppi_install() -> Path:
    snapshot, package = _dfppi_snapshot()
    if (package is None or snapshot["availability"] != "installed"
            or snapshot["installed_version"] != "0.1.0"
            or snapshot["declared_git_revision"] != DATAFUSIONPPI_COMMIT
            or not snapshot["matches_pinned_source"]
            or snapshot["import_resolution"] != "verified_distribution"):
        raise MethodUnavailableError("DataFusionPPI install has unverified source or import resolution")
    return package


MethodFactory = Callable[..., CATEMethod]
_FACTORIES: dict[str, MethodFactory] = {}
_CUSTOM_PROVENANCE: dict[str, str] = {}


def method_names() -> tuple[str, ...]:
    return (
        "rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge",
        "rct_t_extra_trees", "obs_t_extra_trees", "naive_pooled_t_extra_trees",
        "rct_dr_ridge", "rct_dr_extra_trees", "datafusionppi", "cornet_official",
        "integrative_r_linear", "integrative_r_quadratic", "causal_icm_fixed",
        "causal_icm_rct_only", "causal_icm_obs_start",
        "rct_causal_forest",
        *sorted(_FACTORIES),
    )


def register_method(name: str, factory: MethodFactory, *, provenance: str) -> None:
    """Register a custom factory with an author-supplied public version ID.

    The factory must return a CATEMethod. The runner passes only TrainingView
    and evaluation covariates. Registration does not sandbox third-party code.
    The provenance string is an author declaration, not an automatic source
    verification. Custom authors must change it whenever their code changes.
    """
    if not name or name in method_names() or not name.replace("_", "").isalnum():
        raise ValueError("method names must be unique alphanumeric/underscore IDs")
    if not callable(factory) or not provenance or "\n" in provenance:
        raise ValueError("a callable factory and one-line provenance are required")
    _FACTORIES[name] = factory
    _CUSTOM_PROVENANCE[name] = provenance


def custom_provenance() -> dict[str, str]:
    return dict(_CUSTOM_PROVENANCE)


def make_method(name: str, **options: Any) -> CATEMethod:
    """Build an estimator without importing optional estimator dependencies."""
    if name in _FACTORIES:
        return _FACTORIES[name](**options)
    if name == "datafusionppi":
        return DataFusionPPI(**options)
    if name == "cornet_official":
        from .cornet import OfficialCorNet

        return OfficialCorNet(**options)
    if name == "rct_causal_forest":
        from .causal_forest import RCTCausalForest

        return RCTCausalForest(**options)
    if name in {"integrative_r_linear", "integrative_r_quadratic"}:
        if "basis" in options:
            raise ValueError("integrative R-learner basis is fixed by the method ID")
        from .integrative_r import OfficialIntegrativeR

        return OfficialIntegrativeR(basis=name.removeprefix("integrative_r_"), **options)
    if name == "causal_icm_fixed":
        from .causal_icm import OfficialCausalICM

        return OfficialCausalICM(**options)
    if name in {"causal_icm_rct_only", "causal_icm_obs_start"}:
        if "source" in options:
            raise ValueError("Causal-ICM fitting source is fixed by the method ID")
        from .causal_icm import CausalICMSourceControl

        source = "rct" if name == "causal_icm_rct_only" else "obs"
        return CausalICMSourceControl(source=source, **options)
    if name.startswith("rct_dr_"):
        learner = name.removeprefix("rct_dr_")
        if name not in method_names():
            raise ValueError(f"unknown method: {name}")
        return RCTDRLearner(learner=learner, **options)
    prefixes = {"rct_t_": "rct", "obs_t_": "obs", "naive_pooled_t_": "pooled"}
    for prefix, source in prefixes.items():
        if name.startswith(prefix) and name in method_names():
            return TLearner(source=source, learner=name.removeprefix(prefix), **options)
    raise ValueError(f"unknown method: {name}")
