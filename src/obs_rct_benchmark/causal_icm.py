"""Optional, fixed-borrowing recipe using the authors' intact Causal-ICM cell.

The nominal Python helper in the pinned release has unresolved merge markers.
We execute only a hash-locked, intact model-definition cell from a separately
obtained official notebook. No author source or notebook execution is bundled.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from numbers import Real
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

from .data import SourceSample, TrainingView


CAUSAL_ICM_REVISION = "0f0c6ca424b7ca44b84649ca072384b9933b2afe"
CAUSAL_ICM_NOTEBOOK = "rebuttals/extrapolation.ipynb"
CAUSAL_ICM_NOTEBOOK_SHA256 = "0326f655bac0993fd845412ce59fb0cde05e04c6f70890f42ef7e6750fcd855c"
CAUSAL_ICM_CELL_INDEX = 3
CAUSAL_ICM_CELL_SHA256 = "6c85f01d85087e3e69f75c1dbaf926bc8868b6b1be0521a699e00be643f7ca85"
# The separate numerical environment is deliberately narrower than the base
# benchmark. These are the versions used for the prospective native check.
CAUSAL_ICM_NUMERICAL_VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.12.0", "GPy": "1.13.2", "paramz": "0.9.6",
}
# GPy 1.13.2 installs these exact backward-compatibility aliases at import.
# Their canonical module objects and source files belong to paramz, not GPy.
GPY_PARAMZ_ALIASES = {
    "GPy.core.parameterization." + name: "paramz.core." + name
    for name in ("lists_and_dicts", "observable_array", "index_operations")
}


def _verified_gpy_alias(name: str, module: Any) -> bool:
    canonical_name = GPY_PARAMZ_ALIASES.get(name)
    if canonical_name is None or sys.modules.get(canonical_name) is not module:
        return False
    try:
        distribution = importlib.metadata.distribution("paramz")
        if distribution.version != CAUSAL_ICM_NUMERICAL_VERSIONS["paramz"]:
            return False
        initializer = Path(distribution.locate_file("paramz/__init__.py")).resolve()
        root_spec = importlib.util.find_spec("paramz")
        relative = canonical_name.replace(".", "/") + ".py"
        expected = Path(distribution.locate_file(relative)).resolve()
        spec = getattr(module, "__spec__", None)
        filename = getattr(module, "__file__", None)
        return (
            initializer.is_file()
            and root_spec is not None and root_spec.origin is not None
            and Path(root_spec.origin).resolve() == initializer
            and expected.is_relative_to(initializer.parent) and expected.is_file()
            and filename is not None and Path(filename).resolve() == expected
            and getattr(module, "__name__", None) == canonical_name
            and spec is not None and spec.name == canonical_name
            and spec.origin is not None and Path(spec.origin).resolve() == expected
            and not getattr(module, "__path__", ())
        )
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError,
            ImportError, AttributeError, KeyError):
        return False


def _numerical_provenance() -> dict[str, Any]:
    """Inspect metadata and import resolution without importing optional code."""
    result = {}
    for name, expected in CAUSAL_ICM_NUMERICAL_VERSIONS.items():
        entry: dict[str, Any] = {
            "expected_version": expected, "installed_version": None,
            "init_sha256": None, "import_resolution": "unavailable", "verified": False,
            "compatibility_aliases": dict(GPY_PARAMZ_ALIASES) if name == "GPy" else {},
        }
        result[name] = entry
        try:
            distribution = importlib.metadata.distribution(name)
            entry["installed_version"] = distribution.version
            initializer = Path(distribution.locate_file(f"{name}/__init__.py")).resolve()
            entry["init_sha256"] = hashlib.sha256(initializer.read_bytes()).hexdigest()
            spec = importlib.util.find_spec(name)
            correct = (spec is not None and spec.origin is not None
                       and Path(spec.origin).resolve() == initializer)
            package = initializer.parent
            for module_name, module in tuple(sys.modules.items()):
                if module_name == name or module_name.startswith(name + "."):
                    filename = getattr(module, "__file__", None)
                    locations = getattr(module, "__path__", ())
                    if name == "GPy" and module_name in GPY_PARAMZ_ALIASES:
                        if not _verified_gpy_alias(module_name, module):
                            correct = False
                    elif filename is not None and not Path(filename).resolve().is_relative_to(package):
                        correct = False
                    if any(not Path(p).resolve().is_relative_to(package) for p in locations):
                        correct = False
            entry["import_resolution"] = "verified_distribution" if correct else "shadowed"
            entry["verified"] = correct and distribution.version == expected
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError,
                ImportError, AttributeError):
            pass
    return result


def _source_snapshot() -> tuple[dict[str, Any], Path | None, str | None]:
    result: dict[str, Any] = {
        "availability": "unavailable", "expected_revision": CAUSAL_ICM_REVISION,
        "declared_git_revision": None, "tracked_changes": None,
        "source_file": CAUSAL_ICM_NOTEBOOK, "source_file_sha256": None,
        "expected_source_file_sha256": CAUSAL_ICM_NOTEBOOK_SHA256,
        "cell_index": CAUSAL_ICM_CELL_INDEX, "cell_sha256": None,
        "expected_cell_sha256": CAUSAL_ICM_CELL_SHA256,
        "entrypoint": "ICM", "source_verified": False,
        "import_resolution": "isolated_locked_notebook_cell",
        "nominal_helper_status": "unresolved_merge_markers_not_imported",
    }
    raw = os.environ.get("CAUSAL_ICM_SOURCE")
    if not raw:
        return result, None, None
    root = Path(raw).expanduser().resolve()
    try:
        # Requiring the exact Git top-level prevents an arbitrary nested path
        # from inheriting a different repository's revision and clean status.
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top).resolve() != root:
            return result, root, None
        result["declared_git_revision"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        result["tracked_changes"] = bool(changes)
        path = root / CAUSAL_ICM_NOTEBOOK
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            return result, root, None
        contents = path.read_bytes()
        result["source_file_sha256"] = hashlib.sha256(contents).hexdigest()
        notebook = json.loads(contents)
        cell = notebook["cells"][CAUSAL_ICM_CELL_INDEX]
        if cell["cell_type"] != "code":
            return result, root, None
        source = "".join(cell["source"])
        result["cell_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        result["availability"] = "available"
        result["source_verified"] = (
            result["declared_git_revision"] == CAUSAL_ICM_REVISION
            and not changes
            and result["source_file_sha256"] == CAUSAL_ICM_NOTEBOOK_SHA256
            and result["cell_sha256"] == CAUSAL_ICM_CELL_SHA256
        )
        return result, root, source
    except (OSError, subprocess.CalledProcessError, ValueError, TypeError, KeyError,
            IndexError):
        return result, root, None


def causal_icm_provenance() -> dict[str, Any]:
    """Sanitized source-cell and numerical-environment identity; imports no GPy."""
    result = _source_snapshot()[0]
    result["numerical_environment"] = _numerical_provenance()
    result["verified"] = result["source_verified"] and all(
        entry["verified"] for entry in result["numerical_environment"].values()
    )
    return result


def _official_icm():
    """Load only the pinned model cell, never notebook setup or simulation cells."""
    snapshot, _, source = _source_snapshot()
    if not os.environ.get("CAUSAL_ICM_SOURCE"):
        raise RuntimeError("causal_icm_fixed requires CAUSAL_ICM_SOURCE; see README")
    if not snapshot["source_verified"] or source is None:
        raise ValueError("Causal-ICM checkout or native cell does not match the locked source")
    environment = _numerical_provenance()
    if not all(entry["verified"] for entry in environment.values()):
        raise RuntimeError("Causal-ICM requires its locked GPy numerical environment and import resolution")
    tree = ast.parse(source, filename=f"{CAUSAL_ICM_NOTEBOOK}:cell{CAUSAL_ICM_CELL_INDEX}")
    # Hash identity is mandatory; the structural restriction separately guards
    # against accidentally widening execution to notebook imports or setup.
    if (len(tree.body) != 3
            or not isinstance(tree.body[0], ast.Import)
            or ast.unparse(tree.body[0]) != "import GPy"
            or not isinstance(tree.body[1], ast.Import)
            or ast.unparse(tree.body[1]) != "import numpy as np"
            or not isinstance(tree.body[2], ast.FunctionDef)
            or tree.body[2].name != "ICM"
            or tree.body[2].decorator_list):
        raise ValueError("Unexpected executable statements in the locked Causal-ICM cell")
    namespace = {"__name__": "_obsrct_native_causal_icm"}
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        exec(compile(tree, f"{CAUSAL_ICM_NOTEBOOK}:cell{CAUSAL_ICM_CELL_INDEX}", "exec"), namespace)
    except ImportError as exc:
        raise ImportError("Install the causal_icm optional dependencies in a separate environment") from exc
    finally:
        sys.dont_write_bytecode = previous
    # Check cached dependencies again after their actual import, catching a
    # pre-import resolution change or a dependency importing shadowed modules.
    if not all(entry["verified"] for entry in _numerical_provenance().values()):
        raise RuntimeError("Causal-ICM numerical imports did not resolve to their distributions")
    return namespace["ICM"]


@dataclass(frozen=True)
class OfficialCausalICM:
    """Native two-arm GP recipe with prospectively fixed borrowing rho.

    ``rho`` is the authors' borrowing argument, fixed here rather than selected
    by their separate cross-validation notebooks. ``effect_modifier_columns``
    declares the input covariates in advance; ``None`` explicitly uses all X.
    These define the native conditional-mean model, not a truth-selected subset.

    Rank is two. Each arm has RBF x source-coregionalization, source0=RCT and
    source1=OBS, fixed kappa=1e-9 and the native W constraints. Native
    ``optimize()`` calls fit RBF and Gaussian-noise parameters with GPy defaults;
    no scaling, outcome normalization, restarts or query adaptation is added.
    Prediction is the treated-minus-control noiseless mean at source0 for both
    arms. It is a trial-response CATE model; external-population transport needs
    its own assumptions. This is not the paper's selected-rho reproduction.
    Even rho=0 is not an RCT-only control: OBS still fits shared hyperparameters.
    A selected covariate subset additionally requires constant RCT assignment
    probabilities; otherwise conditioning on omitted X may change arm mixtures.
    Only OBS/RCT fitting rows are consumed; tuning/reference rows are ignored.
    """

    rho: float = 0.5
    effect_modifier_columns: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.rho, bool) or not isinstance(self.rho, Real)
                or not np.isfinite(self.rho) or not 0 <= self.rho <= 1):
            raise ValueError("rho must be finite and between zero and one")
        columns = self.effect_modifier_columns
        if columns is not None:
            if (not isinstance(columns, (tuple, list)) or not columns
                    or any(isinstance(c, bool) or not isinstance(c, (int, np.integer))
                           or c < 0 for c in columns)
                    or len(set(columns)) != len(columns)):
                raise ValueError("effect_modifier_columns must be distinct nonnegative column indices")
            object.__setattr__(self, "effect_modifier_columns", tuple(int(c) for c in columns))

    def fit_predict(self, training: TrainingView, eval_X, *, seed: int):
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        query = np.asarray(eval_X, dtype=np.float64)
        if (query.ndim != 2 or query.shape[0] == 0
                or query.shape[1] != training.obs_train.d or not np.isfinite(query).all()):
            raise ValueError("eval_X must be a nonempty finite matrix matching the training covariates")
        obs, rct = training.obs_train, training.rct_train
        for sample in (obs, rct):
            if set(np.unique(sample.T)) != {0.0, 1.0}:
                raise ValueError("Causal-ICM requires both treatment arms in each fitting source")
        columns = self.effect_modifier_columns
        if columns is None:
            columns = tuple(range(obs.d))
        if max(columns) >= obs.d:
            raise ValueError("effect_modifier_columns exceed the training covariate dimension")
        propensity = rct.known_propensity
        if len(columns) < obs.d and not np.all(propensity == propensity[0]):
            raise ValueError("a covariate subset requires constant RCT assignment probabilities")
        native = _official_icm()
        object.__setattr__(self, "provenance_", {
            "native_source": causal_icm_provenance(),
            "recipe": "fixed_rho_native_two_arm_gp", "rho": float(self.rho),
            "effect_modifier_columns": list(columns), "rank": 2,
            "source_encoding": {"rct": 0, "obs": 1},
            "prediction_source": 0, "prediction": "treated_mean_minus_control_mean",
            "noise": "native_GPy_Gaussian_noise_optimized",
            "optimization": "native_GPy_optimize_defaults_no_restarts",
            "rho_zero_is_rct_only": False,
        })
        # Native GPy initialization uses NumPy's legacy RNG. Restore both
        # process RNG states even if native fitting or prediction fails.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            np.random.seed(int(seed))
            random.seed(int(seed))
            m0, m1 = native(
                X_E=np.array(rct.X[:, columns], copy=True),
                X_O=np.array(obs.X[:, columns], copy=True),
                T_E=np.array(rct.T, copy=True), T_O=np.array(obs.T, copy=True),
                Y_E=np.array(rct.Y, copy=True), Y_O=np.array(obs.Y, copy=True),
                r=2, ID=len(columns), AD=list(range(len(columns))), rho=float(self.rho),
            )
            # The final coordinate is SOURCE, not treatment. Separate native
            # models already select treatment arm, so both queries must use 0.
            native_query = np.column_stack((query[:, columns], np.zeros(query.shape[0])))
            means = []
            for model in (m0, m1):
                mean = np.asarray(model.predict_noiseless(native_query.copy())[0], dtype=np.float64)
                if mean.shape != (query.shape[0], 1):
                    raise ValueError("Causal-ICM native means must have shape (n_query, 1)")
                if not np.isfinite(mean).all():
                    raise FloatingPointError("Causal-ICM produced nonfinite native means")
                means.append(mean[:, 0])
            prediction = means[1] - means[0]
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Causal-ICM produced nonfinite CATE predictions")
        return np.asarray(prediction, dtype=np.float64)


@dataclass(frozen=True)
class CausalICMSourceControl:
    """Condition the same native trial-task GP on exactly one fitting source.

    ``source="rct"`` passes no observational rows to native optimization.
    ``source="obs"`` is the trial-task prediction conditional on OBS alone:
    both treatment models are still queried at task0, preserving the fixed
    borrowing prior. It is not the task1 observational contrast and does not
    identify trial CATE from observational data without model assumptions.

    Both controls use every included covariate. They retain the rank-two RBF
    x source kernel, fixed native W/kappa, Gaussian noise and optimizer defaults.
    The native function accepts correctly shaped empty arrays for the excluded
    source. Its values, shape, treatment arms and propensities are never read.
    Tuning labels, target-reference rows and query covariates never enter fit.
    ``fit_predict_sample`` accepts only the included fitting sample; the common
    ``fit_predict`` interface delegates to it for runner compatibility.
    """

    source: str
    rho: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or self.source not in {"rct", "obs"}:
            raise ValueError("source must be rct or obs")
        # Reuse the existing scalar contract without loading native code.
        OfficialCausalICM(rho=self.rho)

    def fit_predict(self, training: TrainingView, eval_X, *, seed: int):
        object.__setattr__(self, "provenance_", None)
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        # Compatibility with the common runner; the pilot uses the single-
        # sample entry point below so excluded real labels never cross its API.
        included = training.rct_train if self.source == "rct" else training.obs_train
        return self.fit_predict_sample(included, eval_X, seed=seed)

    def fit_predict_sample(self, included: SourceSample, eval_X, *, seed: int):
        """Fit using only the declared fitting sample, without another source.

        Source and role are both checked: tuning/test/evaluation samples cannot
        enter this path merely because they use the same numeric array schema.
        """
        object.__setattr__(self, "provenance_", None)
        if not isinstance(included, SourceSample):
            raise TypeError("source controls require a SourceSample")
        if included.source != self.source or included.role != self.source + "_train":
            raise ValueError("sample must have the declared source and its fitting role")
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        query = np.asarray(eval_X, dtype=np.float64)
        if (query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != included.d
                or not np.isfinite(query).all()):
            raise ValueError("eval_X must be a nonempty finite matrix matching the included covariates")
        if set(np.unique(included.T)) != {0.0, 1.0}:
            raise ValueError("Causal-ICM requires both treatment arms in the included source")
        access = {name: {"availability": "not_accessed", "fit_rows": 0,
                         "arm_rows": {"0": 0, "1": 0}} for name in ("rct", "obs")}
        access[self.source] = {
            "availability": "included", "fit_rows": int(included.n),
            "arm_rows": {str(arm): int(np.count_nonzero(included.T == arm)) for arm in (0, 1)},
        }
        record = {
            "recipe": "source_restricted_native_trial_task_gp", "source": self.source,
            "rho": float(self.rho), "supplied_seed": int(seed), "rank": 2,
            "effect_modifier_columns": list(range(included.d)),
            "source_encoding": {"rct": 0, "obs": 1}, "fitting_sources": access,
            "prediction_source": 0, "prediction": "treated_mean_minus_control_mean",
            "known_propensity": "not_used", "rct_tune": "not_accessed",
            "target_reference": "not_accessed", "excluded_source": "empty_native_arrays",
            "rct_only_by_row_exclusion": self.source == "rct",
            "noise": "native_GPy_Gaussian_noise_optimized",
            "optimization": "native_GPy_optimize_defaults_no_restarts", "status": "preparing",
        }
        object.__setattr__(self, "provenance_", record)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            native = _official_icm()
            record["native_source"] = causal_icm_provenance()
            record["status"] = "fitting"
            np.random.seed(int(seed))
            random.seed(int(seed))
            empty_X = np.empty((0, included.d), dtype=np.float64)
            empty_vector = np.empty(0, dtype=np.float64)
            included_X = np.array(included.X, copy=True)
            included_T, included_Y = np.array(included.T, copy=True), np.array(included.Y, copy=True)
            m0, m1 = native(
                X_E=included_X if self.source == "rct" else empty_X.copy(),
                X_O=included_X if self.source == "obs" else empty_X.copy(),
                T_E=included_T if self.source == "rct" else empty_vector.copy(),
                T_O=included_T if self.source == "obs" else empty_vector.copy(),
                Y_E=included_Y if self.source == "rct" else empty_vector.copy(),
                Y_O=included_Y if self.source == "obs" else empty_vector.copy(),
                r=2, ID=included.d, AD=list(range(included.d)), rho=float(self.rho),
            )
            native_query = np.column_stack((query, np.zeros(query.shape[0])))
            means = []
            for model in (m0, m1):
                mean = np.asarray(model.predict_noiseless(native_query.copy())[0], dtype=np.float64)
                if mean.shape != (query.shape[0], 1):
                    raise ValueError("Causal-ICM native means must have shape (n_query, 1)")
                if not np.isfinite(mean).all():
                    raise FloatingPointError("Causal-ICM produced nonfinite native means")
                means.append(mean[:, 0])
            prediction = means[1] - means[0]
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Causal-ICM produced nonfinite CATE predictions")
            record["status"] = "completed"
            return np.asarray(prediction, dtype=np.float64)
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
