"""Local-file adapters and finite-population semi-synthetic OBS/RCT problems.

No data are downloaded or redistributed by this module. IHDP layouts follow
the original CEVAE CSV loader and CFR NPZ loader. ACIC's common CSV export uses
``z,y0,y1,mu0,mu1``; an explicit column map also supports the organizer's dotted
R names. Original treatments and noisy outcomes are validated and discarded.

The empirical distribution puts equal mass on each loaded row. The estimand is
the mean of the selected potential-outcome means conditional on the covariates
actually shown to the estimator, after any explicitly requested coarsening.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import (
    SimulationProblem,
    SimulationTruth,
    SourceSample,
    TargetCovariateSample,
)
from .sensitivity import msm_propensity_bounds


FloatArray = NDArray[np.float64]
EMPIRICAL_DGP_VERSION = "1.0.0"
_MISSING = {"", "na", "nan", "null", "none"}


def _array(values: ArrayLike, ndim: int, label: str) -> FloatArray:
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be a finite {ndim}-dimensional array") from None
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _file_snapshot(path: Path, role: str) -> tuple[bytes, dict[str, str]]:
    """Parse and hash the same captured bytes, including after atomic replacement."""
    content = path.read_bytes()
    return content, {"role": role, "sha256": hashlib.sha256(content).hexdigest()}


def _digest_array(values: NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _safe_provenance(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Limit provenance to structural IDs, file digests, and schema metadata."""
    allowed = {"source_id", "schema", "replication", "pooling", "files", "preprocessing"}
    if set(raw) - allowed:
        raise ValueError("provenance accepts only source/schema/file/preprocessing identifiers")
    result: dict[str, Any] = {}
    for key in ("source_id", "schema", "pooling"):
        if key in raw:
            value = raw[key]
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
                raise ValueError(f"provenance {key} must be a path-free identifier")
            result[key] = value
    if "replication" in raw:
        value = raw["replication"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provenance replication must be a nonnegative integer")
        result["replication"] = value
    if "files" in raw:
        files = []
        for item in raw["files"]:
            if set(item) != {"role", "sha256"}:
                raise ValueError("provenance files require only role and sha256")
            name, digest = item["role"], item["sha256"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError("provenance file roles must be path-free identifiers")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("provenance file sha256 must be a hexadecimal digest")
            files.append({"role": name, "sha256": digest})
        result["files"] = files
    if "preprocessing" in raw:
        prep = raw["preprocessing"]
        permitted = {"encoding", "numeric_columns", "categorical_columns",
                     "categorical_level_counts", "encoding_sha256", "row_alignment"}
        if not isinstance(prep, Mapping) or set(prep) - permitted:
            raise ValueError("unrecognized preprocessing provenance")
        # All fields describe the fixed-population transform, never fitted rows.
        copied = json.loads(json.dumps(dict(prep), allow_nan=False))
        for key in ("encoding", "row_alignment"):
            if key in copied and not re.fullmatch(r"[A-Za-z0-9_.:-]+", copied[key]):
                raise ValueError("preprocessing descriptions must be identifiers")
        for key in ("numeric_columns", "categorical_columns"):
            if key in copied and not all(isinstance(x, str) for x in copied[key]):
                raise ValueError("preprocessing column names must be strings")
        if "encoding_sha256" in copied and not re.fullmatch(
            r"[0-9a-f]{64}", copied["encoding_sha256"]
        ):
            raise ValueError("encoding_sha256 must be a hexadecimal digest")
        if "categorical_level_counts" in copied and not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v > 0
            for k, v in copied["categorical_level_counts"].items()
        ):
            raise ValueError("categorical level counts must be positive integers")
        result["preprocessing"] = copied
    return MappingProxyType(result)


@dataclass(frozen=True)
class EmpiricalPopulation:
    """Evaluation-owned finite population, separate from ``TrainingView``.

    ``feature_groups`` maps encoded columns to original source column names, so
    hiding an ACIC categorical column hides every one-hot indicator together.
    The optional arm means are simulation truth; factual outcomes are not kept.
    """

    X_full: FloatArray
    feature_names: tuple[str, ...]
    mu0: FloatArray | None = None
    mu1: FloatArray | None = None
    feature_groups: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = _array(self.X_full, 2, "X_full")
        names = tuple(self.feature_names)
        groups = tuple(self.feature_groups) or names
        if X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError("an empirical population needs at least two rows and two features")
        if (len(names) != X.shape[1] or len(set(names)) != len(names)
                or any(not isinstance(x, str) or not x for x in names)):
            raise ValueError("feature_names must uniquely name all X_full columns")
        if len(groups) != X.shape[1] or any(not isinstance(x, str) or not x for x in groups):
            raise ValueError("feature_groups must name the source column for every feature")
        if (self.mu0 is None) != (self.mu1 is None):
            raise ValueError("provide both mu0 and mu1, or neither")
        for key in ("mu0", "mu1"):
            if getattr(self, key) is not None:
                value = _array(getattr(self, key), 1, key)
                if value.shape[0] != X.shape[0]:
                    raise ValueError("arm means must match population rows")
                object.__setattr__(self, key, value)
        object.__setattr__(self, "X_full", X)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_groups", groups)
        object.__setattr__(self, "provenance", _safe_provenance(self.provenance))


def _replication_index(replication: int) -> None:
    if isinstance(replication, bool) or not isinstance(replication, int) or replication < 0:
        raise ValueError("replication must be a zero-based nonnegative integer")


def _read_ihdp_npz(path: Path, replication: int, role: str):
    content, digest = _file_snapshot(path, role)
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        required = {"x", "t", "yf", "ycf", "mu0", "mu1"}
        if required - set(data.files):
            raise ValueError(f"IHDP NPZ missing required keys: {sorted(required - set(data.files))}")
        X = np.asarray(data["x"])
        if X.ndim not in {2, 3} or X.shape[1] != 25:
            raise ValueError("IHDP x must have shape (n,25) or (n,25,replications)")
        replications = X.shape[2] if X.ndim == 3 else 1
        if replication >= replications:
            raise ValueError("replication index exceeds the NPZ replication axis")
        rows = X.shape[0]
        selected: dict[str, FloatArray] = {}
        for key in required - {"x"}:
            value = np.asarray(data[key])
            if X.ndim == 3:
                if value.shape != (rows, replications):
                    raise ValueError(f"IHDP {key} must have shape (n,replications)")
                value = value[:, replication]
            elif value.shape == (rows, 1):
                value = value[:, 0]
            elif value.shape != (rows,):
                raise ValueError(f"single-replication IHDP {key} must have shape (n,) or (n,1)")
            selected[key] = _array(value, 1, key)
        if not np.isin(selected["t"], (0, 1)).all():
            raise ValueError("IHDP t must be binary")
        X_selected = X[:, :, replication] if X.ndim == 3 else X
        return _array(X_selected, 2, "x"), selected["mu0"], selected["mu1"], replications, digest


def load_ihdp_npz(
    train_path: str | Path, test_path: str | Path | None = None, *, replication: int = 0
) -> EmpiricalPopulation:
    """Load one CFR-style IHDP replication, pooling supplied train/test rows.

    Historical train/test membership is discarded. All benchmark roles are
    subsequently sampled independently from this explicitly pooled population.
    """
    _replication_index(replication)
    paths = [Path(train_path)] + ([] if test_path is None else [Path(test_path)])
    parts = [_read_ihdp_npz(path, replication, role)
             for path, role in zip(paths, ("ihdp_train", "ihdp_test"))]
    if len({part[3] for part in parts}) != 1:
        raise ValueError("IHDP train and test must have the same replication count")
    names = tuple(f"x{i + 1}" for i in range(25))
    return EmpiricalPopulation(
        np.concatenate([part[0] for part in parts]), names,
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        provenance={"source_id": "ihdp", "schema": "cfr_npz",
                    "replication": replication,
                    "pooling": "train_test" if test_path is not None else "single_file",
                    "files": [part[4] for part in parts]},
    )


def load_ihdp_csv(path: str | Path) -> EmpiricalPopulation:
    """Load headerless CEVAE IHDP CSV: t,yf,ycf,mu0,mu1,x1,...,x25.

    Values are preserved as supplied, including the historical x14 coding.
    No CEVAE-specific feature recoding or outcome standardization is applied.
    """
    path = Path(path)
    content, digest = _file_snapshot(path, "ihdp_csv")
    values = np.loadtxt(io.BytesIO(content), delimiter=",", ndmin=2)
    if values.shape[1] != 30 or not np.isfinite(values).all():
        raise ValueError("CEVAE IHDP CSV must contain exactly 30 finite numeric columns")
    if not np.isin(values[:, 0], (0, 1)).all():
        raise ValueError("IHDP treatment column must be binary")
    return EmpiricalPopulation(
        values[:, 5:], tuple(f"x{i + 1}" for i in range(25)), values[:, 3], values[:, 4],
        provenance={"source_id": "ihdp", "schema": "cevae_csv",
                    "pooling": "single_file", "files": [digest]},
    )


def _read_csv(path: Path, role: str):
    content, digest = _file_snapshot(path, role)
    with io.TextIOWrapper(io.BytesIO(content), newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration as error:
            raise ValueError("CSV files must not be empty") from error
        rows = [[cell.strip() for cell in row] for row in reader]
    if not header or any(not x for x in header) or len(set(header)) != len(header):
        raise ValueError("CSV columns need unique, nonempty headers")
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("CSV rows must be nonempty and match the header width")
    if any(cell.casefold() in _MISSING for row in rows for cell in row):
        raise ValueError("missing CSV values require explicit preprocessing before loading")
    return header, rows, digest


def load_acic2016(
    covariates_path: str | Path,
    response_path: str | Path,
    *,
    column_map: Mapping[str, str] | None = None,
    categorical_columns: Sequence[str] = (),
    row_id_columns: tuple[str, str] | None = None,
) -> EmpiricalPopulation:
    """Load user-provided ACIC x.csv and one zymu response CSV.

    ``column_map`` maps canonical ``z,y0,y1,mu0,mu1`` to actual headers; for an
    organizer R export use e.g. ``{"y0":"y.0", "y1":"y.1",
    "mu0":"mu.0", "mu1":"mu.1"}``. Numeric z in {0,1} is required.
    Rows are positional; optional (x_id,response_id) verifies identical unique
    IDs in the same order. Equal length alone cannot establish record linkage.

    Text columns are one-hot encoded deterministically on the fixed empirical
    population, before any role is drawn. Numeric-coded categories can be named
    explicitly. This declares the population representation; it does not fit a
    transform to a held-out training role or use responses for preprocessing.
    """
    x_path, response_path = Path(covariates_path), Path(response_path)
    x_header, x_rows, x_digest = _read_csv(x_path, "acic_covariates")
    y_header, y_rows, y_digest = _read_csv(response_path, "acic_response")
    if len(x_rows) != len(y_rows):
        raise ValueError("ACIC covariates and response files must have matching row counts")
    if row_id_columns is not None:
        if len(row_id_columns) != 2:
            raise ValueError("row_id_columns requires the covariate and response ID column names")
        x_id, y_id = row_id_columns
        if x_id not in x_header or y_id not in y_header:
            raise ValueError("row ID columns are missing")
        ids = [row[x_header.index(x_id)] for row in x_rows]
        if len(set(ids)) != len(ids) or ids != [row[y_header.index(y_id)] for row in y_rows]:
            raise ValueError("ACIC row IDs must be unique and identical in the same order")
        keep = [i for i, name in enumerate(x_header) if name != x_id]
        x_rows = [[row[i] for i in keep] for row in x_rows]
        x_header = [x_header[i] for i in keep]
    columns = {key: key for key in ("z", "y0", "y1", "mu0", "mu1")}
    if column_map is not None:
        if set(column_map) - set(columns):
            raise ValueError("column_map accepts only z,y0,y1,mu0,mu1")
        columns.update(column_map)
    if len(set(columns.values())) != 5 or any(value not in y_header for value in columns.values()):
        raise ValueError("ACIC response columns are missing or mapped more than once")
    responses = {
        key: _array([row[y_header.index(value)] for row in y_rows], 1, key)
        for key, value in columns.items()
    }
    if not np.isin(responses["z"], (0, 1)).all():
        raise ValueError("ACIC z must be numeric binary 0/1")
    explicit_categories = set(categorical_columns)
    if explicit_categories - set(x_header):
        raise ValueError("categorical_columns contains an unknown covariate")
    encoded, feature_names, feature_groups = [], [], []
    numeric_columns, category_names, level_counts = [], [], {}
    encoding_spec: list[dict[str, Any]] = []
    for index, name in enumerate(x_header):
        raw = [row[index] for row in x_rows]
        numeric, conversions = [], []
        for value in raw:
            try:
                numeric.append(float(value))
                conversions.append(True)
            except ValueError:
                conversions.append(False)
        is_categorical = name in explicit_categories or not any(conversions)
        if not is_categorical and not all(conversions):
            raise ValueError(f"mixed numeric/text covariate {name!r}; declare it categorical explicitly")
        if is_categorical:
            levels = sorted(set(raw))
            category_names.append(name)
            level_counts[name] = len(levels)
            encoding_spec.append({"column": name, "levels": levels})
            for j, level in enumerate(levels):
                encoded.append(np.asarray([value == level for value in raw], dtype=float))
                feature_names.append(f"{name}[level_{j}]")
                feature_groups.append(name)
        else:
            encoded.append(_array(numeric, 1, name))
            numeric_columns.append(name)
            feature_names.append(name)
            feature_groups.append(name)
            encoding_spec.append({"column": name, "numeric": True})
    encoding_digest = hashlib.sha256(json.dumps(encoding_spec, sort_keys=True).encode()).hexdigest()
    return EmpiricalPopulation(
        np.column_stack(encoded), tuple(feature_names), responses["mu0"], responses["mu1"],
        tuple(feature_groups),
        provenance={
            "source_id": "acic2016", "schema": "acic_csv", "pooling": "single_response",
            "files": [x_digest, y_digest],
            "preprocessing": {"encoding": "fixed_population_one_hot",
                              "numeric_columns": numeric_columns,
                              "categorical_columns": category_names,
                              "categorical_level_counts": level_counts,
                              "encoding_sha256": encoding_digest,
                              "row_alignment": "verified_ids" if row_id_columns else "positional_same_length"},
        },
    )


@dataclass(frozen=True)
class EmpiricalDGPConfig:
    name: str
    dataset_id: str = "custom"
    replication: int = 0
    hidden_columns: tuple[str, ...] = ()
    visible_binning: tuple[tuple[str, int], ...] = ()
    allow_no_hidden_ambiguity: bool = False
    outcome_model: Literal["loaded_means", "synthetic"] = "loaded_means"
    gamma: float = 2.0
    marginal_propensity: float = 0.5
    rct_propensity: float = 0.5
    hidden_outcome_loading_0: float = 1.0
    hidden_outcome_loading_1: float = 1.0
    synthetic_baseline_scale: float = 1.0
    synthetic_cate_intercept: float = 1.0
    synthetic_cate_scale: float = 1.0
    noise_std: float = 1.0
    n_obs: int = 2_000
    n_rct_train: int = 200
    n_rct_tune: int = 200
    n_rct_test: int = 500
    n_eval: int = 2_000
    n_target_reference: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name or not isinstance(self.dataset_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.dataset_id)):
            raise ValueError("name must be nonempty and dataset_id must be a path-free identifier")
        _replication_index(self.replication)
        if isinstance(self.hidden_columns, str):
            raise ValueError("hidden_columns must be a sequence of source column names")
        hidden = tuple(self.hidden_columns)
        if (not hidden or len(set(hidden)) != len(hidden)
                or any(not isinstance(x, str) or not x for x in hidden)):
            raise ValueError("hidden_columns must list distinct source column names")
        binning = tuple((name, bins) for name, bins in self.visible_binning)
        if len({name for name, _ in binning}) != len(binning):
            raise ValueError("visible_binning must not repeat columns")
        for name, bins in binning:
            if (not isinstance(name, str) or not name or isinstance(bins, bool)
                    or not isinstance(bins, int) or bins < 1):
                raise ValueError("visible_binning requires source column names and positive integer bins")
        if set(hidden) & {name for name, _ in binning}:
            raise ValueError("a column cannot be both hidden and visibly binned")
        object.__setattr__(self, "hidden_columns", hidden)
        object.__setattr__(self, "visible_binning", binning)
        if self.outcome_model not in {"loaded_means", "synthetic"}:
            raise ValueError("outcome_model must be loaded_means or synthetic")
        if not isinstance(self.allow_no_hidden_ambiguity, bool):
            raise ValueError("allow_no_hidden_ambiguity must be boolean")
        for key in ("gamma", "marginal_propensity", "rct_propensity", "noise_std",
                    "hidden_outcome_loading_0", "hidden_outcome_loading_1",
                    "synthetic_baseline_scale", "synthetic_cate_intercept", "synthetic_cate_scale"):
            if isinstance(getattr(self, key), bool) or not np.isfinite(getattr(self, key)):
                raise ValueError("empirical numeric controls must be finite numbers")
        if self.gamma < 1 or self.noise_std < 0:
            raise ValueError("gamma must be at least one and noise_std nonnegative")
        if not 0 < self.marginal_propensity < 1 or not 0 < self.rct_propensity < 1:
            raise ValueError("marginal and RCT propensities must lie strictly in (0,1)")
        for key in ("n_obs", "n_rct_train", "n_rct_tune", "n_rct_test", "n_eval", "n_target_reference"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "n_target_reference" else 1):
                raise ValueError("sample budgets must be positive integers; target_reference may be zero")


def _group_mean(values: FloatArray, inverse: NDArray[np.int64], counts: NDArray[np.int64]) -> FloatArray:
    return np.bincount(inverse, weights=values, minlength=len(counts)) / counts


def _visible_population(
    population: EmpiricalPopulation, cfg: EmpiricalDGPConfig
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64], FloatArray, list[str], dict[str, Any]]:
    groups = population.feature_groups
    unknown = (set(cfg.hidden_columns) | {name for name, _ in cfg.visible_binning}) - set(groups)
    if unknown:
        raise ValueError(f"unknown empirical source columns: {sorted(unknown)}")
    hidden_mask = np.asarray([name in cfg.hidden_columns for name in groups])
    if hidden_mask.all():
        raise ValueError("at least one empirical source column must remain visible")
    X = np.array(population.X_full[:, ~hidden_mask], copy=True)
    names = [name for name, hide in zip(population.feature_names, hidden_mask) if not hide]
    visible_groups = [name for name, hide in zip(groups, hidden_mask) if not hide]
    binning_summary = {}
    for name, bins in cfg.visible_binning:
        columns = [i for i, group in enumerate(visible_groups) if group == name]
        if len(columns) != 1 or names[columns[0]] != name:
            raise ValueError("visible_binning applies only to a single numeric source column")
        index = columns[0]
        # Interior empirical quantiles; ties collapse deterministically.
        edges = np.unique(np.quantile(X[:, index], np.arange(1, bins) / bins))
        X[:, index] = np.searchsorted(edges, X[:, index], side="right")
        binning_summary[name] = {"requested_bins": bins, "realized_bins": int(np.unique(X[:, index]).size),
                                 "edges_sha256": _digest_array(edges)}
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    hidden = population.X_full[:, hidden_mask]
    # A scalar injective label for distinct hidden vectors avoids cancellation
    # between several masked columns, including one-hot indicators. This is an
    # explicitly chosen synthetic selection mechanism on the fixed population.
    _, hidden_rank = np.unique(hidden, axis=0, return_inverse=True)
    rank_scale = hidden_rank.std()
    score = (hidden_rank - hidden_rank.mean()) / (rank_scale if rank_scale > 0 else 1)
    centered = score - _group_mean(score, inverse, counts)[inverse]
    maximum, minimum = np.full(len(counts), -np.inf), np.full(len(counts), np.inf)
    np.maximum.at(maximum, inverse, score)
    np.minimum.at(minimum, inverse, score)
    active = (maximum - minimum > 1e-12)[inverse]
    if not active.any() and not cfg.allow_no_hidden_ambiguity:
        raise ValueError(
            "hidden columns have no within-visible-group variation: masking unique continuous rows "
            "does not create residual confounding under the finite empirical law; explicitly coarsen "
            "visible columns or set allow_no_hidden_ambiguity=True for a labeled negative control"
        )
    diagnostics = {
        "population_rows": len(X), "visible_groups": len(counts),
        "singleton_row_fraction": float(np.mean(counts[inverse] == 1)),
        "hidden_ambiguity_row_fraction": float(active.mean()),
        "has_hidden_ambiguity": bool(active.any()),
        "visible_binning": binning_summary,
    }
    return X, inverse, counts, centered, names, diagnostics


def generate_empirical_problem(
    population: EmpiricalPopulation, cfg: EmpiricalDGPConfig, seed: int
) -> SimulationProblem:
    """Draw independent roles with replacement and generate fresh T and Y.

    Within every exact visible-X group, ``e_full=e+alpha*centered_hidden``.
    The largest feasible nonnegative groupwise alpha places all rows within the
    marginal sensitivity odds bounds while preserving the exact group mean e.
    A varying group reaches the requested Gamma unless Gamma=1. Groups without
    hidden variation have realized Gamma=1 and are reported explicitly.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if cfg.dataset_id != "custom" and population.provenance.get("source_id") != cfg.dataset_id:
        raise ValueError("dataset_id does not match the loaded population provenance")
    if "replication" in population.provenance and cfg.replication != population.provenance["replication"]:
        raise ValueError("config replication does not match the loaded IHDP replication")
    X, inverse, counts, hidden, visible_names, diagnostics = _visible_population(population, cfg)
    e = np.full(len(X), cfg.marginal_propensity)
    lower, upper = msm_propensity_bounds(e, cfg.gamma)
    positive, negative = np.zeros(len(counts)), np.zeros(len(counts))
    np.maximum.at(positive, inverse, hidden)
    np.maximum.at(negative, inverse, -hidden)
    alpha = np.zeros(len(counts))
    varying = (positive > 1e-12) & (negative > 1e-12)
    alpha[varying] = np.minimum(
        (upper[0] - e[0]) / positive[varying],
        (e[0] - lower[0]) / negative[varying],
    )
    full = e + alpha[inverse] * hidden
    # Check both arm masses separately. Near one, mean(full) can round to e
    # while the rare control arm has a materially different probability.
    treated_mass = _group_mean(full, inverse, counts)
    control_mass = _group_mean(1 - full, inverse, counts)
    if (np.any((full <= 0) | (full >= 1))
            or not np.allclose(treated_mass / e[0], 1, atol=0, rtol=1e-12)
            or not np.allclose(control_mass / (1 - e[0]), 1, atol=0, rtol=1e-12)):
        raise RuntimeError("finite-population propensity calibration failed its arm-mass constraints")
    marginal_log_odds = np.log(treated_mass) - np.log(control_mass)
    log_odds_ratio = np.log(full) - np.log1p(-full) - marginal_log_odds[inverse]
    local_gamma = np.exp(np.abs(log_odds_ratio))
    if np.any(np.abs(log_odds_ratio) > np.log(cfg.gamma) + 1e-12):
        raise RuntimeError("finite-population propensity calibration failed its marginal/odds constraints")
    if cfg.outcome_model == "loaded_means":
        if population.mu0 is None or population.mu1 is None:
            raise ValueError("loaded_means requires both population arm means")
        mu0, mu1 = population.mu0, population.mu1
    else:
        scale = X.std(axis=0)
        z = (X - X.mean(axis=0)) / np.where(scale > 0, scale, 1)
        visible_signal = z.mean(axis=1)
        mu0 = cfg.synthetic_baseline_scale * visible_signal + cfg.hidden_outcome_loading_0 * hidden
        mu1 = (cfg.synthetic_baseline_scale * visible_signal + cfg.synthetic_cate_intercept
               + cfg.synthetic_cate_scale * visible_signal + cfg.hidden_outcome_loading_1 * hidden)
    target0, target1 = _group_mean(mu0, inverse, counts), _group_mean(mu1, inverse, counts)
    # Normalize selection weights before multiplying outcomes: mu*e can
    # underflow even when the conditional mean mu itself is representable.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        selected0 = _group_mean(mu0 * ((1 - full) / control_mass[inverse]), inverse, counts)
        selected1 = _group_mean(mu1 * (full / treated_mass[inverse]), inverse, counts)
    if not np.isfinite(selected0).all() or not np.isfinite(selected1).all():
        raise ValueError("empirical conditional means are not representable in float64")
    bias0, bias1 = selected0 - target0, selected1 - target1
    group_gamma = np.ones(len(counts))
    np.maximum.at(group_gamma, inverse, local_gamma)
    group_truth = {
        "mu0_target": target0, "mu1_target": target1, "tau_target": target1 - target0,
        "mu0_obs": target0, "mu1_obs": target1,
        "mu0_obs_regression": selected0, "mu1_obs_regression": selected1,
        "confounding_bias_0": bias0, "confounding_bias_1": bias1,
        "regression_discrepancy_0": bias0, "regression_discrepancy_1": bias1,
        "source_shift_0": np.zeros(len(counts)), "source_shift_1": np.zeros(len(counts)),
        "confounding_cate_bias": bias1 - bias0, "source_cate_shift": np.zeros(len(counts)),
        "bias_magnitude": np.sqrt((bias0 ** 2 + bias1 ** 2) / 2),
        "obs_propensity": treated_mass,
        "gamma_required": group_gamma,
        "obs_density": np.ones(len(counts)), "rct_density": np.ones(len(counts)),
    }
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]

    def draw(rng: np.random.Generator, n: int, source: str, role: str) -> tuple[SourceSample, NDArray[np.int64]]:
        indices = rng.integers(0, len(X), size=n)
        propensity = full[indices] if source == "obs" else np.full(n, cfg.rct_propensity)
        treatment = rng.binomial(1, propensity).astype(float)
        outcome = np.where(treatment == 1, mu1[indices], mu0[indices]) + rng.normal(0, cfg.noise_std, n)
        sample = SourceSample(X[indices], treatment, outcome, source, role,
                              known_propensity=propensity if source == "rct" else None)
        return sample, indices

    obs, obs_indices = draw(streams[0], cfg.n_obs, "obs", "obs_train")
    rct_train, _ = draw(streams[1], cfg.n_rct_train, "rct", "rct_train")
    rct_tune, _ = draw(streams[2], cfg.n_rct_tune, "rct", "rct_tune")
    rct_test, _ = draw(streams[3], cfg.n_rct_test, "rct", "rct_test")
    eval_indices = streams[4].integers(0, len(X), size=cfg.n_eval)
    eval_truth = {key: value[inverse[eval_indices]] for key, value in group_truth.items()}
    obs_truth = {key: value[inverse[obs_indices]] for key, value in group_truth.items()}
    obs_truth.update({"individual_treatment_probability": full[obs_indices],
                      "hidden_treatment_signal": hidden[obs_indices],
                      "individual_mu0": mu0[obs_indices], "individual_mu1": mu1[obs_indices]})
    reference = None
    if cfg.n_target_reference:
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x454D5052,)))
        reference = TargetCovariateSample(X[rng.integers(0, len(X), size=cfg.n_target_reference)])
    diagnostics.update({"gamma_requested": cfg.gamma, "gamma_realized_max": float(local_gamma.max()),
                        "gamma_varying_group_fraction": float(varying.mean()),
                        "max_group_marginal_error": float(np.max(np.abs(treated_mass - e[0]))),
                        "mean_abs_confounding_cate_bias": float(np.mean(np.abs((bias1 - bias0)[inverse])))})
    provenance = json.loads(json.dumps(dict(population.provenance)))
    metadata = {
        "generator": "empirical_v1", "generator_version": EMPIRICAL_DGP_VERSION,
        "config": asdict(cfg), "dataset_provenance": provenance,
        "target_population": "uniform loaded empirical rows with explicitly coarsened visible covariates",
        "estimand": "E[mu1(row)-mu0(row) | displayed X] under the uniform empirical row distribution",
        "sampling": "all roles independently sample population rows with replacement; fresh treatment and outcome noise",
        "outcome_reassignment": "original T,yf,ycf,y0,y1 are discarded; Y is generated from the selected arm mean",
        "hidden_score_definition": "standardized lexicographic rank of distinct hidden covariate vectors, centered within displayed-X groups",
        "visible_feature_names": visible_names, "diagnostics": diagnostics,
        "population_covariates_sha256": _digest_array(population.X_full),
        "visible_transform_sha256": _digest_array(X),
        "potential_means_sha256": _digest_array(np.column_stack((mu0, mu1))),
    }
    return SimulationProblem(
        cfg.name, int(seed), obs, rct_train, rct_tune, rct_test,
        X[eval_indices], eval_truth["tau_target"], eval_truth["bias_magnitude"],
        metadata=metadata,
        truth=SimulationTruth(eval_truth, obs_truth, cfg.n_eval, cfg.n_obs,
                              "exact finite-population conditional averages over rows sharing displayed X; "
                              "confounding_bias_t=E[mu_t|X,T=t]-E[mu_t|X]; source_shift_t=0"),
        target_reference=reference,
    )
