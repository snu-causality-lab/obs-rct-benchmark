"""Optional bridge to the unmodified Wu--Yang integrative R-learner.

Obtain the pinned R checkout separately and set INTEGRATIVE_R_SOURCE. Set
INTEGRATIVE_R_RSCRIPT to select an Rscript executable (default: Rscript).
The R packages caret, glmnet, stringr and ncvreg must already be installed.
No R implementation or optional Python dependency is bundled or imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .data import TrainingView


INTEGRATIVE_R_REVISION = "a46ce2c1cc814865898b863d6a8e130bbc2c802b"
INTEGRATIVE_R_FILES = {
    "R/intRlearner.R": "98563e2423f41153fcbb2548f3b54ac779c72bcb8d154608bf51f778086fdce8",
    "R/utils.R": "2b85e2e56db83e33427f6f43aa1b59d995bafb6d46bfc1aec8e5f2e6a215525b",
    "DESCRIPTION": "fa2885ec2a8aef1c1502ab3984647f0ae25815c26494e2444dfd296c8bec4810",
    "NAMESPACE": "736cb2c31c983b3a07d06d09b8ef93d89972f312c1db584ff253a0281d082bac",
}
INTEGRATIVE_R_PACKAGES = ("caret", "glmnet", "stringr", "ncvreg")

# This is interchange/orchestration code, not a copy of an upstream routine.
_R_RUNTIME = r'''
for (pkg in c("caret", "glmnet", "stringr", "ncvreg")) {
    if (!requireNamespace(pkg, quietly=TRUE)) stop("missing required R package")
}
runtime_lines <- function() {
    records <- paste("R", R.version.string, sep="\t")
    for (pkg in sort(loadedNamespaces())) {
        # R excludes its built-in base namespace from getNamespaceInfo(). Its
        # package metadata lives in the R installation that runs this process.
        where <- if (identical(pkg, "base")) file.path(R.home("library"), "base") else
            getNamespaceInfo(asNamespace(pkg), "path")
        where <- normalizePath(where, mustWork=TRUE)
        records <- c(records, paste("package", pkg,
            as.character(utils::packageVersion(pkg, lib.loc=dirname(where))), where, sep="\t"))
    }
    records
}
'''

_R_PROBE = _R_RUNTIME + r'''
cat(paste(runtime_lines(), collapse="\n"), "\n", sep="")
'''

_R_BRIDGE = _R_RUNTIME + r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 11L) stop("invalid bridge argument count")
source_root <- normalizePath(args[[1]], mustWork=TRUE)
work <- normalizePath(args[[2]], mustWork=TRUE)
seed <- as.integer(args[[3]])
basis <- args[[4]]
fold_count <- as.integer(args[[5]])
nr <- as.integer(args[[6]])
no <- as.integer(args[[7]])
nq <- as.integer(args[[8]])
d <- as.integer(args[[9]])
expected_p <- as.integer(args[[10]])
expected_revision <- args[[11]]
read_values <- function(name, count) {
    path <- file.path(work, name)
    if (file.info(path)$size != count * 8) stop("invalid float64 input size")
    values <- readBin(path, what="double", n=count, size=8L, endian="little")
    if (length(values) != count || any(!is.finite(values))) stop("invalid numeric input")
    values
}
native <- new.env(parent=globalenv())
source_files <- file.path(source_root, c("R/utils.R", "R/intRlearner.R"))
for (path in source_files) sys.source(path, envir=native)
if (!identical(environment(native$intrlearner), native) ||
    !identical(environment(native$predict.rlasso), native)) stop("unexpected native function environment")
writeLines(normalizePath(source_files, mustWork=TRUE), file.path(work, "sources.txt"))
x <- matrix(read_values("train_X.bin", (nr+no)*d), ncol=d, byrow=TRUE)
query <- matrix(read_values("query_X.bin", nq*d), ncol=d, byrow=TRUE)
w <- read_values("train_T.bin", nr+no)
y <- read_values("train_Y.bin", nr+no)
foldid <- as.integer(read_values("foldid.bin", nr+no))
s <- c(rep(1, nr), rep(0, no))
if (basis == "quadratic") {
    x <- native$poly_interaction(x, d=2, interact=TRUE)
    # Native poly_interaction drops the matrix dimension for a single row.
    # Repeating a prediction row preserves its monomials without fitting on it.
    if (nrow(query) == 1L && ncol(query) > 1L) query <- rbind(query, query)
    query <- native$poly_interaction(query, d=2, interact=TRUE)
    query <- query[seq_len(nq), , drop=FALSE]
} else if (basis != "linear") stop("unknown basis")
if (ncol(x) != expected_p || ncol(query) != expected_p) stop("invalid native basis dimension")
set.seed(seed)
fit <- native$intrlearner(x=x, w=w, y=y, s=s, p=d+1L,
    k_folds=fold_count, foldid=foldid)
prediction <- as.numeric(native$predict.rlasso(fit, query))
if (length(prediction) != nq || any(!is.finite(prediction))) stop("invalid native prediction")
if (length(fit$tau_beta) != 1L+2L*expected_p ||
    length(fit$standardization$mean) != expected_p ||
    length(fit$standardization$std) != expected_p) stop("invalid native parameter map")
writeBin(prediction, file.path(work, "prediction.bin"), size=8L, endian="little")
writeBin(as.numeric(fit$tau_beta), file.path(work, "tau_beta.bin"), size=8L, endian="little")
writeBin(as.numeric(fit$standardization$mean), file.path(work, "center.bin"), size=8L, endian="little")
writeBin(as.numeric(fit$standardization$std), file.path(work, "scale.bin"), size=8L, endian="little")
writeLines(runtime_lines(), file.path(work, "runtime.tsv"))
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
        stderr=subprocess.DEVNULL, timeout=15,
    ).strip()


def _source_provenance() -> dict:
    result = {
        "expected_revision": INTEGRATIVE_R_REVISION,
        "availability": "unavailable", "declared_git_revision": None,
        "package_file_sha256": {}, "tracked_changes": None,
        "untracked_changes": None, "extra_source_files": [],
        "source_files": ["R/utils.R", "R/intRlearner.R"], "verified": False,
    }
    raw = os.environ.get("INTEGRATIVE_R_SOURCE")
    if not raw:
        return result
    root = Path(raw).expanduser().resolve()
    try:
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            return result
        result["declared_git_revision"] = _git(root, "rev-parse", "HEAD")
        result["tracked_changes"] = bool(_git(
            root, "status", "--porcelain", "--untracked-files=no",
        ))
        all_changes = _git(root, "status", "--porcelain", "--untracked-files=all")
        result["untracked_changes"] = any(
            line.startswith("?? ") for line in all_changes.splitlines()
        )
        for name in INTEGRATIVE_R_FILES:
            path = root / name
            if path.resolve() != path or not path.is_file():
                return result
            result["package_file_sha256"][name] = _sha256(path)
        result["extra_source_files"] = sorted(
            path.relative_to(root).as_posix() for path in (root / "R").rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in INTEGRATIVE_R_FILES
        )
        result["availability"] = "available"
        result["verified"] = (
            result["declared_git_revision"] == INTEGRATIVE_R_REVISION
            and not result["tracked_changes"] and not result["untracked_changes"]
            and not result["extra_source_files"]
            and result["package_file_sha256"] == INTEGRATIVE_R_FILES
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _rscript() -> Path:
    name = os.environ.get("INTEGRATIVE_R_RSCRIPT", "Rscript")
    located = shutil.which(os.path.expanduser(name))
    if located is None:
        raise RuntimeError("Integrative R-learner requires Rscript; set INTEGRATIVE_R_RSCRIPT")
    return Path(located).resolve()


def _runtime_provenance(text: str, executable: Path) -> dict:
    """Keep actual namespace identities, with filesystem locations hashed."""
    result = {
        "availability": "available", "r_version": None,
        "executable_sha256": _sha256(executable),
        "executable_path_sha256": hashlib.sha256(str(executable).encode()).hexdigest(),
        "namespace_packages": {},
    }
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] == "R" and result["r_version"] is None:
            result["r_version"] = parts[1]
        elif len(parts) == 4 and parts[0] == "package":
            _, name, version, raw = parts
            if not name or name in result["namespace_packages"]:
                raise ValueError("invalid R namespace metadata")
            root = Path(raw).resolve(strict=True)
            result["namespace_packages"][name] = {
                "version": version,
                "namespace_path_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
                "description_sha256": _sha256(root / "DESCRIPTION"),
            }
        elif line:
            raise ValueError("invalid R runtime metadata")
    if result["r_version"] is None or not set(INTEGRATIVE_R_PACKAGES).issubset(
            result["namespace_packages"]):
        raise ValueError("incomplete R runtime metadata")
    return result


def integrative_r_provenance() -> dict:
    """Source and available R namespace evidence without public absolute paths.

    This inspects installed R namespaces, but never fits a model. Job-level
    CPU/thread and library settings are inherited unchanged from the caller.
    """
    result = _source_provenance()
    result["runtime"] = {"availability": "unavailable"}
    try:
        executable = _rscript()
        probe = subprocess.run(
            [str(executable), "--vanilla", "-e", _R_PROBE],
            capture_output=True, text=True, timeout=120, check=True,
        )
        result["runtime"] = _runtime_provenance(probe.stdout, executable)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        pass
    return result


def _basis_matrix(X, basis: str) -> np.ndarray:
    """Explicit degree-two monomial map used to check native dimensions."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("covariates must be a nonempty matrix")
    if not np.isfinite(values).all():
        raise ValueError("covariates must be finite")
    if basis == "linear":
        return values
    if basis != "quadratic":
        raise ValueError("basis must be linear or quadratic")
    with np.errstate(over="raise", invalid="raise"):
        columns = [values, values**2]
        # R combn orders pairs lexicographically by their first index.
        columns.extend((values[:, i] * values[:, j])[:, None]
                       for i in range(values.shape[1] - 1)
                       for j in range(i + 1, values.shape[1]))
        return np.column_stack(columns)


def _native_prediction_map(X, tau_beta, center, scale, *, basis: str) -> np.ndarray:
    """Decode the native coefficients; confounding slopes do not enter CATE."""
    values = _basis_matrix(X, basis)
    p = values.shape[1]
    beta = np.asarray(tau_beta, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if beta.shape != (1 + 2 * p,) or center.shape != (p,) or scale.shape != (p,):
        raise ValueError("invalid native coefficient dimensions")
    if (not np.isfinite(beta).all() or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("invalid native coefficients or standardization")
    return beta[0] + ((values - center) / scale) @ beta[1:p+1]


def _stratified_foldid(training: TrainingView, k_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    for sample in (training.rct_train, training.obs_train):
        folds = np.empty(sample.n, dtype=np.int64)
        for arm in (0, 1):
            indices = np.flatnonzero(sample.T == arm)
            if len(indices) < k_folds:
                raise ValueError("each source and treatment arm needs at least k_folds rows")
            folds[rng.permutation(indices)] = np.arange(len(indices)) % k_folds + 1
        result.append(folds)
    return np.concatenate(result)


def _read_float64(path: Path, length: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != length * 8:
        raise ValueError("R returned an invalid float64 file size")
    result = np.fromfile(path, dtype="<f8")
    if result.shape != (length,) or not np.isfinite(result).all():
        raise FloatingPointError("Integrative R-learner returned nonfinite values")
    return result.astype(np.float64, copy=False)


@dataclass
class OfficialIntegrativeR:
    """Fixed native SCAD recipe on an explicitly named finite-dimensional basis.

    ``linear`` is the benchmark's restricted raw-covariate recipe.
    ``quadratic`` uses the author's ``gen_data`` example feature map: the native
    degree-two polynomial/interaction expansion on all covariates. Both jointly
    fit effect and confounding functions. The 44 native scale/gamma searches,
    nuisance models and internally estimated RCT propensities are preserved.
    Source/arm-stratified folds are the disclosed benchmark adaptation. Pooled
    fitting covariates determine the native centering/scaling; query covariates
    affect prediction only. Tuning and target-reference samples are unused.
    """

    basis: str = "linear"
    k_folds: int = 5
    timeout_seconds: int = 1200

    def __post_init__(self):
        if self.basis not in {"linear", "quadratic"}:
            raise ValueError("basis must be linear or quadratic")
        if isinstance(self.k_folds, bool) or not isinstance(self.k_folds, int) or self.k_folds < 3:
            raise ValueError("k_folds must be an integer at least 3")
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, int) or self.timeout_seconds < 1):
            raise ValueError("timeout_seconds must be a positive integer")

    def fit_predict(self, training: TrainingView, eval_X, *, seed: int):
        self.native_parameters_ = None
        self.provenance_ = None
        self.last_run_diagnostics_ = None
        self.attempt_provenance_ = None
        self.last_run_provenance_ = {"status": "not_started"}
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                or seed < 0 or seed >= 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        query = np.asarray(eval_X, dtype=np.float64)
        query_basis = _basis_matrix(query, self.basis)
        if query.shape[1] != training.obs_train.d:
            raise ValueError("query covariate dimension does not match training")
        p = query_basis.shape[1]
        for sample in (training.rct_train, training.obs_train):
            expanded = _basis_matrix(sample.X, self.basis)
            if np.any(np.max(expanded, axis=0) == np.min(expanded, axis=0)):
                raise ValueError("every basis column must vary in each fitting source")
        foldid = _stratified_foldid(training, self.k_folds, int(seed))
        source = _source_provenance()
        if not source["verified"]:
            raise RuntimeError("set INTEGRATIVE_R_SOURCE to the clean locked public checkout")
        executable = _rscript()
        executable_hash = _sha256(executable)
        root = Path(os.environ["INTEGRATIVE_R_SOURCE"]).expanduser().resolve()
        samples = (training.rct_train, training.obs_train)
        native_seed = int(seed) % (2**31 - 1)
        self.attempt_provenance_ = {
            "source": source, "basis": self.basis,
            "basis_origin": ("benchmark_raw_covariate_recipe" if self.basis == "linear"
                             else "author_gen_data_degree_two_feature_map"),
            "k_folds": self.k_folds, "timeout_seconds": self.timeout_seconds,
            "supplied_seed": int(seed), "native_seed": native_seed,
            "executable_sha256": executable_hash,
            "executable_path_sha256": hashlib.sha256(str(executable).encode()).hexdigest(),
            "fold_rule": "source_arm_stratified_numpy_default_rng_v1",
            "foldid_sha256": hashlib.sha256(np.asarray(foldid, dtype="<i8").tobytes()).hexdigest(),
            "known_rct_propensity": "not_used_native_estimation",
            "bridge_sha256": hashlib.sha256(_R_BRIDGE.encode()).hexdigest(),
        }
        with tempfile.TemporaryDirectory(prefix="integrative-r-") as temporary:
            work = Path(temporary)
            inputs = {
                "train_X.bin": np.concatenate([s.X for s in samples]),
                "train_T.bin": np.concatenate([s.T for s in samples]),
                "train_Y.bin": np.concatenate([s.Y for s in samples]),
                "query_X.bin": query, "foldid.bin": foldid,
            }
            for name, values in inputs.items():
                np.asarray(values, dtype="<f8").tofile(work / name)
            bridge = work / "bridge.R"
            bridge.write_text(_R_BRIDGE, encoding="utf-8")
            command = [str(executable), "--vanilla", str(bridge), str(root), str(work),
                       str(native_seed), self.basis, str(self.k_folds),
                       str(training.rct_train.n), str(training.obs_train.n),
                       str(len(query)), str(query.shape[1]), str(p), INTEGRATIVE_R_REVISION]
            status, completed = "launch_error", None
            try:
                with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
                    completed = subprocess.run(command, cwd=work, stdout=stdout,
                                               stderr=stderr, timeout=self.timeout_seconds,
                                               check=False)
                status = "validation_pending" if completed.returncode == 0 else "native_error"
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                raise RuntimeError("Integrative R-learner exceeded timeout_seconds") from exc
            finally:
                # The server admission driver may retain these complete logs on
                # failure. They are deliberately separate from public metadata.
                self.last_run_diagnostics_ = {
                    name: (work / f"{name}.log").read_text(encoding="utf-8", errors="replace")
                    for name in ("stdout", "stderr")
                }
                self.last_run_provenance_ = {
                    **self.attempt_provenance_,
                    "stdout_sha256": _sha256(work / "stdout.log"),
                    "stderr_sha256": _sha256(work / "stderr.log"),
                    "native_seed": native_seed, "status": status,
                    "returncode": None if completed is None else completed.returncode,
                }
            if completed.returncode != 0:
                raise RuntimeError(f"Integrative R-learner R process failed with exit code {completed.returncode}")
            try:
                paths = (work / "sources.txt").read_text(encoding="utf-8").splitlines()
                if paths != [str(root / name) for name in source["source_files"]]:
                    raise RuntimeError("R source paths do not match the verified checkout")
                if _source_provenance() != source or _sha256(executable) != executable_hash:
                    raise RuntimeError("Integrative R-learner source or executable changed during fitting")
                runtime = _runtime_provenance((work / "runtime.tsv").read_text(encoding="utf-8"), executable)
                prediction = _read_float64(work / "prediction.bin", len(query))
                parameters = {
                    "tau_beta": _read_float64(work / "tau_beta.bin", 1 + 2*p),
                    "center": _read_float64(work / "center.bin", p),
                    "scale": _read_float64(work / "scale.bin", p),
                }
                mapped = _native_prediction_map(query, basis=self.basis, **parameters)
                if not np.allclose(prediction, mapped, atol=1e-10, rtol=1e-8):
                    raise RuntimeError("native prediction disagrees with its declared CATE map")
            except Exception:
                self.last_run_provenance_["status"] = "validation_error"
                raise
        self.native_parameters_ = parameters
        self.last_run_provenance_["status"] = "success"
        self.provenance_ = {**self.last_run_provenance_, "runtime": runtime}
        return prediction
