"""RCT-only adapter to a separately built, pinned official grf installation.

Set GRF_SOURCE, GRF_LIBRARY and GRF_BUILD_MANIFEST to a clean upstream checkout,
its dedicated installed R library and the retained build record. GRF_RSCRIPT
selects Rscript. No upstream R/C++ implementation is included here. A recorded
build binds the observed source and installed bytes; it is not a compiler proof.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .data import SourceSample, TrainingView


GRF_REVISION = "5bee99b51471f76cb2d63acbc8a9b0ffec408ba0"
GRF_VERSION = "2.6.1"
# Source identities only. No upstream implementation is copied into this module.
GRF_SOURCE_FILES_SHA256 = {
    "r-package/grf/DESCRIPTION": "30058538ee88d58aef86afe6069528a9da2261f2bdc1dc46bf6cd00abac1369b",
    "r-package/grf/R/causal_forest.R": "55910469846d24075b04e614fe06fb3473f7ed58ea75e698497d6f4f07c16c3d",
    "r-package/grf/R/regression_forest.R": "e071eac1cf248ee7b9efe5fb7bad854591f8af1ece295ef31489eaff400b1502",
    "r-package/grf/R/input_utilities.R": "dc4731cf7c791581611515ef741bdeb5f9aac05f23d6f43fd88b63b8e1f9804c",
    "r-package/grf/R/analysis_tools.R": "3f0eecb8d2749cb0f3c9b03e2140acc68e396162523f635e059f0191d80fb684",
    "r-package/grf/R/grf_options.R": "b929dd6abf3146e94d90e30c63f80728941c1d4c862a31d152fc6f5e8078dba7",
    "r-package/grf/bindings/CausalForestBindings.cpp": "9d817009c1e324dd5686d45cd20a23959d98bc8977d2e002b4abeb440e5c2459",
    "r-package/grf/bindings/RegressionForestBindings.cpp": "fa6c4019f0da6a2ab92170b27dca76272dda0bada1574b56b97d3352644af6be",
    "r-package/grf/bindings/AnalysisToolsBindings.cpp": "4558be1ecfb1cb272a303f81ece4aac8d66b1e87df9c26259877f08e6b71e369",
    "r-package/grf/bindings/RcppUtilities.cpp": "c3a5e07c469d7c997ea9b6c134fb3b07524674f21bdc5d512ae1660c94f03585",
    "r-package/grf/bindings/Makevars": "9a79f7e284cf1c78759c3964ed88e04dfbe1f11022622da0d0c866cb6f397134",
    "r-package/grf/src/RcppExports.cpp": "bd6663a2b62a412cd0a84ae74cbba782d5e062c2142ebd982550429f69eb9d62",
    "core/src/forest/ForestOptions.cpp": "ad6f454a23b4355852a777177cf5fd07b883e6e5b4d5c2aceb192c2947f3a808",
    "core/src/forest/ForestTrainer.cpp": "c01432a97e5cfcb713bfc1e486781a1c59d4f1f1db92df82b8d345165e38ab02",
    "core/src/forest/ForestTrainers.cpp": "e2c940769b65d6c5d9f2f3d9fa1cc72e3cbb54f4ec63bcc6095855706c07b250",
    "core/src/forest/ForestPredictors.cpp": "47a9ab25f6a9d49a64f8346c221ee051196f59dd99d2a32f8b8103bd6dc50af0",
    "core/src/prediction/InstrumentalPredictionStrategy.cpp": "4480418c7ab3c89f09fcc61e773d0e2bd709392539a8fc448145dfafefa1b89f",
    "core/src/prediction/collector/SampleWeightComputer.cpp": "763d8ac8e02f9e946ea76985bc2ccb64d1600a68801ae0a3db6411734199514a",
    "core/src/prediction/collector/OptimizedPredictionCollector.cpp": "c76dc0c5912252cd773e62129ca78272b06c06a4e7b05003366fe4649cb30a2e",
    "core/src/relabeling/InstrumentalRelabelingStrategy.cpp": "49f0cbc76183754eb6121d4869ff9bc76928e20df0a9554eca00b2b392bd6bfc",
    "core/src/splitting/InstrumentalSplittingRule.cpp": "ce9dc1da59abfb6a4bef125de1b8f3eb1ebccebfd3b3fcc53dcec128ee9b6aac",
    "core/src/tree/TreeTrainer.cpp": "8ff5406d3ba7bededc80653ef0570e8a6208538095a2da5599699734e5839e41",
    "core/src/sampling/RandomSampler.cpp": "e414de2794420cfc022a0b726e97441814dc91f3a05a5ac3c3f3fba676eef42e",
}

# Binary interchange and orchestration only; fitting and prediction are native.
_R_RUNTIME = r'''
runtime_lines <- function() {
    ns <- asNamespace("grf")
    root <- normalizePath(getNamespaceInfo(ns, "path"), mustWork=TRUE)
    if (!identical(root, normalizePath(file.path(library_root, "grf"), mustWork=TRUE)))
        stop("grf namespace does not belong to the selected library")
    dll <- getLoadedDLLs()[["grf"]]
    if (is.null(dll)) stop("grf native DLL is not loaded")
    dll_path <- normalizePath(dll[["path"]], mustWork=TRUE)
    if (!startsWith(dll_path, paste0(root, .Platform$file.sep, "libs", .Platform$file.sep)))
        stop("grf DLL does not belong to the selected package")
    records <- c(paste("R", R.version.string, sep="\t"),
        paste("namespace", "grf", as.character(utils::packageVersion("grf", lib.loc=library_root)), root, sep="\t"),
        paste("dll", "grf", dll_path, sep="\t"))
    for (pkg in sort(unique(c(loadedNamespaces(), "Rcpp", "RcppEigen")))) {
        where <- if (identical(pkg, "base")) file.path(R.home("library"), "base") else
            if (pkg %in% loadedNamespaces()) getNamespaceInfo(asNamespace(pkg), "path") else
            find.package(pkg)
        where <- normalizePath(where, mustWork=TRUE)
        records <- c(records, paste("package", pkg,
            as.character(utils::packageVersion(pkg, lib.loc=dirname(where))), where, sep="\t"))
    }
    records
}
load_grf <- function() {
    .libPaths(c(library_root, .libPaths()))
    if (!requireNamespace("grf", quietly=TRUE, lib.loc=library_root)) stop("grf is unavailable")
}
'''

_R_PROBE = _R_RUNTIME + r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("invalid probe argument count")
library_root <- normalizePath(args[[1]], mustWork=TRUE)
load_grf()
cat(paste(runtime_lines(), collapse="\n"), "\n", sep="")
'''

_R_BRIDGE = _R_RUNTIME + r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8L) stop("invalid bridge argument count")
library_root <- normalizePath(args[[1]], mustWork=TRUE)
work <- normalizePath(args[[2]], mustWork=TRUE)
seed <- as.integer(args[[3]])
nr <- as.integer(args[[4]])
nq <- as.integer(args[[5]])
d <- as.integer(args[[6]])
num_trees <- as.integer(args[[7]])
num_threads <- as.integer(args[[8]])
if (anyNA(c(seed,nr,nq,d,num_trees,num_threads)) || seed < 0L ||
    min(nr,nq,d,num_trees,num_threads) < 1L) stop("invalid integer argument")
read_values <- function(name, count) {
    path <- file.path(work, name)
    if (!file.exists(path) || file.info(path)$size != count * 8) stop("invalid input size")
    values <- readBin(path, what="double", n=count, size=8L, endian="little")
    if (length(values) != count || any(!is.finite(values))) stop("invalid numeric input")
    values
}
x <- matrix(read_values("rct_X.bin", nr*d), nrow=nr, ncol=d, byrow=TRUE)
w <- read_values("rct_T.bin", nr)
y <- read_values("rct_Y.bin", nr)
propensity <- read_values("rct_propensity.bin", nr)
query <- matrix(read_values("query_X.bin", nq*d), nrow=nq, ncol=d, byrow=TRUE)
if (any(!w %in% c(0,1)) || length(unique(w)) != 2L ||
    any(propensity <= 0 | propensity >= 1)) stop("invalid randomized fitting sample")
load_grf()
writeLines(runtime_lines(), file.path(work, "runtime_before.tsv"))
options(grf.legacy.seed=FALSE, grf.verbose=FALSE)
set.seed(seed)
fit <- grf::causal_forest(X=x, Y=y, W=w, Y.hat=NULL, W.hat=propensity,
    num.trees=num_trees, sample.weights=NULL, clusters=NULL,
    equalize.cluster.weights=FALSE, sample.fraction=0.5,
    mtry=min(ceiling(sqrt(d) + 20), d), min.node.size=5,
    honesty=TRUE, honesty.fraction=0.5, honesty.prune.leaves=TRUE,
    alpha=0.05, imbalance.penalty=0, stabilize.splits=TRUE,
    ci.group.size=2, tune.parameters="none", compute.oob.predictions=TRUE,
    num.threads=num_threads, seed=seed)
prediction <- as.numeric(stats::predict(fit, newdata=query,
    estimate.variance=FALSE, num.threads=num_threads,
    linear.correction.variables=NULL)$predictions)
if (length(prediction) != nq || any(!is.finite(prediction))) stop("invalid native prediction")
if (length(fit$W.hat) != nr || !isTRUE(all.equal(as.numeric(fit$W.hat), propensity,
    tolerance=0, check.attributes=FALSE))) stop("known propensity was not retained")
if (length(fit$Y.hat) != nr || any(!is.finite(fit$Y.hat))) stop("invalid native outcome nuisance")
writeBin(prediction, file.path(work, "prediction.bin"), size=8L, endian="little")
writeLines(runtime_lines(), file.path(work, "runtime.tsv"))
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_hash(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.DEVNULL, timeout=15).strip()


def _relative_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid recorded relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("invalid recorded relative path")
    return path


def _file_inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("installed package must not contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _source_inventory(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    files, links = {}, {}
    names = _git(root, "ls-files", "-z").split("\0")
    for name in names:
        if not name:
            continue
        path = root / _relative_path(name)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError("source link leaves the selected checkout")
        if path.is_symlink():
            links[name] = os.readlink(path)
        if path.is_file():
            files[name] = _sha256(path)
        elif not path.is_symlink() or not path.is_dir():
            raise RuntimeError("tracked source is not a regular file or internal directory link")
    return files, links


def _pinned_source_inventory(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Hash pinned Git objects independently of index flags and build metadata.

    Raw worktree bytes must reproduce the immutable blob IDs in the pinned
    tree. This bypasses status/index exclusions and archive export attributes.
    """
    files, links = {}, {}
    for entry in _git(root, "ls-tree", "-rz", GRF_REVISION).split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, kind, object_id = metadata.split(" ")
        path = root / _relative_path(name)
        if kind != "blob" or not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise RuntimeError("unsupported pinned Git source object")
        if mode == "120000":
            if not path.is_symlink():
                raise RuntimeError("pinned source symlink changed type")
            links[name] = os.readlink(path)
            raw = os.fsencode(links[name])
        elif mode in {"100644", "100755"}:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("pinned source file changed type")
            raw = path.read_bytes()
        else:
            raise RuntimeError("unsupported pinned Git source mode")
        blob_id = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if blob_id != object_id:
            raise RuntimeError("source bytes differ from a pinned Git blob")
        if path.is_file():
            files[name] = _sha256(path)
    return files, links


def _load_manifest(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate build record field")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite build record value")

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("build record must be an object")
    return result


def _rscript() -> Path:
    requested = os.path.expanduser(os.environ.get("GRF_RSCRIPT", "Rscript"))
    located = shutil.which(requested)
    if located is None:
        raise RuntimeError("grf requires Rscript; set GRF_RSCRIPT")
    return Path(located).resolve(strict=True)


def _required_root(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"set {name} for the separately built official grf package")
    return Path(raw).expanduser().resolve(strict=True)


def _dependency_identity(name: str, declaration: dict, root: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", name) or not isinstance(declaration, dict):
        raise ValueError("invalid dependency build record")
    if not isinstance(declaration.get("version"), str) or not declaration["version"]:
        raise ValueError("dependency version is missing")
    if not isinstance(declaration.get("library_root"), str) or not declaration["library_root"]:
        raise ValueError("dependency library root is required")
    if root.parent != Path(declaration["library_root"]).expanduser().resolve(strict=True):
        raise RuntimeError("dependency resolves outside its recorded library")
    description_hash = _sha256(root / "DESCRIPTION")
    if description_hash != declaration.get("description_sha256"):
        raise RuntimeError("dependency DESCRIPTION differs from the build record")
    files = {}
    expected_files = declaration.get("files_sha256", {})
    if not isinstance(expected_files, dict):
        raise ValueError("dependency file inventory must be an object")
    for relative, expected in expected_files.items():
        path = root / _relative_path(relative)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError("dependency file resolves outside its package")
        files[relative] = _sha256(path)
        if files[relative] != expected:
            raise RuntimeError("dependency bytes differ from the build record")
    return {"version": declaration["version"], "description_sha256": description_hash,
            "files_sha256": files, "namespace_path_sha256": _path_hash(root)}


def _inspect_installation() -> tuple[dict, dict]:
    source = _required_root("GRF_SOURCE")
    library = _required_root("GRF_LIBRARY")
    manifest_path = _required_root("GRF_BUILD_MANIFEST")
    manifest = _load_manifest(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise RuntimeError("unsupported grf build record schema")
    if (Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source
            or _git(source, "rev-parse", "HEAD") != GRF_REVISION
            or _git(source, "status", "--porcelain", "--untracked-files=all")):
        raise RuntimeError("grf source must be the clean pinned checkout")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    files, links = _source_inventory(source)
    pinned_files, pinned_links = _pinned_source_inventory(source)
    if files != pinned_files or links != pinned_links:
        raise RuntimeError("complete source differs from the pinned Git objects")
    if (manifest.get("repo_revision") != GRF_REVISION or manifest.get("repo_tree") != tree
            or manifest.get("source_files_sha256") != files
            or manifest.get("source_symlink_targets", {}) != links):
        raise RuntimeError("source differs from its grf build record")
    if any(files.get(name) != expected for name, expected in GRF_SOURCE_FILES_SHA256.items()):
        raise RuntimeError("critical grf source hashes differ from the pin")
    if (Path(manifest["source_root"]).resolve(strict=True) != source
            or Path(manifest["library_root"]).resolve(strict=True) != library
            or not manifest.get("build_command")):
        raise RuntimeError("build record does not describe the selected source and library")
    package = library / "grf"
    if package.is_symlink() or not package.is_dir():
        raise RuntimeError("grf must be installed as a dedicated package directory")
    installed = _file_inventory(package)
    declaration = manifest.get("installed_package", {})
    if not isinstance(declaration, dict):
        raise ValueError("installed grf build record must be an object")
    if (declaration.get("name") != "grf" or declaration.get("version") != GRF_VERSION
            or declaration.get("files_sha256") != installed):
        raise RuntimeError("complete installed grf inventory differs from its build record")
    required = {"DESCRIPTION", "NAMESPACE", "R/grf.rdb", "R/grf.rdx"}
    if (not required.issubset(installed)
            or not any(name.startswith("libs/") and Path(name).suffix in {".so", ".dll", ".dylib"}
                       for name in installed)):
        raise RuntimeError("installed grf lacks its R namespace or native library")
    executable = _rscript()
    executable_hash = _sha256(executable)
    if manifest.get("rscript_sha256") != executable_hash or not isinstance(manifest.get("r_version"), str):
        raise RuntimeError("Rscript or recorded R version differs from the build record")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict) or not {"Rcpp", "RcppEigen"}.issubset(dependencies):
        raise RuntimeError("grf build record lacks its native dependencies")
    dependency_identities = {}
    for name, declaration in dependencies.items():
        if not isinstance(declaration, dict):
            raise ValueError("invalid dependency build record")
        if not isinstance(declaration.get("library_root"), str) or not declaration["library_root"]:
            raise ValueError("dependency library root is required")
        base = Path(declaration["library_root"]).expanduser().resolve(strict=True)
        dependency_identities[name] = _dependency_identity(name, declaration, (base / name).resolve(strict=True))
    result = {
        "availability": "available", "verified": True,
        "expected_revision": GRF_REVISION, "revision": GRF_REVISION,
        "repo_tree": tree, "version": GRF_VERSION,
        "source_files_sha256": files, "source_symlink_targets": links,
        "installed_package_files_sha256": installed,
        "build_manifest_sha256": _sha256(manifest_path),
        "rscript_sha256": executable_hash, "r_version": manifest["r_version"],
        "source_path_sha256": _path_hash(source), "library_path_sha256": _path_hash(library),
        "executable_path_sha256": _path_hash(executable),
        "dependencies": dependency_identities,
        "build_binding": "recorded_source_and_installed_bytes_not_compiler_proof",
        "verification_scope": (
            "Pinned Git-object source; complete installed grf files; Rscript; selected dependency files; "
            "namespace identities. The complete R runtime and all transitive shared libraries are not authenticated."
        ),
    }
    return result, {"source": source, "library": library, "package": package,
                    "manifest_path": manifest_path, "manifest": manifest, "executable": executable,
                    "expected_dependencies": dependency_identities}


def _runtime_provenance(raw: str, paths: dict) -> dict:
    result = {"r_version": None, "namespace_packages": {}, "namespace": None, "dll": None}
    package, manifest = paths["package"], paths["manifest"]
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] == "R" and result["r_version"] is None:
            result["r_version"] = fields[1]
        elif len(fields) == 4 and fields[0] == "namespace" and fields[1] == "grf" and result["namespace"] is None:
            where = Path(fields[3]).resolve(strict=True)
            if where != package or fields[2] != GRF_VERSION:
                raise RuntimeError("R loaded a different grf namespace")
            result["namespace"] = {"version": fields[2], "path_sha256": _path_hash(where)}
        elif len(fields) == 3 and fields[:2] == ["dll", "grf"] and result["dll"] is None:
            dll = Path(fields[2]).resolve(strict=True)
            if not dll.is_relative_to(package / "libs"):
                raise RuntimeError("R loaded a different grf DLL")
            relative = dll.relative_to(package).as_posix()
            expected = manifest["installed_package"]["files_sha256"].get(relative)
            if expected is None or _sha256(dll) != expected:
                raise RuntimeError("loaded grf DLL differs from the recorded installation")
            result["dll"] = {"relative_path": relative, "sha256": expected,
                             "path_sha256": _path_hash(dll)}
        elif len(fields) == 4 and fields[0] == "package":
            _, name, version, raw_path = fields
            if name in result["namespace_packages"] or not name or not version:
                raise ValueError("duplicate or invalid R package metadata")
            where = Path(raw_path).resolve(strict=True)
            declaration = manifest["dependencies"].get(name)
            if declaration is not None:
                metadata = _dependency_identity(name, declaration, where)
                if version != metadata["version"]:
                    raise RuntimeError("R dependency version differs from its build record")
                if metadata != paths["expected_dependencies"].get(name):
                    raise RuntimeError("R dependency identity differs from the prefit installation")
            else:
                metadata = {"version": version, "description_sha256": _sha256(where / "DESCRIPTION"),
                            "namespace_path_sha256": _path_hash(where)}
            if name == "grf" and (where != package or version != GRF_VERSION):
                raise RuntimeError("R package metadata identifies a different grf")
            result["namespace_packages"][name] = metadata
        elif line:
            raise ValueError("invalid R runtime metadata")
    if (result["r_version"] != manifest["r_version"] or result["namespace"] is None
            or result["dll"] is None or "grf" not in result["namespace_packages"]
            or not set(manifest["dependencies"]).issubset(result["namespace_packages"])):
        raise RuntimeError("incomplete or changed R runtime metadata")
    return result


def causal_forest_provenance() -> dict:
    """Inspect requested native source/build/runtime without fitting a forest."""
    result = {"availability": "unavailable", "verified": False,
              "expected_revision": GRF_REVISION, "version": GRF_VERSION}
    try:
        snapshot, paths = _inspect_installation()
        result.update(snapshot)
        probe = subprocess.run([str(paths["executable"]), "--vanilla", "-e", _R_PROBE,
                                str(paths["library"])], capture_output=True, text=True,
                               timeout=120, check=True)
        result["runtime"] = _runtime_provenance(probe.stdout, paths)
        after, _ = _inspect_installation()
        if after != snapshot:
            raise RuntimeError("grf source or installation changed during inspection")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        result.update(availability="unavailable", verified=False, error_type=type(exc).__name__)
    return result


@contextmanager
def _working_directory():
    raw = os.environ.get("GRF_ARTIFACT_ROOT")
    if raw:
        root = Path(raw).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise RuntimeError("GRF_ARTIFACT_ROOT must be an existing directory")
        yield Path(tempfile.mkdtemp(prefix="grf-rct-", dir=root)), True
    else:
        with tempfile.TemporaryDirectory(prefix="grf-rct-") as temporary:
            yield Path(temporary), False


@dataclass
class RCTCausalForest:
    """Honest grf trial CATE with known assignment and native outcome nuisances.

    All fitting rows, nuisance estimation and forest splitting use rct_train
    only. Evaluation covariates enter native prediction after fitting. The
    multiple-of-four tree count restriction makes both the paired causal tree
    count and the native quarter-sized nuisance forest count exact integers.
    No native variance intervals or common tuning selector are requested.
    """

    num_trees: int = 2000
    num_threads: int = 1
    timeout_seconds: int = 600

    def __post_init__(self):
        for name in ("num_trees", "num_threads", "timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31-1:
                raise ValueError(f"{name} must be a positive signed-32-bit integer")
        if self.num_trees % 4:
            raise ValueError("num_trees must be a multiple of four")

    def _reset(self):
        self.provenance_ = None
        self.attempt_provenance_ = None
        self.last_run_diagnostics_ = None
        self.last_run_artifact_path_ = None
        self.last_run_provenance_ = {"status": "not_started"}

    @contextmanager
    def _attempt_guard(self):
        try:
            yield
        except Exception as exc:
            if self.last_run_provenance_.get("status") in {"not_started", "preparing", "validation_pending"}:
                self.last_run_provenance_["status"] = "preparation_error"
            self.last_run_provenance_["error_type"] = type(exc).__name__
            raise

    def fit_predict(self, training: TrainingView, eval_X, *, seed: int):
        self._reset()
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        return self.fit_predict_sample(training.rct_train, eval_X, seed=seed)

    def fit_predict_sample(self, sample: SourceSample, eval_X, *, seed: int):
        """Source-restricted entry point: excluded source/labels never enter it."""
        self._reset()
        if not isinstance(sample, SourceSample):
            raise TypeError("RCT causal forest requires a SourceSample")
        if sample.source != "rct" or sample.role != "rct_train":
            raise ValueError("sample must have source rct and role rct_train")
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        query = np.asarray(eval_X, dtype=np.float64)
        if (query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != sample.d
                or not np.isfinite(query).all()):
            raise ValueError("eval_X must be a nonempty finite matrix matching rct_train")
        if max(sample.n, sample.d, len(query)) > 2**31-1:
            raise ValueError("array dimensions must fit native signed-32-bit integers")
        if set(np.unique(sample.T)) != {0.0, 1.0}:
            raise ValueError("RCT causal forest requires both treatment arms")
        propensity = sample.known_propensity
        if (propensity is None or propensity.shape != (sample.n,)
                or not np.isfinite(propensity).all()
                or np.any((propensity <= 0) | (propensity >= 1))):
            raise ValueError("known RCT assignment probabilities are required")
        native_seed = int(seed) % (2**31-1)
        try:
            snapshot, paths = _inspect_installation()
        except Exception as exc:
            self.last_run_provenance_.update(status="unavailable", error_type=type(exc).__name__)
            raise
        self.attempt_provenance_ = {
            "native_source": snapshot, "recipe": "rct_causal_forest",
            "supplied_seed": int(seed), "native_seed": native_seed,
            "seed_rule": "uint32_mod_2147483647",
            "num_trees": self.num_trees, "num_threads": self.num_threads,
            "timeout_seconds": self.timeout_seconds, "known_propensity": "supplied_as_W.hat",
            "outcome_nuisance": "native_Y.hat_NULL_RCT_only",
            "fitting_source": "rct_train", "fit_rows": sample.n, "covariates": sample.d,
            "obs_train": "not_accessed", "rct_tune": "not_accessed",
            "target_reference": "not_accessed", "eval_X": "prediction_only",
            "honesty": True, "sample_fraction": .5, "honesty_fraction": .5,
            "min_node_size": 5, "alpha": .05, "ci_group_size": 2,
            "mtry": min(int(np.ceil(np.sqrt(sample.d) + 20)), sample.d),
            "tuning": "none", "estimate_variance": False,
            "compute_oob_predictions": True,
            "bridge_sha256": hashlib.sha256(_R_BRIDGE.encode()).hexdigest(),
        }
        self.last_run_provenance_ = {**self.attempt_provenance_, "status": "preparing"}
        with self._attempt_guard(), _working_directory() as (work, retained):
            if retained:
                # Private operational evidence. This raw path never enters the
                # public provenance dictionaries or prediction records.
                self.last_run_artifact_path_ = str(work)
            self.last_run_provenance_["artifacts_retained"] = retained
            inputs = {"rct_X.bin": sample.X, "rct_T.bin": sample.T,
                      "rct_Y.bin": sample.Y, "rct_propensity.bin": propensity,
                      "query_X.bin": query}
            for name, values in inputs.items():
                np.asarray(values, dtype="<f8").tofile(work / name)
            script = work / "bridge.R"
            script.write_text(_R_BRIDGE, encoding="utf-8")
            command = [str(paths["executable"]), "--vanilla", str(script),
                       str(paths["library"]), str(work), str(native_seed), str(sample.n),
                       str(len(query)), str(sample.d), str(self.num_trees), str(self.num_threads)]
            status, completed = "launch_error", None
            try:
                with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
                    completed = subprocess.run(command, cwd=work, stdout=stdout, stderr=stderr,
                                               timeout=self.timeout_seconds, check=False)
                status = "validation_pending" if completed.returncode == 0 else "native_error"
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                raise RuntimeError("grf causal forest exceeded timeout_seconds") from exc
            finally:
                self.last_run_diagnostics_ = {
                    name: (work / f"{name}.log").read_text(encoding="utf-8", errors="replace")
                    for name in ("stdout", "stderr")}
                self.last_run_provenance_.update(
                    status=status, returncode=None if completed is None else completed.returncode,
                    stdout_sha256=_sha256(work / "stdout.log"),
                    stderr_sha256=_sha256(work / "stderr.log"))
            if completed.returncode != 0:
                raise RuntimeError(f"grf R process failed with exit code {completed.returncode}")
            try:
                runtime = _runtime_provenance((work / "runtime.tsv").read_text(encoding="utf-8"), paths)
                before_path = work / "runtime_before.tsv"
                if not before_path.is_file():
                    raise RuntimeError("grf did not report its prefit runtime")
                before = _runtime_provenance(before_path.read_text(encoding="utf-8"), paths)
                for name, metadata in before["namespace_packages"].items():
                    if runtime["namespace_packages"].get(name) != metadata:
                        raise RuntimeError("R namespace changed during native fitting")
                after, _ = _inspect_installation()
                if after != snapshot:
                    raise RuntimeError("grf source, installation or executable changed during fitting")
                output = work / "prediction.bin"
                if not output.is_file() or output.stat().st_size != len(query) * 8:
                    raise ValueError("grf returned an invalid prediction file size")
                prediction = np.fromfile(output, dtype="<f8")
                if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
                    raise FloatingPointError("grf returned nonfinite predictions")
            except Exception:
                self.last_run_provenance_["status"] = "validation_error"
                raise
        self.last_run_provenance_["status"] = "success"
        self.provenance_ = {**self.last_run_provenance_, "runtime": runtime}
        return prediction.astype(np.float64, copy=False)
