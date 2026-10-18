"""Configuration-driven, resumable public benchmark runs.

Artifacts contain relative source names and hashes, disclosed configuration,
package versions, and sanitized failure types. No training arrays, evaluation
truth arrays, traceback paths, or local/server paths are written.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import tomllib
from typing import Any, Iterator, Mapping

from .evaluation import summarize_records, cate_metrics
from .methods import custom_provenance, datafusionppi_provenance, make_method, method_names
from .paper_dgp import PAPER_DGP_VERSION, generate_paper_problem, make_paper_config
from .structural_dgp import (
    STRUCTURAL_ADDENDUM_VERSION, STRUCTURAL_DGP_VERSION,
    generate_structural_problem, make_structural_addendum_config,
    make_structural_config,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScenarioConfig:
    kind: str
    id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.id}"

    def __post_init__(self) -> None:
        if self.kind not in {"structural", "addendum", "paper"}:
            raise ValueError("scenario kind must be structural, addendum, or paper")
        if not self.id or not self.id.replace("_", "").isalnum():
            raise ValueError("scenario id must contain letters, digits, or underscores")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("scenario parameters must be a mapping")
        if {"name", "mechanism", "cell", "paper_id"} & self.parameters.keys():
            raise ValueError("scenario identity must not be overridden in parameters")
        _validate_public_values(dict(self.parameters))
        # Validate the full DGP config before creating output artifacts.
        build_dgp_config(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    scenarios: tuple[ScenarioConfig, ...]
    seeds: tuple[int, ...] = (0, 1)
    methods: tuple[str, ...] = ("rct_t_ridge", "obs_t_ridge", "naive_pooled_t_ridge")
    reference_method: str = "rct_t_ridge"
    method_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scope: str = "smoke/development-inspected"

    def __post_init__(self) -> None:
        if not self.scenarios or len({s.key for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenarios must be nonempty with unique kind/id pairs")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(isinstance(seed, bool) or not isinstance(seed, int)
                       or seed < 0 or seed >= 2**32 for seed in self.seeds)):
            raise ValueError("seeds must be unique integers in [0, 2**32)")
        if (not self.methods or len(set(self.methods)) != len(self.methods)
                or any(name not in method_names() for name in self.methods)):
            raise ValueError("methods must be unique registered method names")
        if self.reference_method not in self.methods:
            raise ValueError("reference_method must be in methods")
        if set(self.method_options) - set(self.methods):
            raise ValueError("method_options contains an unrequested method")
        _validate_public_values(dict(self.method_options))
        if not self.scope or "\n" in self.scope:
            raise ValueError("scope must be a nonempty one-line description")
        _validate_public_values(self.scope)
        for name in self.methods:
            # Constructors are lightweight and must not import optional packages.
            make_method(name, **dict(self.method_options.get(name, {})))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run": {"seeds": list(self.seeds), "methods": list(self.methods),
                    "reference_method": self.reference_method, "scope": self.scope},
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "method_options": {name: dict(options)
                               for name, options in sorted(self.method_options.items())},
        }

    @property
    def config_hash(self) -> str:
        return _hash(self.payload())


def _validate_public_values(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("configuration keys must be strings")
            _validate_public_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_values(item)
    elif isinstance(value, str):
        if value.startswith(("/", "~/", "file://")) or "\n" in value:
            raise ValueError("public configuration must not contain local paths or newlines")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration numbers must be finite")
    elif value is not None and not isinstance(value, (bool, int)):
        raise ValueError("configuration values must be JSON-compatible primitives")


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) - {"schema_version", "run", "scenarios", "method_options"}:
        raise ValueError("unknown top-level configuration field")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing schema_version")
    run = raw.get("run", {})
    if set(run) - {"seeds", "methods", "reference_method", "scope"}:
        raise ValueError("unknown run field")
    scenarios = []
    for entry in raw.get("scenarios", []):
        if set(entry) - {"kind", "id", "parameters"}:
            raise ValueError("unknown scenario field")
        scenarios.append(ScenarioConfig(**entry))
    converted = dict(run)
    for key in ("seeds", "methods"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return BenchmarkConfig(scenarios=tuple(scenarios),
                           method_options=raw.get("method_options", {}), **converted)


def build_dgp_config(scenario: ScenarioConfig) -> Any:
    parameters = dict(scenario.parameters)
    if scenario.kind == "paper":
        return make_paper_config(name=scenario.key, paper_id=scenario.id, **parameters)
    if scenario.kind == "addendum":
        return make_structural_addendum_config(name=scenario.key, cell=scenario.id,
                                               **parameters)
    return make_structural_config(name=scenario.key, mechanism=scenario.id, **parameters)


def _generate(scenario: ScenarioConfig, seed: int) -> Any:
    cfg = build_dgp_config(scenario)
    if scenario.kind == "paper":
        return generate_paper_problem(cfg, seed)
    return generate_structural_problem(cfg, seed)


def method_seed(scenario: str, seed: int, method: str) -> int:
    """Stable method RNG stream, separate from data and other method streams."""
    return int(_hash(["method_seed_v1", scenario, seed, method])[:8], 16)


def record_id(config: BenchmarkConfig, scenario: str, seed: int, method: str) -> str:
    return _hash(["record_v1", config.config_hash, scenario, seed, method])


def run_benchmark(config: BenchmarkConfig, output: str | Path, *,
                  resume: bool = False) -> dict[str, Any]:
    """Run each requested cell once, or resume a verified incomplete output.

    Failed records are terminal on resume, preventing selective retries from
    hiding failures. Use a new output directory for a fresh, disclosed run.
    Incomplete JSONL tails, duplicate cells, provenance changes, and altered
    configs fail closed. Only the runner receives SimulationProblem objects.
    """
    output = Path(output)
    if output.exists() and not resume and any(output.iterdir()):
        raise ValueError("output is not empty; choose a fresh directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        manifest_path = output / "run_manifest.json"
        source = source_provenance()
        custom = {name: value for name, value in custom_provenance().items()
                  if name in config.methods}
        _validate_public_values(custom)
        environment = environment_provenance()
        external = external_method_provenance(config.methods)
        resolved = resolved_scenarios(config)
        fingerprint = _hash({"source": source, "custom_methods": custom,
                             "environment": environment, "external_methods": external,
                             "resolved_scenarios": resolved})
        if resume:
            if not manifest_path.exists():
                raise ValueError("resume requires an existing run_manifest.json")
            manifest = _load_json(manifest_path)
            if (manifest.get("config_hash") != config.config_hash
                    or manifest.get("source_fingerprint") != fingerprint
                    or manifest.get("config") != config.payload()):
                raise ValueError("resume config or source provenance does not match")
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION, "config_hash": config.config_hash,
                "run_id": config.config_hash[:16], "config": config.payload(),
                "source": source, "source_fingerprint": fingerprint,
                "custom_methods": custom,
                "custom_provenance_policy": "author-supplied identifiers; custom source is not automatically verified",
                "external_methods": external, "environment": environment,
                "resolved_scenarios": resolved,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": config.scope,
                "interpretation": "development smoke example; not a method-ranking study",
            }
            _write_json(manifest_path, manifest)
        records_path = output / "records.jsonl"
        records = _read_records(records_path)
        expected = {
            record_id(config, scenario.key, seed, method): (scenario.key, seed, method)
            for scenario in config.scenarios for seed in config.seeds for method in config.methods
        }
        seen = _validate_resume_records(config, records, expected)
        for scenario in config.scenarios:
            for seed in config.seeds:
                pending = [method for method in config.methods
                           if record_id(config, scenario.key, seed, method) not in seen]
                if not pending:
                    continue
                try:
                    problem = _generate(scenario, seed)
                    training = problem.training_view()
                except Exception as exc:
                    for method in pending:
                        row = _base_record(config, scenario.key, seed, method)
                        row.update(status="failed", metrics=None, stage="simulation",
                                   error_type=type(exc).__name__)
                        _append_record(records_path, row)
                        records.append(row)
                    continue
                for method in pending:
                    row = _base_record(config, scenario.key, seed, method)
                    mechanism_hash = problem.metadata.get("mechanism_sha256")
                    if mechanism_hash is not None:
                        row["mechanism_sha256"] = mechanism_hash
                    try:
                        estimator = make_method(method, **dict(config.method_options.get(method, {})))
                        prediction = estimator.fit_predict(training, problem.eval_X,
                                                           seed=row["method_seed"])
                        row.update(status="ok", metrics=cate_metrics(prediction, problem.eval_tau))
                    except Exception as exc:
                        row.update(status="failed", metrics=None, stage="method_or_metric",
                                   error_type=type(exc).__name__)
                    _append_record(records_path, row)
                    records.append(row)
        summary = summarize_records(
            records, scenarios=[scenario.key for scenario in config.scenarios],
            seeds=config.seeds, methods=config.methods,
            reference_method=config.reference_method,
        )
        summary.update(config_hash=config.config_hash, run_id=manifest["run_id"],
                       scope=config.scope, source_fingerprint=fingerprint)
        _write_json(output / "summary.json", summary)
        _write_text(output / "README.md", _result_readme(summary))
        return summary


def _base_record(config: BenchmarkConfig, scenario: str, seed: int, method: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id(config, scenario, seed, method),
        "config_hash": config.config_hash, "scenario": scenario,
        "seed": seed, "method": method,
        "method_seed": method_seed(scenario, seed, method),
    }


def _validate_resume_records(config: BenchmarkConfig, records: list[dict[str, Any]],
                             expected: Mapping[str, tuple[str, int, str]]) -> set[str]:
    seen: set[str] = set()
    for record in records:
        key = record.get("record_id")
        if key in seen:
            raise ValueError("duplicate record_id in records.jsonl")
        if key not in expected:
            raise ValueError("unrequested or altered record_id in records.jsonl")
        scenario, seed, method = expected[key]
        base = _base_record(config, scenario, seed, method)
        if any(record.get(field) != value for field, value in base.items()):
            raise ValueError("record identity/config/seed does not match manifest")
        seen.add(key)
    # Reuse independent scorer validation for statuses and metric identities.
    summarize_records(records, scenarios=[s.key for s in config.scenarios],
                      seeds=config.seeds, methods=config.methods,
                      reference_method=config.reference_method)
    return seen


def source_provenance() -> dict[str, Any]:
    package = Path(__file__).parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.glob("*.py"))}
    return {"distribution": "obs-rct-benchmark", "version": _version("obs-rct-benchmark"),
            "package_file_sha256": files, "aggregate_sha256": _hash(files)}


def resolved_scenarios(config: BenchmarkConfig) -> dict[str, Any]:
    """Record the complete generator controls, including default mechanism seeds."""
    versions = {"paper": PAPER_DGP_VERSION, "structural": STRUCTURAL_DGP_VERSION,
                "addendum": STRUCTURAL_ADDENDUM_VERSION}
    return {
        scenario.key: {"generator_version": versions[scenario.kind],
                       "config": asdict(build_dgp_config(scenario))}
        for scenario in config.scenarios
    }


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_provenance() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(),
        "packages": {name: _version(name) for name in
                     ("obs-rct-benchmark", "numpy", "scikit-learn", "scipy", "dfppi", "torch")},
    }


def external_method_provenance(methods: tuple[str, ...]) -> dict[str, Any]:
    """Bind requested external implementations to manifest/resume identity.

    This records availability and actual file hashes without training or
    importing external algorithm code. Adapter execution handles unavailable
    or modified dependencies as failed cells, preserving coverage reporting.
    """
    result: dict[str, Any] = {}
    if "datafusionppi" in methods:
        result["datafusionppi"] = datafusionppi_provenance()
    if "cornet_official" in methods:
        from .cornet import cornet_provenance

        result["cornet_official"] = cornet_provenance()
    return result


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError("nonfinite JSON constant")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        raise ValueError("records.jsonl has an incomplete final line; inspect it before resuming")
    result = []
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError("records.jsonl contains an empty line")
        def reject(value: str) -> None:
            raise ValueError("nonfinite JSON constant")
        record = json.loads(line, parse_constant=reject)
        if not isinstance(record, dict):
            raise ValueError("each record must be a JSON object")
        result.append(record)
    return result


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json(dict(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n")


def _write_text(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".benchmark-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _run_lock(output: Path) -> Iterator[None]:
    path = output / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("run lock exists; ensure no writer is active before removing it") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def _result_readme(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    table = ["| Method | Mean CATE MSE | Mean root-PEHE | Successful / expected |",
             "| --- | ---: | ---: | ---: |"]
    for method, description in summary["methods"].items():
        mse, root = description["mean_cate_mse"], description["mean_root_pehe"]
        mse_text = "unavailable" if mse is None else f"{mse:.6g}"
        root_text = "unavailable" if root is None else f"{root:.6g}"
        table.append(f"| `{method}` | {mse_text} | {root_text} | "
                     f"{description['successful']} / {description['expected']} |")
    result_table = "\n".join(table) + "\n\n"
    return (
        "# Benchmark run\n\n"
        f"Scope: {summary['scope']}. This is a development example, not a method-ranking study.\n\n"
        f"Status: {summary['status']}. Successful cells: {coverage['successful']}/"
        f"{coverage['expected']}; failed: {coverage['failed']}; missing: {coverage['missing']}.\n\n"
        "`records.jsonl` stores one terminal record per scenario/seed/method. "
        "`run_manifest.json` stores the normalized configuration, source hashes, and environment. "
        "Requested official adapters also record external implementation hashes; custom "
        "method provenance strings are author-supplied identifiers, not automatic source checks. "
        "`summary.json` stores coverage, descriptive metrics, and paired comparisons.\n\n"
        + result_table +
        "Table means describe successful cells only. Incomplete coverage prevents "
        "inferential comparisons and can make descriptive means misleading.\n\n"
        "CATE MSE is the mean squared CATE error over evaluation rows. Root-PEHE is "
        "its square root for a replicate. Mean root-PEHE and root mean MSE are "
        "reported separately and generally differ.\n\n"
        "Paired differences are method minus reference. For the overall comparison, "
        "scenario differences are averaged within each independent seed block. "
        "95% Student t intervals use a conservative tabulated critical value; "
        "fewer than two seed blocks produces no interval. Two seeds are a smoke check, "
        "not sufficient evidence for a scientific comparison. Failed or missing cells "
        "withhold inferential comparisons; no failure is assigned a winning score.\n\n"
        "Resume accepts only matching configuration and source hashes, rejects duplicates "
        "and partial final JSONL lines, and never retries terminal failed records. "
        "After a hard process kill, inspect the output and confirm no writer remains "
        "before manually removing a stale `.run.lock`.\n"
    )
