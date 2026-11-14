"""Run one declared wave-2 shard, check resume, and write a status file without metrics.

Usage: python run_shard.py <shards dir> <index> <outputs dir> [--discard-metrics] [--check-only]
--discard-metrics (preflight) deletes the records and summaries after the status is written.
"""
import hashlib, json, os, resource, shutil, sys, time
from pathlib import Path

from obs_rct_benchmark.runner import load_config, run_benchmark

run = Path(os.environ["RUN"])
shards, index, outputs = run / sys.argv[1], int(sys.argv[2]), sys.argv[3]
discard = "--discard-metrics" in sys.argv[4:]
manifest = json.loads((shards / "study_manifest.json").read_text())
item = manifest["shards"][index]
path = shards / item["file"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "shard file changed"
config = load_config(path)
assert config.config_hash == item["config_hash"], "shard config hash changed"


def check_environment(config):
    """Refuse to fit when an external dependency is not as pinned: an infrastructure
    failure (no records), never a method failure."""
    from obs_rct_benchmark.runner import external_method_provenance

    problems = []

    def scan(node, where):
        if isinstance(node, dict):
            if node.get("verified") is False:
                problems.append(where + ".verified")
            if node.get("availability") in ("unavailable", "unreadable"):
                problems.append(where + ".availability")
            for key, value in node.items():
                scan(value, f"{where}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                scan(value, f"{where}[{i}]")

    provenance = external_method_provenance(config.methods)
    scan(provenance, "external")
    group = os.environ.get("GROUP")
    want = {"tabpfn": "10.0.0", "tsr_legacy": "2.0.9"}.get(group)
    if want is not None:
        import importlib.metadata
        if importlib.metadata.version("tabpfn") != want:
            problems.append(f"tabpfn version is not {want}")
        from obs_rct_benchmark import tsr
        for name, options in config.method_options.items():
            sha = dict(options).get("model_sha256")
            if not sha:
                continue
            if group == "tabpfn":
                release = {"2": "v2", "25": "v2.5", "3": "v3"}.get(name.split("_v")[-1].split("_")[0])
                if release is None:
                    continue
                file = Path(os.environ["OBS_RCT_TABPFN_WEIGHTS_DIR"]) / tsr.TABPFN_RELEASES[release][1]
            else:
                file = Path(os.environ["OBS_RCT_TABPFN_MODEL_PATH"])
            if not file.is_file() or tsr._file_sha256(file) != sha:
                problems.append(f"{name}: checkpoint missing or hash differs")
    if problems:
        raise SystemExit("environment check failed: " + "; ".join(problems))
    return provenance


check_environment(config)
if "--check-only" in sys.argv[4:]:
    print(f"environment check passed for {item['file']}", flush=True)
    sys.exit(0)
out = run / outputs / path.stem
start = time.monotonic()
summary = run_benchmark(config, out, resume=False)
records = out / "records.jsonl"
before = hashlib.sha256(records.read_bytes()).hexdigest()
resumed = run_benchmark(config, out, resume=True)
assert hashlib.sha256(records.read_bytes()).hexdigest() == before, "resume changed records"
coverage = summary["coverage"]
assert coverage["recorded"] == item["cells"] and coverage["missing"] == 0
failures = {}
for line in records.read_text().splitlines():
    row = json.loads(line)
    if row["status"] != "ok":
        key = f'{row["method"]}:{row.get("stage")}:{row.get("error_type")}'
        failures[key] = failures.get(key, 0) + 1
status = {"index": index, "file": item["file"], "coverage": coverage, "failures": failures,
          "elapsed_seconds": round(time.monotonic() - start, 1), "records_sha256": before,
          "maxrss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "maxrss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
target = run / f"status-{outputs}" / f"{path.stem}.json"
target.parent.mkdir(exist_ok=True)
assert not target.exists(), "status file exists"
target.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
if discard:
    shutil.rmtree(out)
print(json.dumps(status, sort_keys=True), flush=True)
