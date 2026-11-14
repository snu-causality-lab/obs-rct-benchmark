"""Run one declared shard, check that resume preserves records, and write a status file.

The status file holds coverage counts, timing and hashes only; no metric is read.
Usage: python run_shard.py <shard index> <outputs dir name>
"""
import hashlib, json, os, sys, time
from pathlib import Path

from obs_rct_benchmark.runner import load_config, run_benchmark

run = Path(os.environ["RUN"])
index, outputs = int(sys.argv[1]), sys.argv[2]
manifest = json.loads((run / "shards" / "study_manifest.json").read_text())
item = manifest["shards"][index]
path = run / "shards" / item["file"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "shard file changed"
config = load_config(path)
assert config.config_hash == item["config_hash"], "shard config hash changed"
out = run / outputs / path.stem
start = time.monotonic()
summary = run_benchmark(config, out, resume=False)
records = out / "records.jsonl"
before = hashlib.sha256(records.read_bytes()).hexdigest()
resumed = run_benchmark(config, out, resume=True)
assert hashlib.sha256(records.read_bytes()).hexdigest() == before, "resume changed records"
coverage = summary["coverage"]
assert coverage["recorded"] == item["cells"] and coverage["missing"] == 0
status = {"index": index, "file": item["file"], "coverage": coverage, "run_status": summary["status"],
          "elapsed_seconds": round(time.monotonic() - start, 1), "records_sha256": before,
          "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
target = run / f"status-{outputs}" / f"shard-{index:03d}.json"
target.parent.mkdir(exist_ok=True)
assert not target.exists(), "status file exists"
target.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(status, sort_keys=True), flush=True)
