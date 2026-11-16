"""Progress of one block from status files and sacct only; never reads records or metrics.
Usage: python3 jobs/progress.py <status dir> <expected> <job ids comma-separated>"""
import glob, json, subprocess, sys
status_dir, expected, jobs = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.load(open(f)) for f in glob.glob(f"{status_dir}/*.json")]
fail = {}
for r in rows:
    for k, v in r["failures"].items():
        fail[k] = fail.get(k, 0) + v
states = subprocess.run(["sacct", "-n", "-X", "-j", jobs, "--format=State%20"], capture_output=True, text=True).stdout.split()
count = {}
for s in states:
    count[s] = count.get(s, 0) + 1
bad = {k: v for k, v in count.items() if k not in ("COMPLETED", "RUNNING", "PENDING")}
el = sorted(r["elapsed_seconds"] for r in rows)
mem = max((r["maxrss_self_kib"] // 1024 for r in rows), default=0)
print(json.dumps({"done": len(rows), "expected": expected, "method_failures": fail, "job_states": count,
                  "bad_states": bad, "longest_s": el[-1] if el else None, "peak_mib": mem}))
