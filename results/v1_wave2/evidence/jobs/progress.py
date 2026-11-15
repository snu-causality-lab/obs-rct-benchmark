"""Progress from status files only (coverage, failure types, runtime, memory); never metrics.

Usage: python3 jobs/progress.py out-native_r out-gp ... (relative to the run directory)
"""
import collections, glob, json, os, statistics, sys

run = os.environ.get("RUN", "/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01")
for out in sys.argv[1:]:
    shards = os.path.join(run, out.replace("out-", "main-").replace("sout-", "smain-"))
    total = len(json.load(open(os.path.join(shards, "study_manifest.json")))["shards"])
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(run, f"status-{out}", "*.json"))]
    fails = collections.Counter()
    for r in rows:
        fails.update(r["failures"])
    el = [r["elapsed_seconds"] for r in rows]
    interrupted = len(glob.glob(os.path.join(run, out, "*.interrupted-*")))
    pending = len([d for d in glob.glob(os.path.join(run, out, "*")) if ".interrupted-" not in d])
    print(f"{out}: {len(rows)}/{total} done, {pending - len(rows) if pending >= len(rows) else 0} in progress, "
          f"{interrupted} interrupted kept; "
          + (f"median {statistics.median(el):.0f}s max {max(el):.0f}s; peak {max(max(r['maxrss_self_kib'], r['maxrss_children_kib']) for r in rows)//1024} MiB; " if rows else "")
          + f"failures {dict(fails)}")
