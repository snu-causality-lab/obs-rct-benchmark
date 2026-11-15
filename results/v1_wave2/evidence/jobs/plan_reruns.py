"""List native_r shards to rerun whole: no status and not running, or a manifest that
records an external runtime as unavailable. Reads manifests and status files, never metrics.

Usage: python3 jobs/plan_reruns.py <shards dir> <outputs dir> <tag> [--apply]
--apply moves a statused bad run and its status aside as .<tag> (kept) and writes rerun-<outputs>-<tag>.txt.
"""
import glob, json, os, subprocess, sys

shards, out, tag = sys.argv[1:4]
apply = "--apply" in sys.argv[4:]
man = json.load(open(f"{shards}/study_manifest.json"))["shards"]
index = {m["file"][:-5]: i for i, m in enumerate(man)}

RERUN_INDEX_FILES = {
    "w2-native_r-rerun1": "rerun-native_r-envcheck-1.txt",
    "w2-native_r-rerun2": "rerun-native_r-envcheck-2.txt",
    "w2-native_r-rerun3": "rerun-out-native_r-provenance-unavailable-1.txt",
    "sw2-native_r-rerun3": "rerun-sout-native_r-provenance-unavailable-1.txt",
    "w2-native_r-rerun4": "rerun-out-native_r-provenance-unavailable-2.txt",
    "sw2-native_r-rerun4": "rerun-sout-native_r-provenance-unavailable-2.txt",
}
# Running shards: map array tasks to shard indices (offset arrays and INDEX_FILE reruns).
running = set()
q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-t", "RUNNING,PENDING,CONFIGURING", "-o", "%i %j"],
                   capture_output=True, text=True).stdout.split("\n")
for line in q:
    if not line.strip():
        continue
    jid, name = line.split()
    if "native_r" not in name or (shards.startswith("smain") != name.startswith("sw2")):
        continue
    job, _, task = jid.partition("_")
    tasks = []
    if task.startswith("["):
        spec = task.strip("[]").split("%")[0]
        for part in spec.split(","):
            a, _, b = part.partition("-")
            tasks += range(int(a), int(b or a) + 1)
    else:
        tasks = [int(task)]
    cmd = subprocess.run(["sacct", "-j", job, "-n", "-X", "-o", "SubmitLine%400"], capture_output=True, text=True).stdout.split("\n")[0]
    words = cmd.split()
    offset = int(words[-1]) if words[-1].isdigit() else int(words[-2])
    # Rerun jobs read INDEX_FILE from the environment, which the submit line does not show.
    remap_file = RERUN_INDEX_FILES.get(name)
    if "rerun" in name and remap_file is None:
        raise SystemExit(f"unknown rerun job {name}: add its INDEX_FILE to RERUN_INDEX_FILES")
    remap = [int(x) for x in open(remap_file).read().split()] if remap_file else None
    for t in tasks:
        p = t + offset
        running.add(remap[p] if remap else p)

bad, nostatus, statused_bad = [], [], []
for i, item in enumerate(man):  # every declared shard: an env-check failure leaves no directory
    stem = item["file"][:-5]
    d = f"{out}/{stem}"
    if i in running:
        continue
    status = f"status-{out}/{stem}.json"
    m = json.load(open(f"{d}/run_manifest.json")) if os.path.exists(f"{d}/run_manifest.json") else {}
    unavailable = '"unavailable"' in json.dumps(m.get("external_methods"))
    if not os.path.exists(status):
        nostatus.append(i)
    elif unavailable:
        statused_bad.append(i)
print(f"{out}: running {len(running)}; rerun: no status {len(nostatus)}, statused with unavailable runtime {len(statused_bad)}")
if apply:
    for i in statused_bad:
        stem = man[i]["file"][:-5]
        os.rename(f"{out}/{stem}", f"{out}/{stem}.{tag}")
        os.rename(f"status-{out}/{stem}.json", f"status-{out}/{stem}.json.{tag}")
    path = f"rerun-{out}-{tag}.txt"
    assert not os.path.exists(path)
    open(path, "w").write("".join(f"{i}\n" for i in sorted(nostatus + statused_bad)))
    print("wrote", path, len(nostatus) + len(statused_bad))
