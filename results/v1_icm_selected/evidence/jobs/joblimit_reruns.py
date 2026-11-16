"""List, and with --submit start, the job-limit reruns of one block (docs/V1_ICM_SELECTED.md):
a main-run shard that reached its time limit is rerun once with min(2 x limit, 12 h); one that
reached its memory limit is rerun once with 2 x memory. Reads sacct and status files, never metrics.
The driver moves the stopped attempt aside (.interrupted-N, kept) and starts a fresh directory.

Usage: python3 jobs/joblimit_reruns.py core|stress [--submit]
Prints one JSON line: new reruns, other bad main-run states, and rerun jobs with their states.
The main arrays of a block are read from arrays-<block>.json, written when they are submitted:
{job id: {"offset": n}} or {job id: {"index_file": path}} (cpu_array.sbatch's two modes).
"""
import glob, json, os, subprocess, sys

R = "/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01"
EXCLUDE = "b03,b14,b15,b24,b25,b26,b27,b28,b30,b31"
BLOCKS = {  # code snapshot, shard and output dirs, main-run limits (minutes, GiB)
    "core": dict(code="7758408", shards="main-core", out="out-core", time=200, mem=4),
    "stress": dict(code=None, shards="main-stress", out="out-stress", time=720, mem=10),
}
block = sys.argv[1]
submit = "--submit" in sys.argv[2:]
B = BLOCKS[block]
os.chdir(R)
man = json.load(open(f"{B['shards']}/study_manifest.json"))["shards"]
B["code"] = B["code"] or open(f"code-{block}.txt").read().strip()  # the stress freeze, recorded at launch
arrays = json.load(open(f"arrays-{block}.json"))
remap = {j: ([int(x) for x in open(a["index_file"]).read().split()] if "index_file" in a else None)
         for j, a in arrays.items()}

def shard_of(job, task):
    return remap[job][task] if remap[job] is not None else task + arrays[job]["offset"]

def states(jobs):
    rows = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(jobs), "-o", "JobID,State"],
                          capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for row in rows:
        jid, state = row.split("|")
        job, _, task = jid.partition("_")
        if task and not task.startswith("["):
            out[(job, int(task))] = state.split()[0]
    return out

listed = {}  # shard index -> rerun file
for path in sorted(glob.glob(f"rerun-{B['out']}-joblimit-*.txt")):
    for line in open(path).read().split():
        assert int(line) not in listed, f"shard {line} listed twice"
        listed[int(line)] = path

new = {"time": [], "memory": []}
other = {}
for (job, task), state in states(list(arrays)).items():
    i = shard_of(job, task)
    if state in ("COMPLETED", "RUNNING", "PENDING"):
        continue
    kind = {"TIMEOUT": "time", "OUT_OF_MEMORY": "memory"}.get(state)
    status = f"status-{B['out']}/{man[i]['file'][:-5]}.json"
    if kind is None:
        other[i] = state
    elif i not in listed and not os.path.exists(status):
        new[kind].append(i)

reruns = {}  # rerun job -> {state: count}
log = f"rerun-{B['out']}-joblimit-jobs.json"
jobs = json.load(open(log)) if os.path.exists(log) else {}
if jobs:
    for (job, task), state in states(list(jobs)).items():
        reruns.setdefault(job, {}).setdefault(state, 0)
        reruns[job][state] += 1

submitted = {}
if submit:
    for kind, idx in new.items():
        if not idx:
            continue
        n = len(glob.glob(f"rerun-{B['out']}-joblimit-*.txt")) + 1
        path = f"rerun-{B['out']}-joblimit-{n}.txt"
        assert not os.path.exists(path)
        open(path, "w").write("".join(f"{i}\n" for i in sorted(idx)))
        minutes = min(2 * B["time"], 720) if kind == "time" else B["time"]
        mem = 2 * B["mem"] if kind == "memory" else B["mem"]
        env = dict(os.environ, CODE_ID=B["code"], GROUP="gp", RUN_DIR=R, INDEX_FILE=f"{R}/{path}")
        cmd = ["sbatch", "--parsable", "--export=ALL", f"--chdir={R}", f"--output={R}/logs/%x-%A_%a.out",
               f"--array=0-{len(idx) - 1}", f"--mem={mem}G", f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
               f"--exclude={EXCLUDE}", "-J", f"icmsel-{block}-joblimit-{n}",
               "jobs/cpu_array.sbatch", B["shards"], B["out"], "0"]
        jid = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout.strip()
        jobs[jid] = {"file": path, "kind": kind, "shards": sorted(idx), "mem_G": mem, "time_min": minutes}
        json.dump(jobs, open(log, "w"), indent=1)
        submitted[jid] = jobs[jid]
        with open("JOBS.txt", "a") as f:
            f.write(f"  job-limit rerun ({kind}; {block}; INDEX_FILE {path}, {len(idx)} shards; "
                    f"{mem}G {minutes} min; driver keeps the stopped attempt as .interrupted-N)={jid}\n")
print(json.dumps({"block": block, "listed": len(listed), "new": new, "other_bad": other,
                  "rerun_states": reruns, "submitted": submitted}, sort_keys=True))
