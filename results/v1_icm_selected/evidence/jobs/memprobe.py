"""Resident memory around each native Causal-ICM call of one selected fit (seed 69999,
outside the grid). Prints memory only, no metric. Usage: python memprobe.py <code dir> <scenario> [gc]"""
import gc, json, sys, time
from pathlib import Path

from obs_rct_benchmark import causal_icm
from obs_rct_benchmark.runner import _generate, load_config, method_seed
from obs_rct_benchmark.methods import make_method


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024


code, key = Path(sys.argv[1]), sys.argv[2]
collect = len(sys.argv) > 3 and sys.argv[3] == "gc"
block = "stress" if key.startswith("stress") else "core"
config = load_config(code / f"configs/v1_icm_selected_{block}.toml")
scenario = next(s for s in config.scenarios if s.key == key)
problem = _generate(scenario, 69999 if block == "core" else 79999)
native = causal_icm._official_icm()
trace = []


def wrapped(**kw):
    out = native(**kw)
    if collect:
        gc.collect()
    trace.append(rss_mib())
    return out


causal_icm._official_icm = lambda: wrapped
start, t = rss_mib(), time.monotonic()
make_method("causal_icm_selected").fit_predict(problem.training_view(), problem.eval_X,
                                               seed=method_seed(key, 0, "causal_icm_selected"))
print(json.dumps({"scenario": key, "gc_after_each_call": collect, "start_mib": start, "calls": len(trace),
                  "rss_after_calls_mib": trace[::5] + trace[-1:], "end_mib": rss_mib(),
                  "seconds": round(time.monotonic() - t, 1)}))
