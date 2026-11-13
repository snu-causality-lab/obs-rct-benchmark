#!/bin/bash
#SBATCH --job-name=tsr-preflight
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/preflight-%j.out
# Usage: sbatch jobs/preflight.sbatch <code-dir-name>
set -euo pipefail
source jobs/common.sh
CODE=$RUN/code/$1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m pip install --quiet --no-deps --force-reinstall "$CODE"
"$PY" -c "import obs_rct_benchmark, sys; print('package', obs_rct_benchmark.__file__)"
cd "$CODE"
echo "== pytest"; "$PY" -m pytest -q -p no:cacheprovider -x 2>&1 | tail -5
echo "== check_release"; "$PY" scripts/check_release.py 2>&1 | tail -3
cd "$RUN"
DEV=$RUN/dev-$1; rm -rf "$DEV"; "$PY" jobs/make_dev_configs.py "$CODE" "$DEV"
echo "== smoke on gamma generator (status only)"
"$PY" -m obs_rct_benchmark.cli run --config "$DEV/smoke.toml" --output "$DEV/smoke-out" > "$DEV/smoke.log" 2>&1 || echo "smoke exit $?"
"$PY" - "$DEV/smoke-out/records.jsonl" <<'PYEOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
print(collections.Counter((r["method"], r["status"], r.get("error_type")) for r in rows))
PYEOF
echo "== timing (runtime/status only)"
/usr/bin/time -v "$PY" -m obs_rct_benchmark.cli run --config "$DEV/timing.toml" --output "$DEV/timing-out" > "$DEV/timing.log" 2> "$DEV/timing.time" || echo "timing exit $?"
grep -E "Elapsed|Maximum resident" "$DEV/timing.time"
"$PY" - "$DEV/timing-out/records.jsonl" <<'PYEOF'
import json, sys, collections
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
for r in rows:
    r.pop("metrics", None)
with open(path, "w") as fh:
    fh.write("".join(json.dumps(r) + "\n" for r in rows))
print("redacted metrics;", collections.Counter((r["method"], r["status"], r.get("error_type")) for r in rows))
print("stage1", collections.Counter((r.get("fit_info") or {}).get("stage1_device") for r in rows), "gp_conv", collections.Counter((r.get("fit_info") or {}).get("gp_converged") for r in rows))
PYEOF
rm -f "$DEV/timing-out/summary.json" "$DEV/timing-out/README.md"
echo PREFLIGHT_DONE
