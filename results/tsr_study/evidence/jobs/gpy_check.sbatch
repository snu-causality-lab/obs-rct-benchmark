#!/bin/bash
#SBATCH --job-name=tsr-gpy-check
#SBATCH --partition=P2
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/gpy-check-%j.out
# Usage: sbatch jobs/gpy_check.sbatch <code-dir-name>
set -euo pipefail
source jobs/common.sh
W=/shared/s2/lab05/sanghack/work/obsrct-v03-adapters-20261026-r01
GPYPY=""
for e in environment-v2 environment-replay-v1; do
  if "$W/$e/bin/python" -c "import GPy" 2>/dev/null; then GPYPY=$W/$e/bin/python; break; fi
done
echo "GPy interpreter: $GPYPY"
CODE=$RUN/code/$1
cd "$CODE"
PYTHONPATH="$CODE/src" "$GPYPY" -c "import obs_rct_benchmark, GPy, numpy, scipy; print(obs_rct_benchmark.__file__, GPy.__version__, numpy.__version__, scipy.__version__)"
PYTHONPATH="$CODE/src" "$GPYPY" scripts/verify_tsr_gp.py --config configs/tsr_study.toml --output "$RUN/gpy-check-$1.json"
echo GPY_CHECK_DONE
