#!/bin/bash
#SBATCH --job-name=intr-trace
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/diag/intr-trace-%j.out
# Diagnostic only: frozen wave-2 code and shard; writes no records; recorded scores unchanged.
export CODE_ID=80c9129
GROUP=native_r source jobs/common.sh
"$PY" -c "import obs_rct_benchmark; print(obs_rct_benchmark.__file__)"
"$PY" diag/intr_trace.py "$RUN" 70000
