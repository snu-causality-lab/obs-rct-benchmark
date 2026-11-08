# Shared settings for every job in /shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
export GSDS_WORK_ROOT=/shared/s2/lab05/sanghack/work
export XDG_CACHE_HOME="$GSDS_WORK_ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PYTHONUSERBASE="$GSDS_WORK_ROOT/.local"
export TMPDIR="$GSDS_WORK_ROOT/.tmp"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export W=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
export ENV_ROOT=/shared/s2/lab05/sanghack/work/obsrct-v03-adapters-20261026-r01
export PY="$ENV_ROOT/environment-v2/bin/python"
export RSCRIPT="$ENV_ROOT/environment-v2/bin/Rscript"
export INTEGRATIVE_R_SOURCE="$ENV_ROOT/upstreams/intRlearner"
export INTEGRATIVE_R_RSCRIPT="$RSCRIPT"
export CODE="$W/code/obs-rct-benchmark-173af2a/src"
