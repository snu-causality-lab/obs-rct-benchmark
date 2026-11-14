# Sourced by every job. Absolute shared paths only. CODE_ID names the frozen code snapshot.
set -uo pipefail
export RUN=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01
export GSDS_WORK_ROOT=/shared/s2/lab05/sanghack/work
export XDG_CACHE_HOME="$GSDS_WORK_ROOT/.cache" PIP_CACHE_DIR="$GSDS_WORK_ROOT/.cache/pip"
export TMPDIR="$GSDS_WORK_ROOT/.tmp" PYTHONUSERBASE="$GSDS_WORK_ROOT/.local"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
export CORNET_SOURCE=/shared/s2/lab05/sanghack/work/obsrct-public-upgrade-20261025-r01/upstreams/CorNet-official-v1
PY=/shared/s2/lab05/sanghack/work/obsrct-public-upgrade-20261025-r01/environment-v1/bin/python
CODE=$RUN/code/${CODE_ID:?set CODE_ID}
export PYTHONPATH="$CODE/src:$CODE/scripts"
mkdir -p "$TMPDIR" "$RUN/logs"
