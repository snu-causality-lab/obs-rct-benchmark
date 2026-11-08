# Shared settings for every job in /shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
export GSDS_WORK_ROOT=/shared/s2/lab05/sanghack/work
export XDG_CACHE_HOME="$GSDS_WORK_ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PYTHONUSERBASE="$GSDS_WORK_ROOT/.local"
export TMPDIR="$GSDS_WORK_ROOT/.tmp"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export W=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
export PY=/shared/s2/lab05/sanghack/work/obsrct-public-upgrade-20261025-r01/environment-v1/bin/python
export CORNET=/shared/s2/lab05/sanghack/work/obsrct-public-upgrade-20261025-r01/upstreams/CorNet-official-v1
export CODE=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/code/obs-rct-benchmark-173af2a/src
