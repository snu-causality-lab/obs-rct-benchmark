# Sourced by every job. Absolute shared paths only.
export RUN=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
export GSDS_WORK_ROOT=/shared/s2/lab05/sanghack/work
export XDG_CACHE_HOME="$GSDS_WORK_ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PYTHONUSERBASE="$GSDS_WORK_ROOT/.local"
export TMPDIR="$GSDS_WORK_ROOT/.tmp"
export TABPFN_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
ENV=$RUN/environment-v1
PY=$ENV/bin/python
export OBS_RCT_TABPFN_MODEL_PATH=$GSDS_WORK_ROOT/.cache/tabpfn/tabpfn-v2-regressor.ckpt
