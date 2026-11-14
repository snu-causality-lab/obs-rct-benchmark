# Sourced by every wave-2 job. Absolute shared paths only.
# Needs CODE_ID (code snapshot) and GROUP (native_r | gp | tabpfn | tsr_legacy | stress).
set -uo pipefail
W=/shared/s2/lab05/sanghack/work
export RUN=${RUN_DIR:-$W/obsrct-v1-wave2-20261114-r01}
export GSDS_WORK_ROOT=$W XDG_CACHE_HOME="$W/.cache" PIP_CACHE_DIR="$W/.cache/pip"
export TMPDIR="$W/.tmp" PYTHONUSERBASE="$W/.local" HF_HOME="$W/.cache/huggingface"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg TABPFN_DISABLE_TELEMETRY=1
CODE=$RUN/code/${CODE_ID:?set CODE_ID}
export PYTHONPATH="$CODE/src:$CODE/scripts"
case "${GROUP:?set GROUP}" in
  native_r)
    PY=$W/obsrct-public-upgrade-20261025-r01/environment-v1/bin/python
    export INTEGRATIVE_R_SOURCE=$W/obsrct-v03-adapters-20261026-r01/upstreams/intRlearner
    export INTEGRATIVE_R_RSCRIPT=$W/obsrct-gp-source-pilot-20261026-r01/environment-v1/bin/Rscript
    export GRF_SOURCE=$W/obsrct-grf-admission-20261027-r01/upstream-v1
    export GRF_LIBRARY=$W/obsrct-grf-admission-20261027-r01/r-library-v1
    export GRF_BUILD_MANIFEST=$W/obsrct-grf-admission-20261027-r01/build-v2/grf-build-manifest.json
    export GRF_RSCRIPT=$W/obsrct-gp-source-pilot-20261026-r01/environment-v1/bin/Rscript ;;
  gp)
    PY=$W/obsrct-gp-source-pilot-20261026-r01/environment-v1/bin/python
    export CAUSAL_ICM_SOURCE=$W/obsrct-v03-adapters-20261026-r01/upstreams/CausalICM ;;
  tabpfn)
    PY=$W/tabpfn-env-20261114/.venv/bin/python
    export OBS_RCT_TABPFN_WEIGHTS_DIR=$RUN/tabpfn-weights
    export TABPFN_MODEL_CACHE_DIR=$RUN/tabpfn-weights ;;
  tsr_legacy)
    PY=$W/obsrct-tsr-study-20261113-r01/environment-v1/bin/python
    export OBS_RCT_TABPFN_MODEL_PATH=$RUN/tabpfn-weights/tabpfn-v2-regressor.ckpt ;;
  stress)
    PY=$W/obsrct-public-upgrade-20261025-r01/environment-v1/bin/python
    export CORNET_SOURCE=$W/obsrct-public-upgrade-20261025-r01/upstreams/CorNet-official-v1 ;;
  *) echo "unknown GROUP $GROUP"; exit 2 ;;
esac
mkdir -p "$TMPDIR" "$RUN/logs"
