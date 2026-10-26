# Installed environment specifications

**Recorded 2026-10-26; fresh-prefix reconstruction passed on GSDS.** These specifications describe the Linux x86_64 environment used for the native fits and CPU profiles: Python 3.11.17 and R 4.3.3. The packet's full-test run used the separate Python 3.12.3 environment recorded in [tests_summary.json](../tests_summary.json).

[conda-linux-64.explicit.txt](conda-linux-64.explicit.txt) contains `@EXPLICIT` followed by the original **193 conda-forge package URLs**, preserving every version and build. [pip-supplement.txt](pip-supplement.txt) retains the original **18 exact PyPI version entries**. The 15 Python entries with conda build-directory origins are supplied by the conda specification; the editable benchmark installation is supplied from the recorded source. Original-record hashes and these extraction rules are in [provenance.json](provenance.json).

On an allocated Linux x86_64 compute node, set `REPLAY_PREFIX` to a fresh absolute environment directory. From the repository root, use a source checkout matching the 16 package-file hashes in [producer_identity.json](../producer_identity.json):

```bash
: "${REPLAY_PREFIX:?Set REPLAY_PREFIX to a fresh absolute environment directory}"
: "${MAMBA_ROOT_PREFIX:?Set MAMBA_ROOT_PREFIX to a shared-work cache directory}"
micromamba --no-rc create --yes -p "$REPLAY_PREFIX" \
  --file results/native_adapter_admission/environment/conda-linux-64.explicit.txt
"$REPLAY_PREFIX/bin/python" -m pip install --no-deps \
  -r results/native_adapter_admission/environment/pip-supplement.txt
"$REPLAY_PREFIX/bin/python" -m pip install --no-deps -e .

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1
export INTEGRATIVE_R_RSCRIPT="$REPLAY_PREFIX/bin/Rscript"
```

The separately obtained author checkouts and their source environment variables remain required. Their exact revisions and setup are in [V03_ADAPTERS.md](../../../docs/V03_ADAPTERS.md); actual imported distributions and R namespace metadata are in the [runtime fingerprint](../runtime_fingerprint.json). Use the recorded inputs, seeds and numerical rules when replaying the linked native-check drivers.

The commands use the documented [micromamba explicit-file format](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html#explicit-spec-files) and [pip installation options](https://pip.pypa.io/en/stable/cli/pip_install/). GSDS job 720512 reconstructed a fresh prefix from these two specifications and reran all native checks. Predictions and retained Integrative R parameters matched the original environment exactly (maximum absolute discrepancy 0; unchanged tolerances 1e-10 absolute / 1e-8 relative). See [reconstruction.json](reconstruction.json). The [recorded comparison driver](compare_recorded_replay.py) reads the original/reconstructed arrays retained on GSDS; those arrays and full installation logs remain server evidence.
