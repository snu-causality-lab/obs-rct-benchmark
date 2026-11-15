# Stress addendum packet

[Scientific result](SCIENTIFIC_RESULT.md) · [plan](../../docs/V1_STRESS_WAVE2.md) · configurations: [`native_r`](../../configs/v1_stress_wave2_native_r.toml), [`gp`](../../configs/v1_stress_wave2_gp.toml), [`tabpfn`](../../configs/v1_stress_wave2_tabpfn.toml), [`tsr_legacy`](../../configs/v1_stress_wave2_tsr_legacy.toml)

The producer is the freeze commit `61cb584` (package version 0.3.0a7), run from that commit's snapshot in four environments on GSDS:

| Group | Environment |
| --- | --- |
| `native_r` | The wave-1 Python environment, R with the pinned intRlearner and grf builds |
| `gp` | The GP pilot's environment (Python 3.11, GPy 1.13.2) |
| `tabpfn` | `tabpfn==10.0.0` on RTX 3090 GPUs, checkpoints pinned by SHA-256 |
| `tsr_legacy` | The TSR study's environment (`tabpfn==2.0.9`) on RTX 3090 GPUs |

Package lists for each environment are in `evidence/environments/`.

| File | Contents |
| --- | --- |
| `replicate_scores/` | All 94,208 cells, one file per group and dial: status, CATE MSE, failure stage and type, and for `rct_selected` the chosen candidate |
| `condition_scores.csv` | Per-condition mean error and success counts for every recipe |
| `contrasts.csv` | Every declared contrast: paired mean, interval, p-value, Holm-adjusted p-value and verdict |
| `readout.csv` | Each primary family's verdicts from level 0 to 3 for each dial and trial size, with the declared state |
| `dose_response.csv` | Every recipe's error relative to `rct_selected` by dial and level: ratio of means and geometric mean ratio with a pointwise 95% interval |
| `headline.md` | Verdict counts per family, extrapolation conditions separate, and the reference identity per group |
| `identity_check.json` | Reproduction of stress study 1's reference scores and the RCT-only identity across levels |
| `analysis.json` | Coverage and family summaries |
| `checks/` | Independent arithmetic and the per-group reproduction reports |
| `as_frozen/` | The analysis before the slow-node rerun: headline, contrasts, readout, analysis, identity, arithmetic, and the 15 cells that differ |
| `evidence/` | Job scripts including the erratum runner, job record, rerun lists, shard manifests, environment package lists, data check and the split script |
| `DELIVERY_MANIFEST.json` | Freeze commit, code archive hash, job IDs and the SHA-256 of each analysis file as exported, including the unsplit per-seed scores |

The analyzer wrote the per-seed scores as one 8.2 MB file. The repository keeps files under 1 MiB, so it was split by group and dial at publication. `python results/stress_wave2/evidence/split_replicate_scores.py --check` rejoins the files and confirms the export's SHA-256 byte for byte. Full records, run manifests, logs and the moved-aside attempts remain on GSDS under `obsrct-v1-wave2-20261114-r01`.

## Job record

- **Main run.** Arrays 823396–823399, one per group. Reruns of infrastructure failures are listed in `evidence/JOBS.txt` and `evidence/reruns/`.
- **Analysis.**
  - Job 831629 was refused by the source check before writing any output.
  - Job 832361 ran the analysis through the erratum runner. That is the `as_frozen/` analysis.
  - The four slow-node shards were then replaced by job 832379's whole reruns on a normal node, and job 832629 produced the final analysis.
  - Both errata are in the plan's [revisions and errata](../../docs/V1_STRESS_WAVE2.md#revisions-and-errata).
- **Reproduction.** Jobs 831634–831637, one per group.

## Reproduce

Install the group's extras at the freeze commit, as described in the main README and [native settings](../../docs/V03_ADAPTERS.md). Write the shards with [`make_stress_wave2_configs.py`](../../scripts/make_stress_wave2_configs.py) and run each with `python -m obs_rct_benchmark.cli run --config <shard> --output <dir>`; `evidence/jobs/` holds the exact driver and job scripts used. Then:

From the repository root, with the package and its scripts importable:

```sh
export PYTHONPATH=src:scripts
python results/stress_wave2/evidence/jobs/analyze_erratum.py scripts/analyze_stress_wave2.py \
  --configs configs/v1_stress_wave2_{native_r,gp,tabpfn,tsr_legacy}.toml \
  --runs <all 2,048 shard directories> --stress1 results/stress_cornet --output analysis
python scripts/verify_stress_wave2.py arithmetic --analysis analysis --runs <...> --stress1 results/stress_cornet --output arithmetic.json
python scripts/verify_stress_wave2.py reproduce --group <group> --config configs/v1_stress_wave2_<group>.toml --runs <that group's shards> --output reproduce.json
```

The analyzer refuses a missing shard, a cell without a terminal record, or shards whose settings, record identities, source or environment differ within a group.
