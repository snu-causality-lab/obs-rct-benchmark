# Wave 2 packet

[Scientific result](SCIENTIFIC_RESULT.md) · [plan](../../docs/V1_CORE_WAVE2.md) · configurations: [`native_r`](../../configs/v1_wave2_native_r.toml), [`gp`](../../configs/v1_wave2_gp.toml), [`tabpfn`](../../configs/v1_wave2_tabpfn.toml), [`tsr_legacy`](../../configs/v1_wave2_tsr_legacy.toml)

The producer is the freeze commit `80c9129` (package version 0.3.0a7), run from that commit's snapshot in four environments on GSDS:

| Group | Environment |
| --- | --- |
| `native_r` | The wave-1 Python environment, R with the pinned intRlearner and grf builds |
| `gp` | The GP pilot's environment (Python 3.11, GPy 1.13.2) |
| `tabpfn` | `tabpfn==10.0.0` on RTX 3090 GPUs, checkpoints pinned by SHA-256 |
| `tsr_legacy` | The TSR study's environment (`tabpfn==2.0.9`) on RTX 3090 GPUs |

Package lists for each environment are in `evidence/environments/`.

| File | Contents |
| --- | --- |
| `replicate_scores/` | All 182,528 cells, one file per group and preset: status, CATE MSE, failure stage and type, and for `rct_selected` the chosen candidate |
| `condition_scores.csv` | Per-condition mean error and success counts for every recipe |
| `contrasts.csv` | Every declared contrast: paired mean, interval, p-value, Holm-adjusted p-value, verdict and failure-as-loss win count |
| `headline.md` | Verdict counts per family, transport conditions separate |
| `identity_check.json` | Reproduction of wave 1's reference scores per group and the OBS-size identity of RCT-only records |
| `analysis.json` | Coverage, the 192 failures and family summaries |
| `checks/` | Independent arithmetic and the per-group reproduction reports |
| `as_frozen/` | The analysis before the slow-node rerun: headline, contrasts, analysis, identity and arithmetic. It is byte-identical to the final analysis; `changed_cells.csv` is empty |
| `evidence/` | Job scripts including the erratum runner, job record, rerun lists, shard manifests, environment package lists, data check, the Integrative R deadline trace and the split script |
| `DELIVERY_MANIFEST.json` | Freeze commit, code archive hash, job IDs and the SHA-256 of each analysis file as exported, including the unsplit per-seed scores |

The analyzer wrote the per-seed scores as one 19.9 MB file. The repository keeps files under 1 MiB, so it was split by group and preset at publication. `python results/v1_wave2/evidence/split_replicate_scores.py --check` rejoins the files and confirms the export's SHA-256 byte for byte. Full records, run manifests, logs and the moved-aside attempts remain on GSDS under `obsrct-v1-wave2-20261114-r01`.

## Job record

- **Main run.** Arrays 822652–822657. Reruns of infrastructure failures (failed environment checks, unavailable runtime provenance, slow-storage nodes) are listed in `evidence/JOBS.txt` and `evidence/reruns/`.
- **Analysis.**
  - Job 831628 was refused by the source check before writing any output.
  - Job 832360 ran the analysis through the erratum runner. That is the `as_frozen/` analysis.
  - Job 832671 then reran seven slow-node shards whole on normal nodes, and job 832672 produced the final analysis. A first submission of the rerun (832628) found the old status files and did nothing, so its dependent analysis (832630) refused the incomplete set; the status files were then moved aside too.
  - Both errata are in the plan's [revisions and errata](../../docs/V1_CORE_WAVE2.md#revisions-and-errata).
- **Reproduction.** Jobs 831630–831633, one per group.
- **Diagnostic.** Job 832380 traced the Integrative R deadline on the CorNet design.

## Reproduce

Install the group's extras at the freeze commit, as described in the main README and [native settings](../../docs/V03_ADAPTERS.md). Write the shards with [`make_v1_wave2_configs.py`](../../scripts/make_v1_wave2_configs.py) and run each with `python -m obs_rct_benchmark.cli run --config <shard> --output <dir>`; `evidence/jobs/` holds the exact driver and job scripts used. Then:

From the repository root, with the package and its scripts importable:

```sh
export PYTHONPATH=src:scripts
python results/v1_wave2/evidence/jobs/analyze_erratum.py scripts/analyze_v1_wave2.py \
  --configs configs/v1_wave2_{native_r,gp,tabpfn,tsr_legacy}.toml \
  --runs <all 7,936 shard directories> --wave1 results/v1_core --output analysis
python scripts/verify_v1_wave2.py arithmetic --analysis analysis --runs <...> --output arithmetic.json
python scripts/verify_v1_wave2.py reproduce --group <group> --config configs/v1_wave2_<group>.toml --runs <that group's shards> --output reproduce.json
```

The analyzer refuses a missing shard, a cell without a terminal record, or shards whose settings, record identities, source or environment differ within a group.
